"""The Archiver over the 30 minutes before each PM capture: every pv of every family, as the Archiver stored it.

A PM capture is taken when the beam trips, and its time is event_ts of its waveforms. For each capture the window
runs from event_ts minus 30 minutes to event_ts. The Archiver stores one row per pv each time its value changes, so
a value holds until the next row. Each window gets the rows of the window and, per pv, the last row before it: the
value every pv holds when the window opens. Those rows are flagged `held`, so a pv that does not change in the
30 minutes still has its value.

    <out>/pm_windows.parquet           one row per PM capture: event_id, event_ts, window_start, window_end,
                                       archiver_from (start of the Archiver data that runs up to the capture),
                                       covered_s (seconds of the window inside an Archiver folder), rows, pvs, pvs
                                       that change in the window, first and last sample in the window

Rows of a window from before archiver_from are the last change of a pv before the Archiver folder starts, not a
record of those minutes.
    <out>/<event_id>/<family>.parquet  one row per pv and sample of one Archiver family (LLRF, MAGNETS, CRYO, ...):
                                       event_id, family, pv, ts, seconds_before (event_ts minus ts), value, text,
                                       severity, status, held

`value` holds the numeric Archiver columns (data_Float64, data_Int64), `text` any other data_ column. All times are
UTC. Read one capture with pd.read_parquet("<out>/<event_id>"), one family of one capture with
pd.read_parquet("<out>/<event_id>/LLRF.parquet"), one family over all captures with
pd.concat(pd.read_parquet(f) for f in Path(out).glob("*/LLRF.parquet")).

    python ess_pm_archiver_window.py <raw> <out> [minutes]
"""

import io
import re
import shutil
import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

NS = 1_000_000_000
COLUMNS = ["event_id", "family", "pv", "ts", "seconds_before", "value", "text", "severity", "status", "held"]

# Filled in the parent before the workers fork: one row per PM capture, and where the parts go.
WINDOWS = pd.DataFrame()
PARTS = Path()


def tar_members(path: Path, suffix: str) -> list[tarfile.TarInfo]:
    with tarfile.open(path) as tar:
        return [m for m in tar.getmembers() if m.name.endswith(suffix)]


def read_at(path: str, offset: int, size: int) -> pq.ParquetFile:
    """A parquet file read from its offset in an uncompressed tar."""
    with open(path, "rb") as f:
        f.seek(offset)
        return pq.ParquetFile(io.BytesIO(f.read(size)))


def capture_time(path: Path) -> dict:
    """event_id and event_ts of a PM capture, from the smallest parquet of its tar."""
    member = min(tar_members(path, ".parquet"), key=lambda m: m.size)
    pf = read_at(str(path), member.offset_data, member.size)
    row = pf.read_row_group(0, columns=["event_id", "event_ts"]).slice(0, 1).to_pylist()[0]
    ts = pd.Timestamp(row["event_ts"])
    return {"event_id": int(row["event_id"]), "event_ts": ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC"),
            "capture_window": member.name.split("/")[1], "tar": path.name}


def last_sample(job: tuple[str, int, int]) -> int:
    """Time in ns of the last sample of one Archiver file."""
    pf = read_at(*job)
    return pc.max(pf.read(columns=["data_ts"])["data_ts"]).cast(pa.timestamp("ns", "UTC")).value or 0


def folder_spans(members: list[tuple[str, tarfile.TarInfo]], workers: int) -> dict[str, tuple[int, int]]:
    """Start and end in ns of each Archiver folder Archiver_Data/<start>_<end>/.

    The folder names are in UTC for 2025 and in the local time of ESS for 2026 (folder 20260629_1825_20260629_2225
    holds 16:25 to 20:25 UTC), so the end is the last sample of the folder and the start is that end minus the
    length the name gives."""
    with Pool(workers) as pool:
        last = pool.map(last_sample, [(path, m.offset_data, m.size) for path, m in members], chunksize=4)
    spans = {}
    for (_, m), ts in zip(members, last):
        folder = Path(m.name).parent.name
        a, b, c, d = folder.split("_")
        length = (pd.Timestamp(f"{c}{d}") - pd.Timestamp(f"{a}{b}")).value
        end = max(spans.get(folder, (0, 0))[1], ts)
        spans[folder] = (end - length, end)
    return spans


def coverage(spans: list[tuple[int, int]], start: int, end: int) -> tuple[pd.Timestamp, float]:
    """Start of the Archiver data that runs without a break up to end, and the seconds of start to end it covers."""
    pieces = sorted((max(a, start), min(b, end)) for a, b in spans if a < end and b > start)
    merged = []
    for a, b in pieces:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    reaches_end = merged and merged[-1][1] >= end
    return (pd.Timestamp(merged[-1][0], tz="UTC") if reaches_end else pd.NaT,
            sum(b - a for a, b in merged) / NS)


def window_rows(job: tuple[int, str, int, int, str, list[int]]) -> list[str]:
    """Rows of one Archiver file that fall in a window or are the last of their pv before it, one part per window.

    An Archiver folder starts each pv with its last change before the folder, however old: a pv that never changes
    has that one row. So the last row before a window, in a folder that overlaps the window, is the value held when
    the window opens."""
    n, tar_path, offset, size, family, windows = job
    pf = read_at(tar_path, offset, size)
    names = pf.schema_arrow.names
    data = [c for c in names if c.startswith("data_") and c != "data_ts"]
    t = pf.read(columns=["pv", "data_ts", "severity", "status"] + data)
    numeric = [c for c in data if pa.types.is_integer(t[c].type) or pa.types.is_floating(t[c].type)]
    other = [c for c in data if c not in numeric]
    value = pc.coalesce(*[t[c].cast(pa.float64()) for c in numeric]) if numeric else pa.nulls(len(t), pa.float64())
    text = pc.coalesce(*[t[c].cast(pa.string()) for c in other]) if other else pa.nulls(len(t), pa.string())
    d = pd.DataFrame({
        "pv": t["pv"].to_pandas(), "ts": t["data_ts"].cast(pa.timestamp("ns", "UTC")).to_pandas(),
        "value": value.to_pandas(), "text": text.to_pandas(),
        "severity": t["severity"].to_pandas(), "status": t["status"].to_pandas()})
    d = d.sort_values("ts", kind="stable", ignore_index=True)
    ts = d.ts.astype("int64").to_numpy()
    if not len(ts):
        return []
    written = []
    for w in WINDOWS.iloc[windows].itertuples():
        start, end = w.window_start.value, w.event_ts.value
        inside = d[(ts >= start) & (ts <= end)]
        before = d[ts < start].drop_duplicates("pv", keep="last")
        if inside.empty and before.empty:
            continue
        part = pd.concat([before.assign(held=True), inside.assign(held=False)], ignore_index=True)
        part.insert(0, "family", family)
        part.insert(0, "event_id", w.event_id)
        out = PARTS / str(w.event_id) / f"{n}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(out, index=False)
        written.append(str(out))
    return written


def build_window(w: dict, out: Path) -> dict:
    """One capture: its parts joined, the held value of each pv taken from the latest file, samples counted once,
    one file per family."""
    files = sorted((PARTS / str(w["event_id"])).glob("*.parquet"))
    if not files:
        return {"event_id": w["event_id"], "rows": 0, "pvs": 0, "pvs_changing": 0, "first_ts": pd.NaT, "last_ts": pd.NaT}
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    held = d[d.held].sort_values("ts").drop_duplicates("pv", keep="last")
    # The same sample sits in every Archiver folder that covers its time.
    inside = d[~d.held].drop_duplicates(["pv", "ts"])
    d = pd.concat([held, inside], ignore_index=True).sort_values(["ts", "pv"], ignore_index=True)
    d["seconds_before"] = (w["event_ts"] - d.ts).dt.total_seconds()
    d["event_id"] = d.event_id.astype("int64")
    d[["family", "pv"]] = d[["family", "pv"]].astype("category")
    for family, part in d.groupby("family", observed=True):
        (out / str(w["event_id"])).mkdir(exist_ok=True)
        part[COLUMNS].to_parquet(out / str(w["event_id"]) / f"{family}.parquet", index=False, compression="zstd")
    shutil.rmtree(PARTS / str(w["event_id"]), ignore_errors=True)
    return {"event_id": w["event_id"], "rows": len(d), "pvs": d.pv.nunique(), "pvs_changing": inside.pv.nunique(),
            "first_ts": inside.ts.min() if len(inside) else pd.NaT,
            "last_ts": inside.ts.max() if len(inside) else pd.NaT}


def main(raw: Path, out: Path, minutes: float = 30, workers: int = 32) -> None:
    global WINDOWS, PARTS
    out.mkdir(parents=True, exist_ok=True)
    PARTS = out / "_parts"
    shutil.rmtree(PARTS, ignore_errors=True)

    pm_tars = sorted(raw.glob("SDS_Data_*_PM_*.tar"))
    with Pool(workers) as pool:
        captures = pd.DataFrame(pool.map(capture_time, pm_tars))
    captures["window_start"] = captures.event_ts - pd.Timedelta(minutes=minutes)
    WINDOWS = captures.sort_values("event_ts", ignore_index=True)
    print(f"{len(WINDOWS)} PM captures, {WINDOWS.event_ts.min()} to {WINDOWS.event_ts.max()}, {minutes} min windows")

    starts = WINDOWS.window_start.map(lambda t: t.value).to_numpy()
    ends = WINDOWS.event_ts.map(lambda t: t.value).to_numpy()
    members = [(str(path), m) for path in sorted(raw.glob("Archiver_Data_*.tar")) for m in tar_members(path, ".parquet")]
    spans = folder_spans(members, workers)
    for folder, (first, last) in sorted(spans.items()):
        print(f"folder {folder}: {pd.Timestamp(first, tz='UTC')} to {pd.Timestamp(last, tz='UTC')}")
    jobs = []
    for path, member in members:
        first, last = spans[Path(member.name).parent.name]
        windows = [int(i) for i in ((starts <= last) & (ends >= first)).nonzero()[0]]
        family = re.sub(r"(_\d+)+$", "", Path(member.name).stem)
        if windows:
            jobs.append((len(jobs), path, member.offset_data, member.size, family, windows))
    print(f"{len(jobs)} Archiver files overlap a window, {len({j[4] for j in jobs})} families")
    with Pool(workers) as pool:
        parts = sum(1 for written in pool.imap_unordered(window_rows, jobs, chunksize=2) for _ in written)
    print(f"{parts} parts written")

    with Pool(workers) as pool:
        counts = pool.starmap(build_window, [(w, out) for w in WINDOWS.to_dict("records")])
    shutil.rmtree(PARTS, ignore_errors=True)
    index = WINDOWS.merge(pd.DataFrame(counts), on="event_id")
    index["window_end"] = index.event_ts
    covered = pd.DataFrame([coverage(list(spans.values()), a, b) for a, b in zip(starts, ends)],
                           columns=["archiver_from", "covered_s"])
    index[["archiver_from", "covered_s"]] = covered[["archiver_from", "covered_s"]].astype({"covered_s": float})
    index = index[["event_id", "event_ts", "window_start", "window_end", "archiver_from", "covered_s", "capture_window",
                   "tar", "rows", "pvs", "pvs_changing", "first_ts", "last_ts"]]
    index.to_parquet(out / "pm_windows.parquet", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(index.drop(columns=["tar", "window_end"]).to_string(index=False))
    print(f"captures without Archiver rows in their window: {(index.pvs_changing == 0).sum()}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), *(float(a) for a in sys.argv[3:4]))
