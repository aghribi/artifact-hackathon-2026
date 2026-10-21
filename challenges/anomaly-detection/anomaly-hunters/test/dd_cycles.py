"""Save the data referenced by dd.list as one pandas DataFrame per cycle_id.

Usage:
    python dd_cycles.py
    from dd_cycles import CycleDataFrames
    dfs_by_cycle = CycleDataFrames("dd_by_cycle")
    df = dfs_by_cycle[15643300178]

All original columns and full waveform arrays are retained. Repeated captures
are retained too, with source columns identifying the original archive/row.
"""

import argparse
import json
import tarfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

import pandas as pd


def resolve_archive(path, list_path):
    """Accept paths from either the host or the notebook's /workspace mount."""
    path = Path(path)
    if not path.is_absolute():
        path = list_path.parent / path
    candidates = [path]
    if path.is_relative_to("/workspace/lustre"):
        candidates.append(Path("/") / path.relative_to("/workspace"))
    elif path.is_relative_to("/lustre"):
        candidates.append(Path("/workspace") / path.relative_to("/"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Archive from {list_path} not found: {path}")


def scan_sources(list_path):
    """Read only schemas and cycle IDs to plan when each cycle is complete."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    archives, sources, schemas = [], [], []
    last_source, expected = {}, Counter()
    for line_number, line in enumerate(list_path.read_text().splitlines(), 1):
        line = line.strip()
        if line and not line.startswith("#"):
            archives.append((line_number, resolve_archive(line, list_path)))
    if not archives:
        raise ValueError(f"No archives listed in {list_path}")
    for line_number, path in archives:
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".parquet"):
                    continue
                with archive.extractfile(member) as stream:
                    parquet = pq.ParquetFile(stream)
                    schema = parquet.schema_arrow.remove_metadata()
                    required = {"cycle_id", "cycle_ts", "pv", "data_ts", "data_array"}
                    if not required.issubset(schema.names):
                        raise ValueError(f"Required columns missing: {path}:{member.name}")
                    ids = parquet.read(columns=["cycle_id"]).column("cycle_id")
                    if ids.null_count:
                        raise ValueError(f"Null cycle_id in {path}:{member.name}")
                    counts = Counter(ids.to_pylist())
                    source_index = len(sources)
                    for cycle_id in counts:
                        last_source[cycle_id] = source_index
                    expected.update(counts)
                    schemas.append(schema)
                    sources.append(dict(archive=path, member=member.name,
                                        list_line=line_number, counts=counts,
                                        rows=parquet.metadata.num_rows))
    if not sources:
        raise ValueError(f"No Parquet data found in {list_path}")
    schema = pa.unify_schemas(schemas)
    leading = ["cycle_id", "cycle_ts", "pv", "data_ts", "data_array"]
    names = leading + [name for name in schema.names if name not in leading]
    schema = pa.schema([schema.field(name) for name in names])
    reserved = {"source_archive", "source_parquet", "source_list_line", "source_row"}
    if reserved.intersection(schema.names):
        raise ValueError("Source data already contains reserved provenance columns")
    return archives, sources, schema, last_source, expected


def build_cycle_dataframes(list_path="dd.list", output_dir="dd_by_cycle", batch_size=128):
    """Write cycle_id.pkl files and an index, without loading all cycles at once.

    Each row is one original PV capture; data_array remains a NumPy float32
    array. source_row is the zero-based row in the original Parquet member.
    Existing output directories are never overwritten.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    list_path = Path(list_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    archives, sources, schema, last_source, expected = scan_sources(list_path)
    print(f"{len(archives)} archives / {len(sources)} Parquet files / "
          f"{sum(expected.values()):,} rows / {len(expected)} cycles", flush=True)
    output_dir.mkdir(parents=True)
    pending, summary = defaultdict(list), []
    processed = 0
    source_index = 0
    for line_number, path in archives:
        with tarfile.open(path, "r:*") as archive:
            # Member order and list order are the same as the metadata scan.
            for member in archive:
                if not member.isfile() or not member.name.endswith(".parquet"):
                    continue
                source = sources[source_index]
                if (path, member.name) != (source["archive"], source["member"]):
                    raise RuntimeError("Source archive changed after metadata scan")
                offset = 0
                with archive.extractfile(member) as stream:
                    parquet = pq.ParquetFile(stream)
                    for batch in parquet.iter_batches(batch_size=batch_size, use_threads=False):
                        table = pa.Table.from_batches([batch]).select(schema.names).cast(schema)
                        frame = table.to_pandas()
                        frame["source_archive"] = str(path)
                        frame["source_parquet"] = member.name
                        frame["source_list_line"] = line_number
                        frame["source_row"] = range(offset, offset + len(frame))
                        for cycle_id, group in frame.groupby("cycle_id", sort=False):
                            pending[int(cycle_id)].append(group.copy())
                        offset += len(frame)
                if offset != source["rows"]:
                    raise RuntimeError(f"Source row count changed: {path}:{member.name}")
                for cycle_id in source["counts"]:
                    if last_source[cycle_id] != source_index:
                        continue
                    cycle_df = pd.concat(pending.pop(cycle_id), ignore_index=True)
                    if len(cycle_df) != expected[cycle_id]:
                        raise RuntimeError(f"Row count mismatch for cycle_id={cycle_id}")
                    filename = f"{cycle_id}.pkl"
                    cycle_df.to_pickle(output_dir / filename, protocol=5)
                    arrays = cycle_df["data_array"]
                    summary.append(dict(
                        cycle_id=cycle_id, cycle_ts=cycle_df["cycle_ts"].min(),
                        rows=len(cycle_df), unique_pvs=cycle_df["pv"].nunique(),
                        source_archives=cycle_df["source_archive"].nunique(),
                        waveform_samples=sum(len(a) for a in arrays if a is not None),
                        empty_or_null_waveforms=sum(a is None or len(a) == 0 for a in arrays),
                        file=filename,
                    ))
                    del cycle_df
                processed += offset
                source_index += 1
        print(f"[{source_index}/{len(sources)} files] {processed:,} rows read; "
              f"{len(summary)}/{len(expected)} DataFrames saved", flush=True)
    if pending or processed != sum(expected.values()) or len(summary) != len(expected):
        raise RuntimeError("Incomplete conversion")
    cycle_summary = pd.DataFrame(summary).sort_values("cycle_id").reset_index(drop=True)
    cycle_summary.to_csv(output_dir / "cycle_summary.csv", index=False)
    manifest = dict(
        list_path=str(list_path), archives=len(archives), parquet_files=len(sources),
        rows=processed, cycles=len(summary), format="pandas.DataFrame pickle (protocol 5)",
        duplicates="preserved", columns=schema.names + [
            "source_archive", "source_parquet", "source_list_line", "source_row"],
        sources=[dict(archive=str(s["archive"]), member=s["member"],
                      list_line=s["list_line"], rows=s["rows"]) for s in sources],
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(summary)} DataFrames to {output_dir}", flush=True)
    return cycle_summary


class CycleDataFrames(Mapping):
    """Dictionary-like access to the saved DataFrames, loading one per access.

    No frames are cached, so this object itself does not hold waveform data.
    Use pd.read_pickle on an individual .pkl file for direct pandas access.
    """

    def __init__(self, directory="dd_by_cycle"):
        self.directory = Path(directory).resolve()
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Completed dataset not found: {self.directory}")
        self.manifest = json.loads(manifest_path.read_text())
        self.summary = pd.read_csv(self.directory / "cycle_summary.csv",
                                   parse_dates=["cycle_ts"])
        self._files = dict(zip(self.summary["cycle_id"].map(int), self.summary["file"]))

    def __getitem__(self, cycle_id):
        return pd.read_pickle(self.directory / self._files[cycle_id])

    def __iter__(self):
        return iter(self._files)

    def __len__(self):
        return len(self._files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", default="dd.list", dest="list_path")
    parser.add_argument("--output", default="dd_by_cycle", dest="output_dir")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    build_cycle_dataframes(args.list_path, args.output_dir, args.batch_size)
