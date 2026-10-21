"""Per-pulse features of the ESS cavity waveforms in the SDS captures, with their timing against the beam
and the beam trips.

Input is the raw directory holding SDS_Data_*_<DD|PM>_*.tar and Archiver_Data_*.tar. DD captures are taken on demand
during operation and hold one cycle in fourteen; PM captures are taken when the beam trips and hold consecutive
cycles up to the trip. Both are read the same way. A capture parquet is long format:
one row per pv and cycle, the waveform in data_array. A cavity has six waveforms per cycle:
<cavity>:Dwn0 (field), Dwn5 (forward), Dwn6 (reflected), each as Cmp0 (amplitude) and Cmp1 (phase, degrees).

    <out>/cavity_pulses.parquet        one row per event, cycle and cavity
    <out>/beam_pulses.parquet          one row per event, cycle and BCM device
    <out>/context_correlation.parquet  Spearman and Pearson r of each pulse feature against beam, trip and cycle context
    <out>/mps_transitions.parquet      one row each time a machine protection interlock changes: pv, data_ts, value (0 = not ok)
    <out>/cavity_comovement.parquet    Spearman and Pearson r of a feature between two cavities, the second 0, 1 or 2 captured cycles
                                       later, as deviations from the mean of the capture

Pulse features, on the field amplitude unless named otherwise. Lengths are in samples, *_us in microseconds
after cycle_ts.

    bins        field_amp_bins, fwd_amp_bins, refl_amp_bins, field_phase_bins, fwd_phase_bins, refl_phase_bins:
                bin means over the whole record, phases as circular means in degrees
    length      pulse_len (above half level), rise_len (10 to 90 %), flat_len (above 98 %), fall_len (90 to 10 %),
                decay_tau (exponential fit of the fall)
    flatness    over the flat top with 5 % trimmed at each end, relative to the level: flat_std_rel, flat_p2p_rel,
                flat_tilt_rel (linear drift end to end), flat_max_dev_rel and its position flat_max_dev_pos (0 to 1),
                flat_step_max_rel (largest jump between consecutive 1/64 blocks); phase_std_deg, phase_p2p_deg,
                phase_tilt_deg
    power       fwd_peak, fwd_flat, fwd_flat_std_rel, refl_flat, refl_fill_peak, refl_end_peak, refl_ratio
    precursor   field_end_after_rf_off (negative: the field left the flat top before the forward power went off);
                <feature>_drift: this cycle minus the median of the earlier half of the cycles of the same capture;
                shape_drift: RMS distance of field_amp_bins to the mean bins of that earlier half, relative to the level
    context     cycle_rank_from_last (0 = last cycle of the capture), seconds_to_event, seconds_to_next_trip,
                seconds_since_last_trip, beam_level, beam_start_us, beam_len_us, beam_after_flat_start_us

A trip is FBIS::beam_permit_ok going to 0 in the Archiver MPS files, which log an interlock only when it changes. Trip times are not bounded to the capture
window: the next trip can be days away.

    python ess_dd_features.py <raw> <out> <DD|PM> <n_bins> [<max_events>]

<max_events> keeps that many captures, evenly spread over the sorted file list.
"""

import os
import re
import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Sampling steps from PV_naming.xlsx of the challenge repository. Shorter cavity records are decimated by an
# undocumented factor, so they get no *_us columns.
RF_DT_US = 0.12
RF_SAMPLES = 35221
BCM_DT_US = 0.18

# A cavity PV reads <section>-<module>RFC:RFS-DIG-<n>01:Dwn<signal>-Cmp<component>SDS-..., for example
# HBL-010RFC:RFS-DIG-101:Dwn0-Cmp0SDS-PMData. The part before :Dwn names the cavity.
CAVITY_PV = r"(?P<cavity>(?P<section>[A-Za-z]+)-\d+RFC:RFS-DIG-\d+):Dwn(?P<dwn>[056])-Cmp(?P<cmp>[01])SDS-\w+"
BCM_PV = r"(?P<device>[\w-]+:PBI-BCM-\d+):"
SIGNAL = {"0": "field", "5": "fwd", "6": "refl"}
COMPONENT = {"0": "amp", "1": "phase"}
PERMIT = "FBIS::beam_permit_ok"
# Columns read next to the waveform. data_ts is the time of the first sample, cycle_ts the start of the machine
# cycle, event_ts the time of the capture; cycle_id counts machine cycles and event_id names the capture.
META = ["pv", "cycle_id", "event_id", "trig_code", "trig_delay", "beam_mode", "beam_state", "beam_present",
        "beam_len", "beam_current", "data_ts", "cycle_ts", "event_ts"]

TAIL = 50          # the last 1/TAIL of a record is the baseline
FLAT = 0.98        # flat top: field above this share of its level
FLAT_BLOCKS = 64   # the flat top is cut in this many blocks to find a step
MIN_PAIRS = 30     # a correlation needs this many pairs of values
LAGS = (0, 1, 2)   # captured cycles between the two cavities of a co-movement pair
METHODS = ("spearman", "pearson")
CAPTURE_TYPES = ("DD", "PM")   # DD: taken on demand during operation. PM: post mortem, taken when the beam trips.

# SHAPE: features of one pulse. DRIFT: the ones followed across the cycles of a capture. FEATURES: what is
# correlated against CONTEXT. COMOVE: what is correlated between cavities.
SHAPE = ["field_level", "pulse_len", "rise_len", "flat_len", "fall_len", "decay_tau", "flat_std_rel", "flat_p2p_rel",
         "flat_tilt_rel", "flat_max_dev_rel", "flat_step_max_rel", "phase_mean_deg", "phase_std_deg", "phase_p2p_deg",
         "phase_tilt_deg", "fwd_peak", "fwd_flat", "fwd_flat_std_rel", "refl_flat", "refl_fill_peak", "refl_end_peak",
         "refl_ratio", "field_end_after_rf_off"]
DRIFT = ["field_level", "flat_len", "flat_std_rel", "flat_tilt_rel", "phase_mean_deg", "decay_tau", "fwd_flat", "refl_ratio"]
FEATURES = SHAPE + [f"{c}_drift" for c in DRIFT] + ["shape_drift"]
CONTEXT = ["beam_current", "beam_level", "beam_after_flat_start_us", "seconds_to_next_trip", "seconds_since_last_trip",
           "cycle_rank_from_last"]
COMOVE = ["field_level", "flat_std_rel", "phase_mean_deg", "flat_len", "refl_ratio"]


def first_last(mask: np.ndarray) -> tuple[int, int]:
    """First and last sample at which the mask holds."""
    idx = np.flatnonzero(mask)
    return int(idx[0]), int(idx[-1])


def slope(y: np.ndarray) -> float:
    """Least-squares slope of y per sample."""
    x = np.arange(len(y)) - (len(y) - 1) / 2
    return float(x @ (y - y.mean()) / (x @ x))


def bin_means(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Mean of x over n_bins consecutive stretches that cover the whole record."""
    edges = np.linspace(0, len(x), n_bins + 1).astype(int)
    return np.add.reduceat(x, edges[:-1]) / np.diff(edges)


def unit(phase_deg: np.ndarray) -> np.ndarray:
    """Phase as a point on the unit circle, so that means and differences survive the jump from 180 to -180 degrees."""
    return np.exp(1j * np.deg2rad(phase_deg))


def pulse_features(sig: dict[str, np.ndarray], n_bins: int, start_us: float) -> dict:
    """Features of one cavity in one cycle. sig: field_amp, and what exists of the other five waveforms."""
    a = sig["field_amp"].astype(np.float64)
    n = len(a)
    # Baseline: the end of the record, long after the pulse. Level: the median of the samples near the top, taken
    # above the baseline. A record counts as a pulse when the level stands clear of the noise of the baseline.
    tail = a[-(n // TAIL):]
    base = float(np.median(tail))
    level = float(np.median(a[a >= base + 0.9 * (np.percentile(a, 99) - base)])) - base
    out = {"array_size": n, "field_base": base, "field_level": level, "has_pulse": bool(level > 5 * tail.std() and level > 0)}
    # Bins are kept for every record, pulse or not: a missing pulse is itself worth seeing.
    for name in ("field_amp", "fwd_amp", "refl_amp"):
        if name in sig:
            out[f"{name}_bins"] = bin_means(sig[name].astype(np.float64), n_bins).astype(np.float32)
    for name in ("field_phase", "fwd_phase", "refl_phase"):
        if name in sig:
            out[f"{name}_bins"] = np.rad2deg(np.angle(bin_means(unit(sig[name]), n_bins))).astype(np.float32)
    if not out["has_pulse"]:
        return out

    # rel is the field as a share of its level: 0 on the baseline, 1 on the flat top. Each length is the distance
    # between two crossings of a share: 10, 20, 50, 90 or 98 %.
    rel = (a - base) / level
    half_start, half_end = first_last(rel > 0.5)
    rise10, fall10 = first_last(rel > 0.1)
    rise90, fall90 = first_last(rel > 0.9)
    flat_start, flat_end = first_last(rel > FLAT)
    _, fall20 = first_last(rel > 0.2)
    # After the RF goes off the field falls as exp(-t / tau): the log of the fall is a line of slope -1 / tau.
    decay = slope(np.log(np.clip(rel[flat_end:fall20 + 1], 1e-6, None))) if fall20 - flat_end >= 8 else np.nan
    # start_us is the time of the first sample after cycle_ts, so start_us + index * dt places a crossing in the cycle.
    dt = RF_DT_US if n == RF_SAMPLES else np.nan
    out.update(pulse_len=half_end - half_start, rise_len=rise90 - rise10, flat_len=flat_end - flat_start,
               fall_len=fall10 - fall90, decay_tau=-1 / decay if decay < 0 else np.nan,
               pulse_start_us=start_us + half_start * dt, flat_start_us=start_us + flat_start * dt,
               flat_end_us=start_us + flat_end * dt)

    # Flatness is measured inside the flat top, with 5 % cut at each end to stay clear of the two corners.
    trim = (flat_end - flat_start) // 20
    top = slice(flat_start + trim, flat_end - trim + 1)
    w = rel[top]
    if len(w) < FLAT_BLOCKS:
        return out
    dev = np.abs(w - np.median(w))
    # A step shows as a jump between the means of two neighbouring blocks, which sample noise does not produce.
    blocks = w[:len(w) // FLAT_BLOCKS * FLAT_BLOCKS].reshape(FLAT_BLOCKS, -1).mean(axis=1)
    out.update(flat_std_rel=float(w.std()), flat_p2p_rel=float(w.max() - w.min()), flat_tilt_rel=slope(w) * len(w),
               flat_max_dev_rel=float(dev.max()), flat_max_dev_pos=float(dev.argmax() / len(w)),
               flat_step_max_rel=float(np.abs(np.diff(blocks)).max()))

    # Phase, forward and reflected power are read over the same flat-top window as the field, so they are used only
    # when their record has the length of the field record.
    if len(sig.get("field_phase", ())) == n:
        z = unit(sig["field_phase"][top])
        mean = z.mean()
        d = np.rad2deg(np.angle(z * np.conj(mean)))
        out.update(phase_mean_deg=float(np.rad2deg(np.angle(mean))), phase_std_deg=float(d.std()),
                   phase_p2p_deg=float(d.max() - d.min()), phase_tilt_deg=slope(d) * len(d))
    fwd_flat = np.nan
    if len(sig.get("fwd_amp", ())) == n:
        f = sig["fwd_amp"].astype(np.float64)
        f -= np.median(f[-(n // TAIL):])
        fwd_flat = float(np.median(f[top]))
        out.update(fwd_peak=float(f.max()), fwd_peak_idx=int(f.argmax()), fwd_flat=fwd_flat)
        if fwd_flat > 0:
            # The RF goes off where the forward power last stands above half of its flat-top value.
            _, rf_off = first_last(f > 0.5 * fwd_flat)
            out.update(fwd_flat_std_rel=float(f[top].std() / fwd_flat), rf_off_us=start_us + rf_off * dt,
                       field_end_after_rf_off=flat_end - rf_off)
    if len(sig.get("refl_amp", ())) == n:
        r = sig["refl_amp"].astype(np.float64)
        r -= np.median(r[-(n // TAIL):])
        refl_flat = float(np.median(r[top]))
        # The reflected power peaks twice: while the cavity fills, and when the RF goes off and the cavity empties.
        out.update(refl_flat=refl_flat, refl_fill_peak=float(r[:flat_start + 1].max()), refl_end_peak=float(r[flat_end:].max()),
                   refl_ratio=refl_flat / fwd_flat if fwd_flat > 0 else np.nan)
    return out


def beam_features(a: np.ndarray, start_us: float) -> dict:
    """Features of one BCM waveform. The beam pulse is short against the record, so the median is the baseline."""
    d = a.astype(np.float64) - np.median(a)
    # Some devices read the beam as a negative pulse; it is flipped so that the beam always points up.
    polarity = -1 if -d.min() > d.max() else 1
    d *= polarity
    peak = float(d.max())
    # Noise as 1.4826 times the median absolute deviation, which the beam pulse itself does not inflate.
    noise = 1.4826 * np.median(np.abs(d)) or d.std()
    out = {"array_size": len(a), "beam_polarity": polarity, "beam_peak": peak, "has_beam": bool(peak > 8 * noise)}
    if not out["has_beam"]:
        return out
    level = float(np.median(d[d > 0.5 * peak]))
    start, end = first_last(d > 0.5 * level)
    # The beam pulse with a tenth cut at each end, for its flatness.
    core = d[start + (end - start) // 10:end - (end - start) // 10 + 1]
    out.update(beam_level=level, beam_start_us=start_us + start * BCM_DT_US, beam_len_us=(end - start) * BCM_DT_US,
               beam_flat_std_rel=float(core.std() / level), beam_charge=float(d[start:end + 1].sum() * BCM_DT_US))
    return out


def parquet_rows(source, capture_window: str, n_bins: int) -> tuple[list[dict], list[dict]]:
    """Rows of one DD parquet: one per cavity and cycle, one per BCM device and cycle.

    The parquet has one row per pv and cycle; the waveform of that pv in that cycle is the list data_array.
    source is a path or an open file.
    """
    cavity_rows, beam_rows = [], []
    pf = pq.ParquetFile(source)
    # The names are read first: a file with no cavity or beam current waveform (the BPM files) is left unread.
    names = pf.read(columns=["pv"]).column("pv").to_pandas()
    if names.str.extract(CAVITY_PV).cavity.isna().all() and names.str.extract(BCM_PV).device.isna().all():
        return cavity_rows, beam_rows
    table = pf.read(columns=META + ["data_array"])
    meta = table.select(META).to_pandas()
    # data_array is stored as one long array of samples plus the offset at which each row starts. Slicing
    # that array gives the waveform of a row without turning thirty thousand samples into Python objects.
    arrays = table.column("data_array").combine_chunks()
    offsets, values = arrays.offsets.to_numpy(), arrays.values.to_numpy()
    wave = lambda i: values[offsets[i]:offsets[i + 1]]
    meta["start_us"] = (meta.data_ts - meta.cycle_ts).dt.total_seconds() * 1e6
    # Columns that describe the cycle and the capture, the same on every row built from it.
    common = lambda r: {
        "event_id": int(r.event_id), "cycle_id": int(r.cycle_id), "capture_window": capture_window,
        "cycle_ts": r.cycle_ts, "event_ts": r.event_ts, "trig_code": int(r.trig_code), "trig_delay": float(r.trig_delay),
        "beam_mode": r.beam_mode, "beam_state": r.beam_state, "beam_present": bool(r.beam_present),
        "beam_len": int(r.beam_len), "beam_current": int(r.beam_current)}

    # Cavity rows: the pv name gives the cavity, the signal and the component. The six waveforms of one
    # cavity in one cycle are gathered under field_amp, field_phase, fwd_amp, fwd_phase, refl_amp, refl_phase
    # and turned into one row of features. Records shorter than n_bins are empty or cut and are left out.
    cav = meta.join(meta.pv.str.extract(CAVITY_PV)).dropna(subset="cavity")
    for (_, cavity), g in cav.groupby(["cycle_id", "cavity"]):
        sig = {f"{SIGNAL[d]}_{COMPONENT[c]}": wave(i) for i, d, c in zip(g.index, g.dwn, g.cmp) if len(wave(i)) >= n_bins}
        if "field_amp" not in sig:
            continue
        r = g.iloc[0]
        cavity_rows.append({**common(r), "cavity": cavity, "section": r.section, **pulse_features(sig, n_bins, r.start_us)})

    # Beam rows: one waveform per beam current monitor and cycle.
    bcm = meta.join(meta.pv.str.extract(BCM_PV)).dropna(subset="device")
    for i, r in bcm.iterrows():
        if len(wave(i)):
            beam_rows.append({**common(r), "device": r.device, **beam_features(wave(i), r.start_us)})
    return cavity_rows, beam_rows


def process_tar(job: tuple[str, int]) -> tuple[list[dict], list[dict]]:
    """Rows of one capture. Its tar holds SDS_Data/<window>/<DD|PM>/<event_id>/<family>.parquet, with family one of
    SPOKES, MBL, HBL, BCM, BPM."""
    path, n_bins = job
    cavity_rows, beam_rows = [], []
    with tarfile.open(path) as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".parquet"):
                continue
            # The tar is not compressed, so the parquet is read in place, without extracting it to disk.
            cavity, beam = parquet_rows(tar.extractfile(member), member.name.split("/")[1], n_bins)
            cavity_rows += cavity
            beam_rows += beam
    return cavity_rows, beam_rows


def load_mps(raw: Path) -> pd.DataFrame:
    """Interlock change log of the Archiver MPS files: one row each time an interlock changes, value 0 = not ok."""
    frames = [pd.DataFrame({"pv": pd.Series(dtype=str), "data_ts": pd.Series(dtype="datetime64[ns, UTC]"), "value": pd.Series(dtype=float)})]
    for path in sorted(raw.glob("Archiver_Data_*.tar")):
        with tarfile.open(path) as tar:
            for member in tar.getmembers():
                if not re.fullmatch(r"MPS(_\d+)*", Path(member.name).stem) or not member.name.endswith(".parquet"):
                    continue
                # An Archiver parquet has one row per pv and sample. The value sits in data_Int64 or data_Float64
                # depending on the pv, so the first filled one is taken.
                pf = pq.ParquetFile(tar.extractfile(member))
                values = [c for c in pf.schema_arrow.names if c.startswith("data_") and c != "data_ts"]
                d = pf.read(columns=["pv", "data_ts"] + values).to_pandas()
                frames.append(pd.DataFrame({"pv": d.pv, "data_ts": d.data_ts, "value": d[values].bfill(axis=1).iloc[:, 0].astype(float)}))
    # The same sample sits in every window folder that covers its time.
    return pd.concat(frames).dropna(subset="value").drop_duplicates(["pv", "data_ts"]).sort_values(["data_ts", "pv"], ignore_index=True)


def add_cycle_context(df: pd.DataFrame) -> pd.DataFrame:
    """Place each pulse among the cycles of its capture and measure how far it has moved from the first ones."""
    df = df.sort_values(["event_id", "cavity", "cycle_id"], ignore_index=True)
    group = df.groupby(["event_id", "cavity"])
    df["cycles_in_event"] = group.cycle_id.transform("size")
    df["cycle_rank_from_last"] = group.cumcount(ascending=False)
    df["seconds_to_event"] = (df.event_ts - df.cycle_ts).dt.total_seconds()
    # The earlier half of the cycles of a capture is the reference of that cavity in that capture. A drift is a
    # feature minus its median over the reference, so it reads 0 for a cavity that does not change.
    early = (df.cycle_rank_from_last * 2 >= df.cycles_in_event).to_numpy()
    for c in DRIFT:
        df[f"{c}_drift"] = df[c] - df[c].where(early).groupby([df.event_id, df.cavity]).transform("median")
    # shape_drift does the same on the whole binned waveform: mean bins of the reference cycles per capture and
    # cavity (code numbers those groups), then the RMS distance of each pulse to them.
    bins, code = np.stack(df.field_amp_bins).astype(np.float64), group.ngroup().to_numpy()
    total, count = np.zeros((code.max() + 1, bins.shape[1])), np.zeros(code.max() + 1)
    np.add.at(total, code[early], bins[early])
    np.add.at(count, code[early], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        reference = total / count[:, None]
        df["shape_drift"] = np.sqrt(((bins - reference[code]) ** 2).mean(axis=1)) / df.field_level.where(df.has_pulse)
    return df


def add_beam_and_trips(df: pd.DataFrame, beam: pd.DataFrame, trips: pd.DatetimeIndex) -> pd.DataFrame:
    """Attach to each cavity pulse the beam of its cycle and its distance in time to the beam trips."""
    if len(beam) and beam.has_beam.any():
        # One beam per cycle: the median over the monitors that saw it. Cavity and beam rows of a cycle share
        # event_id and cycle_id, and both count their *_us from cycle_ts, so the two starts can be subtracted.
        per_cycle = beam[beam.has_beam].groupby(["event_id", "cycle_id"]).agg(
            beam_level=("beam_level", "median"), beam_start_us=("beam_start_us", "median"),
            beam_len_us=("beam_len_us", "median"), beam_devices=("device", "size")).reset_index()
        df = df.merge(per_cycle, on=["event_id", "cycle_id"], how="left")
        df["beam_after_flat_start_us"] = df.beam_start_us - df.flat_start_us
    # merge_asof takes, for each cycle, the nearest trip after it (forward) or before it (backward).
    df = df.sort_values("cycle_ts", ignore_index=True)
    trip = pd.DataFrame({"trip_ts": trips.astype(df.cycle_ts.dtype)})
    for column, direction, sign in (("seconds_to_next_trip", "forward", 1), ("seconds_since_last_trip", "backward", -1)):
        near = pd.merge_asof(df[["cycle_ts"]], trip, left_on="cycle_ts", right_on="trip_ts", direction=direction).trip_ts
        df[column] = sign * (near - df.cycle_ts).dt.total_seconds()
    return df


def context_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman and Pearson r between a feature, as its deviation within the cavity, and each context column."""
    features = [c for c in FEATURES if c in df]
    context = [c for c in CONTEXT if c in df]
    # Cavities sit at different levels, so a feature is pooled as (value - cavity mean) / cavity std. Within one
    # cavity this keeps the order of the pulses; across cavities it puts them on one scale.
    dev = df.groupby("cavity")[features].transform(lambda s: (s - s.mean()) / s.std())
    frames = []
    for section, idx in [("all", df.index), *df.groupby("section").groups.items()]:
        x, y = dev.loc[idx], df.loc[idx, context].astype(np.float64)
        both = pd.concat([x, y], axis=1)
        # corr() works on pairs of columns, over the rows where both are present, and needs MIN_PAIRS of them.
        # Spearman: each column is replaced by the ranks of its values (ties share their average rank) and the
        # Pearson r of the two rank columns is taken. It is 1 when one column rises whenever the other does, whatever
        # the size of the steps, so one extreme pulse counts as one rank.
        # Pearson: covariance over the product of the standard deviations, on the values themselves. It measures a
        # straight-line relation and a few extreme pulses can set it.
        # The matrix is square over features and context; only the features by context block is kept.
        r = {m: both.corr(method=m, min_periods=MIN_PAIRS).loc[features, context].to_numpy().ravel() for m in METHODS}
        n = x.notna().astype(int).T @ y.notna().astype(int)
        frames.append(pd.DataFrame({"section": section, "feature": np.repeat(features, len(context)),
                                    "context": np.tile(context, len(features)), "spearman_r": r["spearman"],
                                    "pearson_r": r["pearson"], "n": n.to_numpy().ravel()}))
    return pd.concat(frames, ignore_index=True).dropna(subset=["spearman_r", "pearson_r"], how="all")


def cavity_comovement(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman and Pearson r of a feature between cavity and cavity_later, the second taken lag captured cycles later
    in the same capture.

    The two differ on trip pulses: a single pulse that cuts the RF in forty cavities at once sets their Pearson r
    to 1 and moves their Spearman r by one rank.
    """
    section = df.drop_duplicates("cavity").set_index("cavity").section
    frames = []
    for feature in (c for c in COMOVE if c in df):
        # One row per captured cycle, one column per cavity.
        p = df.pivot(index=["event_id", "cycle_id"], columns="cavity", values=feature).sort_index()
        # Without the capture mean removed, the shift between captures dominates and every lag reads the same.
        p -= p.groupby(level="event_id").transform("mean")
        for lag in LAGS:
            # Row t of later holds the values of cycle t + lag of the same capture; the last lag rows of a capture
            # have no partner and drop out of the pairs.
            later = p.groupby(level="event_id").shift(-lag).add_suffix("@later")
            both = pd.concat([p, later], axis=1)
            # Ranks and r as in context_correlation. Only the block of cavity (now) by cavity (later) is kept.
            r = {m: both.corr(method=m, min_periods=MIN_PAIRS).loc[p.columns, later.columns].to_numpy().ravel() for m in METHODS}
            frames.append(pd.DataFrame({"feature": feature, "lag_cycles": lag, "cavity": np.repeat(p.columns, len(p.columns)),
                                        "cavity_later": np.tile(p.columns, len(p.columns)), "spearman_r": r["spearman"],
                                        "pearson_r": r["pearson"]}))
    out = pd.concat(frames, ignore_index=True).dropna(subset=["spearman_r", "pearson_r"], how="all")
    out["same_section"] = out.cavity.map(section).to_numpy() == out.cavity_later.map(section).to_numpy()
    return out[out.cavity != out.cavity_later].reset_index(drop=True)


def main() -> None:
    if len(sys.argv) not in (5, 6) or sys.argv[3] not in CAPTURE_TYPES:
        raise SystemExit(__doc__)
    raw, out, capture_type, n_bins = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    tars = sorted(raw.glob(f"SDS_Data_*_{capture_type}_*.tar"))
    if len(sys.argv) == 6:
        keep = np.unique(np.linspace(0, len(tars) - 1, int(sys.argv[5])).round().astype(int))
        tars = [tars[i] for i in keep]
    out.mkdir(parents=True, exist_ok=True)
    workers = len(os.sched_getaffinity(0))
    print(f"{len(tars)} {capture_type} captures, {workers} workers", flush=True)

    # 1. One worker per capture reads its parquet files and returns the feature rows.
    cavity_rows, beam_rows = [], []
    with Pool(workers) as pool:
        for done, (cav, beam) in enumerate(pool.imap_unordered(process_tar, [(str(t), n_bins) for t in tars]), 1):
            cavity_rows += cav
            beam_rows += beam
            print(f"{done}/{len(tars)} captures, {len(cavity_rows)} cavity pulses, {len(beam_rows)} beam pulses", flush=True)
    # 2. Interlock changes from the Archiver. A beam trip is the beam permit going to 0.
    mps = load_mps(raw)
    trips = pd.DatetimeIndex(mps.data_ts[(mps.pv == PERMIT) & (mps.value == 0)])
    print(f"{len(mps)} interlock changes on {mps.pv.nunique()} interlocks, {len(trips)} beam trips", flush=True)

    # 3. The cavity table gets its drift columns, the beam of the cycle and the trip timings.
    beam = pd.DataFrame(beam_rows)
    df = add_beam_and_trips(add_cycle_context(pd.DataFrame(cavity_rows)), beam, trips)
    # 4. The two correlation tables are computed from the finished cavity table.
    context, comovement = context_correlation(df), cavity_comovement(df)
    # 5. cavity_pulses and beam_pulses are the data; the two correlation tables are summaries of cavity_pulses.
    df.to_parquet(out / "cavity_pulses.parquet", index=False)
    beam.to_parquet(out / "beam_pulses.parquet", index=False)
    context.to_parquet(out / "context_correlation.parquet", index=False)
    comovement.to_parquet(out / "cavity_comovement.parquet", index=False)
    mps.to_parquet(out / "mps_transitions.parquet", index=False)

    pd.set_option("display.width", 200)
    print(f"\ncavity_pulses: {len(df)} rows, {df.cavity.nunique()} cavities, {df.event_id.nunique()} captures, "
          f"{int((~df.has_pulse).sum())} without pulse")
    print(df.groupby("section")[["pulse_len", "flat_len", "flat_std_rel", "flat_tilt_rel", "phase_std_deg", "decay_tau", "refl_ratio"]]
          .median().round(5).to_string())
    print(f"\nbeam_pulses: {len(beam)} rows, {int(beam.has_beam.sum()) if len(beam) else 0} with beam")
    print("\nstrongest feature against context (section all):")
    print(context[context.section == "all"].sort_values("spearman_r", key=np.abs, ascending=False).head(20).to_string(index=False))
    print("\nmean r between cavities:")
    print(comovement.groupby(["feature", "lag_cycles", "same_section"])[["spearman_r", "pearson_r"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
