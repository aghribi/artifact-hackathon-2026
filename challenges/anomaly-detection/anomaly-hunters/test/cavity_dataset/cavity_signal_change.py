"""Direct normal/trip waveform comparisons, matched by beam conditions.

This module computes empirical reference envelopes and descriptive differences.
It does not fit classifiers or use predictions from the classifier benchmark.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from cavity_signal_comparison import SIGNALS, SIGNAL_NAMES, CAVITIES


CONDITION_COLUMNS = ["beam_mode", "beam_len", "trig_delay", "beam_present"]


def attach_beam_metadata(metadata, notebook_dir):
    """Read actual acquisition metadata; match using original source row indices."""
    result = metadata.copy()
    fields = ["capture_window", "beam_mode", "beam_state", "beam_len", "beam_current",
              "beam_present", "trig_code", "trig_delay"]
    for source in ("DD", "PM"):
        mask = result["source"].eq(source)
        table = pd.read_parquet(
            Path(notebook_dir).parent / f"notebooks/ess_{source.lower()}/cavity_pulses.parquet",
            columns=["event_id", "cycle_id", "cavity", *fields],
        ).iloc[result.loc[mask, "source_row"].to_numpy()]
        for key in ("event_id", "cycle_id", "cavity"):
            if not np.array_equal(table[key].to_numpy(), result.loc[mask, key].to_numpy()):
                raise ValueError(f"Beam metadata is not aligned: {source}, {key}")
        for field in fields:
            result.loc[mask, field] = table[field].to_numpy()
    return result


def plot_beam_absent_amplitudes(metadata, signals):
    """Yield raw amplitude overlays for every beam-absent acquisition condition.

    Keep single-label conditions, all available waveforms, original amplitude
    units, and all 64 bins. Thin lines are individual waveforms; thick lines are
    per-label medians. Separate beam_len/trig_delay conditions within each mode.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    amplitudes = ("field_amp", "fwd_amp", "refl_amp")
    if not metadata.index.equals(pd.RangeIndex(len(metadata))):
        metadata = metadata.reset_index(drop=True)
    for signal in amplitudes:
        if signals[signal].shape != (len(metadata), 64):
            raise ValueError(f"Expected 64 aligned amplitude bins for {signal}")
    if not metadata["beam_present"].isin([True, False]).all():
        raise ValueError("Expected boolean beam_present values")
    selected = metadata.loc[metadata["beam_present"].eq(False)]
    condition_columns = ["beam_mode", "beam_len", "trig_delay"]
    for conditions, group in selected.groupby(condition_columns, sort=True, dropna=False):
        context = dict(zip(condition_columns, conditions))
        context["beam_present"] = False
        fig, axes = plt.subplots(3, len(CAVITIES), figsize=(17, 9),
                                 sharex=True, sharey="row", squeeze=False)
        bins = np.arange(1, 65)
        for row, signal in enumerate(amplitudes):
            for col, cavity in enumerate(CAVITIES):
                ax = axes[row, col]
                cavity_rows = group.loc[group["cavity"].eq(cavity)]
                counts = []
                for label, color, style in ((0, "#3574ae", "-"), (1, "#e56b32", "--")):
                    indices = cavity_rows.loc[cavity_rows["label"].eq(label)].index.to_numpy()
                    waves = np.asarray(signals[signal][indices], dtype=float)
                    if not np.isfinite(waves).all():
                        raise ValueError(f"Non-finite amplitude waveform: {context}, {cavity}, {signal}")
                    counts.append(len(waves))
                    if not len(waves):
                        continue
                    ax.plot(bins, waves.T, color=color,
                            alpha=0.14 if label == 0 else 0.35, linewidth=0.7)
                    ax.plot(bins, np.median(waves, axis=0), color=color,
                            linestyle=style, linewidth=2)
                ax.set_title(f"Cavity {cavity[-3:]} | label=0: {counts[0]}, label=1: {counts[1]}", fontsize=10)
                ax.grid(alpha=0.15)
                ax.set_xlim(1, 64)
                if col == 0:
                    ax.set_ylabel(f"{SIGNAL_NAMES[signal]}\n(source units)")
                if row == 2:
                    ax.set_xlabel("Waveform bin (whole record)")
                if not any(counts):
                    ax.text(0.5, 0.5, "No waveforms", ha="center", va="center", transform=ax.transAxes)
        handles = [
            Line2D([0], [0], color="#3574ae", linewidth=2, label="label=0 (DD), median"),
            Line2D([0], [0], color="#e56b32", linewidth=2, linestyle="--", label="label=1 (PM last), median"),
            Line2D([0], [0], color="gray", linewidth=0.7, alpha=0.5, label="Thin lines: every waveform"),
        ]
        fig.suptitle(
            f"{context['beam_mode']} | beam_present=False | "
            f"beam_len={context['beam_len']:g}, trig_delay={context['trig_delay']:.6f}", fontsize=15,
        )
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        yield context, fig


def plot_trigger_delay_amplitudes(metadata, signals, beam_mode="Probe", beam_present=True):
    """Overlay trigger-delay groups, separately for each label and beam length."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    amplitudes = ("field_amp", "fwd_amp", "refl_amp")
    metadata = metadata.reset_index(drop=True)
    for signal in amplitudes:
        if signals[signal].shape != (len(metadata), 64):
            raise ValueError(f"Expected 64 aligned amplitude bins for {signal}")
    selected = metadata.loc[
        metadata["beam_mode"].eq(beam_mode) & metadata["beam_present"].eq(beam_present)
    ]
    if selected.empty:
        raise ValueError("No waveforms for the selected beam mode/presence")
    bins = np.arange(1, 65)
    for (beam_len, label), group in selected.groupby(["beam_len", "label"], sort=True):
        delays = sorted(group["trig_delay"].dropna().unique())
        if not delays:
            raise ValueError("No trigger-delay values")
        colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, max(2, len(delays))))
        # Use the familiar blue/orange pair for the two recorded trigger delays.
        if len(delays) == 2:
            colors = ["#3574ae", "#e56b32"]
        fig, axes = plt.subplots(3, len(CAVITIES), figsize=(17, 9),
                                 sharex=True, sharey="row", squeeze=False)
        for row, signal in enumerate(amplitudes):
            for col, cavity in enumerate(CAVITIES):
                ax = axes[row, col]
                captions = []
                for delay_index, (delay, color) in enumerate(zip(delays, colors)):
                    indices = group.loc[
                        group["cavity"].eq(cavity) & group["trig_delay"].eq(delay)
                    ].index.to_numpy()
                    waves = np.asarray(signals[signal][indices], dtype=float)
                    if not np.isfinite(waves).all():
                        raise ValueError("Non-finite waveform in trigger-delay comparison")
                    captions.append(f"{delay:.3f}: {len(waves)}")
                    if not len(waves):
                        continue
                    ax.plot(bins, waves.T, color=color, linewidth=0.7, alpha=0.18)
                    ax.plot(bins, np.median(waves, axis=0), color=color, linewidth=2,
                            linestyle="-" if delay_index == 0 else "--")
                ax.set(title=f"Cavity {cavity[-3:]} | n " + ", ".join(captions), xlim=(1, 64))
                ax.title.set_fontsize(10)
                ax.grid(alpha=0.15)
                if col == 0:
                    ax.set_ylabel(f"{SIGNAL_NAMES[signal]}\n(source units)")
                if row == 2:
                    ax.set_xlabel("Waveform bin (whole record)")
        handles = [Line2D([0], [0], color=color, linewidth=2,
                          linestyle="-" if i == 0 else "--",
                          label=f"trig_delay={delay:.6f}, median")
                   for i, (delay, color) in enumerate(zip(delays, colors))]
        handles.append(Line2D([0], [0], color="gray", alpha=0.4, linewidth=0.7,
                              label="Thin lines: every waveform"))
        fig.suptitle(f"{beam_mode} | beam_present={beam_present} | beam_len={beam_len:g} | "
                     f"label={int(label)} ({'DD' if label == 0 else 'PM last'})", fontsize=15)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        yield {"beam_mode": beam_mode, "beam_present": beam_present,
               "beam_len": beam_len, "label": int(label)}, fig


def _normal_coordinates(matrix, normal, phase):
    matrix = matrix.astype(float)
    if not phase:
        return matrix, np.full(matrix.shape[1], np.nan)
    phasor = np.exp(1j * np.deg2rad(matrix[normal])).mean(axis=0)
    reference = np.angle(phasor)
    relative = np.rad2deg(np.angle(np.exp(1j * (np.deg2rad(matrix) - reference))))
    return relative, np.abs(phasor)


def _envelope(matrix):
    # These order statistics enclose at least 95% of the observed normal values.
    # For a small reference sample they can be its min/max: report sample counts.
    return (np.quantile(matrix, 0.025, axis=0, method="lower"),
            np.quantile(matrix, 0.975, axis=0, method="higher"))


def _rf_window(signals, indices, normal):
    medians = {s: np.median(signals[s][indices][normal], axis=0)
               for s in ("field_amp", "fwd_amp", "refl_amp")}
    masks = []
    for signal, fraction in (("field_amp", 0.5), ("fwd_amp", 0.1), ("refl_amp", 0.1)):
        wave = medians[signal]
        masks.append(wave >= wave.min() + fraction * np.ptp(wave))
    return np.logical_and.reduce(masks)


def analyze_changes(metadata, signals, bootstrap_samples=2000):
    """Describe changes per cavity within matched beam acquisition conditions.

    A pulse's change fraction is its fraction of bins outside the pointwise normal
    envelope. Average pulses within a capture before averaging captures.
    Bootstrap intervals resample capture-level scores and condition on the fixed
    observed envelope. They are withheld for fewer than five normal captures.
    """
    metadata = metadata.reset_index(drop=True).copy()
    missing = set(CONDITION_COLUMNS) - set(metadata)
    if missing:
        raise ValueError(f"Missing beam metadata: {missing}")
    summaries, profiles, pulse_rows, supports = [], [], [], []
    rng = np.random.RandomState(42)
    groups = metadata.groupby([*CONDITION_COLUMNS, "cavity"], sort=True, dropna=False)
    cohort = 0
    for conditions, group in groups:
        cohort += 1
        context = dict(zip([*CONDITION_COLUMNS, "cavity"], conditions))
        context["cohort"] = cohort
        indices = group.index.to_numpy()
        normal = group["label"].to_numpy() == 0
        trip = ~normal
        normal_events = group.loc[normal, "event_id"].nunique()
        trip_events = group.loc[trip, "event_id"].nunique()
        supported = bool(normal.any() and trip.any())
        context.update(normal_pulses=int(normal.sum()), trip_pulses=int(trip.sum()),
                       normal_events=normal_events, trip_events=trip_events,
                       matched=supported, limited_normal_reference=normal_events < 5)
        supports.append(context.copy())
        if not supported:
            continue
        rf_window = _rf_window(signals, indices, normal)
        if not rf_window.any():
            raise ValueError(f"No common normal RF-on bins in cohort {cohort}")
        for signal in SIGNALS:
            matrix, coherence = _normal_coordinates(signals[signal][indices], normal,
                                                     signal.endswith("_phase"))
            if not np.isfinite(matrix).all():
                raise ValueError(f"Non-finite values in {signal}, cohort {cohort}")
            low, high = _envelope(matrix[normal])
            outside = (matrix < low) | (matrix > high)
            normal_median = np.median(matrix[normal], axis=0)
            trip_median = np.median(matrix[trip], axis=0)
            trip_low, trip_high = _envelope(matrix[trip])
            for bin_index in range(matrix.shape[1]):
                profiles.append({
                    **context, "signal": signal, "bin": bin_index + 1,
                    "rf_on_reference": bool(rf_window[bin_index]),
                    "normal_median": normal_median[bin_index],
                    "normal_low": low[bin_index], "normal_high": high[bin_index],
                    "trip_median": trip_median[bin_index],
                    "trip_low": trip_low[bin_index], "trip_high": trip_high[bin_index],
                    "median_shift": trip_median[bin_index] - normal_median[bin_index],
                    "normal_outside_fraction": outside[normal, bin_index].mean(),
                    "trip_outside_fraction": outside[trip, bin_index].mean(),
                    "normal_phase_coherence": coherence[bin_index],
                })
            for window_name, window in (("whole_record", np.ones(matrix.shape[1], bool)),
                                         ("rf_on_reference", rf_window)):
                fraction = outside[:, window].mean(axis=1)
                pulse = group[["dataset_index", "source", "event_id", "label"]].copy()
                pulse["changed_bin_fraction"] = fraction
                pulse["signal"], pulse["window"], pulse["cohort"] = signal, window_name, cohort
                pulse_rows.append(pulse)
                captures = pulse.groupby(["source", "event_id", "label"])["changed_bin_fraction"].mean().reset_index()
                normal_scores = captures.loc[captures["label"] == 0, "changed_bin_fraction"].to_numpy()
                trip_scores = captures.loc[captures["label"] == 1, "changed_bin_fraction"].to_numpy()
                excess = trip_scores.mean() - normal_scores.mean()
                ci_low = ci_high = np.nan
                if normal_events >= 5 and trip_events >= 3:
                    normal_bootstrap = normal_scores[rng.randint(len(normal_scores), size=(bootstrap_samples, len(normal_scores)))].mean(axis=1)
                    trip_bootstrap = trip_scores[rng.randint(len(trip_scores), size=(bootstrap_samples, len(trip_scores)))].mean(axis=1)
                    ci_low, ci_high = np.quantile(trip_bootstrap - normal_bootstrap, [0.025, 0.975])
                summaries.append({
                    **context, "signal": signal, "window": window_name,
                    "n_bins": int(window.sum()),
                    "normal_changed_fraction": normal_scores.mean(),
                    "trip_changed_fraction": trip_scores.mean(),
                    "excess_changed_fraction": excess,
                    "excess_ci_low": ci_low, "excess_ci_high": ci_high,
                    "max_abs_median_shift": np.max(np.abs(trip_median[window] - normal_median[window])),
                })
    summary = pd.DataFrame(summaries)
    support = pd.DataFrame(supports)
    matched_indices = metadata["dataset_index"].isin(
        pd.concat(pulse_rows, ignore_index=True)["dataset_index"].unique()
    )
    beam_summary = summary.groupby(["beam_mode", "beam_present", "signal", "window"], sort=True).agg(
        normal_changed_fraction=("normal_changed_fraction", "mean"),
        trip_changed_fraction=("trip_changed_fraction", "mean"),
        excess_changed_fraction=("excess_changed_fraction", "mean"),
        minimum_normal_events=("normal_events", "min"),
        minimum_trip_events=("trip_events", "min"),
        matched_cavity_condition_cells=("cohort", "nunique"),
    ).reset_index()
    return {"summary": summary, "profiles": pd.DataFrame(profiles),
            "pulse_scores": pd.concat(pulse_rows, ignore_index=True), "support": support,
            "beam_summary": beam_summary, "matched_pulses": int(matched_indices.sum()),
            "unmatched_pulses": int((~matched_indices).sum()), "metadata": metadata}


def _condition_title(row):
    return (f"{row['beam_mode']} | beam_len={row['beam_len']:g} | "
            f"trig_delay={row['trig_delay']:.3f} | beam_present={bool(row['beam_present'])}")


def plot_profiles(result):
    """Yield amplitude/phase plots for every supported beam-condition stratum."""
    import matplotlib.pyplot as plt
    profiles = result["profiles"]
    conditions = profiles.drop_duplicates(CONDITION_COLUMNS)[CONDITION_COLUMNS]
    for _, condition in conditions.iterrows():
        selected = np.logical_and.reduce([profiles[c].to_numpy() == condition[c] for c in CONDITION_COLUMNS])
        group = profiles.loc[selected]
        for phase in (False, True):
            signals = [s for s in SIGNALS if s.endswith("_phase") == phase]
            fig, axes = plt.subplots(3, 4, figsize=(16, 8), squeeze=False)
            for row, signal in enumerate(signals):
                for col, cavity in enumerate(CAVITIES):
                    ax = axes[row, col]
                    table = group[(group["signal"] == signal) & (group["cavity"] == cavity)].sort_values("bin")
                    if table.empty:
                        ax.set_visible(False)
                        continue
                    x = table["bin"].to_numpy()
                    ax.fill_between(x, table["normal_low"].to_numpy(), table["normal_high"].to_numpy(),
                                    color="#3574ae", alpha=0.25, label="Normal 95% envelope")
                    ax.plot(x, table["normal_median"], color="#3574ae", linewidth=1.2, label="Normal median")
                    ax.plot(x, table["trip_median"], color="#e56b32", linewidth=1.2, label="Trip median")
                    ax.plot(x, table["trip_low"], color="#e56b32", alpha=0.55, linewidth=0.7, linestyle=":")
                    ax.plot(x, table["trip_high"], color="#e56b32", alpha=0.55, linewidth=0.7, linestyle=":", label="Trip 95% envelope")
                    ax.set_title(f"{SIGNAL_NAMES[signal]} | {cavity[-3:]} | DD {table.normal_pulses.iloc[0]}, PM {table.trip_pulses.iloc[0]}", fontsize=8.5)
                    gate = table.loc[table["rf_on_reference"], "bin"].to_numpy()
                    if len(gate):
                        ax.axvspan(gate.min() - 0.5, gate.max() + 0.5, color="gray", alpha=0.07)
                    if col == 0:
                        ax.set_ylabel("Relative phase (degrees)" if phase else "Amplitude (source units)", fontsize=9)
                    if row == 2:
                        ax.set_xlabel("Waveform bin")
                    ax.grid(alpha=0.15)
            axes[0, 0].legend(fontsize=7, loc="best")
            fig.suptitle(_condition_title(condition) + (" | phase relative to normal circular mean" if phase else " | amplitudes"), fontsize=11)
            fig.tight_layout()
            yield fig


def plot_change_profiles(result):
    """Show where trip waveforms exceed each matched normal envelope."""
    import matplotlib.pyplot as plt
    profiles = result["profiles"]
    conditions = profiles.drop_duplicates(CONDITION_COLUMNS)[CONDITION_COLUMNS]
    for _, condition in conditions.iterrows():
        selected = np.logical_and.reduce([profiles[c].to_numpy() == condition[c] for c in CONDITION_COLUMNS])
        group = profiles.loc[selected]
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
        for ax, cavity in zip(axes.flat, CAVITIES):
            for signal in SIGNALS:
                table = group[(group["signal"] == signal) & (group["cavity"] == cavity)].sort_values("bin")
                ax.plot(table["bin"], 100 * (table["trip_outside_fraction"] - table["normal_outside_fraction"]),
                        linewidth=1.1, label=SIGNAL_NAMES[signal])
            ax.set(title=cavity[-3:], xlabel="Waveform bin", ylabel="Excess outside normal envelope (pp)")
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.grid(alpha=0.15)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(_condition_title(condition))
        fig.tight_layout()
        yield fig


def plot_change_heatmaps(result):
    import matplotlib.pyplot as plt
    summary = result["summary"].query("window == 'rf_on_reference'")
    for (mode, present), group in summary.groupby(["beam_mode", "beam_present"], sort=True):
        table = group.pivot_table(index="signal", columns="cavity", values="excess_changed_fraction", aggfunc="mean").reindex(SIGNALS)
        values = 100 * table.to_numpy()
        limit = max(10, float(np.nanmax(np.abs(values))))
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        image = ax.imshow(values, vmin=-limit, vmax=limit, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(np.arange(len(table.columns)), table.columns.str[-3:])
        ax.set_yticks(np.arange(len(table.index)), [SIGNAL_NAMES[s] for s in table.index])
        for row in range(table.shape[0]):
            for col in range(table.shape[1]):
                ax.text(col, row, f"{values[row, col]:+.1f}", ha="center", va="center",
                        color="white" if abs(values[row, col]) > 0.65 * limit else "black")
        flag = "\nLimited normal-event reference" if group["limited_normal_reference"].any() else ""
        ax.set(xlabel="Cavity", title=f"{mode} | beam_present={bool(present)}{flag}")
        fig.colorbar(image, ax=ax, label="Extra changed-bin fraction vs normal (percentage points)")
        fig.tight_layout()
        yield fig


def plot_deviation_profiles(result, beam_mode="RfTest", beam_present=True):
    """Zoom into RF-on deviations in original units, centered on normal medians."""
    import matplotlib.pyplot as plt
    profiles = result["profiles"]
    chosen = profiles[(profiles["beam_mode"] == beam_mode)
                      & (profiles["beam_present"] == beam_present)]
    conditions = chosen.drop_duplicates(CONDITION_COLUMNS)[CONDITION_COLUMNS]
    for _, condition in conditions.iterrows():
        select = np.logical_and.reduce([chosen[c].to_numpy() == condition[c] for c in CONDITION_COLUMNS])
        group = chosen.loc[select]
        for phase in (False, True):
            signals = [s for s in SIGNALS if s.endswith("_phase") == phase]
            fig, axes = plt.subplots(3, 4, figsize=(16, 8))
            for row, signal in enumerate(signals):
                for col, cavity in enumerate(CAVITIES):
                    ax = axes[row, col]
                    table = group[(group["signal"] == signal) & (group["cavity"] == cavity)
                                  & group["rf_on_reference"]].sort_values("bin")
                    if table.empty:
                        ax.set_visible(False)
                        continue
                    x = table["bin"].to_numpy()
                    center = table["normal_median"].to_numpy()
                    ax.fill_between(x, table["normal_low"].to_numpy() - center,
                                    table["normal_high"].to_numpy() - center,
                                    color="#3574ae", alpha=0.3, label="Normal 95% envelope")
                    ax.plot(x, table["trip_median"].to_numpy() - center,
                            color="#e56b32", linewidth=1.2, label="Trip median - normal median")
                    ax.plot(x, table["trip_low"].to_numpy() - center, color="#e56b32",
                            linewidth=0.7, alpha=0.6, linestyle=":")
                    ax.plot(x, table["trip_high"].to_numpy() - center, color="#e56b32",
                            linewidth=0.7, alpha=0.6, linestyle=":", label="Trip 95% envelope")
                    ax.axhline(0, color="#3574ae", linewidth=0.8)
                    ax.set_title(f"{SIGNAL_NAMES[signal]} | {cavity[-3:]}", fontsize=9)
                    if col == 0:
                        ax.set_ylabel("Phase difference (degrees)" if phase else "Amplitude difference (source units)", fontsize=9)
                    if row == 2:
                        ax.set_xlabel("Original waveform bin (RF-on reference)")
                    ax.grid(alpha=0.15)
            axes[0, 0].legend(fontsize=7)
            fig.suptitle(_condition_title(condition) + "\nDeviation from the matched normal median", fontsize=11)
            fig.tight_layout()
            yield fig
