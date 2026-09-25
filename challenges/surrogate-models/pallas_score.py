"""Score a PALLAS hackathon submission against an answer key.

Run: SUBMISSION=sub.csv KEY=key.csv /path/to/python pallas_score.py
Design: reports/hackathon/CHALLENGE1_REVIEW.md; scales and baselines in DATA_CARD.md.
A position-dependent task is scored against the PER-Z MEAN, not the global mean.
"""

import os

import numpy as np
import pandas as pd

# Submit PHYSICAL values. The scale each target is scored on is applied here, so
# nobody has to remember which quantities are logged -- and so every submission is
# scored the way the published reference numbers were.
#   (log10, floor in the target's own unit, factor applied before the floor)
SCALES = {
    "E_med_MeV": (False, None, 1.0),
    "E_mean_MeV": (False, None, 1.0),
    "dE_mad": (True, 1e-3, 100.0),      # scored as dE_pct, percent
    "q_end": (True, 0.1, 1e12),         # scored as log10 of charge in pC
    "p_1": (False, None, 1.0),
    "a_0": (False, None, 1.0),
    "c_N2": (False, None, 1.0),
    "E_MeV": (False, None, 1.0),
    "q_pC": (True, 0.1, 1.0),
    "dE_pct": (True, 1e-3, 1.0),
    "emit_um": (True, 1e-4, 1.0),
    "div_mrad": (True, 1e-3, 1.0),
    "sigz_um": (True, 1e-3, 1.0),
    "q_tot_pC": (True, 0.1, 1.0),
}
# The moment targets of the plane-holdout file: six bunch widths on log, six
# centroids on their own linear scale. sigma_* are RMS widths, mean_* centroids;
# zeta/y/z are lengths in um, u_* are normalised momenta (dimensionless).
for _coord, _unit in (("zeta", "um"), ("y", "um"), ("z", "um"),
                      ("ux", ""), ("uy", ""), ("uz", "")):
    SCALES[f"sigma_{_coord}"] = (True, 1e-6, 1.0)
    SCALES[f"mean_{_coord}"] = (False, None, 1.0)

KEY_COLUMNS = ("row_id", "config", "z_mm", "plane_mm", "L_inj")   # anything not a target

PLASMA_END_OFFSET_MM = float(os.environ.get("PLASMA_END_OFFSET_MM", 3.2))
# The plasma ends at L_inj + 3.2 mm, PER CONFIGURATION (Mykyta, 2026-09-10): that
# expression is the shipped density profile's own support, measured, and it runs from
# 3.601 to 3.999 mm across the corpus. A key or submission carrying L_inj is split on
# its own end; the flat fallback below is only for a table that carries no L_inj, and
# it mislabels a median 2 and up to 5 of the 40 in-plasma positions.
PLASMA_END_MM = float(os.environ.get("PLASMA_END_MM", 3.8))

ZONES = ("pooled", "in_plasma", "drift")

# Scored INSIDE THE PLASMA ONLY (Mykyta, 2026-09-10). Past the plasma end these two
# columns measure a definition rather than the beam: div_mrad is
# sqrt(emit_g/beta_x + emit_g/beta_y), a waist-proxy that falls as beta grows through
# the drift, and emit_um is n_emit_x, which falls as the halo leaves the diagnostic.
# Rewarding a model for reproducing either in the drift rewards the wrong thing.
IN_PLASMA_ONLY = ("emit_um", "div_mrad")

# A position must carry at least this many configurations of the zone to be scored;
# below it the per-z mean is a mean over a handful of rows and the position is dropped
# from both sums. 30 is the floor this project already uses for a stratum.
MIN_ZONE_MEMBERS = int(os.environ.get("MIN_ZONE_MEMBERS", 30))

POSITION_COLUMNS = ("z_mm", "plane_mm")


def to_scored_scale(name, values):
    """Physical values on the scale the leaderboard scores them on."""
    use_log, floor, factor = SCALES.get(name, (False, None, 1.0))
    values = np.asarray(values, float) * factor
    return np.log10(np.maximum(values, floor)) if use_log else values


def align(submission, key):
    """Join a submission to its key and return (merged, join columns, targets)."""
    sub = pd.read_csv(submission) if isinstance(submission, str) else submission.copy()
    ans = pd.read_csv(key) if isinstance(key, str) else key.copy()

    join = [c for c in KEY_COLUMNS if c in ans.columns and c in sub.columns]
    if not join:
        raise ValueError(f"submission has no join column; expected one of {KEY_COLUMNS}")
    targets = [c for c in ans.columns if c not in KEY_COLUMNS]
    extra = [c for c in sub.columns if c not in ans.columns and c not in KEY_COLUMNS]
    missing = [c for c in targets if c not in sub.columns]
    if missing:
        raise ValueError(f"submission is missing target column(s) {missing}")

    for c in join:                       # a CSV round trip must not break the join
        if c in POSITION_COLUMNS:
            ans[c], sub[c] = ans[c].astype(float).round(3), sub[c].astype(float).round(3)
    merged = ans.merge(sub[join + targets + extra], on=join, suffixes=("", "_pred"))
    if len(merged) != len(ans):
        raise ValueError(f"submission covers {len(merged)} of {len(ans)} rows")
    return merged, join, targets


def plasma_end(frame):
    """Where the plasma ends for each row, in mm.

    From the row's own L_inj when the table carries it, which is the ruled definition;
    the flat constant only when it does not.
    """
    if "L_inj" in frame.columns:
        return frame["L_inj"].to_numpy() + PLASMA_END_OFFSET_MM
    return np.full(len(frame), PLASMA_END_MM)


def zone_mask(frame, zone):
    """Rows of `frame` inside the plasma, in the drift after it, or all of them."""
    if zone == "pooled":
        return np.ones(len(frame), bool)
    position = "z_mm" if "z_mm" in frame.columns else "plane_mm"
    if position not in frame.columns:
        raise ValueError(f"zone {zone!r} needs a z_mm or plane_mm column")
    inside = frame[position].to_numpy() <= plasma_end(frame)
    return inside if zone == "in_plasma" else ~inside


def scored_targets(targets, zone):
    """The targets that count in this zone. Two are in-plasma only; see IN_PLASMA_ONLY."""
    if zone == "in_plasma":
        return list(targets)
    return [t for t in targets if t not in IN_PLASMA_ONLY]


def r2(true, pred):
    """R2 on values already carried to the scored scale, against the global mean.

    Used for the tasks with no position axis. For the position-dependent tasks see
    r2_per_z, which is the baseline the published anchors use.
    """
    resid = np.sum((true - pred) ** 2)
    spread = np.sum((true - true.mean()) ** 2)
    return float(1.0 - resid / spread)


def r2_per_z(true, pred, position):
    """R2 for a position-dependent task, against the PER-Z MEAN (Mykyta/Chat1, 2026-09-10).

    The denominator is the spread across CONFIGURATIONS at each position, summed over
    positions -- not the spread over every cell at once. On a trajectory most of the
    total variance is the z-trend itself, and a global-mean denominator credits a model
    for reproducing the average curve shape, which is trivial. This baseline asks the
    question a surrogate is for: at this position, can you tell the configurations
    apart? It is also what every published phase30 curve_r2 uses.

    In zone mode the mean is taken over the configurations that are IN THIS ZONE at
    this z, and with a per-config plasma end that membership changes across
    3.6-4.0 mm. A position carrying fewer than MIN_ZONE_MEMBERS configurations is
    DROPPED from both sums rather than contributing a mean over a handful of rows.
    Returns (r2, n_positions_scored, n_positions_dropped).
    """
    position = np.asarray(position)
    resid, spread, scored, dropped = 0.0, 0.0, 0, 0
    for here in np.unique(position):
        at = position == here
        if int(at.sum()) < MIN_ZONE_MEMBERS:
            dropped += 1
            continue
        here_true, here_pred = true[at], pred[at]
        resid += float(np.sum((here_true - here_pred) ** 2))
        spread += float(np.sum((here_true - here_true.mean()) ** 2))
        scored += 1
    if spread <= 0.0:
        return float("nan"), scored, dropped
    return float(1.0 - resid / spread), scored, dropped


def score_submission(submission, key, zone="pooled"):
    """R2 per target and the mean, on the agreed scale.

    `submission` and `key` are paths or DataFrames. They are joined on whichever of
    row_id / (config, z_mm) / (config, plane_mm) the key carries, so a submission in
    a different row order still scores correctly. `zone` restricts the rows to one
    side of the plasma end for the position-dependent tasks.
    """
    merged, _, targets = align(submission, key)
    keep = zone_mask(merged, zone)
    if not keep.any():
        raise ValueError(f"zone {zone!r} selects no rows")
    merged = merged[keep]
    targets = scored_targets(targets, zone)
    at = next((c for c in POSITION_COLUMNS if c in merged.columns), None)

    out, scored, dropped = {}, None, None
    for name in targets:
        true = to_scored_scale(name, merged[name])
        pred = to_scored_scale(name, merged[f"{name}_pred"])
        if at is None:
            out[name] = r2(true, pred)
        else:
            out[name], scored, dropped = r2_per_z(true, pred, merged[at].to_numpy())
    out["mean"] = float(np.mean([out[t] for t in targets]))
    out["n_rows"] = int(len(merged))
    if at is not None:
        out["n_z_scored"], out["n_z_dropped"] = scored, dropped
    return out


def score_zones(submission, key):
    """The same submission scored pooled, inside the plasma, and in the drift.

    The three columns are the point of the trajectory task: a model that looks
    strong pooled and weak in-plasma has learned the coasting, not the acceleration.
    """
    scored = {z: score_submission(submission, key, zone=z) for z in ZONES}
    _, _, targets = align(submission, key)
    extra = [c for c in ("n_rows", "n_z_scored", "n_z_dropped")
             if any(c in v for v in scored.values())]
    table = pd.DataFrame(scored).T.reindex(columns=list(targets) + ["mean"] + extra)
    table.index.name = "zone"
    # A blank cell is "not scored in this zone", not a failure: see IN_PLASMA_ONLY.
    return table


def score_budget(submissions, key, zone="pooled", target=0.90):
    """A data-budget curve: one submission per training-set size.

    `submissions` maps the number of training configs to a submission (path or
    frame). Returns the curve and the budget at which the mean score first reaches
    `target`, interpolated linearly between the two rungs that bracket it -- the
    coordinator's number for "how much data does this approach need".
    """
    budgets = sorted(submissions)
    rows = [{"n_train_configs": b,
             **score_submission(submissions[b], key, zone=zone)} for b in budgets]
    table = pd.DataFrame(rows).set_index("n_train_configs")

    means = table["mean"].to_numpy()
    reached = np.where(means >= target)[0]
    if reached.size == 0:
        budget_at = float("nan")
    elif reached[0] == 0:
        budget_at = float(budgets[0])
    else:
        i = reached[0]
        lo, hi = means[i - 1], means[i]
        frac = (target - lo) / (hi - lo)
        budget_at = float(budgets[i - 1] + frac * (budgets[i] - budgets[i - 1]))
    return table, budget_at


def score_ood(in_dist, in_dist_key, ood, ood_key, zone="pooled"):
    """One model on held-out configs inside its training range and beyond it.

    Both scores come from the SAME trained model; the drop is the answer to the
    organisers' third question. Nothing in the training set appears in either.
    """
    a = score_submission(in_dist, in_dist_key, zone=zone)
    b = score_submission(ood, ood_key, zone=zone)
    names = [k for k in a if k != "n_rows"]
    table = pd.DataFrame({"in_distribution": [a[k] for k in names],
                          "out_of_distribution": [b[k] for k in names]}, index=names)
    table["drop"] = table["in_distribution"] - table["out_of_distribution"]
    table.index.name = "target"
    return table, {"n_rows_in_distribution": a["n_rows"], "n_rows_ood": b["n_rows"]}


def score_calibration(submission, key, level=0.9, zone="pooled"):
    """Coverage and sharpness of predictive intervals, for the inverse task.

    The submission carries `<target>_lo` and `<target>_hi` beside each point
    prediction, an interval meant to hold the truth `level` of the time. Coverage
    alone is gameable -- an infinitely wide interval covers everything -- so the
    mean width on the scored scale travels with it, and the two are read together.
    """
    merged, _, targets = align(submission, key)
    keep = zone_mask(merged, zone)
    merged = merged[keep]
    targets = scored_targets(targets, zone)

    rows = []
    for name in targets:
        bounds = [f"{name}_lo", f"{name}_hi"]
        if any(b not in merged.columns for b in bounds):
            raise ValueError(f"submission is missing interval column(s) for {name}")
        true = to_scored_scale(name, merged[name])
        lo = to_scored_scale(name, merged[bounds[0]])
        hi = to_scored_scale(name, merged[bounds[1]])
        covered = (true >= np.minimum(lo, hi)) & (true <= np.maximum(lo, hi))
        rows.append({"target": name,
                     "coverage": float(covered.mean()),
                     "nominal": float(level),
                     "coverage_error": float(covered.mean() - level),
                     "mean_width_scored": float(np.mean(np.abs(hi - lo)))})
    table = pd.DataFrame(rows).set_index("target")
    return table, {"n_rows": int(len(merged))}


# Particle clouds are scored by a sliced Wasserstein-1 distance, the one this
# project's own particle models are judged by: both clouds are whitened on the TRUE
# cloud (the six coordinates differ in scale by ~100x, so an unwhitened distance
# measures uz and nothing else), the truth is charge-weighted, and the distance is
# divided by the truth's own resolution floor -- two random halves of the true cloud,
# rescaled by sqrt(2) to full size -- so a perfect prediction scores about 1.
# The submission is first redrawn (by its weights) to the true cloud's count, so
# submitting more particles does not lower the score. A plane with fewer than
# MIN_TRUE_PARTICLES true electrons is not scored: its floor is noise.
SW_PROJECTIONS = int(os.environ.get("SW_PROJECTIONS", 256))
SW_FLOOR_REPEATS = int(os.environ.get("SW_FLOOR_REPEATS", 4))
MIN_TRUE_PARTICLES = int(os.environ.get("MIN_TRUE_PARTICLES", 50))


def _probabilities(w, n):
    if w is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(w, dtype=np.float64)
    return w / w.sum() if np.isfinite(w.sum()) and w.sum() > 0 else np.full(n, 1.0 / n)


def _w1_1d(x1, p1, x2, p2):
    o1, o2 = np.argsort(x1), np.argsort(x2)
    x1, p1, x2, p2 = x1[o1], p1[o1], x2[o2], p2[o2]
    grid = np.sort(np.concatenate([x1, x2]))
    F1 = np.concatenate([[0.0], np.cumsum(p1)])[np.searchsorted(x1, grid, side="right")]
    F2 = np.concatenate([[0.0], np.cumsum(p2)])[np.searchsorted(x2, grid, side="right")]
    return float(np.sum(np.abs(F1[:-1] - F2[:-1]) * np.diff(grid)))


def sliced_wasserstein(X1, w1, X2, w2, directions):
    """Mean 1D Wasserstein-1 over the given unit directions, weights normalised."""
    p1, p2 = _probabilities(w1, len(X1)), _probabilities(w2, len(X2))
    P1, P2 = X1 @ directions, X2 @ directions
    return float(np.mean([_w1_1d(P1[:, j], p1, P2[:, j], p2) for j in range(P1.shape[1])]))


def cloud_ratio(X_true, w_true, X_pred, w_pred=None, seed=0):
    """One (configuration, plane): the distance to the truth over the truth's floor."""
    X_true = np.asarray(X_true, dtype=np.float64)
    X_pred = np.asarray(X_pred, dtype=np.float64)
    p = _probabilities(w_true, len(X_true))
    mean = p @ X_true
    cov = ((X_true - mean) * p[:, None]).T @ (X_true - mean)
    cov += np.eye(6) * 1e-12 * max(np.trace(cov) / 6, 1e-300)
    W = np.linalg.inv(np.linalg.cholesky(cov))   # whitening matrix of the truth
    Zt, Zp = (X_true - mean) @ W.T, (X_pred - mean) @ W.T
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((6, SW_PROJECTIONS))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    draw = np.random.default_rng([seed, 1])
    n = len(Zt)
    if w_pred is None and len(Zp) >= n:
        Zp = Zp[draw.choice(len(Zp), n, replace=False)]
    else:
        Zp = Zp[draw.choice(len(Zp), n, replace=True, p=_probabilities(w_pred, len(Zp)))]
    sw = sliced_wasserstein(Zt, w_true, Zp, None, directions)
    halves = []
    for _ in range(SW_FLOOR_REPEATS):
        m = rng.random(len(Zt)) < 0.5
        if m.sum() >= 2 and (~m).sum() >= 2:
            halves.append(sliced_wasserstein(Zt[m], p[m], Zt[~m], p[~m], directions))
    floor = float(np.mean(halves)) / np.sqrt(2.0) if halves else np.nan
    return {"sw": sw, "floor": floor, "ratio": sw / floor if floor > 0 else np.nan}


def load_cloud_file(path):
    """plane_mm, and one (N, 6) cloud with its weights per plane, lost particles dropped."""
    with np.load(path) as z:
        planes = np.asarray(z["plane_mm"], dtype=np.float64)
        state = np.asarray(z["state"], dtype=np.float64)
        present = z["present"] if "present" in z.files else np.isfinite(state).all(axis=2)
        weight = z["weight"] if "weight" in z.files else None
    clouds = {}
    for i, plane in enumerate(planes):
        keep = np.asarray(present[i], dtype=bool) & np.isfinite(state[i]).all(axis=1)
        w = None if weight is None else (weight[i] if np.ndim(weight) == 2 else weight)[keep]
        clouds[round(float(plane), 4)] = (state[i][keep], w)
    return clouds


def score_particles(submission_dir, key_dir, seed=0):
    """Median truth-referenced distance over every (configuration, plane) of the key.

    Both folders hold one `Config_<id>.npz` per configuration, with `plane_mm` and
    `state` (planes, particles, 6) in the pack's coordinate order; a submission may add
    `weight`, otherwise its particles count equally. A missing configuration or plane
    is an error, not a skip. A key plane with no particle present (the bunch is injected
    later) is not scored and is counted in `n_rows_no_bunch`; one with fewer than
    MIN_TRUE_PARTICLES electrons is counted in `n_rows_sparse`.
    """
    rows, skipped, sparse = [], 0, 0
    for name in sorted(os.listdir(key_dir)):
        if not name.endswith(".npz"):
            continue
        sub_path = os.path.join(submission_dir, name)
        if not os.path.exists(sub_path):
            raise ValueError(f"submission is missing {name}")
        truth, pred = load_cloud_file(os.path.join(key_dir, name)), load_cloud_file(sub_path)
        for plane, (X_true, w_true) in truth.items():
            if len(X_true) < 2:          # the bunch is not injected yet at this plane
                skipped += 1
                continue
            if len(X_true) < MIN_TRUE_PARTICLES:
                sparse += 1
                continue
            if plane not in pred:
                raise ValueError(f"{name} is missing plane {plane} mm")
            X_pred, w_pred = pred[plane]
            row = cloud_ratio(X_true, w_true, X_pred, w_pred, seed=seed)
            rows.append({"config": int(name[len("Config_"):-len(".npz")]),
                         "plane_mm": plane, "n_true": len(X_true), **row})
    table = pd.DataFrame(rows)
    return {"median_ratio": float(table.ratio.median()),
            "mean_ratio": float(table.ratio.mean()),
            "n_rows": int(len(table)), "n_config": int(table.config.nunique()),
            "n_rows_no_bunch": skipped, "n_rows_sparse": sparse}, table


if os.environ.get("PARTICLES_SUBMISSION") and os.environ.get("PARTICLES_KEY"):
    summary, _ = score_particles(os.environ["PARTICLES_SUBMISSION"], os.environ["PARTICLES_KEY"])
    for key_name, value in summary.items():
        print(f"{key_name:13s}  {value:.4f}" if isinstance(value, float)
              else f"{key_name:13s}  {value}")

if os.environ.get("SUBMISSION") and os.environ.get("KEY"):
    result = score_submission(os.environ["SUBMISSION"], os.environ["KEY"],
                              zone=os.environ.get("ZONE", "pooled"))
    width = max(len(k) for k in result)
    for key_name, value in result.items():
        print(f"{key_name:{width}s}  {value:.4f}" if isinstance(value, float)
              else f"{key_name:{width}s}  {value}")
