"""Inspect MAD sensitivity and review waveforms without changing any labels."""
import collections
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import waveform_anomaly_comparison as w
from matplotlib import pyplot as plt


METHODS = {"iforest": "PV-specific Isolation Forest", "pca_knn": "PV-specific PCA+kNN",
           "global_pca_knn": "Global PCA+kNN"}
TRIALS = {"iforest": 26., "pca_knn": 20000., "global_pca_knn": 450.}


def analyze(root=w.HERE / "waveform_anomaly_comparison"):
    root = Path(root)
    output = root / "threshold_diagnostics"
    output.mkdir(exist_ok=True)
    with (root / "classification.csv").open() as stream:
        records = list(csv.DictReader(stream))
    with (root / "thresholds.csv").open() as stream:
        thresholds = {(r["pv"], r["detector"]): r for r in csv.DictReader(stream)}
    valid = [r for r in records if r["status"] == "modeled"]
    positions = {int(r["original_row"]): j for j, r in enumerate(valid)}
    unique = np.asarray([not r["duplicate_of"] for r in valid])
    scores, medians, mads, ratios, orders = {}, {}, {}, {}, {}
    selected_grid = [1, 2, 3, 5, 6, 8, 10, 15, 20, 26, 30, 50, 100, 200, 450, 500,
                     1000, 2000, 5000, 10000, 20000, 50000, 100000, 500000, 1000000, 2500000]
    sweep, trials, review, examples, summaries = [], [], [], [], {}
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    for axis, (method, title) in zip(axes, METHODS.items()):
        scores[method] = np.asarray([float(r[method + "_score"]) for r in valid])
        stats = [thresholds["" if method == "global_pca_knn" else r["pv"], method] for r in valid]
        medians[method] = np.asarray([float(t["median"]) for t in stats])
        mads[method] = np.asarray([float(t["mad"]) for t in stats])
        delta = scores[method] - medians[method]
        z = np.divide(delta, mads[method], out=np.zeros_like(delta), where=mads[method] > 0)
        z[(mads[method] == 0) & (delta > 0)] = np.inf
        ratios[method] = z
        order = np.flatnonzero(unique)[np.argsort(z[unique], kind="stable")[::-1]]
        orders[method] = order
        for multiplier in selected_grid:
            candidate = scores[method] > medians[method] + multiplier * mads[method]
            sweep.append(dict(detector=method, mad_multiplier=multiplier,
                              original_rows=int(candidate.sum()), unique_waveforms=int((candidate & unique).sum())))
        active = scores[method] > np.asarray([float(t["threshold"]) for t in stats])
        assert np.array_equal(active, np.asarray([r[method + "_label"] == "-1" for r in valid]))
        multiplier = TRIALS[method]
        candidate = scores[method] > medians[method] + multiplier * mads[method]
        chosen = [r for r, selected in zip(valid, candidate & unique) if selected]
        cycles = collections.Counter(r["cycle_id"] for r in chosen)
        for j in np.flatnonzero(candidate):
            r = valid[j]
            trials.append(dict(detector=method, mad_multiplier=multiplier, original_row=int(r["original_row"]),
                               pv=r["pv"], source_pv=r["source_pv"], cycle_id=r["cycle_id"], beam_mode=r["beam_mode"],
                               array_size=int(r["array_size"]), duplicate_of=r["duplicate_of"],
                               score=scores[method][j], median=medians[method][j], mad=mads[method][j],
                               threshold=medians[method][j] + multiplier * mads[method][j], mad_multiple=z[j]))
        top = [valid[j] for j in order[:100]]
        summary = dict(trial_multiplier=multiplier, trial_original_rows=int(candidate.sum()),
                       trial_unique_waveforms=len(chosen), trial_unique_cycles=len(cycles),
                       top100_cycles=dict(collections.Counter(r["cycle_id"] for r in top)),
                       top100_modes=dict(collections.Counter(r["beam_mode"] for r in top)),
                       top100_unique_pvs=len({r["pv"] for r in top}),
                       mad_multiple_at_rank100=float(z[order[99]]),
                       trial_is_validated=False, trial_is_applied=False)
        summaries[method] = summary
        for rank, j in enumerate(order[:200], 1):
            r = valid[j]
            review.append(dict(detector=method, rank=rank, original_row=int(r["original_row"]),
                               pv=r["pv"], source_pv=r["source_pv"], cycle_id=r["cycle_id"],
                               cycle_ts=r["cycle_ts"], beam_mode=r["beam_mode"], array_size=int(r["array_size"]),
                               score=scores[method][j], median=medians[method][j], mad=mads[method][j],
                               mad_multiple=z[j], mode_count=int(r["mode_count"])))
        ordinary = np.flatnonzero(unique & (z > 1))
        ordinary = ordinary[np.argmin(z[ordinary])]
        for name, j in [("Highest MAD ratio", order[0]), ("Rank 100 for review", order[99]),
                        ("Just above median + MAD", ordinary)]:
            examples.append((method, name, int(j)))
        finite_max = max(float(np.max(z[np.isfinite(z)])), 2.)
        grid = np.unique(np.r_[np.geomspace(1, finite_max * 1.05, 130), 1, multiplier])
        counts = [int(((scores[method] > medians[method] + k * mads[method]) & unique).sum()) for k in grid]
        axis.plot(grid, np.maximum(counts, .5), color="#2563eb", linewidth=2)
        axis.axhline(100, color="#b91c1c", linestyle="--", label="100: domain expectation")
        axis.scatter([multiplier], [len(chosen)], color="#b91c1c", zorder=4)
        axis.annotate(f"k={multiplier:g}: {len(chosen)}", (multiplier, len(chosen)),
                      xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8)
        axis.set(xscale="log", yscale="log", ylim=(.5, 250000), xlabel="MAD multiplier k",
                 ylabel="Candidate unique waveforms", title=title)
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    figure.savefig(output / "mad_sensitivity.png", dpi=150)
    plt.close(figure)
    cache = w.RawCache(w.HERE / "anomaly-pca-reuse-20261015")
    figure, axes = plt.subplots(3, 3, figsize=(20, 13), layout="constrained")
    example_rows = []
    for axis, (method, category, j) in zip(axes.flat, examples):
        row = valid[j]
        options = [r for r in valid if not r["duplicate_of"] and r["original_row"] != row["original_row"]
                   and r["pv"] == row["pv"] and r["beam_mode"] == row["beam_mode"] and r["array_size"] == row["array_size"]]
        same_source = [r for r in options if r["source_pv"] == row["source_pv"]]
        if same_source:
            options = same_source
        center = np.median([float(r[method + "_score"]) for r in options])
        reference = min(options, key=lambda r: abs(float(r[method + "_score"]) - center))
        raw = cache.read([int(row["original_row"]), int(reference["original_row"])])
        axis.plot(raw[1], color="#64748b", linewidth=1., alpha=.85, label=f"Reference cycle {reference['cycle_id']}")
        axis.plot(raw[0], color="#dc2626", linewidth=.85, alpha=.8, label=f"Selected cycle {row['cycle_id']}")
        axis.set_title(f"{METHODS[method]} | {category}\n{row['pv']}\n{row['beam_mode']} | MAD ratio {ratios[method][j]:.4g}", fontsize=8)
        axis.set_xlabel("Original sample index")
        axis.set_ylabel("Original amplitude")
        axis.grid(alpha=.2)
        axis.legend(fontsize=6)
        example_rows.append(dict(detector=method, category=category, original_row=int(row["original_row"]),
                                 pv=row["pv"], cycle_id=row["cycle_id"], source_pv=row["source_pv"],
                                 beam_mode=row["beam_mode"], mad_multiple=ratios[method][j],
                                 reference_original_row=int(reference["original_row"]), reference_cycle_id=reference["cycle_id"],
                                 reference_source_pv=reference["source_pv"], reference_is_confirmed_nominal=False))
    figure.savefig(output / "waveform_review_examples.png", dpi=135)
    plt.close(figure)
    w.write_csv(output / "mad_sensitivity.csv", sweep)
    w.write_csv(output / "trial_candidates.csv", trials)
    w.write_csv(output / "ranked_review.csv", review)
    w.write_csv(output / "waveform_review_examples.csv", example_rows)
    source_hash = hashlib.sha256((root / "classification.csv").read_bytes()).hexdigest()
    summary = dict(source_classification_sha256=source_hash, valid_original_rows=len(valid), unique_waveforms=int(unique.sum()),
                   user_expected_anomalous_waveforms_approximately=100, models=summaries,
                   interpretation="Sensitivity analysis only. Trial coefficients illustrate the domain count expectation; "
                                  "they have not been validated with nominal/fault labels and are not applied. "
                                  "Ranked lists contain 200 unique waveforms per method for review, not confirmed anomalies.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = "".join(f'<tr><td>{html.escape(METHODS[d])}</td><td>{s["trial_multiplier"]:g}</td><td>{s["trial_original_rows"]}</td><td>{s["trial_unique_waveforms"]}</td><td>{s["trial_unique_cycles"]}</td></tr>' for d, s in summaries.items())
    cycle_rows = "".join(f'<tr><td>{html.escape(METHODS[d])}</td><td>{cycle}</td><td>{count}</td></tr>' for d, s in summaries.items() for cycle, count in sorted(s["top100_cycles"].items(), key=lambda item: -item[1])[:5])
    page = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MAD閾値の診断</title><link rel="stylesheet" href="../viewer.css"><main><a href="../index.html">モデル比較へ</a><h1>MAD閾値の診断</h1><p>全PV合計で異常波形は100件程度、という見込みに対し、中央値＋1×MADでは候補が多すぎます。閾値を中央値＋k×MADとした場合の件数を調べました。元データ{len(valid):,}行、完全一致の重複を除くと{int(unique.sum()):,}波形です。</p><p class="note">以下の倍率は100件前後になる感度の例です。異常を正しく識別する閾値としては未検証で、既存モデルの判定には適用していません。正常・異常の実例を用いた検証が必要です。</p><table><thead><tr><th>手法</th><th>倍率の例 k</th><th>元データ行数</th><th>重複除外後の波形数</th><th>サイクル数</th></tr></thead><tbody>{table}</tbody></table><p><a href="mad_sensitivity.csv">倍率ごとの件数CSV</a> · <a href="trial_candidates.csv">例示した倍率での確認候補CSV</a> · <a href="ranked_review.csv">各手法の上位200波形の確認リスト</a> · <a href="summary.json">解析条件</a></p><img class="overview" src="mad_sensitivity.png" alt="MAD倍率と候補数の関係"><h2>上位100波形が集中するサイクル</h2><p>これは上位波形の分布の確認です。異常件数を100件と確定したものではありません。</p><table><thead><tr><th>手法</th><th>cycle_id</th><th>上位100波形中の数</th></tr></thead><tbody>{cycle_rows}</tbody></table><h2>元波形と同条件の代表例</h2><p>赤は確認対象、灰色は同じPV・運転モード・波形長でスコアが中央付近の実際の波形です。可能な場合は元PV名のPMData／DODDataも合わせています。代表例は正常と確定したデータではありません。全サンプルを表示し、振幅の正規化はしていません。</p><img class="overview" src="waveform_review_examples.png" alt="上位候補と中央付近の波形の比較"><p><a href="waveform_review_examples.png">波形図を拡大</a> · <a href="waveform_review_examples.csv">図のPV・サイクル対応表</a></p></main></html>'''
    (output / "index.html").write_text(page)
    print("DIAGNOSTICS", json.dumps(summary), flush=True)
    print(f"Review: {output / 'index.html'}", flush=True)


if __name__ == "__main__":
    analyze()
