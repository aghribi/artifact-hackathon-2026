"""Export every scored waveform and the actual MAD decision boundary, offline."""
import argparse
import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
METHODS = ("iforest", "pca_knn", "global_pca_knn")
TITLES = ("PV-specific Isolation Forest", "PV-specific PCA + kNN", "Global PCA + kNN")


def encode(values, dtype):
    values = np.asarray(values, dtype=dtype)
    return {"dtype": values.dtype.str, "data": base64.b64encode(values.tobytes()).decode("ascii")}


def build(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    columns = ["original_row", "pv", "cycle_id", "cycle_ts", "beam_mode", "array_size",
               "status", "duplicate_of", "global_PC1", "global_PC2", "confirmed_label"]
    columns += [m + suffix for m in METHODS for suffix in ("_score", "_label")]
    frame = pd.read_csv(root / "classification.csv", usecols=columns,
                        dtype={"cycle_id": str, "pv": str}, float_precision="round_trip")
    data = frame.loc[frame.status == "modeled"].reset_index(drop=True)
    thresholds = pd.read_csv(root / "thresholds.csv", float_precision="round_trip").fillna({"pv": ""})
    pvs = sorted(data.pv.unique())
    modes = sorted(data.beam_mode.unique())
    cycles = sorted(data.cycle_id.unique())
    pv_codes = pd.Categorical(data.pv, categories=pvs).codes
    mode_codes = pd.Categorical(data.beam_mode, categories=modes).codes
    cycle_codes = pd.Categorical(data.cycle_id, categories=cycles).codes
    timestamps = data.drop_duplicates("cycle_id").set_index("cycle_id").cycle_ts.to_dict()
    flags = np.zeros(len(data), dtype=np.uint8)
    flags[data.confirmed_label.eq(-1)] |= 8
    flags[data.duplicate_of.notna()] |= 16
    payload = dict(count=len(data), unscored=len(frame) - len(data), pvs=pvs, modes=modes,
                   cycles=[[c, timestamps[c]] for c in cycles], methods=[], arrays={})
    arrays = payload["arrays"]
    for key, values, dtype in [
        ("pc1", data.global_PC1, "<f4"), ("pc2", data.global_PC2, "<f4"),
        ("original", data.original_row, "<u4"), ("pv", pv_codes, "<u2"),
        ("mode", mode_codes, "u1"), ("cycle", cycle_codes, "<u2"),
        ("length", data.array_size, "<u4"),
    ]:
        arrays[key] = encode(values, dtype)
    unique = data.duplicate_of.isna().to_numpy()
    known = data.confirmed_label.eq(-1).to_numpy()
    ratios, sigma_ratios, known_checks, map_checks, sigma_stats, sigma_counts = {}, {}, [], [], [], []
    training = data.loc[unique]
    training_groups = training.groupby("pv", sort=False).indices
    sigma_report = frame[["original_row", "pv", "cycle_id", "cycle_ts", "beam_mode", "array_size", "status", "duplicate_of", "confirmed_label"]].copy()
    sigma_union = np.zeros(len(data), dtype=bool)
    for j, method in enumerate(METHODS):
        table = thresholds.loc[thresholds.detector.eq(method)].set_index("pv")
        assert table.threshold_method.eq("median_mad").all()
        k_values = table.mad_multiplier.unique()
        assert len(k_values) == 1, "A horizontal boundary requires a shared k per detector."
        k = float(k_values[0])
        ordered = table.loc[pvs] if method != "global_pca_knn" else table.loc[[""]]
        median_values, mad_values = ordered["median"].to_numpy(), ordered["mad"].to_numpy()
        medians = median_values[pv_codes] if len(table) > 1 else median_values[0]
        mads = mad_values[pv_codes] if len(table) > 1 else mad_values[0]
        # A zero MAD needs a separate representation, rather than an invented scale.
        assert np.all(mads > 0), "Zero MAD groups require a separate zero-scale panel."
        scores = data[method + "_score"].to_numpy()
        z = (scores - medians) / mads
        ratios[method] = z
        candidates = data[method + "_label"].eq(-1).to_numpy()
        assert np.isfinite(z).all()
        assert np.array_equal(candidates, scores > medians + k * mads)
        assert np.array_equal(candidates, z > k)
        assert int(candidates.sum()) == summary[method + "_candidates"]
        flags[candidates] |= 1 << j
        arrays[method] = encode(scores, "<f8")
        training_scores = training[method + "_score"].to_numpy()
        standard_deviations = (np.asarray([np.std(training_scores[training_groups[pv]], ddof=0) for pv in pvs])
                               if method != "global_pca_knn" else np.asarray([np.std(training_scores, ddof=0)]))
        assert np.all(standard_deviations > 0)
        stds = standard_deviations[pv_codes] if len(table) > 1 else standard_deviations[0]
        sigma_ratios[method] = (scores - medians) / stds
        sigma_thresholds = {str(n): (median_values + n * standard_deviations).tolist() for n in (3, 4, 5, 6)}
        payload["methods"].append(dict(key=method, title=TITLES[j], k=k,
            median=median_values.tolist(), mad=mad_values.tolist(),
            std=standard_deviations.tolist(), sigma_thresholds=sigma_thresholds,
            threshold=ordered.threshold.tolist(), candidates=int(candidates.sum())))
        for p, pv in enumerate(pvs if method != "global_pca_knn" else [""]):
            sigma_stats.append(dict(scope="normalized_pv" if pv else "global", pv=pv, detector=method,
                training_count=int(table.loc[pv, "training_count"]), median=float(median_values[p]),
                std=float(standard_deviations[p]), std_ddof=0,
                **{f"threshold_{n}sigma": sigma_thresholds[str(n)][p] for n in (3, 4, 5, 6)}))
        for n in (3, 4, 5, 6):
            sigma_candidates = scores > medians + n * stds
            assert np.array_equal(sigma_candidates, sigma_ratios[method] > n)
            sigma_counts.append(dict(detector=method, sigma_multiplier=n, candidates=int(sigma_candidates.sum()),
                unique_candidates=int((sigma_candidates & unique).sum()), confirmed_detected=int((sigma_candidates & known).sum()),
                confirmed_total=int(known.sum()), center="median", sigma_definition="np.std(training_scores, ddof=0)"))
            if n == 5:
                sigma_union |= sigma_candidates
                sigma_report.loc[frame.status.eq("modeled"), method + "_score"] = scores
                sigma_report.loc[frame.status.eq("modeled"), method + "_threshold"] = medians + n * stds
                sigma_report.loc[frame.status.eq("modeled"), method + "_sigma_multiple"] = sigma_ratios[method]
                labels = pd.Series(pd.NA, index=frame.index, dtype="Int64")
                labels.loc[frame.status.eq("modeled")] = np.where(sigma_candidates, -1, 1)
                sigma_report[method + "_label"] = labels
        min_known = float(z[known].min()) if known.any() else None
        minimum_count = int((unique & (z >= min_known)).sum()) if min_known is not None else None
        map_checks.append(dict(detector=method, k=k, candidates=int(candidates.sum()),
            unique_candidates=int((candidates & unique).sum()), confirmed_detected=int((known & candidates).sum()),
            minimum_unique_candidates_retaining_known_with_common_k=minimum_count))
        for i in np.flatnonzero(known):
            t = table.loc["" if method == "global_pca_knn" else data.pv.iloc[i]]
            known_checks.append(dict(detector=method, original_row=int(data.original_row.iloc[i]),
                pv=data.pv.iloc[i], cycle_id=data.cycle_id.iloc[i], score=float(scores[i]),
                median=float(t["median"]), mad=float(t["mad"]), mad_multiple=float(z[i]),
                k=k, threshold=float(t["threshold"]), detected=bool(candidates[i]),
                unique_mad_rank=int((unique & (z > z[i])).sum()) + 1,
                std=float(stds[i] if np.ndim(stds) else stds), sigma_multiple=float(sigma_ratios[method][i]),
                threshold_5sigma=float((medians + 5 * stds)[i] if np.ndim(medians) else medians + 5 * stds),
                detected_5sigma=bool(sigma_ratios[method][i] > 5)))
    pd.DataFrame(sigma_stats).to_csv(root / "sigma_thresholds.csv", index=False)
    pd.DataFrame(sigma_counts).to_csv(root / "sigma_comparison.csv", index=False)
    sigma_report.to_csv(root / "sigma5_classification.csv", index=False)
    union_originals = data.original_row.to_numpy()[sigma_union]
    sigma_candidates = sigma_report.loc[sigma_report.original_row.isin(union_originals)].copy()
    sigma_candidates.to_csv(root / "sigma5_candidates.csv", index=False)
    cycle_reports = []
    for cycle, group in sigma_candidates.groupby("cycle_id", sort=True):
        cycle_reports.append(dict(cycle_id=cycle, cycle_ts=group.cycle_ts.iloc[0],
            beam_modes="|".join(sorted(group.beam_mode.unique())), candidate_waveforms=len(group),
            unique_candidate_waveforms=int(group.duplicate_of.isna().sum()), pvs=int(group.pv.nunique()),
            confirmed_anomalies=int(group.confirmed_label.eq(-1).sum()),
            **{m + "_candidates": int(group[m + "_label"].eq(-1).sum()) for m in METHODS}))
    cycle_report = pd.DataFrame(cycle_reports)
    cycle_report.to_csv(root / "sigma5_anomalous_cycles.csv", index=False)
    (root / "sigma5_cycle_ids.txt").write_text("".join(str(c["cycle_id"]) + "\n" for c in cycle_reports))
    for m in METHODS:
        cycle_report.loc[cycle_report[m + "_candidates"].gt(0)].to_csv(root / (m + "_sigma5_cycles.csv"), index=False)
    sigma_summary = dict(threshold="score > median(training scores) + 5 * std(training scores, ddof=0)",
        center="median", sigma_definition="Population standard deviation of the unique training scores (ddof=0)",
        local_scope="One score distribution per normalized PV; modes and lengths not split",
        global_scope="One score distribution for all unique training waveforms",
        rows=len(frame), modeled=len(data), unscored=len(frame)-len(data),
        union_candidate_waveforms=len(sigma_candidates),
        union_unique_candidate_waveforms=int(sigma_candidates.duplicate_of.isna().sum()),
        union_candidate_cycles=len(cycle_report),
        methods={m: dict(next(r for r in sigma_counts if r["detector"] == m and r["sigma_multiplier"] == 5),
                         candidate_cycles=int(cycle_report[m + "_candidates"].gt(0).sum())) for m in METHODS},
        comparison=sigma_counts, interpretation="Candidates, not confirmed anomalies; all scored waves are in-sample. No assumed Gaussian tail probability.")
    (root / "sigma5_summary.json").write_text(json.dumps(sigma_summary, indent=2) + "\n")
    arrays["flags"] = encode(flags, "u1")
    script = (HERE / "waveform_comparison_viewer" / "all_waveforms_map.js").read_text()
    document = '''<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>全波形の集合と判定閾値</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:#172b42;font:15px/1.65 system-ui,sans-serif}
main{max-width:1550px;margin:auto;padding:24px}h1{font-size:26px;margin:12px 0}a{color:#1460ad}
.note{color:#53657a;font-size:13px}.controls{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}
label{font-size:13px}select,input,button{font:inherit;padding:6px 9px;border:1px solid #b9c8d9;border-radius:6px;background:white;color:#172b42}
#pv{width:min(470px,90vw)}.card{background:white;border:1px solid #dce4ed;border-radius:12px;padding:16px;margin:12px 0}
.legend{display:flex;flex-wrap:wrap;gap:20px;font-size:13px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
#plot{width:100%;height:620px;display:block;touch-action:none;cursor:crosshair}
.plotbox{position:relative}#tooltip{position:absolute;display:none;pointer-events:none;background:#172b42ed;color:white;padding:9px 12px;border-radius:6px;font-size:12px;white-space:pre-line;max-width:390px;z-index:2}
#counts{font-variant-numeric:tabular-nums}#detail{white-space:pre-line;overflow-wrap:anywhere}#condition{overflow-wrap:anywhere}
@media(max-width:700px){main{padding:10px}.card{padding:8px}#plot{height:500px}h1{font-size:22px}}
</style><main><a href="index.html">モデル比較へ</a> · <a href="all_waveforms_5sigma_map.png">中央値＋5σの全体図 PNG</a> · <a href="all_waveforms_threshold_map.png">MADの全体図 PNG</a>
<h1>全波形の集合と判定閾値</h1>
<p>横軸は全PV共通の生波形PCAの位置、縦軸は <b id="axisExplanation">（判定スコア − 中央値）÷ σ</b> です。
<b>赤い破線より上が異常候補</b>です。すべての点を含め、点の重なりは濃さで示します。</p>
<p class="note">PV別モデルの中央値・σ・MADは各PVの学習スコアから求めています。全PV共通モデルは全体で1組です。
σは標準偏差（ddof=0）です。中心は中央値を使います。同一サイクルの完全一致波形を除いた学習集合から統計量を求めています。
縦軸をスコアにすることで実際の判定境界を表示しています。PCAの2次元投影に境界を推定した図ではありません。
PCA＋kNNのスコアは元のモデル空間での近傍10波形への平均距離、IFのスコアはIsolation Forestの異常度です。</p>
<div class="controls">
<label>手法 <select id="method"><option value="2">全PV共通 PCA＋kNN</option><option value="0">PV別 Isolation Forest</option><option value="1">PV別 PCA＋kNN</option></select></label>
<label>閾値 <select id="rule"><option value="5">中央値＋5σ</option><option value="3">中央値＋3σ</option><option value="4">中央値＋4σ</option><option value="6">中央値＋6σ</option><option value="mad">中央値＋k×MAD（暫定倍率）</option></select></label>
<label>PV <input id="pv" list="pvlist" placeholder="すべて（PV名で絞り込み）"><datalist id="pvlist"></datalist></label>
<label>beam_mode <select id="mode"><option value="">すべて</option></select></label>
<label>表示 <select id="filter"><option value="all">全波形</option><option value="candidate">異常候補</option><option value="known">確認済み異常</option></select></label>
<label>横軸 <select id="component"><option value="pc1">共通PCA PC1</option><option value="pc2">共通PCA PC2</option></select></label>
<label>縦軸 <select id="scale"><option value="log">符号付き対数</option><option value="linear">線形</option></select></label>
<button id="reset">表示をリセット</button></div>
<div class="legend"><span><i class="dot" style="background:#526d89"></i>その他の波形（正常未確定）</span>
<span><i class="dot" style="background:#d12f3a"></i>異常候補</span><span style="color:#936100">★ ユーザー確認済み異常</span><span>赤破線：判定閾値</span></div>
<div class="card"><div id="condition"></div><div id="counts"></div><div class="plotbox"><canvas id="plot" aria-label="全波形の判定スコアと実際の閾値の散布図"></canvas><div id="tooltip"></div></div>
<div class="note">ホイールで拡大・縮小、ドラッグで移動。点をクリックすると詳細と元波形へのリンクを表示します。絞り込み中も判定閾値は固定です。</div></div>
<div class="card"><b>選択した波形</b><div id="detail">点をクリックしてください。</div><a id="waveLink" hidden target="_blank" rel="noopener">この波形をPV別ビューアで開く（ビューア内の判定表示はMAD基準）</a></div>
<div class="card"><b>中央値＋nσの比較</b>__SIGMA_TABLE__
<p><a href="sigma5_anomalous_cycles.csv">5σ：候補サイクルの集計CSV</a> · <a href="sigma5_cycle_ids.txt">5σ：cycle_idのみ</a> · <a href="sigma5_candidates.csv">5σ：候補波形・PV・スコア・閾値CSV</a> · <a href="sigma5_classification.csv">5σ：全判定CSV</a></p>
<p><a href="sigma_comparison.csv">3σ・4σ・5σ・6σの件数CSV</a> · <a href="sigma_thresholds.csv">各PV・手法の中央値・σ・閾値CSV</a> · <a href="known_anomaly_checks.csv">確認済み異常の検出状況</a></p></div>
<p class="note" id="provenance"></p><p class="note">__KNOWN_NOTE__点群は全件を描画し、間引いていません。同一点に重なる波形も件数に含みます。σ基準の結果は別CSVに保存し、従来のMAD基準の判定も比較用に保持しています。</p>
</main><script id="map-data" type="application/json">__PAYLOAD__</script><script>__SCRIPT__</script></html>'''
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    path = root / "all_waveforms_threshold_map.html"
    known_note = (f"確認済み異常{int(known.sum())}波形を使ってMAD基準の倍率を暫定調整しました。σ基準でも同じ実例の検出状況を確認しています。他の波形の正常・異常は未確認です。"
                  if known.any() else "正常・異常の確定ラベルはありません。")
    sigma_table = '<table style="width:100%;text-align:left"><thead><tr><th>手法</th><th>倍率</th><th>候補波形</th><th>重複除外</th><th>確認済み異常の検出</th></tr></thead><tbody>'
    for row in sigma_counts:
        sigma_table += f'<tr><td>{TITLES[METHODS.index(row["detector"])]}</td><td>{row["sigma_multiplier"]}σ</td><td>{row["candidates"]:,}</td><td>{row["unique_candidates"]:,}</td><td>{row["confirmed_detected"]}/{row["confirmed_total"]}</td></tr>'
    sigma_table += '</tbody></table>'
    path.write_text(document.replace("__KNOWN_NOTE__", known_note).replace("__SIGMA_TABLE__", sigma_table).replace("__PAYLOAD__", encoded).replace("__SCRIPT__", script))

    x = data.global_PC1.to_numpy()
    for rule, coordinates in [("MAD", ratios), ("5sigma", sigma_ratios)]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.6), layout="constrained")
        for j, (axis, method) in enumerate(zip(axes, METHODS)):
            z = coordinates[method]
            k = payload["methods"][j]["k"] if rule == "MAD" else 5
            candidate = z > k
            axis.set_yscale("symlog", linthresh=1)
            axis.scatter(x[~candidate], z[~candidate], s=1.8, c="#526d89", alpha=.15,
                         linewidths=0, rasterized=True, label="Other (unlabelled)")
            axis.scatter(x[candidate], z[candidate], s=10, c="#d12f3a", alpha=.8,
                         linewidths=0, rasterized=True, label="Candidate")
            axis.scatter(x[known], z[known], s=120, marker="*", c="#edb52e", edgecolors="#614500",
                         linewidths=.8, zorder=5, label="Confirmed anomaly")
            axis.axhline(k, color="#bc2431", lw=1.6, ls="--", label=f"Threshold: k = {k:g}")
            bottom, top = axis.get_ylim()
            axis.axhspan(k, top, color="#d12f3a", alpha=.04, zorder=0)
            axis.set_ylim(bottom, top)
            axis.set_title(f"{TITLES[j]}\n{candidate.sum():,} candidates / {len(data):,} waves", fontsize=12)
            axis.set_xlabel("Shared raw-waveform PCA: PC1")
            axis.set_ylabel("(Score - training median) / " + ("raw MAD" if rule == "MAD" else "standard deviation"))
            axis.grid(alpha=.18)
            axis.legend(loc="upper left", fontsize=8)
            if method == "global_pca_knn":
                for i in np.flatnonzero(known):
                    channel = "Dwn0" if ":Dwn0-" in data.pv.iloc[i] else "Dwn6"
                    axis.annotate(channel, (x[i], z[i]), xytext=(15, -18 if channel == "Dwn6" else 14),
                                  textcoords="offset points", fontsize=9, color="#684a00",
                                  arrowprops=dict(arrowstyle="-", color="#936100"))
        title = "All waveforms: collection and actual decision thresholds"
        fig.suptitle(title + (" (median + 5 standard deviations)" if rule == "5sigma" else " (MAD)"), fontsize=16)
        global_t = payload["methods"][2]
        cutoff = global_t["threshold"][0] if rule == "MAD" else global_t["sigma_thresholds"]["5"][0]
        fig.supxlabel(f"Above each dashed line = candidate. Signed logarithmic y-axis; no subsampling. "
                      f"Global distance cutoff = {cutoff:.9f}.\n"
                      f"Gold stars: {int(known.sum())} user-confirmed anomalies; other waves remain unlabelled. "
                      "Statistics exclude exact duplicate training captures.", fontsize=10)
        fig.savefig(root / ("all_waveforms_threshold_map.png" if rule == "MAD" else "all_waveforms_5sigma_map.png"), dpi=180)
        plt.close(fig)
    pd.DataFrame(known_checks).to_csv(root / "known_anomaly_checks.csv", index=False)
    checks = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_classification_sha256=hashlib.file_digest((root / "classification.csv").open("rb"), "sha256").hexdigest(),
        original_rows=len(frame), mapped_rows=len(data), unscored_rows=len(frame)-len(data),
        normalized_pvs=len(pvs), cycles=len(cycles), beam_modes=modes,
        training_duplicates=int((~unique).sum()), confirmed_anomalies=int(known.sum()),
        all_points_included=True, decision_flags_verified_from_original_float64_scores=True,
        threshold_transform="(score - training median) / raw MAD; no consistency correction",
        detectors=map_checks, calibration_note="Known positives used to select provisional thresholds, not held-out validation.",
        sigma_definition="Median(training score) + n * std(training score, ddof=0); same unique training captures as MAD",
        sigma_comparison=sigma_counts, sigma5_candidate_cycles=len(cycle_report),
        sigma5_union_candidate_waveforms=len(sigma_candidates), html_bytes=path.stat().st_size)
    (root / "all_waveforms_threshold_map_validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "waveform_anomaly_comparison")
    build(parser.parse_args().output)
