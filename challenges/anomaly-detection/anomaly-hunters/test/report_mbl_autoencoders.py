"""Export learning curves, MSE comparisons and an offline MBL comparison report."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


def report(directory):
    root = Path(directory).resolve()
    results = pd.read_csv(root/"comparison.csv")
    official_path = root/"kickoff_score_comparison.csv"
    official = pd.read_csv(official_path if official_path.exists() else root/"official_holdout_comparison.csv")
    config = json.loads((root/"configuration.json").read_text())
    histories = [(name, pd.read_csv(root/f"{name}_history.csv")) for name in results.method
                 if (root/f"{name}_history.csv").is_file()]
    fig, axes = plt.subplots(int(np.ceil(len(histories)/3)), 3,
                            figsize=(16, 3.8*int(np.ceil(len(histories)/3))), squeeze=False, layout="constrained")
    for ax, (name, history) in zip(axes.flat, histories):
        ax.plot(history.epoch, history.train_mse, label="Training")
        ax.plot(history.epoch, history.validation_mse, label="Validation")
        best = int(history.loc[history.validation_mse.idxmin(), "epoch"])
        ax.axvline(best, ls="--", lw=.8, color="gray")
        ax.set(title=f"{name} (best epoch {best})", xlabel="Epoch", ylabel="Training-target MSE", yscale="log")
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(histories):]:
        ax.set_visible(False)
    fig.savefig(root/"learning_curves/all_models.png", dpi=160)
    fig.savefig(root/"learning_curves/all_models.pdf")
    plt.close(fig)
    with PdfPages(root/"learning_curves/learning_curves.pdf") as pdf:
        for name, history in histories:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
            axes[0].plot(history.epoch, history.train_mse, label="Training")
            axes[0].plot(history.epoch, history.validation_mse, label="Validation")
            axes[0].set(title=name, xlabel="Epoch", ylabel="Training-target MSE", yscale="log")
            axes[0].legend()
            axes[1].plot(history.epoch, history.validation_waveform_mse, color="tab:orange")
            axes[1].set(title="Full native waveform validation", xlabel="Epoch", ylabel="Fixed per-PV normalized MSE", yscale="log")
            pdf.savefig(fig)
            plt.close(fig)
    native = results.sort_values("test_native_mse")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), layout="constrained")
    axes[0].barh(native.method, native.test_native_mse)
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Native-grid normalized test MSE (log)", xscale="log")
    axes[1].barh(official.method, official.official_holdout_mse)
    axes[1].axvline(.0194, color="tab:red", ls="--", label="Published reference 0.0194")
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Official 42-event normalized MSE (log)", xscale="log")
    axes[1].legend()
    fig.savefig(root/"mse_comparison.png", dpi=170)
    fig.savefig(root/"mse_comparison.pdf")
    plt.close(fig)
    native_holdout_path = root/"heldout_native_comparison.csv"
    native_holdout = pd.read_csv(native_holdout_path) if native_holdout_path.exists() else pd.DataFrame()
    if not native_holdout.empty:
        ordered = native_holdout.sort_values("mean_native_normalized_mse")
        fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
        ax.barh(ordered.method, ordered.mean_native_normalized_mse,
                color=["tab:blue" if grid == "256" else "tab:orange" for grid in ordered.input_grid])
        ax.invert_yaxis()
        ax.set(xlabel="Native-grid normalized MSE on the held-out 42 captures", xscale="log",
               title="Blue: 256-point input; orange: full native input (additional raw samples)")
        fig.savefig(root/"heldout_native_comparison.png", dpi=160)
        fig.savefig(root/"heldout_native_comparison.pdf")
        plt.close(fig)
    splits = pd.read_csv(root/"sample_splits.csv")
    fig, axes = plt.subplots(int(np.ceil(len(results)/3)), 3,
                            figsize=(16, 3.3*int(np.ceil(len(results)/3))), squeeze=False, layout="constrained")
    for ax, row in zip(axes.flat, results.itertuples()):
        scores = np.load(root/f"{row.method}_scores.npy")
        logs = np.log10(np.maximum(scores, 1e-14))
        for split, color in [("calibration", "tab:blue"), ("test", "tab:orange")]:
            ax.hist(logs[splits.split.eq(split)], bins=45, alpha=.5, color=color, density=True, label=split)
        ax.axvline(np.log10(max(row.threshold, 1e-14)), color="tab:red", ls="--", label="Median + 5 std")
        ax.set(title=row.method, xlabel="log10(native normalized MSE)", ylabel="Density")
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(results):]:
        ax.set_visible(False)
    fig.savefig(root/"score_distributions.png", dpi=150)
    plt.close(fig)
    cells = [
        '<!doctype html><html lang="ja"><meta charset="utf-8"><title>MBL AE comparison</title>',
        '<style>body{font-family:system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#172033}img{max-width:100%}table{border-collapse:collapse;font-size:14px}th,td{border:1px solid #d5dae2;padding:6px}th{background:#eef2f8}section{margin:32px 0}.note{background:#fff5db;padding:16px}</style>',
        '<h1>MBL Dwn0-Cmp1 オートエンコーダー比較</h1>',
        '<p>36PV共通のモデルを各手法で学習。公式holdoutの42 cycleは学習・検証・閾値校正・PCAのfitから除外しています。</p>',
        '<p class="note">元データはPCA＋kNNで選別した正常候補です。上流の選別には全データを使っているため、holdoutから完全に独立した選別ではありません。異常ラベル付きMBLデータは未提供であり、候補率は確定した誤検知率ではありません。PV・beam_modeを加えた公式42件での値は、メタデータ入力を追加した評価です。W&amp;Bへは送信していません。</p>',
        '<h2>データ分割</h2>', pd.DataFrame({'waveforms':config['counts'],'cycles':config['cycles']}).to_html(),
        '<h2>MSE比較</h2><img src="mse_comparison.png">',
        '<p>左は全手法を同じ元波形の点列と学習側で固定したPV別スケールで評価したMSEです。右は256点モデルだけを公式42件・公式の正規化で評価したMSEです。</p>',
        '<h2>全手法の独自テスト結果</h2>', results.to_html(index=False, float_format=lambda x:f'{x:.6g}'),
        '<h2>公式42件とkickoffの掲載スコアとの比較</h2><p>kickoffは掲載値0.0194との比較のみです。kickoffモデルの再学習は行っていません。改善率は(1 − 今回のMSE / 0.0194) × 100です。</p>', official.to_html(index=False, float_format=lambda x:f'{x:.6g}'),
        '<p><a href="kickoff_score_comparison.csv">kickoff掲載値との比較CSV</a></p>',
        '<h2>同じ42件の元波形での評価</h2><p>元波形入力モデルには5032点／35221点の追加サンプルを渡しています。この表のMSEは元波形の点列・PV別スケールでの値であり、kickoff掲載値0.0194とは直接比較しません。</p>',
        native_holdout.to_html(index=False, float_format=lambda x:f'{x:.6g}'),
        '<img src="heldout_native_comparison.png">',
        '<h2>ラーニングカーブ</h2><p>各手法の学習対象に対するMSEです。PCA＋AEは標準化した主成分係数のMSEを学習lossとして使います。</p>',
        '<p><a href="learning_curves/learning_curves.pdf">全学習曲線のPDF</a> · <a href="comparison.csv">比較CSV</a></p>',
        '<img src="learning_curves/all_models.png">',
        '<h2>誤差分布と閾値</h2><img src="score_distributions.png">',
    ]
    cells.append('<p>モデル表示：<select id="model-choice"><option value="all">すべて</option>'+''.join(f'<option value="{html.escape(name)}">{html.escape(name)}</option>' for name, _ in histories)+'</select></p>')
    for name, history in histories:
        safe = html.escape(name)
        cells.extend([f'<section data-model="{safe}"><h2>{safe}</h2><a href="{safe}_history.csv">epochごとのloss CSV</a>',
                      f'<img src="learning_curves/{safe}.png"><img src="{safe}_reconstructions.png"></section>'])
    cells.append('<script>document.getElementById("model-choice").addEventListener("change",function(){document.querySelectorAll("section[data-model]").forEach(section=>{section.hidden=this.value!=="all"&&section.dataset.model!==this.value;});});</script></html>')
    (root/"report.html").write_text('\n'.join(cells))
    print(root/"report.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="mbl_autoencoder_comparison")
    report(parser.parse_args().directory)
