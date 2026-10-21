"""Export scientific figures and an offline report for the AE comparison."""
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from waveform_anomaly_comparison import RawCache

HERE=Path(__file__).resolve().parent


def report(root=HERE/"waveform_autoencoder_comparison"):
    root=Path(root)
    comparison=pd.read_csv(root/"comparison.csv")
    config=json.loads((root/"configuration.json").read_text())
    known=pd.read_csv(HERE/"waveform_anomaly_comparison/known_anomalies.csv")
    cache=RawCache(HERE/"anomaly-pca-reuse-20261015")
    waves=cache.read(known.original_row.astype(int).tolist())
    labels={"raw":"元波形 AE","decimate4":"4点ごとに間引き → AE","mean4":"4点区間平均 → AE",
        "mean16":"16点区間平均 → AE","minmax16":"16点最小・最大値 → AE"}
    for k in [32,64,128]:labels[f"pca{k}"]=f"PCA {k}成分（参考）";labels[f"pca{k}_ae"]=f"PCA {k}成分 → AE"
    for row_index,item in enumerate(known.itertuples()):
        fig,axes=plt.subplots(4,3,figsize=(18,13),layout="constrained")
        for ax,result in zip(axes.flat,comparison.to_dict("records")):
            name=result['method']
            with np.load(root/(name+"_known_reconstructions.npz")) as saved:recon=saved[str(item.original_row)]
            x=waves[row_index];step=max(1,int(np.ceil(len(x)/4000)));positions=np.arange(0,len(x),step)
            ax.plot(positions,x[::step],color="#2563eb",lw=.8,label="Original")
            if recon.shape[0]==2:
                ax.fill_between(positions,np.minimum(recon[0],recon[1])[::step],np.maximum(recon[0],recon[1])[::step],color="#e58c24",alpha=.5,label="Reconstructed bounds")
            else:ax.plot(positions,recon[0,::step],color="#dc5726",lw=.8,label="Reconstructed")
            score=np.load(root/(name+"_scores.npy"))[item.original_row]
            ax.set_title(f"{name}: {'DETECTED' if score>result['threshold'] else 'MISSED'}\nscore {score:.4g} / threshold {result['threshold']:.4g}",fontsize=10)
            ax.set_xlabel("Original sample index");ax.grid(alpha=.2);ax.legend(fontsize=7)
        for ax in axes.flat[len(comparison):]:ax.set_visible(False)
        fig.suptitle(f"User-confirmed anomaly: {item.pv}\ncycle_id {item.cycle_id}; original units and sample positions",fontsize=15)
        fig.savefig(root/f"known_{item.original_row}_reconstructions.png",dpi=150);plt.close(fig)
    # Every scored source capture is shown, including excluded cycles evaluated only after fitting.
    coordinates=pd.read_csv(HERE/'waveform_anomaly_comparison/classification.csv',usecols=['original_row','global_PC1'])
    x=coordinates.global_PC1.to_numpy()
    fig,axes=plt.subplots(4,3,figsize=(18,16),layout='constrained')
    for ax,result in zip(axes.flat,comparison.to_dict('records')):
        scores=np.load(root/(result['method']+'_scores.npy'))
        if result['calibration_std']<=0:raise ValueError('Zero calibration standard deviation')
        z=(scores-result['calibration_median'])/result['calibration_std']
        valid=np.isfinite(z);candidate=scores>result['threshold']
        ax.set_yscale('symlog',linthresh=1)
        ax.scatter(x[valid&~candidate],z[valid&~candidate],s=1.4,c='#526d89',alpha=.12,linewidths=0,rasterized=True)
        ax.scatter(x[valid&candidate],z[valid&candidate],s=5,c='#d12f3a',alpha=.6,linewidths=0,rasterized=True)
        idx=known.original_row.to_numpy(dtype=int)
        ax.scatter(x[idx],z[idx],s=85,c='#edb52e',edgecolors='#614500',marker='*',linewidths=.7,zorder=5)
        ax.axhline(5,color='#ba2431',ls='--',lw=1.4)
        ax.set_title(f"{result['method']} | known {result['known_detected']}/2 | candidates {result['all_candidates']:,}",fontsize=10)
        ax.set_xlabel('Shared waveform PC1 (visualization only)');ax.set_ylabel('(Error - calibration median) / std')
        ax.grid(alpha=.15)
    for ax in axes.flat[len(comparison):]:ax.set_visible(False)
    fig.suptitle('All waveforms and actual median + 5 std thresholds; gold stars = confirmed anomalies',fontsize=16)
    fig.supxlabel('All 218,527 nonempty source waveforms, no subsampling. Coordinates reused only for visualization; PCA inputs were fitted on training data only.',fontsize=10)
    fig.savefig(root/'all_waveforms_thresholds.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(18,8),layout='constrained')
    learned=comparison.loc[~comparison.method.str.fullmatch(r'pca\d+')]
    for ax,item in zip(axes.flat,learned.itertuples()):
        history=pd.read_csv(root/(item.method+'_history.csv'))
        ax.plot(history.epoch,history.train_loss,label='Training');ax.plot(history.epoch,history.validation_loss,label='Validation')
        ax.set_title(item.method);ax.set_xlabel('epoch');ax.set_ylabel('Representation MSE');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(root/'learning_curves.png',dpi=150);plt.close(fig)
    rows=[]
    for result in comparison.to_dict('records'):
        name=result['method']
        variance='—' if pd.isna(result['pca_explained_variance']) else f"{result['pca_explained_variance']:.2%}"
        rows.append(f'<tr><td>{html.escape(labels[name])}</td><td>{result["known_detected"]}/2</td><td>{result["test_candidates"]:,}/{result["test_waveforms"]:,} ({result["test_candidate_rate"]:.2%})</td><td>{result["all_candidates"]:,}</td><td>{result["candidate_cycles"]}</td><td>{result["epochs"]}</td><td>{result["training_seconds"]:.1f}</td><td>{variance}</td><td><a href="{name}_candidates.csv">候補CSV</a></td></tr>')
    pics=''.join(f'<h2>{html.escape(item.pv)} ／ cycle_id {item.cycle_id}</h2><a href="known_{item.original_row}_reconstructions.png"><img src="known_{item.original_row}_reconstructions.png" alt="元波形と復元波形"></a>' for item in known.itertuples())
    page='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>波形AEの比較</title>
<style>body{font:15px/1.7 system-ui,sans-serif;background:#f4f7fb;color:#172b42;margin:0}main{max-width:1550px;margin:auto;padding:24px}h1{font-size:26px}h2{font-size:19px}table{border-collapse:collapse;background:white;font-size:13px}td,th{padding:9px 12px;border-bottom:1px solid #dce4ed;text-align:left}img{width:100%;background:white}.table{overflow:auto}a{color:#1263b2}.note{color:#53657a;font-size:13px}</style>
<main><h1>元波形・間引き・区間集約・PCAによるオートエンコーダーの比較</h1>
<p>AEは全PV共通で、PV名（末尾PMData／DODDataを統合）とbeam_modeを条件入力にしています。元波形AEはサンプル数を変更せずに入力し、同じ長さで復元します。長さごとにバッチをまとめますがモデルは分けません。各手法の分割は共通です。</p>
<p>学習入力は <code>dd_all_cycles_without_global_pca_knn_5sigma_cycles.pkl</code>。6空波形を除き、学習141,567波形、モデル選択19,295波形、閾値校正19,588波形、テスト19,561波形です。同一アーカイブ・同一サイクルは複数の分割に入りません。除外49サイクル18,516波形は評価だけに使っています。</p>
<p>閾値は校正用データの復元誤差の<b>中央値＋5σ</b>（σ＝標準偏差、ddof=0）です。AEとPCAと標準化係数は学習用データだけで学習しました。PCA＋AEはPCAで落ちた成分も含めた、元波形に対する復元誤差で判定します。min/maxだけは復元した上下包絡の外側への誤差です。</p>
<p class="note">残したデータは正常候補で、正常確認済みではありません。テスト候補率は確定した誤検知率ではありません。確認済み異常は2波形だけで、これだけから一般的な検出精度は推定しません。2波形はAE学習・モデル選択・閾値校正に使っていません。PCA単体は波形だけの参考基準です。学習時間はAE学習のみで、共通PCAの前処理時間を含みません。30epochまで改善を続けたモデルは追加学習で確認が必要です。</p>
<div class="table"><table><thead><tr><th>手法</th><th>既知異常の検出</th><th>未確認テスト候補</th><th>全データ候補</th><th>候補サイクル</th><th>epoch</th><th>AE学習秒</th><th>PCA保持分散</th><th>出力</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p><a href="comparison.csv">比較CSV</a> · <a href="known_anomaly_results.csv">既知異常のスコア・閾値</a> · <a href="predictions.csv">全波形の全判定</a> · <a href="sample_splits.csv">分割と元データ行ID</a> · <a href="configuration.json">学習条件</a> · <a href="../kickoff_autoencoder_comparison.ipynb">比較ノートブック</a></p>
<img src="comparison.png" alt="候補率と学習時間"><h2>学習曲線</h2><img src="learning_curves.png" alt="学習と検証の復元誤差"><h2>全波形の集合と実際の閾値</h2><p>赤破線より上が候補、金色の星は確認済み異常です。横軸は表示だけの共通座標です。</p><a href="all_waveforms_thresholds.png"><img src="all_waveforms_thresholds.png" alt="全波形と閾値"></a>__PICS__</main></html>'''
    (root/'index.html').write_text(page.replace('__ROWS__',''.join(rows)).replace('__PICS__',pics))
    print(comparison[['method','known_detected','test_candidates','all_candidates','candidate_cycles','epochs']].to_string(index=False))
    print(root/'index.html')


if __name__=='__main__':report()
