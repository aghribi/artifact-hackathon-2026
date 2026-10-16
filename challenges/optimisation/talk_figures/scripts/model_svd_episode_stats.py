"""Export current adaptive SVD episode results and talk histograms.

Run from the repository root; intermediate tracking files are not required.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/clara-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':8,
    'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.5,
    'figure.constrained_layout.use':True,'savefig.dpi':300})


def histogram(ax,data,method,metric,bins):
    values=np.array([e['methods'][method][metric] for e in data['per_episode']])
    color={'bo':'#2475ad','rcds':'#df8738'}[method]
    counts,edges,_=ax.hist(values,bins=bins,color=color,alpha=.8,edgecolor='white',linewidth=.45)
    assert int(counts.sum())==len(values), 'Histogram omitted an episode'
    ax.axvline(values.mean(),color='#263238',lw=1,label=f'Mean {values.mean():.2f}')
    ax.axvline(np.median(values),color='#702b79',lw=1,ls='--',label=f'Median {np.median(values):.2f}')
    ax.set(title=f'{method.upper()} · {len(values)} episodes',ylabel='Episodes',
        xlabel='Steps / shots per episode (reset excluded)' if metric=='shots' else 'Final benchmark score (lower is better)')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if metric=='shots':ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.legend(frameon=False)


def make(source,out):
    data=json.loads(source.read_text());assert data['episodes']==len(data['per_episode'])==100
    assert len({e['seed'] for e in data['per_episode']})==100
    for m in ('bo','rcds'):
        for metric,key in [('shots','shots'),('score','score'),('wall_time_s','wall_time_s')]:
            values=[e['methods'][m][metric] for e in data['per_episode']]
            assert np.isclose(np.mean(values),data['summary'][m][key]['mean'])
    metrics=source.parent/'metrics.json';batch=json.loads(metrics.read_text()) if metrics.exists() else None
    metadata=dict(policy='Adaptive model-assisted SVD BO/RCDS',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        episodes_per_method=100,backend='cheetah',seed_range=[42,141],target_rms_m=[.001,.001],target_centroid_m=[0.,0.],particles_per_shot=10000,
        steps_definition='One env.step per shot, including every correction and restoration; one reset reading per branch is excluded. No threading replay.',
        score_definition='Last actual single-shot SimEnv benchmark_score, including mismatch, loss and shot-count penalty; lower is better.',
        time_definition='Local simulation/model/optimizer compute wall time; no real-machine setting, settling, trigger or camera acquisition latency.',
        figures=[f'model_svd_opt_100_episodes_{name}.{ext}' for name in ('steps','score','all') for ext in ('png','pdf')])
    if batch:
        metadata.update(parallel_workers=batch['metadata']['workers'],batch_compute_wall_time_s=batch['metadata']['wall_time_including_combined_json_export_s'],
            total_steps=batch['aggregate']['total_steps'],reset_readings=batch['aggregate']['total_reset_shots'],screen_changes=batch['aggregate']['total_screen_changes'])
    export=dict(metadata=metadata,**data);out.mkdir(parents=True,exist_ok=True)
    (out/'model_svd_opt_100_episodes.json').write_text(json.dumps(export,indent=2)+'\n')
    bins={
        'shots':np.arange(.5,max(e['methods'][m]['shots'] for e in data['per_episode'] for m in ('bo','rcds'))+1.5),
        'score':np.linspace(0,np.ceil(max(e['methods'][m]['score'] for e in data['per_episode'] for m in ('bo','rcds'))*10)/10,18),
    }
    for metric,name in [('shots','steps'),('score','score')]:
        fig,axes=plt.subplots(1,2,figsize=(7.2,2.6),sharex=True,sharey=True)
        for ax,m in zip(axes,('bo','rcds')):histogram(ax,data,m,metric,bins[metric])
        for ext in ('png','pdf'):fig.savefig(out/f'model_svd_opt_100_episodes_{name}.{ext}')
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.1),sharey='row')
    for row,metric in enumerate(('shots','score')):
        for ax,m in zip(axes[row],('bo','rcds')):histogram(ax,data,m,metric,bins[metric])
    for ext in ('png','pdf'):fig.savefig(out/f'model_svd_opt_100_episodes_all.{ext}')
    plt.close(fig)
    return export

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path,nargs='?',default=Path('model-svd_opt/results/adaptive_100_episodes/adaptive_summary.json'))
    p.add_argument('out',type=Path,nargs='?',default=Path('talk_figures/model_svd_opt'))
    a=p.parse_args();d=make(a.source,a.out);print(json.dumps(d['metadata'],indent=2))
