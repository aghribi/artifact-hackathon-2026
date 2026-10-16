"""Compact adaptive-shot results and comparison with the retained fixed-budget run."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/clara-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def stats(values):
    x=np.array(values,float)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),std=float(x.std(ddof=1)) if len(x)>1 else 0.,min=float(x.min()),max=float(x.max()),p10=float(np.quantile(x,.1)),p90=float(np.quantile(x,.9)))


def report(directory,baseline):
    directory=Path(directory);base=json.loads(Path(baseline).read_text());metrics=json.loads((directory/'metrics.json').read_text());episodes=[];summary={}
    for episode in metrics['episodes']:
        path=directory/f"episode_{episode['episode']:03d}/summary.json";workflow=json.loads(path.read_text());item=dict(episode=episode['episode'],seed=episode['seed'],methods={})
        for method in ('bo','rcds'):
            branch=episode['branches'][method];s=workflow['branches'][method];i=branch['final'];target=np.array(episode['config']['target'])
            item['methods'][method]=dict(shots=branch['counts']['steps'],score=branch['final_score'],physical_mismatch=i['scored_mismatch'],relative_error=i['scored_relative_error'],transmission=i['transmission'],on_screen_fraction=i['on_screen_fraction'],sigma_m=[i['sigma_x'],i['sigma_y']],centroid_m=[i['mu_x'],i['mu_y']],target_met=s['target_met'],stop_reasons=s['stop_reasons'],wall_time_s=branch['wall_time']['total_s'],status=branch['status'])
        episodes.append(item)
    for method in ('bo','rcds'):
        values=[e['methods'][method] for e in episodes];old={e['seed']:e['branches'][method] for e in base['episodes'] if e['branches'][method]['status']=='completed'};paired=[e for e in episodes if e['seed'] in old]
        summary[method]=dict(shots=stats([v['shots'] for v in values]),score=stats([v['score'] for v in values]),physical_mismatch=stats([v['physical_mismatch'] for v in values]),wall_time_s=stats([v['wall_time_s'] for v in values]),statuses=dict(Counter(v['status'] for v in values)),shot_histogram=dict(sorted(Counter(v['shots'] for v in values).items())),strict_final_target_met=sum(v['target_met'] for v in values),benchmark_hit_rates={str(t):float(np.mean([v['relative_error']<=t for v in values])) for t in (.1,.25,.5)},shot_reduction_vs_192=1-np.mean([v['shots'] for v in values])/192,
            optics_stops=dict(Counter(v['stop_reasons']['optics'] for v in values)),centering_stops=dict(Counter(v['stop_reasons']['centering'] for v in values)),
            paired=dict(n=len(paired),old_physical_mismatch_mean=float(np.mean([old[e['seed']]['final'].get('scored_mismatch_mean',old[e['seed']]['final']['scored_mismatch']) for e in paired])),new_physical_mismatch_mean=float(np.mean([e['methods'][method]['physical_mismatch'] for e in paired]))))
    output=dict(episodes=len(episodes),summary=summary,per_episode=episodes,notes=['Single-shot, stochastic final readings; fixed-budget reference used three-shot final averages.','All same 100 episode seeds and the fixed centered 1 mm target; no discovery or replay acquisition overhead.','Completed means beam at the IP; strict final target_met additionally requires each size within 15%, each centroid within 0.1 target sigma, and on-screen fraction >=90%.','Focus stopping emulates MBRL aggregate mismatch<=0.15 or patience=3; BO also uses MF-BO expected improvement versus marginal benchmark shot cost.','Wall times are simulator/model/optimizer compute durations, not real accelerator operating times.'])
    (directory/'adaptive_summary.json').write_text(json.dumps(output,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained');colors=['#2475ad','#df8738']
    maximum=max(v['shots'] for e in episodes for v in e['methods'].values());bins=np.arange(-.5,maximum+1.5)
    for method,color in zip(('bo','rcds'),colors):
        v=summary[method]['shots'];axes[0].hist([e['methods'][method]['shots'] for e in episodes],bins=bins,color=color,alpha=.55,label=f'{method.upper()}: mean {v["mean"]:.2f}, median {v["median"]:g}')
    axes[0].set(title='Adaptive shots (previous budget: 192 each)',xlabel='SimEnv steps per method (reset excluded)',ylabel='Episodes');axes[0].legend(fontsize=9)
    x=np.arange(2);w=.36
    old_means=[summary[m]['paired']['old_physical_mismatch_mean'] for m in ('bo','rcds')];new_means=[summary[m]['paired']['new_physical_mismatch_mean'] for m in ('bo','rcds')]
    for values,offset,label,color in [(old_means,-w/2,'Previous fixed budget','#9ba2a8'),(new_means,w/2,'Adaptive','#4caa91')]:
        bars=axes[1].bar(x+offset,values,w,label=label,color=color);axes[1].bar_label(bars,fmt='%.3f',padding=3)
    axes[1].set(xticks=x,xticklabels=['BO','RCDS'],ylabel='Mean physical mismatch (shot penalty excluded)',title='Residual quality on matched episode seeds');axes[1].legend(fontsize=9)
    from matplotlib.ticker import MaxNLocator
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[1].set_ylim(0,max(old_means+new_means)*1.3)
    fig.suptitle('100 Cheetah episodes · single-shot adaptive stopping',fontsize=14)
    fig.savefig(directory/'adaptive_shots_and_quality.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2));return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--baseline',default='model-svd_opt/results/cheetah_100_episodes/metrics.json');a=p.parse_args();report(a.directory,a.baseline)
