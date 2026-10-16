"""Plot retained ensemble metrics without needing intermediate shot logs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/clara-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def describe(values):
    x=np.asarray(values,float)
    return dict(n=len(x),mean=float(x.mean()),std=float(x.std(ddof=1)) if len(x)>1 else 0.,median=float(np.median(x)),q25=float(np.quantile(x,.25)),q75=float(np.quantile(x,.75)),min=float(x.min()),max=float(x.max())) if len(x) else dict(n=0)


def summarize(directory):
    directory=Path(directory);data=json.loads((directory/'metrics.json').read_text());methods=['bo','rcds'];colors=['#2374ab','#df8632'];report={}
    for m in methods:
        branches=[e['branches'][m] for e in data['episodes']];complete=[b for b in branches if b['status']=='completed']
        report[m]=dict(status_counts=dict(Counter(b['status'] for b in branches)),
            all_attempt_shot_histogram=dict(sorted(Counter(b['counts']['steps'] for b in branches).items())),
            completed_shots=describe([b['counts']['steps'] for b in complete]),
            completed_shots_including_reset=describe([b['counts']['shots_including_resets'] for b in complete]),
            completed_branch_wall_time_s=describe([b['wall_time']['total_s'] for b in complete if b['wall_time']]))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    positions=np.arange(4);width=.36
    for i,(m,c) in enumerate(zip(methods,colors)):
        counts=report[m]['all_attempt_shot_histogram'];y=[counts.get(k,0) for k in (0,4,6,192)]
        bars=axes[0].bar(positions+(i-.5)*width,y,width,color=c,label=m.upper())
        axes[0].bar_label(bars,labels=[str(v) if v else '' for v in y],padding=3,fontsize=10)
    axes[0].set(xticks=positions,xticklabels=['0\nnot started','4\nreplay stopped','6\nreplay stopped','192\ncompleted'],ylabel='Number of episodes',xlabel='Shots per method (reset excluded)',title='Observed shot-count distribution',ylim=(0,110))
    axes[0].legend();axes[0].grid(axis='y',alpha=.2);axes[0].set_axisbelow(True)
    left=np.zeros(2)
    for name,values,c in [('Threading replay',[6,6],'#929da5'),('Optics search + verification',[108,108],'#51a99a'),('Final centering + verification',[78,78],'#966eae')]:
        axes[1].barh([0,1],values,left=left,label=name,color=c)
        for j,v in enumerate(values):
            if v>15:axes[1].text(left[j]+v/2,j,str(v),ha='center',va='center',color='white',weight='bold')
        left+=values
    axes[1].set(yticks=[0,1],yticklabels=['BO','RCDS'],xlim=(0,207),xlabel='Shots',title='Completed-run budget: 192 shots each')
    for j in range(2):axes[1].text(6,j+.28,'replay: 6',fontsize=9,ha='left')
    axes[1].legend(loc='upper left',bbox_to_anchor=(0,-.13),fontsize=9,frameon=False)
    fig.suptitle('100 stochastic Cheetah episodes · fixed (1, 1) mm target',fontsize=14)
    fig.savefig(directory/'shot_count_distribution.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    values=[]
    for m in methods:
        values.append([e['branches'][m]['wall_time']['total_s'] for e in data['episodes'] if e['branches'][m]['status']=='completed' and e['branches'][m]['wall_time']])
    bins=np.linspace(min(map(min,values))-.2,max(map(max,values))+.2,19)
    for m,c,x in zip(methods,colors,values):ax.hist(x,bins=bins,alpha=.55,color=c,label=f'{m.upper()} (n={len(x)})',edgecolor='white')
    ax.set(xlabel='Simulation compute wall time [s] — not machine time',ylabel='Number of episodes',title='Compute-time distribution for completed branches')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend()
    fig.savefig(directory/'wall_time_distribution.png',dpi=180);plt.close(fig)
    report['notes']=['Wall times are local simulator/optimizer compute times, not accelerator acquisition time; they include no hardware setting, settling or trigger latency.','Completed branches have 192 shots excluding reset, or 193 including reset.','The two methods use fixed search budgets; this does not estimate shots needed to converge.','192 = 6 replay + (35*3+3) optics + (25*3+3) final centering.','Excluding replay, optics and final centering use 186 shots; optics alone uses 108.','Shared threading discovery costs another 6 shots per episode; it is not added twice to branch counts.','BO completed 99 branches but full branch wall time was retained for only 98; its completed shot counts remain available for all 99.']
    (directory/'shot_count_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');summarize(p.parse_args().directory)
