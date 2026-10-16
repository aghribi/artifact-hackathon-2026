"""Plot per-node transmission in both L4H configurations and updated orbit SVDs."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT=Path(__file__).parent;RESULTS=ROOT/'results'


def transmission(case):
    out=RESULTS/case;data={}
    for backend in ['cheetah','xsuite']:
        with np.load(out/f'transmission_{backend}.npz') as d:data[backend]={k:d[k] for k in d.files}
    axis=data['cheetah']['spot_sizes_m']*1e3
    t=[100*data[b]['transmission'] for b in ['cheetah','xsuite']]
    difference=t[1]-t[0];limit=max(float(np.max(abs(difference))),.001)
    fig,axes=plt.subplots(1,3,figsize=(19,7),layout='constrained')
    for index,(ax,z,title) in enumerate(zip(axes,[*t,difference],['Cheetah: second-order tracking','Xsuite: thick-element tracking','Xsuite − Cheetah [percentage points]'])):
        kwargs=dict(cmap='viridis',vmin=0,vmax=100) if index<2 else dict(cmap='RdBu_r',vmin=-limit,vmax=limit)
        image=ax.pcolormesh(axis,axis,z.T,shading='nearest',**kwargs)
        ax.set(xscale='log',yscale='log',xlabel=r'Requested $\sigma_x$ [mm]',ylabel=r'Requested $\sigma_y$ [mm]',title=title)
        ax.set_xticks(axis,[f'{v:.2g}' for v in axis],rotation=45,fontsize=8)
        ax.set_yticks(axis,[f'{v:.2g}' for v in axis],fontsize=8)
        for i,x in enumerate(axis):
            for j,y in enumerate(axis):
                color=('white' if z[i,j]<45 else 'black') if index<2 else ('white' if abs(z[i,j])>.65*limit else 'black')
                label=f'{z[i,j]:.1f}' if index<2 else f'{z[i,j]:+.2f}'
                ax.text(x,y,label,ha='center',va='center',fontsize=6.8,color=color)
        fig.colorbar(image,ax=ax,label='Transmission [% of launched bunch]' if index<2 else 'Difference [percentage points]',shrink=.85)
    off=case.endswith('l4h_off')
    fig.suptitle('CLARA rematched 12 × 12 optics: '+('L4H replaced by a drift' if off else 'supplied lattice, L4H powered')+'\n'
        'VBC 131.69935 mrad · 200,000 identical launched particles · apertures on · no wakefields',fontsize=15)
    fig.supxlabel('Same rematched quadrupole settings in both L4H cases; labels identify requested targets, not achieved sizes.',fontsize=10)
    fig.savefig(out/'transmission_comparison.png',dpi=180);fig.savefig(out/'transmission_comparison.pdf');plt.close(fig)
    with (out/'transmission_nodes.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['sigma_x_m','sigma_y_m','cheetah_transmission','xsuite_transmission','xsuite_minus_cheetah_percentage_points','cheetah_on_chip_fraction','xsuite_on_chip_fraction'])
        for i,x in enumerate(axis):
            for j,y in enumerate(axis):writer.writerow([x*1e-3,y*1e-3,t[0][i,j]/100,t[1][i,j]/100,difference[i,j],data['cheetah']['on_screen_fraction'][i,j],data['xsuite']['on_screen_fraction'][i,j]])
    summary={b:dict(min=float(d['transmission'].min()),median=float(np.median(d['transmission'])),max=float(d['transmission'].max()),nodes_below_10percent=int((d['transmission']<.1).sum()),nodes_at_least_95percent=int((d['transmission']>=.95).sum())) for b,d in data.items()}
    summary['max_backend_difference_percentage_points']=float(np.max(abs(difference)))
    (out/'comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def orbit():
    out=RESULTS/'vbc_on_l4h_off/orbit';fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    for row,plane in enumerate('xy'):
        fields=[];matrices=[]
        for backend in ['cheetah','xsuite']:
            with np.load(out/f'{backend}_{plane}_grid_svd.npz') as d:
                axis=d['spot_sizes_m']*1e3;fields.append(d['singular_values'][:,:,0]);matrices.append(d['weighted_response'])
        difference=100*np.linalg.norm(matrices[1]-matrices[0],axis=(2,3))/np.linalg.norm(matrices[1],axis=(2,3))
        for j,z in enumerate([*fields,difference]):
            ax=axes[row,j];image=ax.pcolormesh(axis,axis,z.T,shading='nearest',norm=LogNorm(),cmap='viridis')
            ax.set(xscale='log',yscale='log',xlabel=r'Requested $\sigma_x$ [mm]',ylabel=r'Requested $\sigma_y$ [mm]',
                title=('Horizontal' if plane=='x' else 'Vertical')+': '+(['Cheetah $s_1$','Xsuite $s_1$','relative matrix difference'][j]))
            fig.colorbar(image,ax=ax,label='m/rad' if j<2 else '%')
    fig.suptitle('VBC-on, L4H-drift orbit responses across all 144 settings\nIP squared-error weight 5; each BPM weight 1',fontsize=15)
    fig.savefig(out/'orbit_svd_comparison.png',dpi=180);fig.savefig(out/'orbit_svd_comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(15,9),layout='constrained');i,j=5,7
    for col,plane in enumerate('xy'):
        for b,backend in enumerate(['cheetah','xsuite']):
            with np.load(out/f'{backend}_{plane}_grid_svd.npz') as d:
                s=d['singular_values'][i,j];v=d['Vt'][i,j,0];names=d['correctors'].tolist();axis=d['spot_sizes_m']*1e3
            axes[0,col].semilogy(np.arange(1,len(s)+1),s,'o-' if b==0 else 'x--',label=backend,ms=4)
            x=np.arange(len(names));axes[1,col].bar(x+(b-.5)*.38,v,width=.38,label=backend)
        axes[0,col].set(title=('Horizontal' if plane=='x' else 'Vertical')+' singular values',xlabel='Sorted mode',ylabel='m/rad');axes[0,col].legend();axes[0,col].grid(alpha=.2)
        labels=[n.removeprefix('cla_').replace('_mag_hvcor_','/C').replace('_mag_vcor_','/VC').upper() for n in names]
        axes[1,col].set_xticks(np.arange(len(names)),labels,rotation=90,fontsize=7)
        axes[1,col].set(title='Leading right-singular-vector coefficients',ylabel='Dimensionless angle-direction coefficient');axes[1,col].legend()
    fig.suptitle(f'Example grid node: {axis[i]:.5g} × {axis[j]:.5g} mm requested RMS\nVBC on; L4H drift; weighted SVD',fontsize=15)
    fig.savefig(out/'example_modes.png',dpi=180);fig.savefig(out/'example_modes.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',choices=['vbc_on','vbc_on_l4h_off','all'],default='all')
    args=parser.parse_args();summary={}
    for case in (['vbc_on','vbc_on_l4h_off'] if args.case=='all' else [args.case]):summary[case]=transmission(case)
    orbit();print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
