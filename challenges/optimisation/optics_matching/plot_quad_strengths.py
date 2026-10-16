"""Plot the complete matched tune; run as python -m optics_matching.plot_quad_strengths."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import MatchedOptics, build_cheetah


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sigma-mm', type=float, default=np.sqrt(2.0))
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent/'results')
    args = parser.parse_args()
    model = MatchedOptics()
    tune = model.evaluate(args.sigma_mm*1e-3, args.sigma_mm*1e-3)
    segment = build_cheetah(args.sigma_mm*1e-3, args.sigma_mm*1e-3, model=model)
    positions = {}; s = 0.0
    for element in segment.elements:
        length = float(element.length)
        if element.name in tune['settings']:
            positions[element.name] = s + length/2
        s += length
    names = model.quadrupoles
    assert set(names) == set(positions) and len(names) == 46
    z = np.array([positions[n] for n in names]); k = tune['k1']
    labels = [n.removeprefix('cla_').replace('_mag_quad_', '/Q').upper() for n in names]
    colors = ['#2474a6']*42 + ['#dc7432']*4
    with plt.rc_context({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False}):
        fig, (ax, detail) = plt.subplots(2,1, figsize=(15,8), gridspec_kw={'height_ratios':[1.2,1]}, layout='constrained')
        for axis, x in [(ax,z),(detail,np.arange(46))]:
            axis.axhline(0,color='0.4',lw=.8)
            axis.vlines(x,0,k,color=colors,lw=1.4)
            axis.scatter(x,k,c=colors,s=27,zorder=3)
            axis.set_ylabel(r'Quadrupole strength $k_1$ [m$^{-2}$]')
            axis.grid(axis='y',alpha=.2)
        ax.set_xlim(0,s*1.015)
        ax.set_xlabel('Quadrupole centre position from benchmark entrance [m]')
        ax.axvline(s,color='0.5',ls='--',lw=1)
        ax.text(s,.97,'IP proxy',transform=ax.get_xaxis_transform(),ha='right',va='top',fontsize=10)
        ax.scatter([],[],color=colors[0],label='Common upstream tune (42 quads)')
        ax.scatter([],[],color=colors[-1],label='Size-dependent final focus (4 quads)')
        ax.legend(loc='lower left',fontsize=10)
        detail.set_xticks(np.arange(46),labels,rotation=90,fontsize=8)
        detail.set_xlim(-1,46)
        detail.set_xlabel('Quadrupoles in beamline order (even spacing for identification)')
        fig.suptitle(f'CLARA matched quadrupole strengths — symmetric IP target {args.sigma_mm:.4g} mm × {args.sigma_mm:.4g} mm\n'
                     'Zero VBC excitation · wakefields off · L4H off',fontsize=14)
        args.output_dir.mkdir(parents=True,exist_ok=True)
        stem=args.output_dir/'symmetric_quad_strengths'
        fig.savefig(stem.with_suffix('.png'),dpi=180)
        fig.savefig(stem.with_suffix('.pdf'))
        plt.close(fig)
    with stem.with_suffix('.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['quadrupole','s_centre_m','k1_m_minus2','target_sigma_x_m','target_sigma_y_m'])
        writer.writerows((n,positions[n],float(v),args.sigma_mm*1e-3,args.sigma_mm*1e-3) for n,v in zip(names,k))
    print(stem.with_suffix('.png'))


if __name__ == '__main__':
    main()
