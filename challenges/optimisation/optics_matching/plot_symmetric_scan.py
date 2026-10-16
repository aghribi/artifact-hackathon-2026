"""Plot k1 versus equal requested IP RMS sizes: python -m optics_matching.plot_symmetric_scan."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import MatchedOptics


def main():
    model = MatchedOptics()
    sizes_mm = np.geomspace(0.1, 20.0, 401)
    strengths = np.array([model.evaluate(s*1e-3, s*1e-3)['k1'] for s in sizes_mm])
    output = Path(__file__).parent/'results'/'symmetric_target_strength_scan'
    colors = list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('tab20b').colors) + list(plt.get_cmap('tab20c').colors)
    colors[42:46] = ['#0072B2', '#009E73', '#D55E00', '#CC79A7']
    with plt.rc_context({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False}):
        fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True,
                                 gridspec_kw={'height_ratios':[1.1,1]})
        handles = []
        for j, name in enumerate(model.quadrupoles):
            label = name.removeprefix('cla_').replace('_mag_quad_', '/Q').upper()
            style = ['-', '--', ':'][j//20] if j < 42 else ['-', '--', '-.', ':'][j-42]
            line, = axes[0].plot(sizes_mm, strengths[:,j], color=colors[j],
                                ls=style, lw=1.4, label=label)
            handles.append(line)
            if j >= 42:
                axes[1].plot(sizes_mm, strengths[:,j], color=colors[j], ls=style,
                             lw=2, label=label)
        axes[0].set_title('All 46 quadrupoles (42 constant upstream strengths)',loc='left')
        axes[1].set_title('Detail: four varying final-focus quadrupoles',loc='left')
        for ax in axes:
            ax.set_xscale('log')
            ax.set_xlim(.1,20)
            ax.set_ylabel(r'$k_1$ [m$^{-2}$]')
            ax.axhline(0,color='0.5',lw=.6,zorder=0)
            ax.grid(which='both',alpha=.18)
        axes[1].set_xticks([.1,.2,.5,1,2,5,10,20], ['0.1','0.2','0.5','1','2','5','10','20'])
        axes[1].set_xlabel(r'Symmetric target RMS size $\sigma_x=\sigma_y$ [mm]')
        axes[1].legend(ncol=2,fontsize=10)
        fig.suptitle('CLARA quadrupole strengths versus symmetric IP target size\n'
                     'Smooth corrected spline family · zero VBC · wakes off · L4H off',fontsize=15,y=.985)
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.015),
                   ncol=6,fontsize=9,title='All quadrupoles in beamline order')
        fig.subplots_adjust(top=.91,bottom=.275,hspace=.2,left=.09,right=.98)
        fig.savefig(output.with_suffix('.png'),dpi=180)
        fig.savefig(output.with_suffix('.pdf'))
        plt.close(fig)
    with output.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['target_sigma_x_mm','target_sigma_y_mm']+[n+'_k1_m_minus2' for n in model.quadrupoles])
        writer.writerows([float(s),float(s),*map(float,k)] for s,k in zip(sizes_mm,strengths))
    print(output.with_suffix('.png'))


if __name__ == '__main__':
    main()
