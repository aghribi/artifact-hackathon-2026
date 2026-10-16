"""Render the 2D SVD and a symmetric-target slice."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from . import SizeModes


def main():
    out=Path(__file__).parent/'results'/'size_svd'
    with np.load(out/'grid_svd.npz') as d:
        axis=d['spot_sizes_m']*1e3;s=d['singular_values'][:,:,:36];vt=d['Vt'];names=d['quadrupoles'].tolist()
    fraction=np.cumsum(s**2,axis=2)/np.sum(s**2,axis=2,keepdims=True)
    count=(fraction<.99).sum(axis=2)+1
    gap=np.min((s[:,:,:-1]-s[:,:,1:])/s[:,:,:-1],axis=2)
    fields=[s[:,:,0],s[:,:,1],s[:,:,2],count,fraction[:,:,4],gap]
    titles=['Largest singular value $s_1$ [m²]','$s_2$ [m²]','$s_3$ [m²]',
            'Modes for 99% of squared sensitivity','Top-five squared sensitivity fraction','Smallest relative adjacent singular gap']
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    for j,(ax,z,title) in enumerate(zip(axes.flat,fields,titles)):
        kwargs={'norm':LogNorm(vmin=z.min(),vmax=z.max())} if j in [0,1,2,5] else {}
        im=ax.pcolormesh(axis,axis,z.T,shading='auto',cmap='viridis',**kwargs)
        ax.set(xscale='log',yscale='log',xlabel=r'Target $\sigma_x$ [mm]',ylabel=r'Target $\sigma_y$ [mm]',title=title)
        colorbar=fig.colorbar(im,ax=ax)
        if j == 3:colorbar.set_ticks(np.arange(int(z.min()),int(z.max())+1))
    fig.suptitle('CLARA weighted fractional-beta response SVD versus IP target sizes\n'
                 '49 × 49 exact-response samples · 36 resolved modes · zero VBC · wakes off',fontsize=14)
    fig.savefig(out/'size_modes_summary.png',dpi=170);fig.savefig(out/'size_modes_summary.pdf');plt.close(fig)
    model=SizeModes();sizes=np.geomspace(.1,20,301);spectra=[];vectors=[];reference=None
    for size in sizes:
        r=model.evaluate(size*1e-3,size*1e-3,reference_Vt=reference)
        reference=r['Vt'][:2];spectra.append(r['singular_values']);vectors.append(r['Vt'][:2])
    spectra=np.array(spectra);vectors=np.array(vectors)
    fig,axes=plt.subplots(3,1,figsize=(15,12),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[1.1,1,1]})
    for j in range(36):
        axes[0].plot(sizes,spectra[:,j],lw=1.4 if j<5 else .6,alpha=1 if j<5 else .45,label=f'Mode {j+1}' if j<5 else None)
    axes[0].set(yscale='log',ylabel='Singular value [m²]',title='All 36 non-zero singular values')
    axes[0].legend(ncol=5);axes[0].grid(alpha=.2,which='both')
    labels=[n.removeprefix('cla_').replace('_mag_quad_','/Q').upper() for n in names]
    for mode,ax in enumerate(axes[1:]):
        limit=np.max(abs(vectors[:,mode]));im=ax.pcolormesh(sizes,np.arange(46),vectors[:,mode].T,cmap='RdBu_r',vmin=-limit,vmax=limit,shading='auto')
        ax.set_yticks(np.arange(0,46,2),labels[::2],fontsize=7)
        ax.set(ylabel='Quadrupole (alternate labels)',title=f'Mode {mode+1}: signed k1-direction coefficients')
        fig.colorbar(im,ax=ax,label='Dimensionless coefficient')
    for ax in axes:ax.set_xscale('log');ax.set_xlim(.1,20)
    axes[-1].set_xlabel(r'Symmetric target $\sigma_x=\sigma_y$ [mm]')
    fig.suptitle('Symmetric-target slice of the 2D SVD family',fontsize=15)
    fig.savefig(out/'symmetric_size_modes.png',dpi=170);fig.savefig(out/'symmetric_size_modes.pdf');plt.close(fig)
    np.savez_compressed(out/'symmetric_slice.npz',spot_sizes_m=sizes*1e-3,singular_values=spectra,leading_Vt=vectors,quadrupoles=np.array(names))
    print(out/'size_modes_summary.png')

if __name__=='__main__':main()
