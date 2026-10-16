"""Build and validate 2D size-parametrized SVD interpolation.

Run from the repository root: python -m optics_matching.build_size_modes
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from . import MatchedOptics

ROOT=Path(__file__).parent
OUT=ROOT/'results'/'size_svd'


def compile_surface(axis, values):
    first=make_interp_spline(axis,values,k=3,axis=0)
    second=make_interp_spline(axis,first.c,k=3,axis=1)
    return first.t,second.t,np.moveaxis(second.c,0,1)


def align_modes(vectors):
    """Maximum-overlap spanning-tree gauge; retain sorted mode indices."""
    n=len(vectors);rank=vectors.shape[2];v=vectors.copy()
    node=np.arange(n*n).reshape(n,n)
    a=np.r_[node[:-1].ravel(),node[:,:-1].ravel()]
    b=np.r_[node[1:].ravel(),node[:,1:].ravel()]
    original=v.reshape(n*n,rank,46)
    for mode in range(rank):
        overlap=np.einsum('ij,ij->i',original[a,mode],original[b,mode])
        graph=coo_matrix((-np.maximum(abs(overlap),1e-15),(a,b)),shape=(n*n,n*n)).tocsr()
        tree=minimum_spanning_tree(graph)
        tree=(tree+tree.T).tocsr()
        signs=np.zeros(n*n);root=(n//2)*n+n//2
        row=original[root,mode];signs[root]=1 if row[np.argmax(abs(row))]>=0 else -1
        stack=[root]
        while stack:
            i=stack.pop()
            for j in tree.indices[tree.indptr[i]:tree.indptr[i+1]]:
                if signs[j]==0:
                    signs[j]=signs[i]*(1 if original[i,mode]@original[j,mode]>=0 else -1)
                    stack.append(j)
        original[:,mode]*=signs[:,None]
    return v


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--points',type=int,default=49)
    parser.add_argument('--source',type=Path,help='Reuse an existing grid_svd/probe npz')
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    model=MatchedOptics();rank=36
    reference=model.optics_response(.001,.001)
    if args.source:
        with np.load(args.source) as d:
            axis=d['axis'].copy() if 'axis' in d else d['spot_sizes_m'].copy()
            spectra=d['s'].copy() if 's' in d else d['singular_values'].copy()
            raw=d['Vt'].copy();ip_rows=d['ip_rows'].copy()
    else:
        axis=np.geomspace(1e-4,.02,args.points)
        spectra=np.empty((len(axis),len(axis),46));raw=np.empty((len(axis),len(axis),rank,46));ip_rows=np.empty((len(axis),len(axis),2,46))
        for i,x in enumerate(axis):
            for j,y in enumerate(axis):
                a=model.optics_response(x,y)['weighted_response']
                _,s,vt=np.linalg.svd(a,full_matrices=False)
                spectra[i,j]=s;raw[i,j]=vt[:rank];ip_rows[i,j]=a[-2:]
            if i%8==0:print(f'Computed row {i+1}/{len(axis)}',flush=True)
    n=len(axis)
    if not np.all(np.sum(spectra > spectra[:,:,:1]*1e-10,axis=2)==rank):
        raise ValueError("Resolved rank changes on this grid; revise the mode bundle")
    dx=np.einsum('ijmk,ijmk->ijm',raw[1:],raw[:-1])
    dy=np.einsum('ijmk,ijmk->ijm',raw[:,1:],raw[:,:-1])
    flux=dx[:,:-1]*dy[1:]*dx[:,1:]*dy[:-1]
    aligned=align_modes(raw)
    tx,ty,response_c=compile_surface(np.log(axis),ip_rows)
    _,_,gauge_c=compile_surface(np.log(axis),aligned)
    labels=[f'{screen}:ln_beta_{plane}' for screen in reference['screens'] for plane in ('x','y')]
    np.savez_compressed(ROOT/'data'/'size_modes.npz',rank=rank,spot_sizes_m=axis,
        knots_x=tx,knots_y=ty,response_coefficients=response_c,gauge_coefficients=gauge_c,
        constant_weighted_response=reference['weighted_response'],quadrupoles=np.array(model.quadrupoles),
        row_labels=np.array(labels),row_weights=reference['row_weights'])
    np.savez_compressed(OUT/'grid_svd.npz',spot_sizes_m=axis,singular_values=spectra,Vt=aligned,
        ip_rows=ip_rows,quadrupoles=np.array(model.quadrupoles),row_weights=reference['row_weights'],
        negative_sign_plaquettes=flux<0)
    from .size_modes import SizeModes
    modes=SizeModes();rng=np.random.default_rng(501)
    points=np.exp(rng.uniform(np.log(axis[0]),np.log(axis[-1]),(128,2)))
    rows=[];orth=[];reconstruction=[]
    for x,y in points:
        result=modes.evaluate(x,y);interp=modes.evaluate(x,y,exact_response=False)
        a=result['weighted_response'];u,s,vt=np.linalg.svd(a,full_matrices=False)
        vi=interp['Vt'];dots=np.abs(np.einsum('ij,ij->i',vi,vt[:rank]))
        angle=np.degrees(np.arccos(np.clip(dots,-1,1)))
        # A subspace metric avoids false failures caused solely by near-degenerate rotations.
        top2_angle=np.degrees(np.arccos(np.clip(np.linalg.svd(vi[:2]@vt[:2].T,compute_uv=False).min(),-1,1)))
        rows.append([x,y,result['relative_interpolation_error'],np.max(abs(interp['singular_values']/s[:rank]-1)),np.max(angle),np.max(angle[:2]),top2_angle,float(np.min(result['gauge_overlap']))])
        orth.append(np.linalg.norm(result['Vt']@result['Vt'].T-np.eye(rank)))
        reconstruction.append(np.linalg.norm((result['U']*result['singular_values'])@result['Vt']-a)/np.linalg.norm(a))
    rows=np.array(rows)
    header=['sigma_x_m','sigma_y_m','raw_interpolation_response_relative_error','raw_interpolation_max_singular_relative_error','raw_interpolation_max_right_vector_angle_deg','raw_interpolation_top2_right_vector_angle_deg','raw_interpolation_top2_subspace_angle_deg','minimum_gauge_overlap']
    np.savetxt(OUT/'heldout_validation.csv',rows,delimiter=',',header=','.join(header),comments='')
    gaps=(spectra[:,:,:rank-1]-spectra[:,:,1:rank])/spectra[:,:,:rank-1]
    edge_dots=np.r_[np.einsum('ijmk,ijmk->ijm',aligned[1:],aligned[:-1]).reshape(-1,rank),np.einsum('ijmk,ijmk->ijm',aligned[:,1:],aligned[:,:-1]).reshape(-1,rank)]
    summary=dict(grid_shape=[n,n],resolved_rank=rank,relative_rank_cutoff=1e-10,
        minimum_resolved_singular_value=float(spectra[:,:,rank-1].min()),
        maximum_discarded_singular_value=float(spectra[:,:,rank:].max()),
        minimum_relative_adjacent_gap_by_pair=np.min(gaps,axis=(0,1)).tolist(),
        negative_sign_plaquettes_by_mode=np.sum(flux<0,axis=(0,1)).tolist(),
        remaining_negative_gauge_edges_by_mode=np.sum(edge_dots<0,axis=0).tolist(),
        maximum_unoriented_adjacent_angle_deg_by_mode=np.degrees(np.arccos(np.clip(np.min(abs(edge_dots),axis=0),0,1))).tolist(),
        heldout_points=len(points),raw_interpolation_max_response_relative_error=float(rows[:,2].max()),
        raw_interpolation_max_singular_relative_error=float(rows[:,3].max()),
        raw_interpolation_max_right_vector_angle_deg=float(rows[:,4].max()),
        raw_interpolation_top2_max_right_vector_angle_deg=float(rows[:,5].max()),
        default_max_orthogonality_error=float(max(orth)),default_max_relative_reconstruction_error=float(max(reconstruction)),
        top2_minimum_squared_fraction=float(np.min(np.sum(spectra[:,:,:2]**2,axis=2)/np.sum(spectra[:,:,:rank]**2,axis=2))),
        method='Cubic log-size response/vector gauge interpolation, sorted SVD, default exact analytic response refinement. No new particle tracking.',
        continuity_caveat='Negative plaquettes indicate unresolved rotations or degeneracies; they do not alone prove a true degeneracy. Sorted singular vectors have local smooth gauges, not a certified global smooth orientation.')
    # Closed-loop local transport, with midpoint subdivision to resolve sign transport.
    path=np.array([[.001,.001],[.002,.001],[.002,.002],[.001,.002],[.001,.001]])
    transported=modes.transport_path(path,n_modes=rank)
    summary['closed_path_transport']={k:v for k,v in transported.items() if k!='results'}
    summary['closed_path_transport']['return_signs']=np.sign(np.einsum('ij,ij->i',transported['results'][0]['Vt'],transported['results'][-1]['Vt'])).astype(int).tolist()
    (OUT/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    examples=[dict(sigma_x_mm=x*1e3,sigma_y_mm=y*1e3,variables=modes.ranked_variables(x,y)) for x,y in [(1e-4,1e-4),(.001,.001),(.001,.01),(.02,.02)]]
    (OUT/'dominant_variables_examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
