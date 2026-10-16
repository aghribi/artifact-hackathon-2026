"""Compile complete per-node VBC-on/L4H-drift responses and weighted SVD controls."""
from pathlib import Path
import csv,json,shutil,hashlib
import numpy as np
from .interpolation import align_modes,compile_surface

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'orbit_matching/results/vbc_on_l4h_off/orbit';DATA=ROOT/'orbit_matching/data'
BACKENDS=['cheetah','xsuite']


def compile_results():
    from optics_matching_vbc_on import MatchedOptics
    optics=MatchedOptics(l4h_off=True);axis=optics.axis;n=len(axis)
    k1=optics.data['k1'];quadrupoles=np.array(optics.quadrupoles)
    transmission_paths=[OUT.parent/f'transmission_{backend}.npz' for backend in BACKENDS]
    if all(p.exists() for p in transmission_paths):
        identity=json.loads((OUT.parent/'transmission_inputs.json').read_text())
        for name,expected in identity.items():
            if name=='particles_requested':continue
            actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            if actual!=expected:raise RuntimeError('Transmission inputs changed: rerun transmission before attaching validity metadata')
    all_data={b:[] for b in BACKENDS}
    for backend in BACKENDS:
        for i in range(n):
            row=[]
            for j in range(n):
                with np.load(OUT/'checkpoints'/f'{backend}_{i:02d}_{j:02d}.npz',allow_pickle=False) as archive:row.append({k:archive[k] for k in archive.files})
            all_data[backend].append(row)
    first=all_data[BACKENDS[0]][0][0]
    response_bundle=dict(schema_version=2,rank_relative_cutoff=1e-8,k1=k1,quadrupoles=quadrupoles,backends=np.array(BACKENDS),spot_sizes_m=axis,
        monitors=first['x_monitors'],row_weights=first['x_row_weights'],vbc_angle_rad=optics.vbc_angle_rad,l4h_off=True,
        monitor_s_m=np.array([all_data[b][0][0]['x_monitor_s_m'] for b in BACKENDS]))
    summary=dict(schema_version=2,rank_relative_cutoff=1e-8,l4h='drift',vbc_angle_rad=optics.vbc_angle_rad,
        grid_shape=[n,n],particles_per_response=229,weights=dict(bpm=1.,ip=5.,ip_row_multiplier=float(np.sqrt(5))),backends={})
    # Archive the former VBC-off physics, keeping it available for comparison.
    old=DATA/'orbit_response_model.npz'
    if old.exists():
        with np.load(old) as d:legacy='schema_version' not in d or int(d['schema_version'])!=2
        if legacy:
            archive=DATA/'legacy_vbc_off';archive.mkdir(exist_ok=True)
            for name in ['orbit_response_model.npz','orbit_modes_x.npz','orbit_modes_y.npz','calibration.json','mode_calibration.json']:
                if (DATA/name).exists():shutil.move(str(DATA/name),str(archive/name))
    for plane in 'xy':
        merged={};coefficients=[];gauges=[];gauge_ranks=[]
        for backend in BACKENDS:
            grid=all_data[backend]
            keys=['response','weighted_response','response_step','response_half_step','cross_plane_response','U','singular_values','Vt','rank','reference_orbit_m','step_relative_change']
            arrays={key:np.array([[grid[i][j][f'{plane}_{key}'] for j in range(n)] for i in range(n)]) for key in keys}
            arrays['rank']=np.sum(arrays['singular_values']>arrays['singular_values'][:,:,:1]*1e-8,axis=2)
            minimum=int(arrays['rank'].min());maximum=int(arrays['rank'].max())
            raw=arrays['Vt'][:,:,:minimum];aligned=align_modes(raw)
            signs=np.sign(np.einsum('ijmc,ijmc->ijm',aligned,raw))
            arrays['Vt'][:,:,:minimum]=aligned;arrays['U'][:,:,:,:minimum]*=signs[:,:,None,:]
            tx,ty,rc=compile_surface(np.log(axis),arrays['weighted_response'])
            _,_,gc=compile_surface(np.log(axis),arrays['Vt'])
            coefficients.append(rc);gauges.append(gc);gauge_ranks.append(minimum)
            for key,value in arrays.items():merged.setdefault(key,[]).append(value)
            metadata=dict(rank_relative_cutoff=1e-8,spot_sizes_m=axis,k1=k1,quadrupoles=quadrupoles,vbc_angle_rad=optics.vbc_angle_rad,l4h_off=True,correctors=first[f'{plane}_correctors'],monitors=first[f'{plane}_monitors'],row_weights=first[f'{plane}_row_weights'],monitor_s_m=grid[0][0][f'{plane}_monitor_s_m'])
            np.savez_compressed(OUT/f'{backend}_{plane}_grid_svd.npz',**arrays,**metadata)
            weights=first[f'{plane}_row_weights'];sv=arrays['singular_values'];fraction=np.cumsum(sv**2,axis=2)/np.sum(sv**2,axis=2,keepdims=True)
            counts=(fraction<.99).sum(axis=2)+1
            reconstruction=np.einsum('ijrm,ijm,ijmc->ijrc',arrays['U'],sv,arrays['Vt'])
            reconstruction_error=np.linalg.norm(reconstruction-arrays['weighted_response'],axis=(2,3))/np.linalg.norm(arrays['weighted_response'],axis=(2,3))
            summary['backends'].setdefault(backend,{})[plane]=dict(shape=list(arrays['response'].shape[2:]),rank_range=[minimum,maximum],
                n_modes_99percent_range=[int(counts.min()),int(counts.max())],max_step_relative_change=float(arrays['step_relative_change'].max()),
                max_reconstruction_relative_error=float(reconstruction_error.max()),max_cross_plane_response_m_per_rad=float(np.max(abs(arrays['cross_plane_response']))))
            with (OUT/f'{backend}_{plane}_response_rows.csv').open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['sigma_x_m','sigma_y_m','monitor','row_weight',*metadata['correctors']])
                for i,x in enumerate(axis):
                    for j,y in enumerate(axis):
                        for r,name in enumerate(metadata['monitors']):writer.writerow([x,y,name,weights[r],*arrays['response'][i,j,r]])
            with (OUT/f'{backend}_{plane}_right_singular_vectors.csv').open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['sigma_x_m','sigma_y_m','mode','singular_value_m_per_rad','resolved',*metadata['correctors']])
                for i,x in enumerate(axis):
                    for j,y in enumerate(axis):
                        for mode,s in enumerate(sv[i,j]):writer.writerow([x,y,mode+1,s,mode<int(arrays['rank'][i,j]),*arrays['Vt'][i,j,mode]])
        for key in ['response','response_step','response_half_step','reference_orbit_m']:response_bundle[f'{plane}_{key}']=np.array(merged[key])
        response_bundle[f'{plane}_correctors']=first[f'{plane}_correctors']
        np.savez_compressed(DATA/f'orbit_modes_{plane}.npz',schema_version=2,rank_relative_cutoff=1e-8,backends=np.array(BACKENDS),spot_sizes_m=axis,
            correctors=first[f'{plane}_correctors'],monitors=first[f'{plane}_monitors'],row_weights=first[f'{plane}_row_weights'],
            knots_x=tx,knots_y=ty,response_coefficients=np.array(coefficients),gauge_coefficients=np.array(gauges),
            rank=int(np.max(merged['rank'])),**{('rank_grid' if key=='rank' else key):np.array(value) for key,value in merged.items()})
    transmission_paths=[OUT.parent/f'transmission_{backend}.npz' for backend in BACKENDS]
    if all(p.exists() for p in transmission_paths):
        tracked=[]
        for p in transmission_paths:
            with np.load(p) as d:tracked.append({k:d[k] for k in d.files})
        response_bundle.update(full_bunch_transmission=np.array([d['transmission'] for d in tracked]),
            bpm_valid=np.array([d['bpm_valid'] for d in tracked]),
            ip_readout_valid=np.array([d['screen_stats'][:,:,4].astype(bool) for d in tracked]),
            particles_per_transmission=np.array([d['num_particles'] for d in tracked]))
    np.savez_compressed(DATA/'orbit_response_model.npz',**response_bundle)
    from .compatibility import record_calibration,record_modes
    record_calibration(DATA,optics=optics);record_modes(DATA)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':compile_results()
