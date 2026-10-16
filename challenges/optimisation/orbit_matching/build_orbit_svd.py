"""Tracked horizontal/vertical BPM+IP orbit response to corrector angles."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from nominal_model import xsuite_model as model

ROOT=Path(__file__).parent


def inventory(converted):
    names=[name for name,_ in converted.boundaries]
    horizontal=[n for n in names if '_mag_hvcor_' in n or '_mag_hcor_' in n]
    vertical=[n for n in names if '_mag_hvcor_' in n or '_mag_vcor_' in n]
    monitors=[n for n in names if '_dia_bpm_' in n]+[model.READOUT]
    return horizontal,vertical,monitors


def tracked_response(converted,step=1e-6):
    line=converted.line;horizontal,vertical,monitors=inventory(converted)
    boundaries=dict(converted.boundaries)
    indices=np.array([boundaries[n] for n in monitors])
    positions=np.asarray(line.get_table().s)[indices]
    def orbit():
        particle=model.electron_particles(converted.initial_energy_ev)
        line.track(particle,turn_by_turn_monitor='ONE_TURN_EBE')
        record=line.record_last_track
        if int(particle.state[0])!=1:raise RuntimeError('Reference probe was lost')
        return np.array([np.asarray(record.x)[0,indices],np.asarray(record.y)[0,indices]])
    reference=orbit();results={}
    for plane,correctors in [('x',horizontal),('y',vertical)]:
        field='knl' if plane=='x' else 'ksl';sign=-1 if plane=='x' else 1
        matrices=[]
        for h in [step,step/2]:
            response=np.empty((2,len(monitors),len(correctors)))
            for j,name in enumerate(correctors):
                coefficients=getattr(line[name],field);original=float(coefficients[0])
                try:
                    coefficients[0]=original+sign*h;plus=orbit()
                    coefficients[0]=original-sign*h;minus=orbit()
                    response[:,:,j]=(plus-minus)/(2*h)
                finally:coefficients[0]=original
            matrices.append(response)
        richardson=(4*matrices[1]-matrices[0])/3
        selected=0 if plane=='x' else 1
        r=richardson[selected];cross=richardson[1-selected]
        weights=np.ones(len(monitors));weights[-1]=5.
        weighted=np.sqrt(weights)[:,None]*r
        u,s,vt=np.linalg.svd(weighted,full_matrices=True)
        for j in range(len(s)):
            if vt[j,np.argmax(abs(vt[j]))]<0:vt[j]*=-1;u[:,j]*=-1
        rank=int(np.sum(s>s[0]*1e-10))
        causal=np.array([[boundaries[c]>=boundaries[b] for c in correctors] for b in monitors])
        reconstruction=(u[:,:len(s)]*s)@vt[:len(s)]
        results[plane]=dict(response=r,weighted_response=weighted,cross_plane_response=cross,
            response_step=matrices[0][selected],response_half_step=matrices[1][selected],
            U=u,singular_values=s,Vt=vt,rank=rank,correctors=np.array(correctors),
            monitors=np.array(monitors),row_weights=weights,monitor_s_m=positions,
            reference_orbit_m=reference[selected],
            step_relative_change=float(np.linalg.norm(matrices[1][selected]-matrices[0][selected])/np.linalg.norm(r)),
            reconstruction_relative_error=float(np.linalg.norm(weighted-reconstruction)/np.linalg.norm(weighted)),
            causal_zero_max_abs_m_per_rad=float(np.max(abs(r[causal]),initial=0)),
            cross_plane_max_abs_m_per_rad=float(np.max(abs(cross),initial=0)))
    return results


def save_results(results,out,metadata):
    out.mkdir(parents=True,exist_ok=True);summary=dict(metadata)
    for plane,d in results.items():
        np.savez_compressed(out/f'{plane}_orbit_svd.npz',**d)
        for filename,key in [('response.csv','response'),('weighted_response.csv','weighted_response')]:
            with (out/f'{plane}_{filename}').open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['monitor','s_m','weight',*d['correctors']])
                writer.writerows([name,s,w,*row] for name,s,w,row in zip(d['monitors'],d['monitor_s_m'],d['row_weights'],d[key]))
        s=d['singular_values'];fraction=np.cumsum(s**2)/np.sum(s**2)
        with (out/f'{plane}_singular_values.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['mode','singular_value_m_per_rad','cumulative_squared_fraction'])
            writer.writerows(zip(range(1,len(s)+1),s,fraction))
        summary[plane]=dict(shape=list(d['response'].shape),rank=d['rank'],singular_values_m_per_rad=s.tolist(),
            n_modes_99percent=int(np.searchsorted(fraction,.99)+1),
            step_relative_change=d['step_relative_change'],reconstruction_relative_error=d['reconstruction_relative_error'],
            causal_zero_max_abs_m_per_rad=d['causal_zero_max_abs_m_per_rad'],cross_plane_max_abs_m_per_rad=d['cross_plane_max_abs_m_per_rad'])
        assert d['reconstruction_relative_error']<1e-12
        assert d['causal_zero_max_abs_m_per_rad']<1e-9
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step',type=float,default=1e-6)
    parser.add_argument('--sigma-x-mm',type=float)
    parser.add_argument('--sigma-y-mm',type=float)
    parser.add_argument('--output',type=Path,default=ROOT/'results'/'nominal')
    args=parser.parse_args()
    if args.step<=0:parser.error('--step must be positive')
    with np.load(model.INITIAL_DISTRIBUTION) as beam:energy=float(beam['energy_ev'])
    converted=model.build_line(energy,wakefields=False)
    metadata=dict(optics='original nominal',wakefields=False,ip_proxy=model.READOUT,
        weights=dict(each_bpm=1.,ip=5.,ip_row_multiplier=float(np.sqrt(5))),
        corrector_coordinate='positive local reference-angle kick in rad; horizontal knl=-theta, vertical ksl=theta',
        observable='reference-orbit x or y in metres, about zero entrance reference coordinates',
        initial_energy_ev=energy,step_rad=args.step,
        derivative='Richardson-extrapolated central difference from h and h/2',
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [model.LATTICE_JSON,Path(model.__file__)]})
    if args.sigma_x_mm is not None or args.sigma_y_mm is not None:
        if args.sigma_x_mm is None or args.sigma_y_mm is None:parser.error('Provide both target sizes')
        from optics_matching import apply_to_xsuite
        apply_to_xsuite(converted,args.sigma_x_mm*1e-3,args.sigma_y_mm*1e-3)
        metadata.update(optics='matched size family',sigma_x_mm=args.sigma_x_mm,sigma_y_mm=args.sigma_y_mm)
    summary=save_results(tracked_response(converted,args.step),args.output,metadata)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
