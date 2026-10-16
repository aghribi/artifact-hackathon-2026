"""Validate simultaneous injected probes against independent magnet finite differences."""
from pathlib import Path
import json
import numpy as np
import torch
import cheetah
from nominal_model import xsuite_model as nominal
from .vbc_reference import ReferenceOrbitEngine

OUT=Path(__file__).parent/'results/vbc_on_l4h_off/orbit'


def single_orbit(engine):
    if engine.backend=='xsuite':
        p=nominal.electron_particles(engine.energy)
        engine.line.track(p,turn_by_turn_monitor='ONE_TURN_EBE')
        record=engine.line.record_last_track
        indices=[engine.boundaries[n] for n in engine.monitors]
        if int(p.state[0])<1:raise RuntimeError('Independent reference probe lost')
        return np.array([np.asarray(record.x)[0,indices],np.asarray(record.y)[0,indices]])
    beam=cheetah.ParticleBeam(particles=torch.tensor([[0.,0.,0.,0.,0.,0.,1.]],dtype=torch.float64),energy=torch.tensor(engine.energy,dtype=torch.float64))
    values=np.empty((2,len(engine.monitors)))
    with torch.no_grad():
        for chunk,(name,end) in zip(engine.chunks,engine.events):
            beam=chunk.track(beam)
            if name in engine.monitors:values[:,engine.monitors.index(name)]=[float(beam.x[0]),float(beam.y[0])]
    return values


def main():
    torch.set_num_threads(1);records=[]
    for backend in ['cheetah','xsuite']:
        engine=ReferenceOrbitEngine(backend)
        for i,j in [(0,0),(6,6),(11,11)]:
            x,y=engine.optics.axis[[i,j]];packet=engine.evaluate(x,y)
            for plane in 'xy':
                names=engine.correctors[plane];columns=sorted(set([0,len(names)//2,len(names)-2,len(names)-1]));row=0 if plane=='x' else 1
                actual=np.empty((len(engine.monitors),len(columns)))
                for c,column in enumerate(columns):
                    name=names[column]
                    if backend=='xsuite':target=getattr(engine.line[name],'knl' if plane=='x' else 'ksl');original=float(target[0]);sign=-1 if plane=='x' else 1
                    else:target=getattr(next(e for e in engine.segment.elements if e.name==name),'horizontal_angle' if plane=='x' else 'vertical_angle');original=float(target);sign=1
                    derivatives=[]
                    try:
                        for step in [1e-7,5e-8]:
                            if backend=='xsuite':target[0]=original+sign*step
                            else:target.fill_(original+step)
                            plus=single_orbit(engine)
                            if backend=='xsuite':target[0]=original-sign*step
                            else:target.fill_(original-step)
                            minus=single_orbit(engine)
                            derivatives.append((plus[row]-minus[row])/(2*step))
                        actual[:,c]=(4*derivatives[1]-derivatives[0])/3
                    finally:
                        if backend=='xsuite':target[0]=original
                        else:target.fill_(original)
                expected=packet[plane]['response'][:,columns];weights=np.sqrt(packet[plane]['row_weights'])[:,None]
                relative=float(np.linalg.norm(weights*(actual-expected))/np.linalg.norm(weights*actual))
                assert relative<1e-7,(backend,i,j,plane,relative)
                records.append(dict(backend=backend,i=i,j=j,plane=plane,correctors=[names[c] for c in columns],weighted_relative_error=relative,max_abs_error_m_per_rad=float(np.max(abs(actual-expected)))))
            print(f'{backend} independent check {i},{j} passed',flush=True)
    report=dict(checks=records,max_weighted_relative_error=max(r['weighted_relative_error'] for r in records))
    (OUT/'independent_reference_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
