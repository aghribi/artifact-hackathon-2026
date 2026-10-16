"""Direct corrector-kick reference probes: all columns in one tracking pass.

Each corrector has independent +/-h and +/-h/2 probe particles. They receive
local canonical-angle increments immediately after that corrector; all other
probes continue unperturbed until their own injection point. No prefix inverse
or fixed-quadrupole approximation is used.
"""
import re
import numpy as np
import torch
import cheetah
from nominal_model import xsuite_model as nominal
from optics_matching_vbc_on import MatchedOptics,build_xsuite,build_cheetah,apply_to_xsuite,apply_to_cheetah


class ReferenceOrbitEngine:
    def __init__(self,backend='xsuite',optics=None):
        if backend not in ('xsuite','cheetah'):raise ValueError('Unknown backend')
        self.backend=backend;self.optics=optics or MatchedOptics(l4h_off=True)
        with np.load(nominal.INITIAL_DISTRIBUTION) as d:self.energy=float(d['energy_ev'])
        x,y=self.optics.axis[0],self.optics.axis[0]
        if backend=='xsuite':
            self.converted=build_xsuite(x,y,model=self.optics);self.line=self.converted.line
            self.boundaries=dict(self.converted.boundaries);names=list(self.boundaries)
        else:
            self.segment=build_cheetah(x,y,model=self.optics,dtype=torch.float64)
            self.boundaries={e.name:i+1 for i,e in enumerate(self.segment.elements)};names=list(self.boundaries)
        self.correctors={p:[n for n in names if re.fullmatch(r'.*_mag_(hvcor|'+('hcor' if p=='x' else 'vcor')+r')_[0-9]+',n)] for p in 'xy'}
        self.monitors=[n for n in names if '_dia_bpm_' in n]+[nominal.READOUT]
        self.controls=[(p,n) for p in 'xy' for n in self.correctors[p]]
        self.injections={}
        for j,(p,n) in enumerate(self.controls):self.injections.setdefault(n,[]).append((p,np.arange(1+4*j,5+4*j)))
        names=[n for n in names if n in self.injections or n in self.monitors]
        self.events=[(n,self.boundaries[n]) for n in names]
        if backend=='cheetah':
            self.chunks=[];start=0
            for name,end in self.events:
                self.chunks.append(cheetah.Segment(self.segment.elements[start:end]));start=end
        if backend=='xsuite':
            ends=np.r_[0,np.cumsum([float(getattr(self.line[n],'length',0.)) for n in self.line.element_names])]
            self.monitor_s_m=ends[[self.boundaries[n] for n in self.monitors]]
        else:
            ends=np.r_[0,np.cumsum([float(e.length) for e in self.segment.elements])]
            self.monitor_s_m=ends[[self.boundaries[n] for n in self.monitors]]

    def evaluate(self,sigma_x_m,sigma_y_m,step=1e-7):
        if self.backend=='xsuite':apply_to_xsuite(self.converted,sigma_x_m,sigma_y_m,model=self.optics)
        else:
            # Match SimEnv's float32-written magnet settings, then evaluate
            # derivatives with float64 probes to avoid cancellation round-off.
            tune=self.optics.evaluate(sigma_x_m,sigma_y_m)
            elements={e.name:e for e in self.segment.elements}
            for n,k in tune['settings'].items():elements[n].k1.copy_(elements[n].k1.new_tensor(float(np.float32(k))))
        number=1+4*len(self.controls);record=np.empty((2,len(self.monitors),number));orbit=np.empty((2,len(self.monitors)))
        if self.backend=='xsuite':particles=nominal.electron_particles(self.energy,x=np.zeros(number))
        else:
            coords=torch.zeros((number,7),dtype=torch.float64);coords[:,-1]=1
            beam=cheetah.ParticleBeam(particles=coords,energy=torch.tensor(self.energy,dtype=torch.float64))
        start=0
        for event,(name,end) in enumerate(self.events):
            if self.backend=='xsuite':
                self.line.track(particles,ele_start=start,ele_stop=end);start=end
                ids=np.asarray(particles.particle_id);order=np.argsort(ids)
                if np.any(np.asarray(particles.state)[order]<1):raise RuntimeError(f'Reference probe lost at {name}; cannot define a full linear response')
                # Probe identity is retained explicitly even if xtrack reorganizes.
                for plane,slots in self.injections.get(name,[]):
                    index=np.searchsorted(ids[order],slots);actual=order[index]
                    getattr(particles,'px' if plane=='x' else 'py')[actual]+=np.array([step,-step,step/2,-step/2])
                if name in self.monitors:
                    i=self.monitors.index(name)
                    record[:,i]=np.array([np.asarray(particles.x)[order],np.asarray(particles.y)[order]])
            else:
                with torch.no_grad():
                    beam=self.chunks[event].track(beam)
                    if torch.any(beam.survival_probabilities<.5):raise RuntimeError(f'Reference probe lost at {name}')
                    for plane,slots in self.injections.get(name,[]):beam.particles[slots,1 if plane=='x' else 3]+=torch.tensor([step,-step,step/2,-step/2],dtype=torch.float64)
                if name in self.monitors:
                    i=self.monitors.index(name);record[:,i]=np.array([beam.x.numpy(),beam.y.numpy()])
        weights=np.r_[np.ones(len(self.monitors)-1),5.]
        result={}
        for plane in 'xy':
            columns=[j for j,(p,n) in enumerate(self.controls) if p==plane]
            base=np.array([1+4*j for j in columns])
            coarse=(record[:,:,base]-record[:,:,base+1])/(2*step)
            fine=(record[:,:,base+2]-record[:,:,base+3])/step
            rich=(4*fine-coarse)/3;row=0 if plane=='x' else 1
            response=rich[row]
            result[plane]=dict(response=response,weighted_response=np.sqrt(weights)[:,None]*response,
                response_step=coarse[row],response_half_step=fine[row],cross_plane_response=rich[1-row],
                correctors=self.correctors[plane],monitors=self.monitors,row_weights=weights,
                monitor_s_m=self.monitor_s_m,reference_orbit_m=record[row,:,0],step_rad=step,
                step_relative_change=float(np.linalg.norm(fine[row]-coarse[row])/np.linalg.norm(response)))
        return result
