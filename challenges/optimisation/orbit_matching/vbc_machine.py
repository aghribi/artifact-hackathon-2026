"""Current VBC-on physical lattice and SimEnv-equivalent aperture/diagnostic tracking."""
from pathlib import Path
import numpy as np
import cheetah
from nominal_model import xsuite_model as model
from .backend_transport import CheetahTransport,XsuiteTransport

ROOT=Path(__file__).resolve().parents[1]
GRID=ROOT/'optics_matching_vbc_on/results/rematched_grid_vbc_on.npz'
LATTICE=ROOT/'optics_matching_vbc_on/CLARA_cheetah_second_order.json'
VBC_DIPOLES=tuple(f'cla_vbc_mag_dip_{i:02d}' for i in range(1,5))


class VBCMachine:
    def __init__(self,backend,num_particles=200000,*,lattice_json=None):
        self._cached_prefix=None
        self.use_cached_prefix=True
        self.backend=backend
        lattice_json=Path(lattice_json) if lattice_json else LATTICE
        coordinates,self.energy_ev,_=model.load_distribution()
        self._coordinates=coordinates[:num_particles]
        self.segment=model.cheetah_segment(lattice_json,apertures=True)
        self.diagnostics={e.name:e for e in self.segment.elements if isinstance(e,cheetah.Screen)}
        self.stops=[n for n in self.diagnostics if any(tag in n for tag in ('dia_bpm','dia_scr','sim_aper'))]
        self.bpm_names=[n for n in self.diagnostics if 'dia_bpm' in n]
        if backend=='cheetah':self.transport=CheetahTransport(self.segment,self.energy_ev,self.stops)
        elif backend=='xsuite':self.transport=XsuiteTransport(model,self.segment,lattice_json,model.START,model.READOUT,self.energy_ev,self.stops,True)
        else:raise ValueError('Unknown backend')
        with np.load(GRID) as d:angle=float(d['vbc_angle_rad'])
        for n,sign in zip(VBC_DIPOLES,(-1,1,1,-1)):self.transport.write(n,'angle',sign*angle)

    def _run_cached_xsuite(self):
        transport=self.transport;split=transport.converted.line.element_names.index('cla_vbc_mag_dip_04')+1
        # The supplied grid has exactly the same pre-VBC settings at every node.
        if self._cached_prefix is None:
            particles=model._particles(self._coordinates,self.energy_ev);at=0;records=[]
            for name,index in transport._stops:
                if index>=split:break
                if index>at:transport._line.track(particles,ele_start=at,ele_stop=index)
                at=index
                x,y,alive=transport._ordered(particles,len(self._coordinates))
                records.append((name,x.copy(),y.copy(),alive.copy()))
            if split>at:transport._line.track(particles,ele_start=at,ele_stop=split)
            self._cached_prefix=(particles.copy(),records)
        cached,records=self._cached_prefix
        yield from records
        particles=cached.copy();at=split
        for name,index in transport._stops:
            if index<split:continue
            if index>at:transport._line.track(particles,ele_start=at,ele_stop=index)
            at=index
            x,y,alive=transport._ordered(particles,len(self._coordinates))
            yield name,x,y,alive

    def _simulate(self):
        number=len(self._coordinates);live=np.ones(number,dtype=bool);minimum=.1*number
        valid=np.zeros(len(self.bpm_names),dtype=np.int8);bpm_x=np.full(len(valid),np.nan);bpm_y=bpm_x.copy()
        iterator=self._run_cached_xsuite() if self.backend=='xsuite' and self.use_cached_prefix else self.transport.run(self._coordinates,self.stops)
        for name,x,y,alive in iterator:
            live&=alive;live&=x*x+y*y<=.05**2
            if name in self.bpm_names and live.sum()>=minimum:
                slot=self.bpm_names.index(name);valid[slot]=1;bpm_x[slot]=x[live].mean();bpm_y[slot]=y[live].mean()
            if name==model.READOUT:
                e=self.diagnostics[name];extent=tuple(float(v) for v in e.extent)
                on=live & (np.abs(x)<=extent[1]) & (np.abs(y)<=extent[3])
                fraction=float(on.sum())/number
                stats=(float(x[on].mean()),float(y[on].mean()),float(x[on].std()),float(y[on].std()),True) if on.sum()>=minimum else (0.,0.,0.,0.,False)
        self._readings=dict(transmission=float(live.sum())/number,on_screen_fraction=fraction,stats=stats,bpm_valid=valid,bpm_x=bpm_x,bpm_y=bpm_y)
