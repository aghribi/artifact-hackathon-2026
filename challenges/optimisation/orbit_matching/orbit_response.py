"""Current VBC-on/L4H-drift orbit responses for both tracking backends."""
from pathlib import Path
from collections import OrderedDict
import numpy as np
from optics_matching_vbc_on import MatchedOptics
from .compatibility import CalibrationGuard,StaleCalibrationError

DEFAULT=Path(__file__).parent/'data'/'orbit_response_model.npz'


class OrbitResponse:
    def __init__(self,path=DEFAULT,*,optics=None,backend='xsuite'):
        with np.load(path,allow_pickle=False) as d:self.data={k:d[k] for k in d.files}
        if int(self.data.get('schema_version',0))!=2:raise StaleCalibrationError('This response is from the previous VBC-off model. Run python -m orbit_matching.build_vbc_orbits.')
        self.backend=backend;self._backend=self.data['backends'].tolist().index(backend)
        self._matching=optics or MatchedOptics(l4h_off=True)
        self.monitors=self.data['monitors'].tolist();self.row_weights=self.data['row_weights'].copy()
        self.axis=self.data['spot_sizes_m'];self._engine=None;self._cache=OrderedDict()
        self._guard=CalibrationGuard(Path(path).parent,self._matching);self.check_compatibility()

    def check_compatibility(self):self._guard.check()

    def _node(self,x,y):
        ix=np.flatnonzero(np.isclose(x,self.axis,rtol=1e-10,atol=0));iy=np.flatnonzero(np.isclose(y,self.axis,rtol=1e-10,atol=0))
        return (int(ix[0]),int(iy[0])) if len(ix) and len(iy) else None

    def _metadata(self,plane):
        return dict(plane=plane,backend=self.backend,correctors=self.data[f'{plane}_correctors'].tolist(),monitors=self.monitors,
            row_weights=self.row_weights.copy(),monitor_s_m=self.data['monitor_s_m'][self._backend].copy())

    def evaluate(self,sigma_x_m,sigma_y_m,plane='x'):
        self.check_compatibility()
        if plane not in ('x','y'):raise ValueError("plane must be 'x' or 'y'")
        # Validate the requested domain through the public provider contract.
        tune=self._matching.evaluate(sigma_x_m,sigma_y_m)
        node=self._node(sigma_x_m,sigma_y_m)
        if node is not None:
            i,j=node;r=self.data[f'{plane}_response'][self._backend,i,j].copy()
            result=dict(**self._metadata(plane),response=r,weighted_response=np.sqrt(self.row_weights)[:,None]*r,
                at_grid_node=True,reference_orbit_m=self.data[f'{plane}_reference_orbit_m'][self._backend,i,j].copy())
            if 'full_bunch_transmission' in self.data:
                result.update(transmission=float(self.data['full_bunch_transmission'][self._backend,i,j]),
                    bpm_valid=self.data['bpm_valid'][self._backend,i,j].copy(),
                    ip_readout_valid=bool(self.data['ip_readout_valid'][self._backend,i,j]))
            return result
        key=(float(sigma_x_m),float(sigma_y_m))
        if key not in self._cache:
            self._ensure_engine();self._cache[key]=self._engine.evaluate(*key)
            if len(self._cache)>16:self._cache.popitem(last=False)
        result=dict(self._cache[key][plane]);result.update(backend=self.backend,at_grid_node=False)
        return result

    def _ensure_engine(self):
        if self._engine is None:
            from .vbc_reference import ReferenceOrbitEngine
            # Geometry is part of the calibration; injected providers need only
            # the original named-settings/fingerprint contract.
            geometry=MatchedOptics(l4h_off=True);provider=self._matching
            class Bound:
                axis=geometry.axis
                vbc_angle_rad=geometry.vbc_angle_rad
                l4h_off=True
                def evaluate(self,x,y,**kwargs):return provider.evaluate(x,y,**kwargs)
            self._engine=ReferenceOrbitEngine(self.backend,optics=Bound())

    def at_k1(self,k1,plane='x'):
        """Evaluate a supplied full strength vector in the provider's quadrupole order."""
        self.check_compatibility()
        if plane not in ('x','y'):raise ValueError("plane must be 'x' or 'y'")
        values=np.asarray(k1,dtype=float)
        names=self.data['quadrupoles'].tolist()
        if values.shape!=(len(names),) or not np.isfinite(values).all():raise ValueError('One finite strength per quadrupole is required')
        self._ensure_engine();original=self._engine.optics
        settings=dict(zip(names,map(float,values)))
        class Fixed:
            def evaluate(self,x,y):return dict(settings=settings,k1=values)
        try:
            self._engine.optics=Fixed()
            result=self._engine.evaluate(float(self.axis[0]),float(self.axis[0]))[plane]
        finally:self._engine.optics=original
        return dict(result,backend=self.backend,at_grid_node=False)
