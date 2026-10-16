"""Weighted orbit SVD controls for VBC-on rematched optics with L4H a drift."""
from pathlib import Path
import numpy as np
from scipy.interpolate import NdBSpline
from .orbit_response import OrbitResponse
from .compatibility import check_modes,StaleCalibrationError


class OrbitModes:
    def __init__(self,plane='x',bundle=None,*,optics=None,backend='xsuite'):
        if plane not in ('x','y'):raise ValueError("plane must be 'x' or 'y'")
        self.plane=plane;self.backend=backend
        path=Path(bundle) if bundle else Path(__file__).parent/'data'/f'orbit_modes_{plane}.npz'
        check_modes(path);self._bundle_path=path
        with np.load(path,allow_pickle=False) as d:self.data={k:d[k] for k in d.files}
        d=self.data
        if int(d.get('schema_version',0))!=2:raise StaleCalibrationError('Legacy orbit modes: rebuild for VBC-on/L4H drift.')
        self._backend=d['backends'].tolist().index(backend)
        self.rank=int(np.max(d['rank_grid'][self._backend]));self.axis=d['spot_sizes_m']
        self.correctors=d['correctors'].tolist();self.monitors=d['monitors'].tolist();self.row_weights=d['row_weights'].copy()
        knots=(d['knots_x'],d['knots_y'])
        self._response=NdBSpline(knots,d['response_coefficients'][self._backend],(3,3),extrapolate=False)
        self._gauge=NdBSpline(knots,d['gauge_coefficients'][self._backend],(3,3),extrapolate=False)
        self._exact=OrbitResponse(path.parent/'orbit_response_model.npz',optics=optics,backend=backend)

    def evaluate(self,sigma_x_m,sigma_y_m,n_modes=None,*,reference_Vt=None,exact_response=True):
        """Resolved U/s/Vt: exact saved nodes; direct reference probes off grid.

        Off-grid strengths are interpolated, but their transmission and achieved
        sizes are not validated. exact_response=False is diagnostic interpolation.
        """
        self._exact.check_compatibility();check_modes(self._bundle_path)
        xy=np.array([sigma_x_m,sigma_y_m],dtype=float)
        if not np.isfinite(xy).all() or np.any(xy<self.axis[0]) or np.any(xy>self.axis[-1]):raise ValueError('Both sizes must be in [0.1,20] mm')
        interpolated=np.asarray(self._response(np.log(xy)))
        node=self._exact._node(*xy);a=interpolated.copy()
        if exact_response and node is not None:
            i,j=node;b=self._backend
            a=self.data['weighted_response'][b,i,j].copy();u=self.data['U'][b,i,j].copy()
            s=self.data['singular_values'][b,i,j].copy();vt=self.data['Vt'][b,i,j].copy();rank=int(self.data['rank_grid'][b,i,j])
        else:
            if exact_response:a=self._exact.evaluate(*xy,self.plane)['weighted_response']
            u,s,vt=np.linalg.svd(a,full_matrices=False);rank=int(np.sum(s>s[0]*float(self.data.get('rank_relative_cutoff',1e-8))))
        count=rank if n_modes is None else n_modes
        if not isinstance(count,(int,np.integer)) or not 1<=count<=rank:raise ValueError(f'n_modes must be between 1 and {rank}')
        gauge=np.asarray(self._gauge(np.log(xy)));overlap=np.einsum('ij,ij->i',vt[:rank],gauge[:rank])
        gauge_overlap=abs(overlap)/np.maximum(np.linalg.norm(gauge[:rank],axis=1),1e-30)
        if reference_Vt is not None:
            reference=np.asarray(reference_Vt,dtype=float)
            if reference.ndim!=2 or reference.shape[1]!=len(self.correctors) or not 1<=len(reference)<=rank or not np.isfinite(reference).all():raise ValueError('Invalid reference_Vt')
            overlap[:len(reference)]=np.einsum('ij,ij->i',vt[:len(reference)],reference)
        signs=np.where(overlap<0,-1.,1.);vt[:rank]*=signs[:,None];u[:,:rank]*=signs
        result=dict(plane=self.plane,backend=self.backend,target_sigma_x_m=float(xy[0]),target_sigma_y_m=float(xy[1]),
            U=u[:,:count],singular_values=s[:count],Vt=vt[:count],rank=rank,weighted_response=a,
            response=a/np.sqrt(self.row_weights)[:,None],row_weights=self.row_weights.copy(),
            cumulative_squared_fraction=np.cumsum(s[:count]**2)/np.sum(s[:rank]**2),gauge_overlap=gauge_overlap[:count],
            relative_singular_gaps=(s[:rank-1]-s[1:rank])/s[:rank-1],discarded_fraction=float(np.linalg.norm(s[rank:])/np.linalg.norm(s)),
            relative_interpolation_error=float(np.linalg.norm(a-interpolated)/np.linalg.norm(a)),exact_response=exact_response,at_grid_node=node is not None,
            vbc_angle_rad=float(self._exact.data['vbc_angle_rad']),l4h_off=True)
        if node is not None and 'full_bunch_transmission' in self._exact.data:
            b=self._backend;i,j=node
            result.update(transmission=float(self._exact.data['full_bunch_transmission'][b,i,j]),
                bpm_valid=self._exact.data['bpm_valid'][b,i,j].copy(),
                ip_readout_valid=bool(self._exact.data['ip_readout_valid'][b,i,j]))
        return result

    def ranked_variables(self,sigma_x_m,sigma_y_m,n_modes=5,**kwargs):
        r=self.evaluate(sigma_x_m,sigma_y_m,n_modes,**kwargs)
        return [dict(mode=i+1,sensitivity_m_per_rad=float(s),
            cumulative_squared_fraction=float(r['cumulative_squared_fraction'][i]),
            coefficients={name:float(v) for name,v in zip(self.correctors,r['Vt'][i])},
            largest_components=[dict(corrector=self.correctors[j],coefficient=float(r['Vt'][i,j])) for j in np.argsort(-abs(r['Vt'][i]))[:5]])
            for i,s in enumerate(r['singular_values'])]

    def to_kicks(self,amplitudes,sigma_x_m,sigma_y_m,**kwargs):
        """Return delta kick angles in rad; these are not absolute lattice settings."""
        values=np.asarray(amplitudes,dtype=float)
        if values.ndim!=1 or not 1<=len(values)<=self.rank or not np.isfinite(values).all():raise ValueError('Invalid modal amplitudes')
        return self.evaluate(sigma_x_m,sigma_y_m,len(values),**kwargs)['Vt'].T@values

    def from_kicks(self,delta_kicks,sigma_x_m,sigma_y_m,**kwargs):
        values=np.asarray(delta_kicks,dtype=float)
        if values.shape!=(len(self.correctors),) or not np.isfinite(values).all():raise ValueError('One finite angle per corrector is required')
        return self.evaluate(sigma_x_m,sigma_y_m,**kwargs)['Vt']@values

    def projector(self,sigma_x_m,sigma_y_m,first_mode=1,last_mode=2):
        if not isinstance(first_mode,(int,np.integer)) or not isinstance(last_mode,(int,np.integer)) or not 1<=first_mode<=last_mode<=self.rank:raise ValueError('Invalid mode group')
        v=self.evaluate(sigma_x_m,sigma_y_m,last_mode)['Vt'][first_mode-1:last_mode]
        return v.T@v

    def transport_path(self,target_sizes_m,n_modes=None,**kwargs):
        """Align sorted-vector signs along a path; see SizeModes.transport_path."""
        from .interpolation import transport_path
        return transport_path(self,target_sizes_m,n_modes,**kwargs)

    def n_modes_for_fraction(self,sigma_x_m,sigma_y_m,fraction=.99):
        if not 0<fraction<=1:raise ValueError('fraction must be in (0,1]')
        r=self.evaluate(sigma_x_m,sigma_y_m)
        return min(self.rank,int(np.searchsorted(r['cumulative_squared_fraction'],fraction))+1)
