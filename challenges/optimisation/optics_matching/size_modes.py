"""Sorted quadrupole SVD modes versus two desired IP RMS sizes (metres).

A cubic response interpolant is re-diagonalized to retain orthonormal modes.
An interpolated vector field fixes signs; use reference_Vt for path transport.
Individual sorted modes need not be globally smooth near degeneracies in 2D.
"""
from pathlib import Path
import numpy as np
from scipy.interpolate import NdBSpline

DEFAULT = Path(__file__).parent/'data'/'size_modes.npz'


class SizeModes:
    def __init__(self, bundle=DEFAULT):
        with np.load(bundle, allow_pickle=False) as data:
            self.data = {k:data[k] for k in data.files}
        d = self.data
        self.rank = int(d['rank'])
        self.axis = d['spot_sizes_m']
        self.quadrupoles = d['quadrupoles'].tolist()
        self.row_labels = d['row_labels'].tolist()
        self.row_weights = d['row_weights'].copy()
        knots = (d['knots_x'], d['knots_y'])
        self._response = NdBSpline(knots, d['response_coefficients'], (3,3), extrapolate=False)
        self._gauge = NdBSpline(knots, d['gauge_coefficients'], (3,3), extrapolate=False)
        self._matching = None

    def evaluate(self, sigma_x_m, sigma_y_m, n_modes=None, *, reference_Vt=None,
                 exact_response=True):
        """Return sorted U, singular_values, Vt; Vt rows are k1 directions.

        Default: refine the interpolated SVD using the inexpensive analytic
        response at the corrected matched strengths. No particle tracking occurs.
        exact_response=False uses only the compiled response interpolation.
        A reference aligns signs locally, without permuting sorted modes.
        """
        xy = np.array([sigma_x_m, sigma_y_m], dtype=float)
        if not np.all(np.isfinite(xy)) or np.any(xy < self.axis[0]) or np.any(xy > self.axis[-1]):
            raise ValueError('Both RMS sizes must be between 0.1 and 20 mm')
        count = self.rank if n_modes is None else n_modes
        if not isinstance(count, (int,np.integer)) or not 1 <= count <= self.rank:
            raise ValueError(f'n_modes must be an integer between 1 and {self.rank}')
        point = np.log(xy)
        a = self.data['constant_weighted_response'].copy()
        a[-2:] = self._response(point)
        interpolated = a.copy()
        if exact_response:
            if self._matching is None:
                from . import MatchedOptics
                self._matching = MatchedOptics()
            a = self._matching.optics_response(*xy)['weighted_response']
        u,s,vt = np.linalg.svd(a, full_matrices=False)
        gauge = np.asarray(self._gauge(point))
        overlap = np.einsum('ij,ij->i',vt[:self.rank],gauge)
        gauge_overlap = np.abs(overlap)/np.maximum(np.linalg.norm(gauge,axis=1),1e-30)
        if reference_Vt is not None:
            reference = np.asarray(reference_Vt,dtype=float)
            if reference.ndim != 2 or reference.shape[1] != len(self.quadrupoles) or not 1 <= len(reference) <= self.rank or not np.isfinite(reference).all():
                raise ValueError('reference_Vt must contain 1 to rank finite rows and 46 columns')
            overlap[:len(reference)] = np.einsum('ij,ij->i',vt[:len(reference)],reference)
        signs = np.where(overlap<0,-1.,1.)
        vt[:self.rank] *= signs[:,None]
        u[:,:self.rank] *= signs[None,:]
        gaps = (s[:self.rank-1]-s[1:self.rank])/s[:self.rank-1]
        return dict(target_sigma_x_m=float(xy[0]),target_sigma_y_m=float(xy[1]),
            U=u[:,:count],singular_values=s[:count],Vt=vt[:count],
            weighted_response=a,interpolated_weighted_response=interpolated,
            relative_interpolation_error=float(np.linalg.norm(a-interpolated)/np.linalg.norm(a)),
            cumulative_squared_fraction=np.cumsum(s[:count]**2)/np.sum(s[:self.rank]**2),
            relative_singular_gaps=gaps,
            gauge_overlap=gauge_overlap[:count],
            discarded_fraction=float(np.linalg.norm(s[self.rank:])/np.linalg.norm(s)),
            exact_response=exact_response)

    def transport_path(self, target_sizes_m, n_modes=None, *, minimum_overlap=.98, max_depth=12, max_log_step=.05):
        """Continuously transport signs along a specified path in log-size space.

        Adaptive midpoint checks limit rotations, including through cells where
        a global spline sign convention has a seam. Actual degeneracies cannot
        be removed by sign alignment; unresolved segments are reported.
        Returns modes at requested points, plus internal-check diagnostics.
        """
        points = np.asarray(target_sizes_m,dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points)<1:
            raise ValueError('target_sizes_m must have shape (N,2), N>=1')
        if not 0 < minimum_overlap < 1 or max_depth < 0 or max_log_step <= 0:
            raise ValueError('Require 0 < minimum_overlap < 1 and max_depth >= 0')
        results=[self.evaluate(*points[0],n_modes)]
        stats=dict(internal_evaluations=0,unresolved_segments=0,minimum_accepted_overlap=1.)
        def advance(left_xy,left,right_xy,depth):
            right=self.evaluate(*right_xy,n_modes,reference_Vt=left['Vt'])
            stats['internal_evaluations']+=1
            overlap=float(np.min(np.einsum('ij,ij->i',left['Vt'],right['Vt'])))
            needs_subdivision = overlap<minimum_overlap or np.max(abs(np.log(right_xy/left_xy)))>max_log_step
            if needs_subdivision and depth<max_depth:
                middle_xy=np.exp((np.log(left_xy)+np.log(right_xy))/2)
                middle=advance(left_xy,left,middle_xy,depth+1)
                return advance(middle_xy,middle,right_xy,depth+1)
            if needs_subdivision:stats['unresolved_segments']+=1
            stats['minimum_accepted_overlap']=min(stats['minimum_accepted_overlap'],overlap)
            return right
        for i in range(1,len(points)):
            results.append(advance(points[i-1],results[-1],points[i],0))
        return dict(results=results,**stats)

    def ranked_variables(self, sigma_x_m, sigma_y_m, n_modes=8, **kwargs):
        result=self.evaluate(sigma_x_m,sigma_y_m,n_modes,**kwargs)
        return [dict(mode=i+1,sensitivity_m2=float(s),
            cumulative_squared_fraction=float(result['cumulative_squared_fraction'][i]),
            coefficients={q:float(v) for q,v in zip(self.quadrupoles,result['Vt'][i])},
            largest_components=[dict(quadrupole=self.quadrupoles[j],coefficient=float(result['Vt'][i,j]))
                                for j in np.argsort(-abs(result['Vt'][i]))[:5]])
            for i,s in enumerate(result['singular_values'])]

    def to_k1(self, amplitudes, sigma_x_m, sigma_y_m, **kwargs):
        """Matched strengths plus Vt.T @ amplitudes (amplitudes in m^-2).

        A finite modal perturbation generally changes the requested spot sizes
        and may violate apertures; this is a local response coordinate system.
        """
        amplitudes=np.asarray(amplitudes,dtype=float)
        if amplitudes.ndim!=1 or not 1<=len(amplitudes)<=self.rank or not np.isfinite(amplitudes).all():
            raise ValueError('amplitudes must be a finite vector with 1 to rank entries')
        modes=self.evaluate(sigma_x_m,sigma_y_m,len(amplitudes),**kwargs)
        if self._matching is None:
            from . import MatchedOptics
            self._matching=MatchedOptics()
        return self._matching.evaluate(sigma_x_m,sigma_y_m)['k1']+modes['Vt'].T@amplitudes

    def from_k1(self, k1, sigma_x_m, sigma_y_m, **kwargs):
        """Project an absolute tune onto resolved modes about the local match."""
        k1=np.asarray(k1,dtype=float)
        if k1.shape!=(len(self.quadrupoles),) or not np.isfinite(k1).all():
            raise ValueError('k1 must contain one finite setting per quadrupole')
        modes=self.evaluate(sigma_x_m,sigma_y_m,**kwargs)
        if self._matching is None:
            from . import MatchedOptics
            self._matching=MatchedOptics()
        return modes['Vt']@(k1-self._matching.evaluate(sigma_x_m,sigma_y_m)['k1'])

    def projector(self, sigma_x_m, sigma_y_m, first_mode=1, last_mode=2):
        """Sign-independent right-mode subspace projector (1-based inclusive).

        Smooth wherever the selected group is spectrally separated from its
        complement, including when its internal singular values coincide.
        """
        if not (isinstance(first_mode,(int,np.integer)) and isinstance(last_mode,(int,np.integer)) and 1<=first_mode<=last_mode<=self.rank):
            raise ValueError('Require 1 <= first_mode <= last_mode <= rank')
        result=self.evaluate(sigma_x_m,sigma_y_m,last_mode)
        basis=result['Vt'][first_mode-1:last_mode]
        return basis.T@basis

    def n_modes_for_fraction(self, sigma_x_m, sigma_y_m, fraction=.99):
        """Smallest sorted prefix accounting for a squared-response fraction."""
        if not 0 < fraction <= 1:
            raise ValueError('fraction must lie in (0,1]')
        result=self.evaluate(sigma_x_m,sigma_y_m)
        return min(self.rank,int(np.searchsorted(result['cumulative_squared_fraction'],fraction))+1)
