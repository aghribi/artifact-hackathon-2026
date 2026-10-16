"""C2 quadrupole strength surface versus desired IP rms sizes (metres).

Logarithmic axes, no extrapolation. The common pre-S07 matching section is fixed.
Use validation diagnostics before claiming accuracy between the measured grid nodes.
"""
from pathlib import Path
import numpy as np
from scipy.interpolate import NdBSpline, PchipInterpolator
DEFAULT=Path(__file__).resolve().parent/'data'/'matching_surface.npz'

class MatchedOptics:
    def __init__(self,path=DEFAULT):
        with np.load(path) as data:
            self.data={key:data[key] for key in data.files}
        self.quadrupoles=self.data['quadrupoles'].tolist();self.axis=self.data['spot_sizes_m']
        self.logaxis=np.log(self.axis);self.k1=self.data['k1'];self.prefix=len(self.quadrupoles)-4
        self.directory=Path(path).resolve().parent
        with np.load(self.directory/'strength_splines.npz') as compiled:
            self.spline=NdBSpline((compiled['knots_x'],compiled['knots_y']),compiled['coefficients'],(3,3),extrapolate=False)

    def evaluate(self,sigma_x_m,sigma_y_m,derivatives=False,refine=True):
        x,y=float(sigma_x_m),float(sigma_y_m)
        if not (np.isfinite(x) and np.isfinite(y) and self.axis[0]<=x<=self.axis[-1] and self.axis[0]<=y<=self.axis[-1]):
            raise ValueError('Both requested rms spot sizes must lie between 0.1 and 20 mm')
        a,b=np.log(x),np.log(y)
        point=np.array([a,b])
        k=np.asarray(self.spline(point)).copy();k[:self.prefix]=self.k1[0,0,:self.prefix]
        dx=np.asarray(self.spline(point,nu=(1,0))).copy();dx[:self.prefix]=0
        dy=np.asarray(self.spline(point,nu=(0,1))).copy();dy[:self.prefix]=0
        if refine:
            import torch
            with torch.inference_mode(False), torch.enable_grad():
                if not hasattr(self,'_runtime'):
                    try:
                        from .runtime_model import RuntimeFinal
                    except ImportError:
                        from runtime_model import RuntimeFinal
                    self._runtime=RuntimeFinal(self.directory/'beamlet_runtime.npz')
                target=torch.tensor([a,b],dtype=torch.float64,requires_grad=derivatives)
                initial=torch.tensor(k[-4:],dtype=torch.float64)
                if derivatives:
                    initial=initial+torch.tensor(np.column_stack([dx[-4:],dy[-4:]]),dtype=torch.float64)@(target-target.detach())
                corrected=self._runtime.correct(initial,target,derivatives=derivatives)
                if derivatives:
                    D=torch.stack([torch.autograd.grad(corrected[j],target,retain_graph=True)[0] for j in range(4)]).detach().numpy()
                    dx[-4:]=D[:,0];dy[-4:]=D[:,1]
                k[-4:]=corrected.detach().numpy()
        return dict(k1=k,settings=dict(zip(self.quadrupoles,k)),d_k1_d_log_sigma_x=dx if derivatives else None,d_k1_d_log_sigma_y=dy if derivatives else None,d_k1_d_sigma_x=dx/x if derivatives else None,d_k1_d_sigma_y=dy/y if derivatives else None,
            target_sigma_x_m=x,target_sigma_y_m=y,vbc_theta_rad=0.)


    def optics_response(self,sigma_x_m,sigma_y_m):
        """Return the weighted/unweighted 46x46 beta response at the matched strengths."""
        if not hasattr(self,'_response'):
            try:
                from .response_surface import OpticsResponse
            except ImportError:
                from response_surface import OpticsResponse
            self._response=OpticsResponse(self.directory/'optics_response_model.npz')
        return self._response.evaluate(self.evaluate(sigma_x_m,sigma_y_m)['k1'])


class CompressionEffects:
    """No-wake VBC compression for this family's fixed upstream optics.

    This does not extend the spot-size match away from zero VBC excitation.
    """
    def __init__(self,path=DEFAULT.parent/'compression_strength.npz'):
        with np.load(path) as data:
            self.columns=data['columns'].tolist();values=data['values']
        self.curve=PchipInterpolator(values[:,0],values,axis=0,extrapolate=False)
    def evaluate(self,strength_fraction):
        u=float(strength_fraction)
        if not np.isfinite(u) or not -1<=u<=1:
            raise ValueError('VBC strength fraction must be in [-1,1]')
        return dict(zip(self.columns,map(float,self.curve(u))))
