"""Locate numerical degeneracies and check vector sign holonomy around them."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from . import SizeModes


def main():
    model=SizeModes();out=Path(__file__).parent/'results'/'size_svd'
    with np.load(out/'grid_svd.npz') as d:
        s=d['singular_values'];axis=d['spot_sizes_m'];g=(s[:,:,:35]-s[:,:,1:36])/s[:,:,:35]
    records=[]
    for j in [6,8,19]:
        idx=np.unravel_index(np.argmin(g[:,:,j]),g[:,:,j].shape)
        def objective(z):
            q=model.evaluate(*np.exp(z))['singular_values']
            return ((q[j]-q[j+1])/q[j])**2
        result=minimize(objective,np.log([axis[idx[0]],axis[idx[1]]]),method='Nelder-Mead',
            bounds=[(np.log(1e-4),np.log(.02))]*2,
            options={'maxiter':250,'xatol':1e-10,'fatol':1e-22})
        angles=np.linspace(0,2*np.pi,65)
        path=np.exp(result.x[None,:]+.002*np.column_stack([np.cos(angles),np.sin(angles)]))
        transport=model.transport_path(path,n_modes=j+2,minimum_overlap=.98)
        dots=np.einsum('ij,ij->i',transport['results'][0]['Vt'],transport['results'][-1]['Vt'])
        records.append(dict(modes=[j+1,j+2],sigma_x_m=float(np.exp(result.x[0])),sigma_y_m=float(np.exp(result.x[1])),
            relative_singular_gap=float(np.sqrt(result.fun)),loop_log_radius=.002,
            loop_return_signs=np.sign(dots).astype(int).tolist(),
            unresolved_segments=transport['unresolved_segments'],
            minimum_accepted_overlap=transport['minimum_accepted_overlap']))
    (out/'degeneracies.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
