"""Check every current node response, weighted SVD and modal-coordinate round trip."""
from pathlib import Path
import json
import numpy as np
from . import OrbitModes

OUT=Path(__file__).parent/'results/vbc_on_l4h_off/orbit'


def main():
    report={}
    for backend in ['cheetah','xsuite']:
        for plane in 'xy':
            model=OrbitModes(plane,backend=backend)
            with np.load(OUT/f'{backend}_{plane}_grid_svd.npz') as d:data={key:d[key] for key in d.files}
            response_error=0.;reconstruction=0.;orthogonality=0.;singular_error=0.
            for i,x in enumerate(model.axis):
                for j,y in enumerate(model.axis):
                    r=model.evaluate(float(x),float(y));rank=r['rank'];a=r['weighted_response']
                    response_error=max(response_error,float(np.linalg.norm(a-data['weighted_response'][i,j])/np.linalg.norm(a)))
                    singular_error=max(singular_error,float(np.max(abs(r['singular_values']/data['singular_values'][i,j,:rank]-1))))
                    reconstruction=max(reconstruction,float(np.linalg.norm((r['U']*r['singular_values'])@r['Vt']-a)/np.linalg.norm(a)))
                    orthogonality=max(orthogonality,float(np.linalg.norm(r['Vt']@r['Vt'].T-np.eye(rank))))
                    assert 'transmission' in r and r['bpm_valid'].shape==(31,)
            x,y=float(model.axis[5]),float(model.axis[7]);a=np.array([1.,-2.,3.])*1e-6
            recovered=model.from_kicks(model.to_kicks(a,x,y),x,y)
            roundtrip=float(np.max(abs(recovered[:3]-a)))
            assert response_error<1e-12 and singular_error<1e-12 and reconstruction<1e-8 and orthogonality<1e-12 and roundtrip<1e-18
            report[f'{backend}_{plane}']=dict(nodes_checked=144,response_relative_error=response_error,singular_value_relative_error=singular_error,
                resolved_svd_reconstruction_relative_error=reconstruction,right_vector_orthogonality_error=orthogonality,modal_roundtrip_max_abs_error_rad=roundtrip)
    (OUT/'api_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
