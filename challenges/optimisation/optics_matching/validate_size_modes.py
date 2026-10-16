"""Check compiled SVD knots, modal round trips, projectors and domain guards."""
import json
from pathlib import Path
import numpy as np
from . import SizeModes


def main():
    m=SizeModes();out=Path(__file__).parent/'results'/'size_svd'
    maximum_response_error=0.;maximum_spectrum_error=0.;minimum_overlap=1.
    with np.load(out/'grid_svd.npz') as grid:
        axis=grid['spot_sizes_m']
        for i,x in enumerate(axis):
            for j,y in enumerate(axis):
                result=m.evaluate(x,y,exact_response=False)
                expected=m.data['constant_weighted_response'].copy();expected[-2:]=grid['ip_rows'][i,j]
                maximum_response_error=max(maximum_response_error,float(np.linalg.norm(result['weighted_response']-expected)/np.linalg.norm(expected)))
                maximum_spectrum_error=max(maximum_spectrum_error,float(np.max(abs(result['singular_values']/grid['singular_values'][i,j,:m.rank]-1))))
                minimum_overlap=min(minimum_overlap,float(np.min(result['gauge_overlap'])))
    amplitudes=np.array([1,-2,3,-4,5])*1e-5
    tune=m.to_k1(amplitudes,.001,.003)
    projected=m.from_k1(tune,.001,.003)
    roundtrip=float(np.max(abs(projected[:5]-amplitudes)))
    residual=float(np.max(abs(projected[5:])))
    projector=m.projector(.001,.003,7,8)
    projector_error=float(np.linalg.norm(projector@projector-projector))
    for x,y in [(0,.001),(.021,.001),(float('nan'),.001)]:
        try:m.evaluate(x,y)
        except ValueError:pass
        else:raise AssertionError('Out-of-domain lookup accepted')
    assert maximum_response_error<1e-12 and maximum_spectrum_error<1e-7
    assert roundtrip<1e-12 and residual<1e-12 and projector_error<1e-12
    summary=dict(knots_checked=len(axis)**2,knot_max_relative_response_error=maximum_response_error,
        knot_max_relative_spectrum_error=maximum_spectrum_error,knot_minimum_vector_gauge_overlap=minimum_overlap,
        modal_roundtrip_max_abs_error=roundtrip,modal_roundtrip_unused_amplitudes_max_abs=residual,
        projector_idempotence_error=projector_error,domain_checks_passed=True)
    (out/'api_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
