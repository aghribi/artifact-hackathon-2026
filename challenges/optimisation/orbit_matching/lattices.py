"""Apply orbital SVD kick increments to Xsuite or Cheetah lattices."""
import numpy as np
from .orbit_modes import OrbitModes


def _tune(plane,sigma_x_m,sigma_y_m,amplitudes,model):
    model=model or OrbitModes(plane)
    if model.plane!=plane:raise ValueError('Requested plane differs from the mode model')
    kicks=model.to_kicks(amplitudes,sigma_x_m,sigma_y_m)
    return dict(plane=plane,delta_kicks_rad=kicks,settings=dict(zip(model.correctors,map(float,kicks))))


def apply_to_xsuite(line,sigma_x_m,sigma_y_m,amplitudes,*,plane='x',model=None):
    """Add modal kick increments; accepts Line or ConvertedLine.

    Quadrupoles/RF are untouched: first apply optics_matching's matched tune.
    Repeated calls add increments repeatedly. Missing controls fail before edits.
    """
    tune=_tune(plane,sigma_x_m,sigma_y_m,amplitudes,model)
    line=getattr(line,'line',line)
    missing=set(tune['settings'])-set(line.element_names)
    if missing:raise ValueError(f'Missing correctors: {sorted(missing)}')
    field='knl' if plane=='x' else 'ksl';sign=-1 if plane=='x' else 1
    targets=[]
    for name,kick in tune['settings'].items():
        values=getattr(line[name],field)
        if len(values)<1:raise ValueError(f'{name}: missing dipole coefficient')
        targets.append((values,float(values[0])+sign*kick))
    for values,value in targets:values[0]=value
    return tune


def apply_to_cheetah(segment,sigma_x_m,sigma_y_m,amplitudes,*,plane='x',model=None):
    """Add modal angular increments, preserving tensor dtype/device."""
    import torch
    tune=_tune(plane,sigma_x_m,sigma_y_m,amplitudes,model)
    def leaves(node):
        for element in node.elements:
            if hasattr(element,'elements'):yield from leaves(element)
            else:yield element
    elements={e.name:e for e in leaves(segment)}
    missing=set(tune['settings'])-elements.keys()
    if missing:raise ValueError(f'Missing correctors: {sorted(missing)}')
    field='horizontal_angle' if plane=='x' else 'vertical_angle'
    targets=[(getattr(elements[n],field),kick) for n,kick in tune['settings'].items()]
    with torch.no_grad():
        for value,kick in targets:value.add_(value.new_tensor(kick))
    return tune
