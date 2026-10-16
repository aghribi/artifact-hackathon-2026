"""Apply the saved k1 family to compatible CLARA lattices (SI units)."""
from pathlib import Path
import json
from .matched_optics import MatchedOptics

DATA = Path(__file__).resolve().parent / 'data'
START = 'cla_s02_sim_aper_01'
IP = 'cla_feh_dia_scr_01'


def apply_to_xsuite(line, sigma_x_m, sigma_y_m, *, model=None):
    """Set all 46 quadrupoles on an existing line; return the tune.

    Accepts xtrack.Line or nominal_model's ConvertedLine. Existing RF, VBC,
    wakes and reference energy are untouched and must match the saved family.
    Missing quadrupoles are rejected before any strengths are changed.
    """
    tune = (model or MatchedOptics()).evaluate(sigma_x_m, sigma_y_m)
    line = getattr(line, 'line', line)
    missing = set(tune['settings']) - set(line.element_names)
    if missing:
        raise ValueError(f'Missing quadrupoles: {sorted(missing)}')
    for name, k1 in tune['settings'].items():
        line[name].k1 = float(k1)
    return tune


def _elements(segment):
    for element in segment.elements:
        if hasattr(element, 'elements'):
            yield from _elements(element)
        else:
            yield element


def apply_to_cheetah(segment, sigma_x_m, sigma_y_m, *, model=None):
    """Set all 46 k1 buffers, preserving their dtype and device; return the tune.

    As for apply_to_xsuite, this changes quadrupoles only. The lookup is a
    NumPy API; use evaluate(derivatives=True) for spot-size derivatives.
    """
    import torch
    tune = (model or MatchedOptics()).evaluate(sigma_x_m, sigma_y_m)
    elements = {element.name: element for element in _elements(segment)}
    missing = set(tune['settings']) - elements.keys()
    if missing:
        raise ValueError(f'Missing quadrupoles: {sorted(missing)}')
    with torch.no_grad():
        for name, k1 in tune['settings'].items():
            elements[name].k1.copy_(elements[name].k1.new_tensor(float(k1)))
    return tune


def write_lattice_json(path, sigma_x_m, sigma_y_m, *, model=None):
    """Export a complete Cheetah-format lattice usable by either converter.

    Starts with the saved anchor geometry/RF, at zero VBC and zero L4H voltage.
    The Xsuite converter must still be called with wakefields=False.
    """
    tune = (model or MatchedOptics()).evaluate(sigma_x_m, sigma_y_m)
    lattice = json.loads((DATA / 'anchor_lattice.json').read_text())
    for name, k1 in tune['settings'].items():
        lattice['elements'][name][1]['k1'] = float(k1)
    lattice['info'] = (f'Matched IP RMS targets: {sigma_x_m:.17g}, {sigma_y_m:.17g} m; '
                       'VBC=0, L4H off; Xsuite requires wakefields=False.')
    path = Path(path)
    path.write_text(json.dumps(lattice, indent=2) + '\n')
    return tune


def build_cheetah(sigma_x_m, sigma_y_m, *, model=None, device=None, dtype=None):
    """Build the benchmark slice with the saved RF/geometry and matched k1.

    Native Cheetah tracking is a different approximation to the Xsuite model;
    the Xsuite full-bunch spot-size validation does not transfer to this backend.
    """
    import cheetah
    import torch
    full = cheetah.Segment.from_lattice_json(str(DATA / 'anchor_lattice.json'))
    segment = full.partition_at(START, mode='after')[-1].partition_at(IP, mode='after')[0]
    segment = segment.to(device=device, dtype=dtype or torch.float64)
    # Xsuite treats the disabled harmonic cavity as a drift.
    segment.elements = torch.nn.ModuleList([
        cheetah.Drift(length=e.length.clone(), name=e.name,
                      device=e.length.device, dtype=e.length.dtype)
        if e.name == 'cla_l4h_lin_cav_01' else e
        for e in segment.elements
    ])
    apply_to_cheetah(segment, sigma_x_m, sigma_y_m, model=model)
    return segment


def build_xsuite(sigma_x_m, sigma_y_m, *, initial_energy_ev=None, model=None):
    """Build nominal_model.ConvertedLine with wakes disabled and matched k1.

    Requires this repository's nominal_model converter and its dependencies.
    The reference energy defaults to the saved nominal bunch energy.
    """
    from nominal_model import xsuite_model
    if initial_energy_ev is None:
        import numpy as np
        with np.load(xsuite_model.INITIAL_DISTRIBUTION) as bunch:
            initial_energy_ev = float(bunch['energy_ev'])
    converted = xsuite_model.build_line(
        initial_energy_ev, lattice_json=DATA / 'anchor_lattice.json', wakefields=False
    )
    apply_to_xsuite(converted, sigma_x_m, sigma_y_m, model=model)
    return converted
