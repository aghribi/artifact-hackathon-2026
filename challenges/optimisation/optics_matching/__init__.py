"""Smooth CLARA IP spot-size optics, with sizes in m and k1 in m^-2."""
from .matched_optics import MatchedOptics, CompressionEffects
from .lattices import (
    apply_to_xsuite, apply_to_cheetah, build_xsuite, build_cheetah,
    write_lattice_json,
)

__all__ = [
    'MatchedOptics', 'CompressionEffects', 'apply_to_xsuite', 'apply_to_cheetah',
    'build_xsuite', 'build_cheetah', 'write_lattice_json',
]

from .size_modes import SizeModes
__all__.append("SizeModes")
