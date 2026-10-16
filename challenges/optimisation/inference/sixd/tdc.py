"""Transverse deflecting cavity for the xsuite CLARA line.

The lattice file carries the TDC with voltage 0 and frequency 0, and the xsuite conversion
turns it into a drift. `install_tdc` replaces that drift by drift(L/2), a thin deflecting
kick, drift(L/2).

Kick model (thin, vertical deflection, Hamiltonian H = -a y sin(phi + k zeta)):

    py    += a sin(phi + k zeta)
    delta += a k y cos(phi + k zeta)        (Panofsky-Wenzel, from the same Hamiltonian)

with a = e V / (p0 c) and k = 2 pi f / c. zeta = -tau is positive for particles ahead of the
reference. phi = 0 is the zero crossing (pure streak). The sign of the real machine's
streak relative to this convention is a calibration choice, not a physics unknown.
"""

from __future__ import annotations

import math
from typing import ClassVar

import xobjects as xo
import xtrack as xt

SPEED_OF_LIGHT = 299792458.0
TDC_NAME = "cla_s07_dia_tdc_01"
TDC_FREQUENCY_HZ = 2998.5e6  # assumed S-band, not confirmed for CLARA
TDC_MAX_VOLTAGE_V = 5e6  # VELA/CLARA TDC design value (assumed)


class VerticalDeflectingKick(xt.BeamElement):
    """Thin RF deflection in y with the Panofsky-Wenzel energy kick."""

    _xofields: ClassVar[dict] = {
        "strength": xo.Float64,  # a = eV/(p0 c)
        "wavenumber": xo.Float64,  # k = 2 pi f / c, 1/m
        "phase": xo.Float64,  # rad
    }
    isthick = False
    has_backtrack = False
    allow_kernel_compilation = True

    _extra_c_sources: ClassVar[list] = [
        r"""
#include "xtrack/headers/track.h"

GPUFUN
void VerticalDeflectingKick_track_local_particle(VerticalDeflectingKickData el,
                                                 LocalParticle* part0){
    double const a = VerticalDeflectingKickData_get_strength(el);
    double const k = VerticalDeflectingKickData_get_wavenumber(el);
    double const phi = VerticalDeflectingKickData_get_phase(el);
    START_PER_PARTICLE_BLOCK(part0, part);
        double const arg = phi + k * LocalParticle_get_zeta(part);
        double const y = LocalParticle_get_y(part);
        LocalParticle_add_to_py(part, a * sin(arg));
        LocalParticle_update_delta(
            part, LocalParticle_get_delta(part) + a * k * y * cos(arg));
    END_PER_PARTICLE_BLOCK;
}
"""
    ]


def kick_strength(voltage_v: float, momentum_ev: float) -> float:
    return voltage_v / momentum_ev


def install_tdc(
    converted,
    voltage_v: float = 0.0,
    phase_rad: float = 0.0,
    frequency_hz: float = TDC_FREQUENCY_HZ,
    p0c_ev: float | None = None,
):
    """Replace the TDC drift of a ConvertedLine in place. Returns the kick element name."""
    line = converted.line
    names = list(line.element_names)
    index = names.index(TDC_NAME)
    length = line[TDC_NAME].length
    if p0c_ev is None:
        p0c_ev = _momentum_at_start_of(converted, index)
    kick = VerticalDeflectingKick(
        strength=kick_strength(voltage_v, p0c_ev),
        wavenumber=2 * math.pi * frequency_hz / SPEED_OF_LIGHT,
        phase=phase_rad,
    )
    elements = [line.element_dict[n] for n in names]
    new_names = (
        names[:index]
        + [TDC_NAME + "..pre", TDC_NAME + "..kick", TDC_NAME]
        + names[index + 1 :]
    )
    new_elements = (
        elements[:index]
        + [xt.Drift(length=length / 2), kick, xt.Drift(length=length / 2)]
        + elements[index + 1 :]
    )
    particle_ref = line.particle_ref
    new_line = xt.Line(elements=new_elements, element_names=new_names)
    new_line.particle_ref = particle_ref
    new_line.build_tracker()
    converted.line = new_line
    converted.boundaries = [
        (n, i + 2 if i > index else i) for n, i in converted.boundaries
    ]
    return TDC_NAME + "..kick"


def _momentum_at_start_of(converted, index: int) -> float:
    """Reference momentum (eV/c) at the TDC (no cavity in between a start and the TDC
    for the S07 segment, so the initial reference momentum applies)."""
    import xsuite_model

    return xsuite_model.momentum_ev(converted.initial_energy_ev)


def set_tdc(
    line,
    voltage_v: float,
    phase_rad: float | None = None,
    p0c_ev: float | None = None,
) -> None:
    """Change voltage (and optionally phase) of an installed TDC on an xtrack line."""
    kick = line[TDC_NAME + "..kick"]
    if p0c_ev is None:
        p0c_ev = float(line.particle_ref.p0c[0])
    kick.strength = kick_strength(voltage_v, p0c_ev)
    if phase_rad is not None:
        kick.phase = phase_rad
