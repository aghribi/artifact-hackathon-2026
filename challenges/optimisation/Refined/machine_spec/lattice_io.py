"""Turning a `MachineSpec` into a Cheetah segment and a beam.

The part of "load a machine" that is the same for every machine: read the
lattice, cut it to the modelled stretch, insert the vacuum chamber, switch off
the elements that are off, and produce a beam. Everything machine-specific is in
the spec rather than here.

This exists because both halves of the project need it and each had grown its
own copy: `nominal_model/xsuite_model.py` had one wired to CLARA's files, and
`Refined/lattice.py` had a second. The two agreed on the chamber semantics and
differed only in that `Refined`'s also accepted a flat `{element: half}` table,
so that is the one kept.

`nominal_model/xsuite_model.py` is left alone — 38 files import it directly and
it is the xsuite model's own business. This module is for the Cheetah side.
"""

from __future__ import annotations

import json
from pathlib import Path

import cheetah
import numpy as np
import torch

DTYPE = torch.float32


# ---------------------------------------------------------------- apertures


def load_apertures(path: str | Path) -> dict:
    """An aperture table as `{"pipe": {...}, "collimators": {...}}`.

    Two formats are accepted. The full one names `pipe` and `collimators` lists,
    as STFC's SimFrame model of CLARA does. The plain one is a flat
    `{element: half_size}` or `{element: [x_max, y_max]}` map, which is all most
    machines have, and is read as a collimator at each named element.
    """
    table = json.loads(Path(path).read_text())
    if "pipe" not in table and "collimators" not in table:
        flat = table.get("elements", table)
        table = {"pipe": [], "collimators": [
            {"element": name,
             **(dict(zip(("x_max", "y_max"), value))
                if isinstance(value, (list, tuple))
                else {"x_max": value, "y_max": value})}
            for name, value in flat.items() if not isinstance(value, str)
        ]}
    return {
        kind: {entry["element"]: entry for entry in table.get(kind, [])}
        for kind in ("pipe", "collimators")
    }


def _aperture_element(entry: dict, name: str) -> cheetah.Aperture:
    return cheetah.Aperture(
        x_max=torch.tensor(float(entry["x_max"]), dtype=DTYPE),
        y_max=torch.tensor(float(entry["y_max"]), dtype=DTYPE),
        shape=entry.get("shape", "elliptical"),
        name=name,
    )


def with_apertures(segment: cheetah.Segment, apertures: dict) -> cheetah.Segment:
    """Insert the vacuum chamber as thin Cheetah apertures.

    Two kinds of entry, which is how a chamber is actually described:

    - `pipe` sets the chamber *from its element onwards* — the beam is checked
      there and at the exit of every element with a length, until the next pipe
      entry supersedes it. A point-like aperture marker read as the chamber of
      the section it starts.
    - `collimators` are checked at the entrance and exit of their own element.

    Apertures are named `<element>_pipe`, `<element>_col_in` and
    `<element>_col_out`, and stay on the design axis when magnets are misaligned.
    """
    pipe, collimators = apertures["pipe"], apertures["collimators"]
    elements, chamber = [], None
    for element in segment.elements:
        name = element.name
        collimator = collimators.get(name)
        if collimator is not None:
            elements.append(_aperture_element(collimator, f"{name}_col_in"))
        elements.append(element)
        if name in pipe:
            chamber = pipe[name]
            elements.append(_aperture_element(chamber, f"{name}_pipe"))
        elif collimator is not None:
            elements.append(_aperture_element(collimator, f"{name}_col_out"))
        elif chamber is not None and float(getattr(element, "length", 0.0)) > 0.0:
            elements.append(_aperture_element(chamber, f"{name}_pipe"))
    return cheetah.Segment(elements, name=segment.name)


# ---------------------------------------------------------------- the segment


def disable(segment: cheetah.Segment, names: set[str]) -> cheetah.Segment:
    """Replace named elements with a drift of their own length.

    An element in the lattice file is not necessarily an element on the machine.
    CLARA's L4H linearizer is in the file as a powered cavity and is switched off
    in reality, so a model that powers it is tuning a different accelerator — the
    chirp, the energy spread and what the dogleg makes of them all change. The
    length and the name stay, so apertures and diagnostics still line up.
    """
    if not names:
        return segment
    elements = []
    for element in segment.elements:
        if element.name in names:
            # Second order, not whatever the original element used: a cavity's
            # tracking method is not one a drift understands, and Cheetah quietly
            # falls back to linear if given it — which tracks a different beam.
            elements.append(cheetah.Drift(
                length=torch.tensor(
                    float(getattr(element, "length", 0.0) or 0.0), dtype=DTYPE
                ),
                tracking_method="second_order",
                name=element.name,
            ))
        else:
            elements.append(element)
    return cheetah.Segment(elements, name=segment.name)


def build_segment(spec) -> cheetah.Segment:
    """The modelled stretch of a spec's lattice, chamber and all.

    `start` is excluded and `end` included, matching how the benchmark describes
    its own stretch. The chamber goes on before the cut so the aperture at
    `start` comes along with it.
    """
    segment = cheetah.Segment.from_lattice_json(str(spec.lattice))
    if spec.apertures:
        segment = with_apertures(segment, load_apertures(spec.apertures))
    if spec.start:
        segment = segment.partition_at(spec.start, mode="after")[-1]
    if spec.end:
        segment = segment.partition_at(spec.end, mode="after")[0]
    return disable(segment, set(spec.disabled))


# ---------------------------------------------------------------- the beam


def load_distribution(path: str | Path) -> tuple[np.ndarray, float, dict]:
    """(particles (N, 6) in Cheetah's convention, reference energy in eV, metadata)."""
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = {
            key: data[key].item() for key in data.files
            if key not in ("particles", "energy_ev")
        }
        return data["particles"], float(data["energy_ev"]), metadata


def beam_from_twiss(twiss: dict, num_particles: int,
                    seed: int = 0) -> tuple[np.ndarray, float, dict]:
    """A beam from Twiss parameters, for a machine with no distribution file.

    The measured distribution in `spec.beam` is always the better input — it
    carries the correlations and the tails a Twiss ellipse does not. This is for
    lattices that simply do not have one, where a matched Gaussian is the honest
    stand-in and is recorded as such.
    """
    wanted = ("beta_x", "alpha_x", "emittance_x",
              "beta_y", "alpha_y", "emittance_y", "energy")
    missing = [key for key in wanted if key not in twiss]
    if missing:
        raise KeyError(f"beam_twiss is missing {missing}")
    # Cheetah's sampler takes no generator, so the global seed is set around it:
    # a machine's beam must be the same every time it is loaded, or two runs of
    # the same seed are two different episodes.
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    tensor = lambda key, default=None: torch.tensor(  # noqa: E731
        float(twiss.get(key, default)), dtype=DTYPE
    )
    beam = cheetah.ParticleBeam.from_twiss(
        num_particles=num_particles,
        beta_x=tensor("beta_x"), alpha_x=tensor("alpha_x"),
        emittance_x=tensor("emittance_x"),
        beta_y=tensor("beta_y"), alpha_y=tensor("alpha_y"),
        emittance_y=tensor("emittance_y"),
        sigma_tau=tensor("sigma_tau", 1e-4), sigma_p=tensor("sigma_p", 1e-3),
        energy=tensor("energy"),
        total_charge=tensor("total_charge", 1e-12),
        dtype=DTYPE,
    )
    torch.random.set_rng_state(state)
    particles = beam.particles[..., :6].detach().cpu().numpy().astype(float)
    return particles, float(beam.energy), {"source": "beam_from_twiss"}


def load_beam(spec, num_particles: int | None = None,
              seed: int = 0) -> tuple[np.ndarray, float, dict]:
    """The spec's beam: its distribution file, or Twiss where there is none."""
    if spec.beam:
        particles, energy_ev, metadata = load_distribution(spec.beam)
    elif spec.beam_twiss:
        particles, energy_ev, metadata = beam_from_twiss(
            spec.beam_twiss, num_particles or 10_000, seed
        )
    else:
        raise ValueError(
            f"{spec.name or 'this spec'} has neither `beam` (a distribution .npz) "
            f"nor `beam_twiss`; there is nothing to track"
        )
    return np.ascontiguousarray(particles[:num_particles]), energy_ev, metadata
