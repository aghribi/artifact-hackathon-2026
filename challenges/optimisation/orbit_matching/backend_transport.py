"""Generated from RL_framework SimEnv transport classes; run sync_backend_transport to refresh.

Physics is supplied by nominal_model; this avoids requiring Gym for batch studies.
"""
from __future__ import annotations
from abc import ABC,abstractmethod
from typing import Iterator
import numpy as np
import torch
import cheetah
DTYPE=torch.float32


VBC_EDGE = {
    "cla_vbc_mag_dip_01": "exit",
    "cla_vbc_mag_dip_02": "entry",
    "cla_vbc_mag_dip_03": "exit",
    "cla_vbc_mag_dip_04": "entry",
}

class Transport(ABC):
    """One of the two models the environment can track its beam through.

    Both carry the same elements under the same names, so the environment works in
    names and in Cheetah's units and leaves each model to find and write its own
    elements. `config.backend` picks between them.
    """

    name: str

    @abstractmethod
    def read(self, element: str, prop: str) -> float:
        """What a magnet is set to, in the lattice file's units."""

    @abstractmethod
    def write(self, element: str, prop: str, value: float) -> None:
        """Set a magnet, in the lattice file's units."""

    @abstractmethod
    def read_shift(self, element: str) -> tuple[float, float]:
        """Where an element sits against the design axis."""

    @abstractmethod
    def shift(self, element: str, dx: float, dy: float) -> None:
        """Move an element off the design axis."""

    @abstractmethod
    def run(
        self, coordinates: np.ndarray, stops: list[str]
    ) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        """Track from the start of the line, yielding (stop, x, y, alive) at each of
        `stops` in turn. `stops` runs from the start of the line to wherever the beam
        is being read, and x, y and alive are in the order the particles were given."""

class CheetahTransport(Transport):
    """Cheetah: first-order magnets and achromatic cavities, tracked in torch.

    The physics `score.py` is scored on, and the faster of the two to set up — there
    is no tracking kernel to compile.
    """

    name = "cheetah"

    def __init__(self, segment: cheetah.Segment, energy_ev: float, stops: list[str]):
        self._segment = segment
        self._elements = {element.name: element for element in segment.elements}
        self._energy = torch.tensor(float(energy_ev), dtype=DTYPE)
        for element in segment.elements:
            if isinstance(element, cheetah.Screen):
                # The environment does its own imaging, and an active screen costs a
                # full camera frame at every diagnostic in the line.
                element.is_active = False
        # One sub-segment per stop, over the same element objects, so each stretch is
        # tracked as one fused transfer map instead of element by element.
        self._chunks, previous = {}, 0
        for stop in stops:
            index = segment.element_index(stop)
            self._chunks[stop] = cheetah.Segment(segment.elements[previous:index])
            previous = index

    def read(self, element: str, prop: str) -> float:
        return float(getattr(self._elements[element], prop))

    # `phase` needs nothing special here: Cheetah stores it on the cavity, in
    # degrees, and rebuilds the transfer map from it on every track.

    def write(self, element: str, prop: str, value: float) -> None:
        setattr(self._elements[element], prop, torch.tensor(value, dtype=DTYPE))
        if prop == "angle" and element in VBC_EDGE:
            face = "dipole_e2" if VBC_EDGE[element] == "exit" else "dipole_e1"
            setattr(self._elements[element], face, torch.tensor(value, dtype=DTYPE))

    def read_shift(self, element: str) -> tuple[float, float]:
        misalignment = self._elements[element].misalignment
        return float(misalignment[0]), float(misalignment[1])

    def shift(self, element: str, dx: float, dy: float) -> None:
        self._elements[element].misalignment = torch.tensor([dx, dy], dtype=DTYPE)

    def run(self, coordinates, stops):
        particles = torch.tensor(
            np.column_stack([coordinates, np.ones(len(coordinates))]), dtype=DTYPE
        )
        beam = cheetah.ParticleBeam(particles=particles, energy=self._energy)
        for stop in stops:
            chunk = self._chunks[stop]
            if len(chunk.elements):
                beam = chunk.track(beam)
            yield (
                stop,
                beam.x.detach().numpy(),
                beam.y.detach().numpy(),
                beam.survival_probabilities.detach().numpy() > 0.5,
            )

class XsuiteTransport(Transport):
    """xsuite: thick elements at each particle's own momentum, and standing-wave
    cavities that focus at the energy the particle has.

    Closer to the machine off momentum, and the model `nominal_model/README.md`
    documents. Building the line compiles a tracking kernel, which takes a minute.
    """

    name = "xsuite"

    # What has to be copied onto each piece of a cavity when it is re-expanded.
    _CAVITY_FIELDS = {
        "AxisymmetricThinLens": ("strength", "chromatic_power"),
        "Cavity": ("voltage", "frequency", "phase"),
        "ReferenceEnergyIncrease": ("Delta_p0c",),
    }

    def __init__(
        self, model, segment, lattice_json, start, end, energy_ev, stops, apertures
    ):
        self.converted = model.build_line(
            energy_ev, lattice_json, start, end, apertures
        )
        self._model = model
        self._segment = segment
        self._line = self.converted.line
        self._index = {name: i for i, name in enumerate(self._line.element_names)}
        self._stops = [(stop, self._index[stop]) for stop in stops]
        # The cavities that are actually powered, in order. A cavity at zero volts
        # converted to a drift and has no phase to set.
        self._cavities = [
            element
            for element in segment.elements
            if isinstance(element, cheetah.Cavity) and float(element.voltage)
        ]

    def _expand_cavities(self) -> None:
        """Rebuild every cavity's slices from the phases the segment now carries.

        A standing-wave cavity becomes some 1300 xsuite elements whose focusing,
        acceleration and reference-energy steps all depend on the phase and on the
        energy the beam arrives with, so changing a phase means redoing that
        arithmetic — and redoing it for every cavity downstream too, because they
        are each handed the energy the one before left. The numbers come from the
        nominal model itself rather than a copy of it here, so the two cannot drift
        apart; only the resulting values are transplanted onto the existing
        elements, which leaves the line's structure untouched.
        """
        energy = self.converted.initial_energy_ev
        for cavity in self._cavities:
            pieces, energy = self._model._standing_wave_cavity(
                cavity.name, cavity, energy
            )
            for piece_name, piece in pieces:
                target = self._line.element_dict[piece_name]
                for field in self._CAVITY_FIELDS[type(piece).__name__]:
                    setattr(target, field, getattr(piece, field))
        self.converted.final_energy_ev = energy

    def _element(self, name: str):
        return self._line.element_dict[name]

    def _segment_element(self, name: str):
        return next(e for e in self._segment.elements if e.name == name)

    def read(self, element: str, prop: str) -> float:
        if prop == "phase":
            # The sliced line has no single phase to read; the segment holds it.
            return float(self._segment_element(element).phase)
        target = self._element(element)
        if prop == "horizontal_angle":
            return -float(target.knl[0])  # Cheetah's kick is the opposite of knl
        if prop == "vertical_angle":
            return float(target.ksl[0])
        return float(getattr(target, prop))

    def write(self, element: str, prop: str, value: float) -> None:
        if prop == "phase":
            self._segment_element(element).phase = torch.tensor(value, dtype=DTYPE)
            self._expand_cavities()
            return
        target = self._element(element)
        if prop == "horizontal_angle":
            target.knl[0] = -value
        elif prop == "vertical_angle":
            target.ksl[0] = value
        elif prop == "angle":
            # Setting `angle` moves the reference curvature but leaves the field
            # behind, so k0 has to follow it, as `xsuite_model._dipole` sets it.
            target.angle = value
            target.k0 = value / float(target.length) if float(target.length) else 0.0
            if element in VBC_EDGE:
                setattr(target, f"edge_{VBC_EDGE[element]}_angle", value)
        else:
            setattr(target, prop, value)

    def read_shift(self, element: str) -> tuple[float, float]:
        target = self._element(element)
        return float(target.shift_x), float(target.shift_y)

    def shift(self, element: str, dx: float, dy: float) -> None:
        target = self._element(element)
        target.shift_x, target.shift_y = dx, dy

    def _ordered(self, particles, count):
        """x, y and alive, in the order the particles were launched in."""
        ids = np.asarray(particles.particle_id)
        slots = np.flatnonzero((ids >= 0) & (ids < count))
        x, y = np.zeros(count), np.zeros(count)
        alive = np.zeros(count, dtype=bool)
        x[ids[slots]] = np.asarray(particles.x)[slots]
        y[ids[slots]] = np.asarray(particles.y)[slots]
        alive[ids[slots]] = np.asarray(particles.state)[slots] > 0
        return x, y, alive

    def run(self, coordinates, stops):
        x0, px0, y0, py0, tau0, delta0 = coordinates.T
        particles = self._model.electron_particles(
            self.converted.initial_energy_ev,
            x=x0, px=px0, y=y0, py=py0, zeta=-tau0, delta=delta0,
        )
        count, at = len(coordinates), 0
        for stop, index in self._stops[: len(stops)]:
            if index > at:
                self._line.track(particles, ele_start=at, ele_stop=index)
                at = index
            yield (stop, *self._ordered(particles, count))
