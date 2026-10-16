"""Many settings of the error-free twin in one vectorised Cheetah track.

`OpticsEnv._probe` tracks one setting at a time. Cheetah broadcasts element
parameters over leading batch dimensions, so setting each knob to a tensor of N
values and tracking the nominal beam once gives all N readings in one pass: the
transfer maps become (N, 7, 7) and the particles (N, n, 7).

    stats, transmission, names = probe_batch(env, {"cla_feh_mag_quad_01": k1s, ...})

`stats` is (N, n_diagnostics, 4): centroid x, y and rms x, y at every BPM and
screen in tracking order, NaN where too little beam arrives, exactly as `_probe`
returns them; `transmission` is (N,). Knobs named in neither `columns` nor `base`
are at the lattice file's strengths.

The batched track runs on its own deep copy of the twin's Cheetah segment, and
every knob is written on every call, so nothing of a batch can stay behind in the
twin that `_probe` and the re-match use.
"""

from __future__ import annotations

import copy

import cheetah
import numpy as np
import torch

# optics_env puts RL_framework on the path, so the names come through it
from .optics_env import BPM_TAG, DTYPE, SCREEN_TAG, VBC_EDGE, OpticsEnv

# Settings per vectorised track: memory goes as chunk x particles x 7.
CHUNK = 512


class _BatchSegment:
    """A private copy of the twin's segment, cut at the same stops."""

    def __init__(self, twin):
        segment = copy.deepcopy(twin.transport._segment)
        self.elements = {element.name: element for element in segment.elements}
        self.chunks, previous = [], 0
        for stop in [s.name for s in twin._stops]:
            index = segment.element_index(stop)
            self.chunks.append((stop, cheetah.Segment(segment.elements[previous:index])))
            previous = index

    def write(self, element: str, prop: str, values: torch.Tensor) -> None:
        """`CheetahTransport.write` with a tensor of values."""
        target = self.elements[element]
        setattr(target, prop, values)
        if prop == "angle" and element in VBC_EDGE:
            face = "dipole_e2" if VBC_EDGE[element] == "exit" else "dipole_e1"
            setattr(target, face, values)


def _segment(env: OpticsEnv) -> _BatchSegment:
    if getattr(env, "_batch_segment", None) is None:
        if env.twin.config.backend != "cheetah":
            raise ValueError("probe_batch needs the Cheetah backend")
        env._batch_segment = _BatchSegment(env.twin)
    return env._batch_segment


def _write(env: OpticsEnv, columns: dict[str, np.ndarray]) -> _BatchSegment:
    """Every knob of the batch segment to its column, or to the lattice file."""
    twin = env.twin
    segment = _segment(env)
    for name, knob in twin._knobs.items():
        values = columns.get(name)
        if values is None:
            tensor = torch.tensor(float(twin._nominal_strengths[name]), dtype=DTYPE)
        else:
            tensor = torch.as_tensor(np.asarray(values, dtype=float), dtype=DTYPE)
        segment.write(knob.element_name, knob.prop, tensor)
        for element, sign in knob.coupled:
            segment.write(element, knob.prop, sign * tensor)
    return segment


def _launch(env: OpticsEnv, coordinates: np.ndarray | None):
    """The beam to track, and which of its particles are alive: all of them."""
    twin = env.twin
    if coordinates is None:
        coordinates = twin._coordinates
    particles = torch.tensor(
        np.concatenate([coordinates, np.ones(coordinates.shape[:-1] + (1,))], axis=-1),
        dtype=DTYPE,
    )
    beam = cheetah.ParticleBeam(particles=particles, energy=twin.transport._energy)
    return beam, torch.ones(coordinates.shape[:-1], dtype=torch.bool)


def _run(env: OpticsEnv, chunks, beam, live, count: int):
    """Track `beam` through `chunks`, reading every BPM and screen on the way.
    `live` is updated in place."""
    radius = env.twin.config.beam_pipe_radius
    launched = live.shape[-1]
    minimum = env.twin.config.min_screen_fraction * launched
    names, stats = [], []
    with torch.no_grad():
        for stop, chunk in chunks:
            if len(chunk.elements):
                beam = chunk.track(beam)
            x = beam.x.expand(count, -1)
            y = beam.y.expand(count, -1)
            live &= (beam.survival_probabilities > 0.5).expand(count, -1)
            if radius is not None:
                live &= x**2 + y**2 <= radius**2
            if BPM_TAG not in stop and SCREEN_TAG not in stop:
                continue
            names.append(stop)
            n = live.sum(dim=1).to(DTYPE)
            w = live.to(DTYPE)
            mx = (x * w).sum(1) / n
            my = (y * w).sum(1) / n
            # population std over the live particles, as numpy's std
            sx = torch.sqrt(((x - mx[:, None]) ** 2 * w).sum(1) / n)
            sy = torch.sqrt(((y - my[:, None]) ** 2 * w).sum(1) / n)
            row = torch.stack([mx, my, sx, sy], dim=1).double().numpy()
            row[(n < minimum).numpy()] = np.nan
            stats.append(row)
    stats = np.stack(stats, axis=1) if stats else np.zeros((count, 0, 4))
    return stats, names, beam


def _track(env: OpticsEnv, columns: dict[str, np.ndarray], count: int,
           coordinates: np.ndarray | None = None):
    """`coordinates`, (count, particles, 6), tracks a different beam per row instead
    of the twin's nominal beam."""
    segment = _write(env, columns)
    beam, live = _launch(env, coordinates)
    live = live.expand(count, -1).clone()
    stats, names, _ = _run(env, segment.chunks, beam, live, count)
    transmission = live.sum(dim=1).double().numpy() / live.shape[-1]
    return stats, transmission, names


class Stations:
    """The nominal beam tracked once with `base`, kept at every diagnostic.

    A batch of settings that moves only magnets from some point of the line on is
    then tracked from the last diagnostic before the first of them (`track`), so a
    knob near the end of the line costs the elements after it, not the whole line.
    Diagnostics upstream of that point read as at `base`.
    """

    def __init__(self, env: OpticsEnv, base: dict[str, float]):
        self.env = env
        self.base = dict(base)
        segment = _write(env, {k: np.array([v]) for k, v in self.base.items()})
        beam, live = _launch(env, None)
        live = live[None].clone()
        # the beam and its live particles at the start of each chunk
        self.beams, self.lives, stats, self.names = [], [], [], []
        for stop, chunk in segment.chunks:
            self.beams.append(beam)
            self.lives.append(live.clone())
            row, names, beam = _run(env, [(stop, chunk)], beam, live, 1)
            stats.append(row)
            self.names += names
        self.stats = np.concatenate(stats, axis=1)
        self.transmission = float(live.sum()) / live.shape[-1]
        # the chunk each element is tracked in
        elements = env.twin.transport._segment
        self._ends = [elements.element_index(stop) for stop, _ in segment.chunks]

    def start(self, names) -> int:
        """The chunk holding the most upstream of these knobs."""
        twin = self.env.twin
        elements = twin.transport._segment
        index = min(elements.element_index(element) for n in names
                    for element in (twin._knobs[n].element_name,
                                    *(e for e, _ in twin._knobs[n].coupled)))
        return next(i for i, end in enumerate(self._ends) if end > index)

    def track(self, columns: dict[str, np.ndarray]):
        """Same returns as `probe_batch`, for `columns` on top of `base`."""
        start = self.start(columns)
        count = len(next(iter(columns.values())))
        full = {**{k: np.full(count, v) for k, v in self.base.items()
                   if k not in columns}, **columns}
        segment = _write(self.env, full)
        live = self.lives[start].expand(count, -1).clone()
        stats, names, _ = _run(self.env, segment.chunks[start:], self.beams[start],
                               live, count)
        upstream = len(self.names) - len(names)
        stats = np.concatenate(
            [np.repeat(self.stats[:, :upstream], count, axis=0), stats], axis=1)
        transmission = live.sum(dim=1).double().numpy() / live.shape[-1]
        return stats, transmission, self.names


def probe_batch(
    env: OpticsEnv, columns: dict[str, np.ndarray], base: dict[str, float] | None = None
):
    """Track every row of `columns` (name -> N values) on top of `base`."""
    count = len(next(iter(columns.values())))
    base = {k: v for k, v in (base or {}).items() if k not in columns}
    stats, transmission, names = [], [], None
    for start in range(0, count, CHUNK):
        part = {k: np.asarray(v)[start : start + CHUNK] for k, v in columns.items()}
        n = len(next(iter(part.values())))
        full = {**{k: np.full(n, v) for k, v in base.items()}, **part}
        s, t, names = _track(env, full, n)
        stats.append(s)
        transmission.append(t)
    return np.concatenate(stats), np.concatenate(transmission), names


# Beams per vectorised track in `probe_beams`: each carries all its particles.
BEAM_CHUNK = 48


def probe_beams(env: OpticsEnv, coordinates: np.ndarray, settings: dict[str, float]):
    """Track N different beams, `coordinates` (N, particles, 6), through one setting
    of the twin (`settings` on top of the lattice file). Same returns as
    `probe_batch`, one row per beam."""
    stats, transmission, names = [], [], None
    for start in range(0, len(coordinates), BEAM_CHUNK):
        part = coordinates[start:start + BEAM_CHUNK]
        s, t, names = _track(env, dict(settings), len(part), part)
        stats.append(s)
        transmission.append(t)
    return np.concatenate(stats), np.concatenate(transmission), names
