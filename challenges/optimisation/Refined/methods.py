"""The tuning methods, on top of a precomputed family. Each builds on the last.

    session = Session(machine, family, spec)
    METHODS["steer"](session)
    session.machine.score

| method   | what it adds                                               |
|----------|------------------------------------------------------------|
| `none`   | nothing; the do-nothing baseline                            |
| `optics` | look the request up in the family and load it: one shot     |
| `steer`  | SVD orbit steering on the BPMs, then place the spot         |
| `mpc`    | a measurement-anchored CEM/Adam tweak against a `Line` twin |
| `bo`     | multi-fidelity Bayesian optimisation of the same knobs      |

Nothing here knows the name of a magnet. Every method works from the `MachineSpec`
and the artifact, so the same code runs on CLARA and on whatever lattice somebody
points `describe.py` at next. The CLARA-specific versions these are lifted from —
`MFopt/steering.py`, `MFopt/mfbo.py`, `MBRL_opt/planner.py` — each carry a few
dozen hard-coded names between them, which is the thing `Refined/` exists to undo.

The one principle running through all of them is **anchoring**: the model is asked
what a change *does*, never what the reading *is*. A model that is wrong by a
constant still predicts the effect of moving a quadrupole perfectly well, and
subtracting its own prediction at the present setting cancels that constant
exactly. It is also what an operator does — you do not trust the model to tell you
where the beam is, you trust it to tell you which way to turn.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from artifact import Family, Tune  # noqa: E402
from machine import KICK, TRANSMISSION_WEIGHT, MachineSpec  # noqa: E402
from machines import Machine  # noqa: E402

# Singular values below this fraction of the largest are dropped from a steering
# solve. Keeping them asks for enormous kicks to correct a direction the magnets
# barely move, which is how a steering correction loses a beam it had.
RCOND = 1e-3
# Raised by this much after a shot that lost ground: fewer modes, gentler kicks.
RCOND_BACKOFF = 10.0
RCOND_CEILING = 0.1
# A corrector counts as "final" — able to move the spot and nothing a BPM sees —
# when its largest response at any BPM is below this fraction of its response at
# the readout. Derived from the response rather than named, so it works on a
# lattice nobody has written a list for.
FINAL_CORRECTOR_RATIO = 1e-2
# Corrector elements the placement solve may use, nearest the readout. Two gives
# four planes, which is two degrees of freedom per plane for a two-degree
# problem — enough to be exact, few enough to disturb nothing upstream.
PLACEMENT_ELEMENTS = 2
# Screens the orbit stage may move to, when there are no BPMs to read instead.
# One, because one recovers the dead BPMs almost exactly and the rest is flat:
# measured over 12 seeds, score 2.028 with none, 1.671 with one, 1.793 with two
# and 1.645 with three, against 1.680 on the same machine with its BPMs working.
# Each costs a shot and about half a minute, so the second is 30 s for nothing.
SCREEN_BUDGET = 1
# Below this the corrector does not move the readout at all and is numerical
# noise rather than a knob, in metres of spot per radian of kick.
USELESS = 1e-6
# CEM, for the MPC tweak.
CEM_POPULATION = 192
CEM_ITERATIONS = 4
CEM_ELITES = 24
# How far one planned shot may move a knob, as a fraction of its usable span, and
# how that leash adapts: a step that improved the reading earns a longer one, a
# step that did not loses half of it.
TRUST = 0.25
TRUST_GROWTH = 1.3
TRUST_DECAY = 0.5
# What a candidate that misses the screen is charged: more than the worst beam
# that lands can cost, so the planner never prefers losing it.
LOST_BEAM_COST = 20.0
SCRAPE_WEIGHT = 1e3


@dataclass
class Session:
    """One episode: the machine, the family it is being tuned from, the budget."""

    machine: Machine
    family: Family
    spec: MachineSpec
    budget: int = 25
    # Particles the planning twin tracks. A tenth of what the machine uses, on
    # purpose: the twin is only ever asked what a *change* does, and anchoring
    # cancels the constant difference between the two. Tracking cost is close to
    # linear in this, and the CEM tracks a population of 192 per shot.
    twin_particles: int = 1000
    # Where the planning twin tracks. The CEM population is the only
    # batch in stage 2 big enough for a GPU to matter, and only once the
    # particle count is high.
    twin_device: str = "cpu"
    # A `Line` at the machine's settings, for the methods that plan against a
    # model. Built lazily, because `optics` and `steer` never need one.
    twin: object | None = None
    tune: Tune | None = None
    log: list[str] = field(default_factory=list)
    # Which stage of the pipeline the next shot belongs to. The methods set it;
    # nothing in the tuning reads it. It exists so a recorder can label a shot
    # without guessing from what the action moved — a guess that gets the
    # restore shot wrong, because restoring re-writes every magnet and so looks
    # exactly like loading the optics.
    stage: str = "start"

    @property
    def target(self) -> np.ndarray:
        return self.machine.target

    @property
    def left(self) -> int:
        return self.budget - self.machine.shots

    def say(self, text: str) -> None:
        self.log.append(text)

    def set(self, settings: dict[str, float]):
        if self.left <= 0:
            raise OutOfShots(f"the episode's {self.budget} shots are spent")
        return self.machine.set(settings)

    def model(self):
        """The twin, built on first use and kept at the machine's settings."""
        if self.twin is None:
            from lattice import Line

            self.twin = Line(self.spec, num_particles=self.twin_particles,
                             device=self.twin_device)
        self.twin.reset()
        self.twin.set({
            name: value for name, value in self.machine.last.settings.items()
            if name in self.twin.knobs
        })
        return self.twin


class OutOfShots(RuntimeError):
    """The budget ran out. Methods let this end the episode rather than catch it."""


# ---------------------------------------------------------------- the methods


def none(session: Session) -> None:
    """Do nothing. The baseline every other number is only meaningful against."""


def optics(session: Session) -> None:
    """Load the matched tune for the request. One shot, no measurement used.

    This is most of the job on a good day and it costs nothing to find: the
    family already knows which strengths give this spot, so the agent sets them
    before it has looked at the machine once.
    """
    target = session.target
    # The family does not just answer the request, it says which setting of each
    # scan axis the request is reachable at — on CLARA the compressor angle, and
    # it is worth a factor of five in the smallest vertical spot the final focus
    # can make. So the axes are chosen for the request rather than left wherever
    # the machine happens to sit, and then set explicitly, so the tune the family
    # solved for is the tune that ends up loaded.
    session.stage = "optics"
    present = dict(session.machine.last.settings)
    tune, axes = session.family.choose(target[2], target[3], present=present)
    axes = {n: session.machine.clip(n, v) for n, v in axes.items()}
    session.tune = tune
    session.say(
        f"optics: model gives {tune.achieved[0] * 1e3:.4f} x "
        f"{tune.achieved[1] * 1e3:.4f} mm at T = {tune.transmission:.2f}, "
        f"confidence {tune.confidence:.2f}"
        + (f", at {', '.join(f'{n}={v:g}' for n, v in axes.items())}" if axes else "")
        + ("  [request outside the grid]" if tune.clamped else "")
    )
    settings = {n: v for n, v in {**axes, **tune.k1}.items()
                if n in session.machine.limits}
    session.set(settings)


def steer(session: Session, orbit_shots: int = 4, spot_shots: int = 2,
          restore: bool = True, screen_budget: int = SCREEN_BUDGET) -> None:
    """Optics, then the orbit, then the spot where it was asked for.

    Both solves use the family's own response matrix, measured in the model at
    this optics, so they already carry the chicane, the cavities and the
    chromatic averaging over the beam's energy spread. The orbit is compared
    against the **model's** orbit, not against zero: with dispersion the
    error-free beam is not on axis, so zero is the wrong place to steer to and
    steering there bends a good orbit into a bad one.

    **Where the monitors come from.** With working BPMs the orbit is read every
    shot for free and the loop below runs as it always has. With none — CLARA at
    the moment — the only monitor is whichever screen is in the beam, and a
    screen intercepts: moving it costs a shot and about half a minute, and while
    it is upstream the request is not being read at all. So:

    - The readout screen is already in the beam when the optics loads, and on
      every seed measured the beam arrives there. Its centroid is all the
      placement solve needs, so the ordinary episode moves no screen at all.
    - If the beam is *not* on the readout, `_find_beam` walks upstream to find
      where it stops. That is the only case where a switch is needed rather than
      chosen.
    - `screen_budget` above zero additionally spends that many switches
      correcting the orbit from upstream screens. It defaults to zero because
      the measurement says not to: `mpc`, which re-optimises the focus against
      its twin, scores *better* with no orbit correction at all (0.396 against
      0.560, and five fewer shots).
    """
    optics(session)
    tune = session.tune
    rows, upstream, final = _split_correctors(session, tune)
    if not upstream and not final:
        session.say("steer: no corrector has a usable response; nothing to do")
        return

    session.stage = "steer"
    if not session.machine.last.on_screen:
        # The beam is not where the request is. Nothing else can proceed until
        # it is found, and finding it is the one thing worth a switch outright.
        _find_beam(session, tune, upstream)

    # Screens substitute for BPMs, they do not supplement them: where the orbit
    # can be read every shot for free, paying half a minute to read two numbers
    # off one screen buys nothing. So this engages only when there is no
    # continuously readable monitor — and only if the beam is actually at the
    # readout, because correcting an orbit that does not arrive is two more
    # switches spent on a measurement that cannot be scored.
    if (screen_budget > 0 and upstream and not rows
            and session.machine.last.on_screen):
        _steer_on_screens(session, tune, upstream, screen_budget)

    rcond = RCOND
    best = session.machine.last
    for _ in range(min(orbit_shots, max(0, session.left - spot_shots - 1))):
        if not upstream or not rows:
            break
        reading = session.machine.last
        error, keep = _orbit_error(session, reading, tune, rows)
        if not keep:
            session.say("steer: no BPM is reading; the beam is not getting there")
            break
        matrix = tune.R[keep][:, upstream]
        kicks = -_pseudo_inverse(matrix, rcond) @ error
        now = reading.settings
        action = {
            name: session.machine.clip(name, now.get(name, 0.0) + float(dk))
            for name, dk in zip([session.family.correctors[i] for i in upstream],
                                kicks)
        }
        after = session.set(action)
        if after.score < best.score:
            best = after
        else:
            # Losing ground means the solve asked for more than the response can
            # honestly support. Fewer modes, gentler kicks, and go back first.
            rcond = min(rcond * RCOND_BACKOFF, RCOND_CEILING)
            session.set(dict(best.settings))
            if rcond >= RCOND_CEILING:
                break

    _place(session, tune, final, shots=spot_shots)
    # Only when `steer` is the whole episode. As a stage of `mpc` or `bo` the
    # restore is a shot spent going back to a tune the next stage overwrites.
    if restore:
        _finish(session)


def _place(session: Session, tune: Tune, final: list[int], shots: int = 2,
           twin=None) -> None:
    """Put the spot where it was asked for, with the correctors the BPMs cannot see.

    This is where most of the score is. The benchmark divides every error by the
    mean requested spot, and this machine is asked for spots of 80 to 400 um at
    offsets of up to 2 mm — so the position term alone is worth 2.5 to 25 spot
    sizes against a score that stops counting at 10. Sizing the beam perfectly
    and leaving it a millimetre off is a failed tune.

    `twin` recomputes the response at the settings in force rather than reading
    it from the family. That matters after the focus has moved: on CLARA
    `cla_feh_mag_hvcor_01` sits between the last two matched quadrupoles, so
    changing the optics changes what its kick does at the readout, and the
    family's response was measured at the optics the family matched.
    """
    if not final:
        return
    session.stage = "final" if twin is not None else "place"
    names = [session.family.correctors[i] for i in final]
    best = session.machine.last
    for _ in range(min(shots, max(0, session.left - 1))):
        reading = session.machine.last
        if not reading.on_screen:
            break
        matrix = (_placement_response(twin, names) if twin is not None
                  else _family_response(session, tune, final))
        error = np.array([reading.mu_x, reading.mu_y]) - session.target[:2]
        # The same truncated inverse the orbit solve uses: it drops singular
        # values that would ask for an enormous kick in a direction the magnets
        # barely move, and guards the NaNs a lost beam leaves in the response.
        kicks = -_pseudo_inverse(matrix, RCOND) @ error
        now = reading.settings
        after = session.set({
            name: session.machine.clip(name, now.get(name, 0.0) + float(dk))
            for name, dk in zip(names, kicks)
        })
        if after.score >= best.score:
            break
        best = after


def _family_response(session: Session, tune: Tune, final: list[int]) -> np.ndarray:
    """d(readout centroid)/d(final corrector), from the precomputed family."""
    readout = session.family.diagnostics.index(session.family.readout)
    return tune.R[[2 * readout, 2 * readout + 1]][:, final]


def _placement_response(twin, names: list[str]) -> np.ndarray:
    """The same, measured in the twin at whatever is set now. One batched track."""
    import torch

    base = np.array([[twin.value(n) for n in names]])
    population = np.repeat(base, len(names) + 1, axis=0)
    for j in range(len(names)):
        population[j + 1, j] += KICK
    with torch.no_grad():
        readings = twin.track(
            overrides=twin.columns(names, population), batch=len(population)
        )
        centroid, _ = readings.at(twin.readout)
    centroid = centroid.detach().cpu().numpy().astype(float)
    return np.nan_to_num((centroid[1:] - centroid[0]).T / KICK)


def mpc(session: Session, shots: int = 10) -> None:
    """Steer, then close the gap with a twin the measurement is anchored on.

    What is left after the family and the orbit solve is the difference between
    the machine the optics was matched for and the one today actually is. The
    twin cannot be told that difference, but it does not need to be: anchoring
    cancels whatever the twin has constant-wrong, and what remains is the error
    in the *derivative*, which is far smaller.

    Size only. Where the spot lands is a corrector's job and a linear one, and
    leaving position in the objective only tempts the quadrupoles into fixing a
    2 mm offset by distorting the optics — which they can do, and which ruins the
    size.

    So the order is optics, steer, focus, **steer again**. The last one is not
    tidying up: the focus moves the matched quadrupoles, the spot moves with
    them, and on CLARA one of the placement correctors sits between the last two
    of those quadrupoles, so the response used to place the spot is stale as soon
    as the focus changes. Placement is also where most of the score is.
    """
    import torch

    steer(session, orbit_shots=3, spot_shots=1, restore=False)
    knobs = _tweak_knobs(session)
    if not knobs:
        session.say("mpc: no knob to tweak")
        return
    twin = session.model()
    low = np.array([session.machine.limits[n][0] for n in knobs])
    high = np.array([session.machine.limits[n][1] for n in knobs])
    rng = np.random.default_rng(0)
    trust = TRUST
    session.stage = "focus"
    best = session.machine.last

    for _ in range(min(shots, max(0, session.left - 2))):
        reading = session.machine.last
        values = np.array([reading.settings.get(n, twin.value(n)) for n in knobs])
        twin.reset()
        twin.set({n: v for n, v in reading.settings.items() if n in twin.knobs})
        # The anchor: what to add to the twin's prediction so it agrees with the
        # shot just taken. Zero size is no information, so a lost beam anchors at
        # nothing and the twin's own absolute prediction is all there is.
        with torch.no_grad():
            _, size = twin.track().beam_at(twin.readout)
        anchor = (np.array([reading.sigma_x, reading.sigma_y]) - size.detach().cpu().numpy()
                  if reading.on_screen else np.zeros(2))

        span = trust * (high - low)
        candidate, cost = _cem(
            twin, knobs, values, session.target, anchor,
            np.maximum(low, values - span), np.minimum(high, values + span), rng,
        )
        after = session.set(dict(zip(knobs, candidate)))
        if after.score < best.score:
            best, trust = after, min(trust * TRUST_GROWTH, 1.0)
        else:
            trust *= TRUST_DECAY
            if trust < 1e-3:
                break
    _final_steer(session)
    _finish(session)


def bo(session: Session, shots: int = 12) -> None:
    """Steer, then multi-fidelity Bayesian optimisation of the same knobs.

    Two fidelities, as `MFopt/mfbo.py` has it: the twin is free, so it is sampled
    densely and a GP fitted to it; the machine is expensive, so it is modelled as
    the twin plus a discrepancy with its own GP. After one shot the discrepancy
    is already a constant offset, which makes the first proposal the model's own
    answer corrected by what today's beam does differently — the same anchoring
    the MPC method does, learnt rather than subtracted.
    """
    try:
        import botorch  # noqa: F401
        import gpytorch  # noqa: F401
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "the `bo` method needs botorch and gpytorch, which are not installed "
            "in this environment:\n"
            "    uv pip install --python .venv/bin/python botorch gpytorch\n"
            f"({error})"
        ) from None
    from bayes import multi_fidelity  # noqa: E402

    steer(session, orbit_shots=3, spot_shots=1, restore=False)
    multi_fidelity(session, _tweak_knobs(session), shots)
    _final_steer(session)
    _finish(session)


METHODS = {
    "none": none,
    "optics": optics,
    "steer": steer,
    "mpc": mpc,
    "bo": bo,
}


# ---------------------------------------------------------------- the pieces


def _final_steer(session: Session, shots: int = 2) -> None:
    """Step 4: place the spot again, against the optics the tuning ended on.

    The focus stage moves the quadrupoles and the spot moves with it, so the
    placement done before it no longer holds. The response is re-measured in the
    twin at the settings now in force rather than read from the family, because
    the family's was measured at a different optics.
    """
    if session.tune is None:
        return
    _, _, final = _split_correctors(session, session.tune)
    _place(session, session.tune, final, shots=shots, twin=session.model())


def _finish(session: Session) -> None:
    """Go back to the best reading taken, if the last one was not it.

    The benchmark scores where the episode *ends*, not the best it ever saw, so a
    method that finds a good tune and then takes one bad exploratory shot books
    the bad one. One shot to go back is nearly always cheaper than that.
    """
    session.stage = "restore"
    if session.left > 0:
        restored = session.machine.restore_best()
        if restored is not None:
            session.say("returned to the best setting")


def _split_correctors(session: Session, tune: Tune):
    """(row map, upstream corrector columns, final corrector columns).

    A "final" corrector is one **no monitor in use can see** — downstream of all
    of them — so it moves the spot at the readout and nothing else. That makes
    placement a small solve that cannot disturb the orbit, which is why it is
    worth separating. Found from the response matrix itself rather than from a
    list of names, so it needs nothing said about the lattice.

    "Monitor" means the BPMs *and* the upstream screens, not the BPMs alone.
    With working BPMs the screens are redundant — a BPM sits near enough every
    screen — and the answer is the same four planes as before. With the BPMs
    dead, reading it as BPMs-only makes every one of the 58 planes "invisible",
    and the placement solve would reach the length of the machine to move the
    spot a millimetre.

    The row map pairs each monitor's rows in the family's response with the same
    monitor's row in whatever the machine reports. The two are *not* the same
    list: the family carries every diagnostic of the line and the machine may
    report a different subset in a different order. Indexing one with the
    other's rows compares a monitor against something else entirely and steers
    confidently to the wrong orbit, so they are matched by name.
    """
    family = session.family
    readout = family.diagnostics.index(family.readout)
    machine_rows = {name: i for i, name in enumerate(session.machine.diagnostics)}
    # Two different questions, and they want different lists.
    #
    # *Where do the final correctors start?* Every monitor upstream of the
    # readout bounds that, whether or not it can be read this shot — a screen
    # exists and will be looked at if the orbit needs it.
    monitors = set(session.spec.bpms) | set(session.spec.screens)
    flat = [r for i, name in enumerate(family.diagnostics)
            if name in monitors and i < readout
            for r in (2 * i, 2 * i + 1)]

    # *Which monitors can I read continuously?* Only those the machine reports
    # every shot, which is the BPMs: a screen reads only while it is the one in
    # the beam, and `steer`'s screen stage handles those one at a time. Mapping
    # a screen to a machine row it does not have gave it index -1, which is the
    # last BPM — steering confidently against the wrong monitor.
    rows = [
        (2 * i, 2 * i + 1, machine_rows[name])
        for i, name in enumerate(family.diagnostics)
        if name in set(session.spec.bpms) and name in machine_rows
    ]
    readout_rows = [2 * readout, 2 * readout + 1]
    at_monitors = (np.abs(tune.R[flat]).max(axis=0) if flat
                   else np.zeros(tune.R.shape[1]))
    at_readout = np.abs(tune.R[readout_rows]).max(axis=0)
    # "Final" means one thing only: no BPM can see it, so it is downstream of all
    # of them and moves the spot without disturbing the orbit.
    #
    # An earlier version also demanded its reach at the readout clear 1 % of the
    # largest reach of *any* corrector, which quietly disabled placement
    # altogether. The largest belongs to a corrector near the start of the line,
    # whose kick the whole downstream optics amplifies: 13.9 m/rad against
    # 0.077 m/rad for the corrector a metre before the screen. Comparing the two
    # is meaningless — the near one is the only one that can place the spot
    # without moving the orbit, whatever the far one's lever arm is.
    invisible = at_monitors <= FINAL_CORRECTOR_RATIO * max(at_monitors.max(), 1e-30)
    final = [j for j in range(tune.R.shape[1])
             if invisible[j] and at_readout[j] > USELESS]
    # Placing a spot is a two-degree-of-freedom problem, so it wants the fewest
    # correctors that can do it, nearest the readout. Being invisible to the
    # monitors is necessary and not sufficient: a corrector upstream of the
    # final-focus quadrupoles is invisible too, and tilting the beam through
    # them changes the *size*, which placement has no business touching. With
    # the BPMs dead the invisible set runs to ten planes, three elements of
    # which sit before the last quadrupoles — using all of them cost 537 um of
    # vertical spot against 422 um for the nearest two elements alone.
    #
    # With working BPMs the last monitor is already inside the final focus, so
    # this changes nothing there: the invisible set is those same two elements.
    final = _nearest_elements(session, final, PLACEMENT_ELEMENTS)
    upstream = [j for j in range(tune.R.shape[1])
                if j not in final and at_monitors[j] > 0]
    return rows, upstream, final


def _upstream_screens(session: Session) -> list[tuple[str, int]]:
    """(screen, family row index) for every screen before the readout, in order."""
    family = session.family
    readout = family.diagnostics.index(family.readout)
    wanted = set(session.spec.screens)
    return [(name, i) for i, name in enumerate(family.diagnostics)
            if name in wanted and i < readout]


def _screen_error(session: Session, reading, tune: Tune, row: int):
    """How far the beam sits from where the model says it should, on one screen."""
    measured = np.array([reading.mu_x, reading.mu_y])
    predicted = tune.orbit.ravel()[[2 * row, 2 * row + 1]]
    return measured - predicted


def _find_beam(session: Session, tune: Tune, upstream: list[int]) -> None:
    """The beam is not on the readout. Walk upstream until something sees it.

    A bisection rather than a sweep, because each look costs a switch: if the
    middle screen of the remaining stretch sees beam the problem is downstream
    of it, and if it does not the problem is upstream. That finds the last
    screen with beam in about log2(21) ~ 5 looks instead of 21.

    Having found it, the orbit is corrected there with the correctors upstream
    of that screen, and the readout is put back in the beam — always, because
    the score is only read there and a budget spent without returning is a
    total loss.
    """
    machine = session.machine
    screens = _upstream_screens(session)
    if not screens:
        session.say("the beam is off the readout and there is no screen to look at")
        return

    low, high, found = 0, len(screens) - 1, None
    while low <= high and session.left > 2:
        middle = (low + high) // 2
        name, row = screens[middle]
        reading = machine.look_at(name)
        if reading.on_screen:
            found = (name, row, reading)
            low = middle + 1          # beam reaches here; look further along
        else:
            high = middle - 1         # it is already lost; look further back
    if found is None:
        session.say("no screen sees the beam; it is lost before the first of them")
        machine.look_at(machine.readout)
        return

    name, row, reading = found
    session.say(f"the beam reaches {name} but not the readout; correcting there")
    usable = [j for j in upstream
              if abs(tune.R[2 * row, j]) + abs(tune.R[2 * row + 1, j]) > USELESS]
    if usable and session.left > 2:
        error = _screen_error(session, reading, tune, row)
        kicks = -_pseudo_inverse(tune.R[[2 * row, 2 * row + 1]][:, usable], RCOND) @ error
        now = reading.settings
        session.set({
            n: machine.clip(n, now.get(n, 0.0) + float(dk))
            for n, dk in zip([session.family.correctors[j] for j in usable], kicks)
        })
    machine.look_at(machine.readout)


def _steer_on_screens(session: Session, tune: Tune, upstream: list[int],
                      budget: int) -> None:
    """Correct the orbit from a few upstream screens, within a switch budget.

    Which screens: the ones that between them pin the correctors down best, by
    the conditioning of the response they give. Greedy on the smallest singular
    value of `R[chosen rows][:, upstream]` — adding the screen that most
    improves how well the solve is determined. The response is the family's
    own, so this costs no tracking, only the looking.

    Each screen is one shot and one switch, and the return to the readout is one
    more. That is why the budget exists: at about half a minute each, three
    screens is two minutes of machine time against roughly a second a shot.
    """
    machine = session.machine
    screens = _upstream_screens(session)
    if not screens or budget <= 0:
        return
    session.stage = "screens"

    chosen: list[tuple[str, int]] = []
    rows: list[int] = []
    for _ in range(min(budget, len(screens), max(0, session.left - 3))):
        best, best_score = None, -np.inf
        for name, row in screens:
            if (name, row) in chosen:
                continue
            trial = rows + [2 * row, 2 * row + 1]
            singular = np.linalg.svd(tune.R[trial][:, upstream], compute_uv=False)
            if singular[-1] > best_score:
                best, best_score = (name, row), float(singular[-1])
        if best is None:
            break
        chosen.append(best)
        rows += [2 * best[1], 2 * best[1] + 1]

    error, keep = [], []
    for name, row in chosen:
        reading = machine.look_at(name)
        if not reading.on_screen:
            session.say(f"{name} sees no beam; skipping it")
            continue
        error += list(_screen_error(session, reading, tune, row))
        keep += [2 * row, 2 * row + 1]
    if keep and session.left > 2:
        kicks = -_pseudo_inverse(tune.R[keep][:, upstream], RCOND) @ np.array(error)
        now = machine.last.settings
        session.set({
            n: machine.clip(n, now.get(n, 0.0) + float(dk))
            for n, dk in zip([session.family.correctors[j] for j in upstream], kicks)
        })
        session.say(f"orbit corrected from {len(keep) // 2} screen(s), "
                    f"{machine.screen_switches} switch(es) so far")
    # Always back to the readout: it is the only place the score is read.
    if not machine.looking_at_readout:
        machine.look_at(machine.readout)


def _nearest_elements(session: Session, columns: list[int], count: int) -> list[int]:
    """The planes of the `count` corrector elements closest to the readout."""
    names = session.family.correctors
    order = {name: i for i, name in enumerate(session.spec.knobs and
                                              [k.name for k in session.spec.knobs])}
    # A plane is `<element>_h` / `_v`; group by element, keep the last few in
    # lattice order, which is the order the spec declares its knobs in.
    by_element: dict[str, list[int]] = {}
    for j in columns:
        by_element.setdefault(names[j].rsplit("_", 1)[0], []).append(j)
    latest = sorted(by_element, key=lambda e: max(
        order.get(f"{e}_h", -1), order.get(f"{e}_v", -1), order.get(e, -1)))
    return [j for element in latest[-count:] for j in by_element[element]]


def _orbit_error(session: Session, reading, tune: Tune, rows):
    """(error vector, response rows) over the BPMs that actually read."""
    valid = np.asarray(reading.bpm_valid, dtype=float) > 0.5
    error, keep = [], []
    for row_x, row_y, machine_row in rows:
        if not valid[machine_row]:
            continue
        measured = np.asarray(reading.bpm, dtype=float)[machine_row]
        error += [measured[0] - tune.orbit.ravel()[row_x],
                  measured[1] - tune.orbit.ravel()[row_y]]
        keep += [row_x, row_y]
    return np.array(error), keep


def _pseudo_inverse(matrix: np.ndarray, cutoff: float) -> np.ndarray:
    """Truncated pseudo-inverse: the classical SVD orbit correction."""
    u, s, vt = np.linalg.svd(np.nan_to_num(matrix), full_matrices=False)
    keep = s > cutoff * (s[0] if len(s) else 1.0)
    inverse = np.zeros_like(s)
    inverse[keep] = 1.0 / s[keep]
    return vt.T @ np.diag(inverse) @ u.T


def _tweak_knobs(session: Session) -> list[str]:
    """What the planner may turn: the matched quadrupoles and the non-linear knobs.

    Not the standing optics — that is the machine the family was built on, and
    moving it invalidates everything the family said. Not the correctors either:
    they are linear and already solved.
    """
    wanted = list(session.spec.matching_quads) + [
        k.name for k in session.spec.nonlinear_knobs()
    ]
    return [n for n in wanted if n in session.machine.limits]


def _cem(twin, knobs, values, target, anchor, low, high, rng):
    """Sample, keep the best, resample around them. Batched, so a population of
    two hundred candidates costs about what one does."""
    import torch

    mean = values.copy()
    deviation = (high - low) / 4
    best, best_cost = values.copy(), np.inf
    for _ in range(CEM_ITERATIONS):
        population = np.clip(
            rng.normal(mean, np.maximum(deviation, 1e-12),
                       (CEM_POPULATION, len(knobs))),
            low, high,
        )
        population[0] = mean  # never lose the incumbent to bad luck
        with torch.no_grad():
            readings = twin.track(
                overrides=twin.columns(knobs, population), batch=len(population)
            )
            _, size = readings.beam_at(twin.readout)
            costs = _cost(
                size.detach().cpu().numpy().astype(float) + anchor,
                readings.transmission.detach().cpu().numpy().astype(float),
                readings.scrape.detach().cpu().numpy().astype(float),
                readings.valid.detach().cpu().numpy().astype(float)[..., twin.readout],
                target,
            )
        costs = np.nan_to_num(costs, nan=np.inf, posinf=np.inf)
        order = np.argsort(costs)
        if costs[order[0]] < best_cost:
            best_cost, best = float(costs[order[0]]), population[order[0]]
        elite = population[order[:CEM_ELITES]]
        mean, deviation = elite.mean(axis=0), elite.std(axis=0)
    return best, best_cost


def _cost(size, transmission, scrape, on_screen, target):
    """What a candidate is worth: the benchmark, plus what the candidate risks.

    The first two terms are the benchmark's own — size error in units of the
    requested spot, and transmission at the weight the benchmark gives it. The
    last two are caution: a tune that puts the beam off the screen is charged
    more than the worst delivered beam can cost, and one merely sitting near the
    chamber wall is charged in proportion, because it will not survive the next
    shot's jitter and because the model is least trustworthy exactly there.
    """
    scale = float(np.mean(target[2:]))
    error = np.abs(size - np.asarray(target[2:])) / scale
    return (
        error.mean(axis=-1)
        + TRANSMISSION_WEIGHT * (1.0 - transmission)
        + (1.0 - on_screen) * LOST_BEAM_COST
        + SCRAPE_WEIGHT * scrape
    )
