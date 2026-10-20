"""Describe a machine: read its lattice, then ask what the lattice cannot say.

    python -I describe.py lattice.json --out machine.json
    python -I describe.py lattice.json --defaults            # ask nothing, take defaults
    python -I describe.py lattice.json --answers prev.json   # replay a past session

An interview. It reads a Cheetah lattice, reports what it found, and then asks about
the things that are facts about the hardware rather than about the file: which dipole
strings are knobs, what shares a power supply, where the limits come from, and — the
one that is easy to forget — whether anything other than the requested spot changes
the optics enough that the precomputed family has to span it.

A matched family answers "which quadrupole strengths give this spot?", and that has
a single answer only if everything else about the beam is pinned. On CLARA the L4H
linearizer is presently run as a drift, so it is; powered, its phase would set the
chirp, and the energy spread, and therefore what the dogleg turns into horizontal
size at the readout — and a family built at one phase would be wrong at every other.

That part is measurable: `sensitivity.py` sweeps each candidate knob with the
quadrupoles held still and ranks it by how far the readout spot moves, which is how
far the family would have been wrong to ignore it. What is not measurable is how
many of them to pay for, because every axis multiplies a precompute counted in
hours and the ranking usually falls away sharply after the leader. So the script
ranks and the user takes the top *k* — one, on a first pass.

Every answer is recorded, so a session can be replayed, scripted or audited. The
interview is the interface; `machine.json` and the answer file are the artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from interview import Interview  # noqa: E402
from machine import (  # noqa: E402
    NONLINEAR_PROPERTIES,
    SETTABLE,
    STEERING_PROPERTIES,
    Knob,
    MachineSpec,
    ScanAxis,
    default_knobs,
    plane_name,
)

# How diagnostics are recognised when the lattice does not type them. Cheetah models
# BPMs, screens and apertures all as `Screen`, so the names are all there is.
TAGS = {
    "bpm": ("dia_bpm", "_bpm", "bpm_"),
    "screen": ("dia_scr", "_scr", "scr_", "_screen"),
    "aperture": ("sim_aper", "_aper", "aper_", "collimator", "_col"),
}
GROUP_GAP = 40  # elements; dipoles closer together than this are one string
# Most members a name-stem group may have before it is assumed to be a naming
# convention rather than a power supply. Eight quadrupoles in a section share a
# stem and emphatically do not share a cable.
MAX_GROUP = 6
# Two strengths count as the same supply if they agree to this, in relative terms.
SAME_STRENGTH = 1e-6
GROUP_MIN = 2
CLOSED_BUMP = 0.05  # a string whose angles sum to within this of its largest closes
# Most scan axes the grid may have. The precompute is a product over them, so four
# is already a job nobody runs twice.
MAX_AXES = 4
# Quadrupoles the family matches with, by default: the last few before the readout.
# Two sizes are being asked for, so two is the minimum that can trade the planes off
# against each other, and four leaves enough slack to hold transmission as well.
DEFAULT_MATCHING = 4


# ---------------------------------------------------------------- reading


def load_lattice(path: Path) -> tuple[dict[str, str], list[str], dict]:
    """(element name -> type, the sequence in order, the raw file)."""
    raw = json.loads(Path(path).read_text())
    root = raw.get("root") or next(iter(raw["lattices"]))
    sequence = raw["lattices"][root]
    types = {n: raw["elements"][n][0] for n in sequence if n in raw["elements"]}
    return types, list(sequence), raw


def classify(names: list[str], types: dict[str, str]) -> dict[str, list[str]]:
    """Split the diagnostics into BPMs, screens and apertures.

    The element type is believed first where it says anything: Cheetah has a
    `BPM` type, and a lattice that uses it has already answered the question.
    CLARA's does not — it models BPMs, screens *and* aperture markers all as
    `Screen` — so for those the name tags decide, which is a convention of that
    lattice and not a fact about Cheetah.
    """
    found = {"bpm": [], "screen": [], "aperture": [], "untagged": []}
    for name in names:
        kind = types.get(name)
        if kind == "BPM":
            found["bpm"].append(name)
            continue
        if kind != "Screen":
            continue
        lowered = name.lower()
        for tag_kind, tags in TAGS.items():
            if any(tag in lowered for tag in tags):
                found[tag_kind].append(name)
                break
        else:
            # A `Screen` the tags do not recognise, in a lattice that types its
            # BPMs separately, is simply a screen.
            (found["screen"] if "BPM" in types.values()
             else found["untagged"]).append(name)
    return found


def common_prefix(names: list[str]) -> str:
    parts = [n.split("_") for n in names]
    shared = []
    for chunk in zip(*parts):
        if len(set(chunk)) == 1:
            shared.append(chunk[0])
        else:
            break
    return "_".join(shared) or re.sub(r"\d+$", "", names[0])


def dipole_groups(sequence, types, raw) -> list[dict]:
    """Strings of nearby dipoles, described rather than judged.

    The useful signal is whether a string closes its own bump — angles summing to
    zero — which is true of a chicane and of a dogleg alike, and false of a net
    bend. Which one it *is* decides knob or geometry, and the lattice cannot say.

    An earlier version of this looked for strictly alternating signs and found
    nothing at all on CLARA, because a four-dipole chicane runs (-, +, +, -).
    """
    dipoles = [n for n in sequence if types.get(n) == "Dipole"]
    if not dipoles:
        return []
    angle = {n: float(raw["elements"][n][1].get("angle", 0.0) or 0.0) for n in dipoles}
    order = {n: i for i, n in enumerate(sequence)}
    runs, run = [], []
    for name in dipoles:
        if run and order[name] - order[run[-1]] > GROUP_GAP:
            runs.append(run)
            run = []
        run.append(name)
    runs.append(run)

    out = []
    for run in runs:
        if len(run) < GROUP_MIN:
            continue
        angles = [angle[n] for n in run]
        biggest = max(abs(a) for a in angles) or 1.0
        total = sum(angles)
        out.append({
            "members": run, "angles": angles,
            "signs": [1.0 if a >= 0 else -1.0 for a in angles],
            "sum": total, "closed": abs(total) <= CLOSED_BUMP * biggest,
            "powered": biggest > 1e-9, "name": common_prefix(run),
        })
    return out


def name_stem(name: str) -> str:
    """A name with its trailing index removed: `..._sext_01` -> `..._sext`."""
    return re.sub(r"[_\-]?\d+$", "", name)


def strength_groups(sequence, types, raw, skip: set[str]) -> list[dict]:
    """Same-type elements sharing a name stem and a strength — likely one supply.

    Proximity is the wrong signal for everything except dipole strings. CLARA's
    two FEA sextupoles sit fifty-one elements apart and are plainly a pair; eight
    quadrupoles in one section sit on top of each other and are plainly not.

    What does carry signal is the naming — accelerators are named by function, so
    `..._sext_01` and `..._sext_02` are the same kind of magnet doing the same job
    — together with the strengths agreeing. Magnets on one supply run at the same
    current, so equal (or exactly mirrored) strengths is the tell, and differing
    strengths rule a group out. Both at zero is uninformative rather than
    negative, so those are proposed too, for the user to confirm.
    """
    by_stem: dict[tuple[str, str], list[str]] = {}
    for name in sequence:
        kind = types.get(name)
        if kind is None or name in skip or kind == "Dipole":
            continue
        props = SETTABLE.get(kind, ())
        if not props or props[0] in STEERING_PROPERTIES:
            continue  # steerers are independent by definition; see machine.py
        by_stem.setdefault((kind, name_stem(name)), []).append(name)

    out = []
    for (kind, stem), members in by_stem.items():
        if not (2 <= len(members) <= MAX_GROUP):
            continue
        prop = SETTABLE[kind][0]
        values = [float(raw["elements"][n][1].get(prop, 0.0) or 0.0) for n in members]
        sizes = [abs(v) for v in values]
        biggest = max(sizes)
        if biggest > 0 and max(sizes) - min(sizes) > SAME_STRENGTH * biggest:
            continue  # different strengths: not one supply
        out.append({
            "name": stem, "kind": kind, "prop": prop, "members": members,
            "values": values, "all_zero": biggest == 0,
            "signs": [1.0 if v >= 0 else -1.0 for v in values] if biggest else
                     [1.0] * len(members),
        })
    return out


# ---------------------------------------------------------------- asking


def ask_readout(ask: Interview, diagnostics, sequence) -> str:
    ask.section("The readout")
    screens = diagnostics["screen"]
    guess = screens[-1] if screens else (sequence[-1] if sequence else "")
    ask.say(f"{len(screens)} screen(s) found; the last in the stretch is", f"  {guess}", "")
    return ask.text(
        "readout", "Which screen is the request about?", default=guess,
        help="The one the tune is scored on. Everything past it is in its shadow,\n"
             "so it is usually the last.",
        allow_empty=False,
    )


def ask_dipoles(ask: Interview, groups, all_dipoles) -> tuple[list[Knob], list[str]]:
    """For each dipole string: a knob, or the geometry of the beamline."""
    knobs: list[Knob] = []
    fixed: list[str] = []
    grouped = {n for g in groups for n in g["members"]}
    fixed += [n for n in all_dipoles if n not in grouped]
    if not groups:
        return knobs, fixed

    ask.section("Dipole strings")
    ask.say(
        "A dipole can be a knob or it can be the shape of the beamline, and the",
        "lattice cannot tell you which: a compressor and a dogleg both close their",
        "own bump and look identical here.",
    )
    for group in groups:
        members = group["members"]
        shape = ("closes its own bump" if group["closed"]
                 else f"net bend {group['sum']:+.4f} rad")
        ask.say(
            "",
            f"{group['name']}: {len(members)} dipoles, {members[0]} ... {members[-1]}",
            f"  {'powered' if group['powered'] else 'at zero'}, {shape}, "
            f"signs {[int(s) for s in group['signs']]}",
        )
        role = ask.choose(
            f"dipoles.{group['name']}.role",
            "Is this string a knob, or fixed geometry?",
            ["fixed", "knob"], default="fixed",
            help="knob:  powered as a tuning handle (a bunch compressor)\n"
                 "fixed: it decides where the beam goes (a dogleg, an arc)",
        )
        if role == "fixed":
            fixed.extend(members)
            continue
        name = ask.text(f"dipoles.{group['name']}.name", "Name for this knob",
                        default=group["name"] + "_angle", allow_empty=False)
        faces = ask.confirm(
            f"dipoles.{group['name']}.faces",
            "Do the pole-face angles follow the body angle?", default=True,
            help="On most chicanes the inner faces turn with the bend. Setting the\n"
                 "angle without the face gives the wrong edge focusing, which the\n"
                 "vertical plane sees.",
        )
        biggest = max(abs(a) for a in group["angles"])
        limit = ask.number(
            f"dipoles.{group['name']}.limit",
            "Largest |angle| it may be set to, in rad",
            default=round(biggest * 1.05, 4) if biggest else 0.1,
        )
        follow: dict[str, list[str]] = {}
        if faces:
            # The faces on the inside of the bump turn with the bend; the outer
            # ones stay square to the beamline. Along the string that alternates:
            # the first magnet's exit, the second's entry, the third's exit, and
            # so on, each magnet presenting one angled face to the detour.
            #
            # Giving the middle magnets *both* faces, which looks like the
            # symmetric choice, is wrong and expensive: it doubles their edge
            # focusing and throws CLARA's beam into the wall as soon as the
            # chicane is moved at all. The error is invisible at the nominal
            # setting, because nothing is written until the knob is turned.
            for index, element in enumerate(members):
                follow[element] = ["dipole_e2" if index % 2 == 0 else "dipole_e1"]
        knobs.append(Knob(
            name=name, prop="angle", elements=members, signs=group["signs"],
            follow=follow, low=-abs(limit), high=abs(limit), nonlinear=True,
            note="declared a knob in the interview; "
                 + ("inner pole faces follow the body angle, alternating "
                    "exit/entry along the string — check this against the magnets"
                    if faces else "faces fixed"),
        ))
    return knobs, fixed


def ask_groups(ask: Interview, knobs: list[Knob], groups: list[dict]) -> list[Knob]:
    """Fold magnets that share a supply into single knobs.

    Two magnets on one power supply are one number, and treating them as two is
    both wrong and expensive: the optimiser spends a dimension discovering a
    constraint the hardware already enforces, and can ask for a pair of settings
    no supply can produce.

    Candidates are proposed from the naming and the strengths; after those, any
    grouping at all can be entered by hand, because a shared supply is a fact
    about cabling and no file records it.
    """
    ask.section("Shared supplies")
    ask.say(
        "Magnets wired to one supply are one knob. Nothing in the lattice says",
        "which are, so these are proposals from the names and the strengths.",
    )
    by_name = {k.name: k for k in knobs}

    def fold(name: str, members: list[str], prop: str, signs: list[float],
             low: float, high: float, nonlinear: bool, note: str) -> None:
        for element in members:
            by_name.pop(plane_name(element, prop), None)
        by_name[name] = Knob(
            name=name, prop=prop, elements=members, signs=signs,
            low=low, high=high, nonlinear=nonlinear, note=note,
        )

    for group in groups:
        members, prop = group["members"], group["prop"]
        ask.say(
            "",
            f"{group['name']}: {len(members)} {group['kind']}s sharing a name stem",
            f"  {', '.join(members)}",
            f"  {prop} = {group['values']}"
            + ("  (all zero, so the strengths say nothing either way)"
               if group["all_zero"] else "  (equal, which is what one supply gives)"),
        )
        if ask.choose(
            f"group.{group['name']}.role", "One supply, or independent?",
            ["independent", "grouped"], default="independent",
            help="grouped: one knob drives them all\n"
                 "independent: a knob each, as the lattice has them",
        ) != "grouped":
            continue
        name = ask.text(f"group.{group['name']}.name", "Name for the knob",
                        default=group["name"], allow_empty=False)
        pattern = ask.choose(
            f"group.{group['name']}.signs", "How are they wired?",
            ["same", "alternating"], default="same",
            help="same: all take the knob value\n"
                 "alternating: +, -, +, - along the string",
        )
        signs = ([1.0] * len(members) if pattern == "same"
                 else [(-1.0) ** i for i in range(len(members))])
        biggest = max(abs(v) for v in group["values"]) or 0.0
        limit = ask.number(
            f"group.{group['name']}.limit", f"Largest |{prop}| it may be set to",
            default=round(biggest * 1.05, 6) if biggest else 10.0,
        )
        fold(name, members, prop, signs, -abs(limit), abs(limit),
             prop in NONLINEAR_PROPERTIES,
             f"grouped in the interview: {pattern} signs on one supply")

    while ask.confirm(
        f"group.extra.{len([k for k in by_name])}.add",
        "Add another group by hand?", default=False,
        help="For supplies the proposals missed. Give the element names.",
    ):
        index = sum(1 for k in by_name.values() if k.is_group)
        raw_members = ask.text(f"group.extra.{index}.members",
                               "Element names, comma separated", allow_empty=False)
        members = [m.strip() for m in raw_members.split(",") if m.strip()]
        prop = ask.text(f"group.extra.{index}.prop", "Which property?",
                        default="k1", allow_empty=False)
        name = ask.text(f"group.extra.{index}.name", "Name for the knob",
                        default=name_stem(members[0]), allow_empty=False)
        raw_signs = ask.text(f"group.extra.{index}.signs",
                             "Signs, comma separated (blank for all +1)", default="")
        signs = ([float(v) for v in raw_signs.split(",")] if raw_signs
                 else [1.0] * len(members))
        limit = ask.number(f"group.extra.{index}.limit",
                           f"Largest |{prop}| it may be set to", default=10.0)
        fold(name, members, prop, signs, -abs(limit), abs(limit),
             prop in NONLINEAR_PROPERTIES, "added by hand in the interview")

    return list(by_name.values())


def _grouped_limits(env, knobs: list[Knob]) -> dict[str, tuple[float, float]]:
    """Limits for knobs the source only knows under another name.

    A grouped knob is named here and named differently there — this interview
    calls the chicane `vbc_chicane` and SimEnv calls it `cla_vbc_chicane` — so a
    lookup by name silently misses it and the knob keeps whatever the interview
    guessed. On CLARA that guess was +-0.135 rad against a real +-0.1317, and the
    precompute happily built optics at angles the machine cannot reach.

    Matched on the element set instead, which is what a knob actually is, and the
    limits are flipped if the two disagree about which way is positive. They do:
    the chicane runs at +0.1317 here and -0.1317 there.
    """
    by_elements = {}
    for name in env.magnet_names:
        knob = env._knobs[name]
        members = ((knob.element_name, 1.0), *knob.coupled)
        by_elements[frozenset(e for e, _ in members)] = (name, dict(members))
    out: dict[str, tuple[float, float]] = {}
    for knob in knobs:
        if knob.name in env.magnet_limits:
            continue
        found = by_elements.get(frozenset(knob.elements))
        if found is None:
            continue
        name, signs = found
        import numpy as np

        theirs = np.array([signs[e] for e in knob.elements])
        flip = float(np.sign(np.dot(theirs, np.array(knob.signs))) or 1.0)
        low, high = env.magnet_limits[name]
        out[knob.name] = tuple(sorted((low * flip, high * flip)))
    return out


def ask_limits(ask: Interview, knobs: list[Knob]) -> list[str]:
    """Where the power supply ranges come from."""
    ask.section("Magnet limits")
    ask.say(
        "The lattice says what the magnets are set to, not what they can reach.",
        "Unbounded, the match will ask for strengths that do not exist.",
    )
    source = ask.choose(
        "limits.source", "Where should the limits come from?",
        ["none", "file", "simenv"], default="none",
        help="file:   a JSON of {knob: [low, high]}\n"
             "simenv: this challenge's RL_framework table (CLARA only)\n"
             "none:   leave unbounded and decide later",
    )
    unresolved: list[str] = []
    table: dict = {}
    if source == "file":
        path = ask.text("limits.path", "Path to the limits JSON", default="",
                        allow_empty=False)
        table = json.loads(Path(path).read_text())
    elif source == "simenv":
        try:
            # This challenge's environment lives two directories up and is not on
            # the path when `describe.py` is run in isolated mode.
            here = Path(__file__).resolve().parent
            for extra in (here.parent, here.parent / "RL_framework"):
                if str(extra) not in sys.path:
                    sys.path.insert(0, str(extra))
            from rl_framework.config import RLConfig
            from rl_framework.envs.sim_env import SimEnv

            config = RLConfig.clara()
            config.num_particles = 100  # nothing is tracked; this is for the table
            env = SimEnv(config)
            table = dict(env.magnet_limits)
            table.update(_grouped_limits(env, knobs))
        except Exception as error:  # noqa: BLE001 — the user chose this source
            unresolved.append(f"could not read the SimEnv limits: {error!r}")
    if source == "none":
        unresolved.append(
            "no magnet limits: every knob is unbounded. Set low/high before "
            "precomputing, or the match will reach for strengths nothing delivers."
        )
        return unresolved
    for knob in knobs:
        if knob.name in table:
            knob.low, knob.high = (float(v) for v in table[knob.name])
    missing = [k.name for k in knobs if k.low is None]
    if missing:
        unresolved.append(
            f"{len(missing)} knob(s) have no limit in that source: {missing[:5]}"
        )
    return unresolved


def ask_points(ask: Interview, index: int) -> int:
    return int(ask.number(
        f"axes.{index}.points", "Points along it", default=5, cast=int,
        bounds=(2, 51),
        help="Every point multiplies the precompute. 5 is plenty for a smooth\n"
             "dependence; more only if the optics turns sharply.",
    ))


def axes_by_hand(ask: Interview, candidates: list[str]) -> list[ScanAxis]:
    """Axes named outright, for when the scan cannot run or is not trusted."""
    count = int(ask.number("axes.count", "How many extra axes?", default=0,
                           cast=int, bounds=(0, MAX_AXES)))
    axes: list[ScanAxis] = []
    for index in range(count):
        ask.say("", f"--- axis {index + 1} of {count}")
        knob = ask.text(f"axes.{index}.knob", "Which knob does it vary?",
                        default=candidates[0] if candidates else "", allow_empty=False)
        low = ask.number(f"axes.{index}.low", "Lowest value", default=0.0)
        high = ask.number(f"axes.{index}.high", "Highest value", default=1.0)
        points = ask_points(ask, index)
        note = ask.text(f"axes.{index}.note", "Why does it change the optics?",
                        default="")
        axes.append(ScanAxis(name=knob, kind="knob", knob=knob, low=low, high=high,
                             points=points, spacing="linear", note=note))
    return axes


def scan_ranking(spec: MachineSpec, particles: int, points: int):
    """Sweep every candidate knob and rank it by how far it moves the spot.

    Imported here rather than at the top: the scan pulls in torch and Cheetah and
    tracks a beam, and an interview that only names its axes should not pay for it.
    """
    import sensitivity
    from lattice import Line

    line = Line(spec, num_particles=particles)
    candidates, matching, steering = sensitivity.candidates_for(spec)
    print(f"\n  {len(matching)} matching quadrupole(s) and {len(steering)} "
          f"corrector(s) are not candidates: the first are what the family solves\n"
          f"  for, the second are a separable orbit with their own stage.")
    effects = sensitivity.measure(line, candidates, points)
    sensitivity.report(effects, spec)
    return sensitivity.rank(effects), sensitivity.AXIS_THRESHOLD


def ask_axes(ask: Interview, spec: MachineSpec, particles: int = 2000,
             points: int = 5) -> list[ScanAxis]:
    """What else the optics depends on, beyond the requested spot.

    The ranking is a measurement and the cut is a judgement, so the script makes
    the first and the user makes the second. `sensitivity.py` sweeps every
    candidate knob with the quadrupoles held still and orders them by how far the
    readout spot moves — which is exactly how far the family would have been wrong
    had it ignored that knob. What it cannot decide is how many of them are worth
    paying for, because each one multiplies a precompute measured in hours and the
    ranking usually falls away sharply after the leader. So the user picks *k*.
    """
    ask.section("What the family must span")
    ask.say(
        "The family answers 'which quadrupole strengths give this spot?'. That has",
        "one answer only if everything else about the beam is pinned.",
        "",
        "If some other knob changes the beam enough to change the answer — an RF",
        "phase that sets the energy spread, a compressor angle that sets the bunch",
        "length — the family has to span it too, or it is right at one value of",
        "that knob and wrong at every other.",
        "",
        "Each axis multiplies the precompute, so the question is not which knobs",
        "matter at all, but which few matter most.",
    )
    candidates = [k.name for k in spec.knobs if k.nonlinear]
    if candidates:
        ask.say("", f"non-linear knobs, the usual suspects: {candidates}")
    source = ask.choose(
        "axes.source", "How should the axes be chosen?",
        ["scan", "manual", "none"], default="scan",
        help="scan:   sweep every candidate knob, rank them, take the top k\n"
             "manual: name them yourself\n"
             "none:   span the requested spot and nothing else",
    )
    if source == "none":
        return []
    if source == "manual":
        return axes_by_hand(ask, candidates)

    try:
        ranked, threshold = scan_ranking(spec, particles, points)
    except Exception as error:  # noqa: BLE001 — the scan is an aid, not a gate
        ask.say("", f"the scan could not run ({error!r}) — naming them by hand")
        spec.unresolved.append(f"the sensitivity scan did not run: {error!r}")
        return axes_by_hand(ask, candidates)

    usable = [e for e in ranked if e.usable]
    if not usable:
        ask.say("", "No candidate knob both holds the beam and moves the spot, so a",
                "family over the requested spot alone is enough.")
        return []

    strong = sum(1 for e in usable if e.proposes_axis)
    ask.say(
        "",
        f"{strong} of {len(usable)} knob(s) move the spot by more than "
        f"{threshold:.0%}.",
        "The ranking normally falls away sharply after the leader, so 1 is the",
        "usual first pass; take more only if the second is close behind the first.",
    )
    top_k = int(ask.number(
        "axes.top_k", "How many of the top-ranked knobs become axes?",
        default=1, cast=int, bounds=(0, min(MAX_AXES, len(usable))),
        help="\n".join(
            f"{i + 1}: {e.knob}  moves the spot {e.size_change:.1%} over "
            f"[{e.low:g}, {e.high:g}]"
            for i, e in enumerate(usable[:MAX_AXES])),
    ))

    axes: list[ScanAxis] = []
    for index, effect in enumerate(usable[:top_k]):
        ask.say("", f"--- axis {index + 1}: {effect.knob}")
        ask.say(
            f"    swept [{effect.low:g}, {effect.high:g}]"
            + ("  (narrowed from its limits to where the beam survives)"
               if effect.narrowed else "")
            + f", moving the spot by {effect.size_change:.1%}"
        )
        low = ask.number(f"axes.{index}.low", "Lowest value", default=effect.low)
        high = ask.number(f"axes.{index}.high", "Highest value", default=effect.high)
        axes.append(ScanAxis(
            name=effect.knob, kind="knob", knob=effect.knob,
            low=low, high=high, points=ask_points(ask, index), spacing="linear",
            note=f"ranked {index + 1} of {len(usable)} by the sensitivity scan: "
                 f"sweeping it over [{effect.low:g}, {effect.high:g}] with the "
                 f"quadrupoles held still moves the readout spot by "
                 f"{effect.size_change:.1%}, so the matched family depends on it",
        ))
    return axes


def ask_matching(ask: Interview, spec: MachineSpec, sequence: list[str]) -> None:
    """Which quadrupoles the family varies to hit a requested spot.

    These are what the precompute solves *for*, so the count is a real choice and
    not a detail. Two constraints are being met — a horizontal and a vertical size
    — and more quadrupoles than that leaves the solution a surface rather than a
    point, which is fine and even useful (the extra freedom buys transmission and
    keeps the strengths reachable) as long as the solver's gauge is pinned. Far
    more than that and the match wanders a long way from the reference tune for no
    gain, each node taking longer and agreeing less with its neighbours.

    The default is the last few quadrupoles before the readout, which is the final
    focus on any machine shaped like a beamline. On CLARA it is the same four
    `MFopt` matches.
    """
    ask.section("The matching quadrupoles")
    order = {name: i for i, name in enumerate(sequence)}
    position = lambda knob: max(order.get(e, -1) for e in knob.elements)  # noqa: E731
    quads = sorted((k for k in spec.knobs if k.prop == "k1"), key=position)
    readout_at = order.get(spec.readout, len(sequence))
    upstream = [k for k in quads if position(k) < readout_at]
    ask.say(
        f"{len(quads)} quadrupole knob(s), {len(upstream)} of them before the readout.",
        "The family varies these to hit the requested spot; everything else is held.",
        "",
        "The last few before the readout are its final focus. More than a handful",
        "and the match drifts far from the reference tune for little gain; fewer",
        "than two and there is nothing to trade the two planes against.",
    )
    if not upstream:
        spec.unresolved.append(
            "no quadrupole knob upstream of the readout: nothing can match a spot"
        )
        return
    count = int(ask.number(
        "matching.count", "How many of the last quadrupoles should match?",
        default=min(DEFAULT_MATCHING, len(upstream)), cast=int,
        bounds=(1, len(upstream)),
        help="the last, in order: "
             + ", ".join(k.name for k in upstream[-DEFAULT_MATCHING * 2:]),
    ))
    spec.matching_quads = [k.name for k in upstream[-count:]]
    ask.say("", f"matching: {spec.matching_quads}")


def ask_grid(ask: Interview, spec: MachineSpec, axes: list[ScanAxis]) -> None:
    ask.section("The spot-size grid")
    spec.spot_low = ask.number("grid.low", "Smallest rms spot to support, metres",
                               default=1e-4)
    spec.spot_high = ask.number("grid.high", "Largest rms spot, metres", default=2e-2)
    spec.spot_points = int(ask.number(
        "grid.points", "Points per plane (the grid is this squared)", default=49,
        cast=int, bounds=(5, 129),
        help="The family is interpolated between these, so this sets how smooth\n"
             "the lookup is. 49 gives the 2401-point grid optics_matching uses.",
    ))
    spec.axes = axes
    ask.say("", f"grid {spec.grid_shape()} = {spec.grid_size():,} matched tunes")
    if spec.grid_size() > 20000:
        ask.say("  that is a large precompute — consider fewer points or one less axis")


# ---------------------------------------------------------------- driving


def run(path: Path, ask: Interview, start=None, end=None, beam=None,
        apertures=None, pipe_radius=None, scan_particles=2000,
        scan_points=5) -> MachineSpec:
    types, sequence, raw = load_lattice(path)
    if start or end:
        lo = sequence.index(start) + 1 if start else 0
        hi = sequence.index(end) + 1 if end else len(sequence)
        sequence = sequence[lo:hi]
        types = {n: t for n, t in types.items() if n in set(sequence)}

    diagnostics = classify(sequence, types)
    ask.section(path.name)
    ask.say(f"{len(sequence)} elements in the modelled stretch")
    for kind, count in sorted(Counter(types.values()).items(), key=lambda kv: -kv[1]):
        ask.say(f"    {count:>4}  {kind}")
    ask.say(
        f"diagnostics: {len(diagnostics['bpm'])} BPMs, "
        f"{len(diagnostics['screen'])} screens, "
        f"{len(diagnostics['aperture'])} apertures, "
        f"{len(diagnostics['untagged'])} untagged"
    )

    readout = ask_readout(ask, diagnostics, sequence)
    knobs = default_knobs(types, sequence)
    dipole_knobs, fixed = ask_dipoles(
        ask, dipole_groups(sequence, types, raw),
        [n for n in sequence if types.get(n) == "Dipole"],
    )
    knobs += dipole_knobs
    knobs = ask_groups(
        ask, knobs, strength_groups(sequence, types, raw, set(fixed))
    )

    dead = [
        k.name for k in knobs
        if k.prop == "phase"
        and not float(raw["elements"][k.elements[0]][1].get("voltage", 0) or 0)
    ]
    if dead:
        ask.section("Unpowered cavities")
        ask.say(f"{len(dead)} cavity/cavities sit at zero volts: {dead[:4]}")
        if ask.confirm("cavities.drop_dead", "Drop their phases as knobs?", True,
                       help="A cavity at zero volts does nothing with its phase."):
            knobs = [k for k in knobs if k.name not in dead]

    unresolved = ask_limits(ask, knobs)

    # The spec is built before the axes are chosen, because choosing them by
    # measurement means tracking a beam through this very machine.
    spec = MachineSpec(
        lattice=str(path), beam=str(beam) if beam else None,
        apertures=str(apertures) if apertures else None,
        start=start, end=end, readout=readout,
        bpms=diagnostics["bpm"], screens=diagnostics["screen"],
        pipe_radius=pipe_radius, knobs=knobs, fixed=fixed,
        name=path.stem, unresolved=unresolved,
    )
    ask_matching(ask, spec, sequence)
    axes = ask_axes(ask, spec, scan_particles, scan_points)
    ask_grid(ask, spec, axes)
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lattice", type=Path)
    parser.add_argument("--out", type=Path, default=Path("machine.json"))
    parser.add_argument("--answers", type=Path, default=None,
                        help="replay a previous session instead of asking")
    parser.add_argument("--record", type=Path, default=None)
    parser.add_argument("--defaults", action="store_true",
                        help="ask nothing; take every default")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--beam", type=Path, default=None)
    parser.add_argument("--apertures", type=Path, default=None)
    parser.add_argument("--pipe-radius", type=float, default=None)
    parser.add_argument("--scan-particles", type=int, default=2000,
                        help="particles tracked by the sensitivity scan")
    parser.add_argument("--scan-points", type=int, default=5,
                        help="values sampled per knob by the sensitivity scan")
    args = parser.parse_args()

    ask = Interview.replay(args.answers, defaults_only=args.defaults)
    spec = run(args.lattice, ask, args.start, args.end,
               args.beam, args.apertures, args.pipe_radius,
               args.scan_particles, args.scan_points)
    spec.save(args.out)
    record = args.record or args.out.with_name(args.out.stem + "_answers.json")
    ask.save(record)

    types, _, _ = load_lattice(args.lattice)
    problems = spec.validate(types, allow_draft=True)

    ask.section("Result")
    ask.say(
        f"{len(spec.knobs)} knobs ({len(spec.linear_knobs())} linear, "
        f"{len(spec.nonlinear_knobs())} non-linear), {len(spec.fixed)} fixed",
        f"grid {spec.grid_shape()} = {spec.grid_size():,} matched tunes",
        f"spec -> {args.out}",
        f"answers -> {record}   ({ask.asked} question(s) asked)",
    )
    for label, items in (("problem", problems), ("unresolved", spec.unresolved)):
        if items:
            ask.say("", f"{len(items)} {label}(s):")
            for item in items:
                ask.say(f"    - {item}")
    ask.say("", "Next: precompute.py" if not (problems or spec.unresolved)
            else "Fix the above, then run precompute.py.")


if __name__ == "__main__":
    main()
