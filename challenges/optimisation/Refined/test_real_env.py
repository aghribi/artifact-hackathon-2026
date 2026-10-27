"""`RealEnv` against a fake control system. Nothing here touches CLARA.

    python -I test_real_env.py

Every placeholder in `RL_framework/rl_framework/envs/hardware.py` is replaced by
a stub that makes up a plausible answer and records that it was called. That is
enough to check the thing the real machine cannot be used to check: that the
environment and the tuners agree about the interface, that a coupled knob
reaches all four of its magnets with the right signs, that a screen switch
moves exactly one mechanism, and that nothing is quietly left unimplemented.

The *scores* here are meaningless — the fake's beam is a Gaussian that moves
where the correctors tell it and nothing else. What is being tested is that
every call the pipeline makes is implemented, and that the unimplemented
placeholders fail by name rather than deep inside something else.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (HERE, ROOT, ROOT / "RL_framework"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from machine import MachineSpec  # noqa: E402

PASSED, FAILED = 0, 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  PASS  {label}" + (f"  — {detail}" if detail else ""))
    else:
        FAILED += 1
        print(f"  FAIL  {label}" + (f"  — {detail}" if detail else ""))


# ---------------------------------------------------------------- the fake


class FakeMachine:
    """A control system that answers plausibly and remembers being asked.

    Deliberately not a simulation: the spot responds to the correctors through
    a made-up gain so that an optimiser has *something* to push against, and
    everything else is constant. A real model of the machine already exists in
    `SimEnv` and using it here would test the fake rather than the plumbing.
    """

    CHIP = (-11e-3, 11e-3, -11e-3, 11e-3)  # a 22 mm chip, as CLARA's are
    PIXELS = 160  # not 128, so the rebinning onto the observation grid is used

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.strengths: dict[str, float] = {}
        self.inserted: str | None = None
        self.log: list[tuple[str, str]] = []
        # A write does not show up in the readback until it has been waited
        # for. This is not invented: a caput to the real CLARA takes about half
        # a second to come back round through EPICS, and reading in that window
        # returns the *previous* setting. `Refined` takes its next step from
        # the settings it read back, so anything that reads before settling
        # steps from the wrong place — modelled here so it cannot regress.
        self._in_flight: dict[str, float] = {}

    # -- magnets

    def SET_REAL_MAGNET_POWER(self, magnet_name: str, value: float) -> None:
        self.calls["SET_REAL_MAGNET_POWER"] += 1
        self._in_flight[magnet_name] = float(value)
        self.log.append(("set", magnet_name))

    def GET_REAL_MAGNET_POWER(self, magnet_name: str) -> float:
        self.calls["GET_REAL_MAGNET_POWER"] += 1
        return self.strengths.get(magnet_name, 0.0)

    def WAIT_FOR_MAGNETS_SETTLED(self, magnet_names) -> None:
        self.calls["WAIT_FOR_MAGNETS_SETTLED"] += 1
        for name in magnet_names:
            if name in self._in_flight:
                self.strengths[name] = self._in_flight.pop(name)

    def demanded(self, magnet_name: str) -> float:
        """What was last asked of a magnet, whether or not it has landed."""
        if magnet_name in self._in_flight:
            return self._in_flight[magnet_name]
        return self.strengths[magnet_name]

    # -- screens

    def INSERT_REAL_SCREEN(self, screen_name: str) -> None:
        self.calls["INSERT_REAL_SCREEN"] += 1
        assert self.inserted is None, (
            f"{screen_name} inserted while {self.inserted} is still in the beam"
        )
        self.inserted = screen_name
        self.log.append(("insert", screen_name))

    def RETRACT_REAL_SCREEN(self, screen_name: str) -> None:
        self.calls["RETRACT_REAL_SCREEN"] += 1
        self.inserted = None
        self.log.append(("retract", screen_name))

    def WAIT_FOR_SCREEN_SETTLED(self, screen_name: str) -> None:
        self.calls["WAIT_FOR_SCREEN_SETTLED"] += 1

    def GET_REAL_SCREEN_EXTENT(self, screen_name: str):
        self.calls["GET_REAL_SCREEN_EXTENT"] += 1
        return self.CHIP

    def GET_REAL_SCREEN_IMAGE(self, screen_name: str) -> np.ndarray:
        self.calls["GET_REAL_SCREEN_IMAGE"] += 1
        # A spot that the correctors can push around, so a steering solve has a
        # response to find. Everything else about it is invented.
        kick = sum(v for n, v in self.strengths.items() if "cor" in n)
        x_min, x_max, y_min, y_max = self.CHIP
        axis = np.linspace(x_min, x_max, self.PIXELS)
        x, y = np.meshgrid(axis, axis)
        mu, sigma = 300e-6 + 50 * kick, 250e-6
        return np.exp(-((x - mu) ** 2 + (y - mu) ** 2) / (2 * sigma**2))

    # -- the rest

    def GET_REAL_BPM_READING(self, bpm_name: str):
        self.calls["GET_REAL_BPM_READING"] += 1
        return 0.0, 0.0, True

    def GET_REAL_TRANSMISSION(self) -> float:
        self.calls["GET_REAL_TRANSMISSION"] += 1
        return 0.95


PLACEHOLDERS = [
    "SET_REAL_MAGNET_POWER", "GET_REAL_MAGNET_POWER", "WAIT_FOR_MAGNETS_SETTLED",
    "INSERT_REAL_SCREEN", "RETRACT_REAL_SCREEN", "WAIT_FOR_SCREEN_SETTLED",
    "GET_REAL_SCREEN_IMAGE", "GET_REAL_SCREEN_EXTENT",
    "GET_REAL_BPM_READING", "GET_REAL_TRANSMISSION",
]

TARGET = (0.0, 0.0, 266e-6, 150e-6)


def build(spec_name: str = "clara_febe_no_bpms.json"):
    from rl_framework.config import RLConfig
    from rl_framework.envs.real_env import RealEnv

    spec = MachineSpec.load(HERE / "machines" / spec_name)
    fake = FakeMachine()
    env = RealEnv(RLConfig(spec=spec, target=TARGET), hardware=fake)
    return spec, fake, env


# ---------------------------------------------------------------- the checks


def test_builds() -> None:
    print("\n1. the environment builds and reads with no control system")
    spec, fake, env = build()
    check("it constructs against a real spec",
          len(env.segment.elements) > 0,
          f"{len(env.segment.elements)} elements, {len(env.screen_names)} screens, "
          f"{len(env.bpm_names)} BPMs")
    check("the extent is the camera's, not the model's",
          env.screen_extent(env.scored_screen) == FakeMachine.CHIP,
          f"{tuple(round(v, 4) for v in env.screen_extent())}")

    observation, info = env.reset()
    check("the observation matches its own space",
          env.observation_space.contains(observation))
    check("a frame of a different resolution is rebinned onto the grid",
          observation["screen_image"].shape == tuple(env.config.screen_shape),
          f"{FakeMachine.PIXELS}x{FakeMachine.PIXELS} -> "
          f"{observation['screen_image'].shape}")

    # Every key `Refined.machines.EnvMachine._record` reads out of `info`.
    needed = ["mu_x", "mu_y", "sigma_x", "sigma_y", "on_screen", "transmission",
              "benchmark_score", "screen"]
    missing = [k for k in needed if k not in info]
    check("info carries everything the tuners read", not missing,
          f"{len(needed)} keys" if not missing else f"missing {missing}")
    check("the spot is measured in metres off the chip centre",
          0 < info["sigma_x"] < 2e-3 and info["on_screen"],
          f"sigma {info['sigma_x'] * 1e6:.0f} x {info['sigma_y'] * 1e6:.0f} um")
    check("transmission is the machine's, and it is charged for",
          abs(info["transmission"] - 0.95) < 1e-9,
          f"T {info['transmission']:.2f}")

    # Calibrations are fetched for the screens actually read, not for all 22.
    # On a real machine each one is a round trip to a camera, and asking about
    # a screen nobody looks at is a conversation with hardware for nothing.
    asked = fake.calls["GET_REAL_SCREEN_EXTENT"]
    check("only the screens actually read are calibrated",
          0 < asked <= 2 < len(env.screen_names),
          f"{asked} of {len(env.screen_names)} screens asked")
    env.screen_extent(env.scored_screen)
    check("and a chip is not re-measured once known",
          fake.calls["GET_REAL_SCREEN_EXTENT"] == asked,
          "the extent is cached after the first read")


def test_reset_is_read_only() -> None:
    print("\n2. reset() does not move the machine")
    spec, fake, env = build()
    before = fake.calls["SET_REAL_MAGNET_POWER"]
    env.reset()
    check("no magnet is written by a reset",
          fake.calls["SET_REAL_MAGNET_POWER"] == before == 0,
          f"{fake.calls['SET_REAL_MAGNET_POWER']} writes")

    from rl_framework.config import RLConfig
    from rl_framework.envs.real_env import RealEnv
    try:
        RealEnv(RLConfig(spec=spec), hardware=FakeMachine())
        check("a missing request is refused", False, "it was accepted")
    except ValueError as error:
        check("a missing request is refused", "config.target" in str(error))


def test_coupled_knob() -> None:
    print("\n3. a coupled knob reaches every magnet it drives")
    spec, fake, env = build()
    knob = spec.knob("cla_vbc_chicane")
    fake.log.clear()
    env.set_magnets({"cla_vbc_chicane": 0.10})
    written = [name for what, name in fake.log if what == "set"]
    check("every dipole of the group is written",
          written == knob.elements, f"{len(written)} of {len(knob.elements)}")
    signs = [np.sign(fake.demanded(n)) for n in knob.elements]
    check("with the spec's alternating signs",
          signs == [np.sign(s) for s in knob.signs],
          f"{[round(fake.demanded(n), 4) for n in knob.elements]}")
    check("a pole face is not written as if it had its own supply",
          all("_e1" not in n and "_e2" not in n for n in written),
          "the iron turns the faces, not a power supply")


def test_corrector_planes_are_separate_supplies() -> None:
    print("\n3b. the two planes of a corrector are two magnets")
    spec, fake, env = build()
    # One element in the model, two magnets on the machine. If the plane does
    # not travel with the name, the hardware layer cannot tell them apart and
    # drives one plane with the other's demand — silently, and in a way that
    # looks like a steering error no corrector can fix.
    knob = next(k for k in spec.knobs if k.name.endswith("_h")
                and "hvcor" in k.elements[0])
    element = knob.elements[0]
    fake.log.clear()
    env.set_magnets({knob.name: 1e-5, knob.name[:-2] + "_v": -2e-5})
    written = [name for what, name in fake.log if what == "set"]
    check("each plane is its own supply name",
          written == [f"{element}_h", f"{element}_v"], f"{written}")
    check("and they hold different values",
          fake.demanded(f"{element}_h") != fake.demanded(f"{element}_v"),
          f"h={fake.demanded(f'{element}_h'):.1e}, "
          f"v={fake.demanded(f'{element}_v'):.1e}")

    # And the CLARA driver turns those into the two real magnets.
    from rl_framework.envs import hardware_CLARA as clara
    check("the CLARA driver maps them to HCOR and VCOR",
          (clara._catap_name(f"{element}_h").endswith("HCOR-01")
           and clara._catap_name(f"{element}_v").endswith("VCOR-01")),
          f"{clara._catap_name(f'{element}_h')} / "
          f"{clara._catap_name(f'{element}_v')}")


def test_settles_before_reading_back() -> None:
    print("\n3c. a setting is never read back before it has landed")
    spec, fake, env = build()
    knob = next(k for k in spec.knobs if k.prop == "k1")
    demanded = 1.75
    env.step({"magnets": {knob.name: demanded}})
    # The fake only makes a write visible once it has been waited for, as the
    # control system does. If the step read before settling, the reported
    # setting would be the one before this.
    reported = env._get_info()["magnets"][knob.name]
    check("the reported setting is the one just applied",
          abs(reported - demanded) < 1e-9,
          f"asked {demanded}, machine says {reported}")
    check("and the wait actually happened",
          fake.calls["WAIT_FOR_MAGNETS_SETTLED"] > 0,
          f"{fake.calls['WAIT_FOR_MAGNETS_SETTLED']} wait(s)")
    check("nothing is left in flight",
          not fake._in_flight, f"{len(fake._in_flight)} unlanded write(s)")


def test_clara_driver_offline() -> None:
    """The CLARA driver, as far as it can be checked with no control system."""
    print("\n8. the CLARA driver maps every supply it will be asked for")
    from rl_framework.envs import hardware_CLARA as clara
    from rl_framework.envs.base_env import PROPERTY_SUFFIX

    sys.path.insert(0, str(ROOT / "data" / "clara" / "lattice_data"))
    import utils

    check("importing it connects to nothing", not clara._factories,
          "factories are built on first use, not at import")
    check("the safety switch is on", clara.USE_VIRTUAL is True,
          "USE_VIRTUAL=True — it cannot move a magnet on CLARA")

    cal = {"corrector": utils.CORRECTOR_CAL, "quadrupole": utils.QUAD_CAL,
           "dipole": utils.DIPOLE_CAL, "sextupole": utils.SEXTUPOLE_CAL}
    spec = MachineSpec.load(HERE / "machines" / "clara_febe.json")
    supplies = {e + PROPERTY_SUFFIX[k.prop] for k in spec.knobs for e in k.elements}
    missing, cavities = [], []
    for supply in sorted(supplies):
        kind = clara._kind(supply)
        if kind == "cavity":
            cavities.append(supply)
        elif clara._catap_name(supply) not in cal[kind]:
            missing.append(supply)
    check("every magnet it can be asked to set has a calibration", not missing,
          f"{len(supplies) - len(cavities)} magnets"
          if not missing else f"no calibration for {missing[:4]}")
    check("cavity phases are refused rather than mis-set", bool(cavities),
          f"{len(cavities)} cavity phase(s) raise NotImplementedError")
    for supply in cavities[:1]:
        try:
            clara.SET_REAL_MAGNET_POWER(supply, 0.0)
            check("  and the refusal is explicit", False, "it was accepted")
        except NotImplementedError as error:
            check("  and the refusal is explicit", "cavity" in str(error))

    # The conversions have to be each other's inverse, or every readback is
    # wrong by however much they disagree.
    worst = 0.0
    for name, forward, inverse, values in (
        ("cla_s02_mag_quad_01", utils.quadrupole_current_from_k,
         utils.quadrupole_k_from_current, (0.5, -2.0, 5.0)),
        ("cla_s02_mag_hvcor_01_h", utils.corrector_current_from_angle,
         utils.corrector_angle_from_current, (1e-5, -2e-4)),
        ("cla_vbc_mag_dip_01", utils.dipole_current_from_angle,
         utils.dipole_angle_from_current, (0.1, -0.1)),
    ):
        catap, p = clara._catap_name(name), clara._momentum(name)
        for value in values:
            back = inverse(catap, forward(catap, value, p), p)
            worst = max(worst, abs(back - value) / abs(value))
    check("strength -> current -> strength round-trips", worst < 1e-9,
          f"worst relative error {worst:.1e}")

    # Momentum is per section, not one number: the linacs accelerate, so S02 is
    # 32.8 MeV/c and FEBE is 250. Using a single value would have set every S02
    # magnet about seven times too strong, with no symptom but a mistuned
    # machine.
    unknown = [s for s in supplies
               if clara._kind(s) != "cavity" and s[4:7] not in utils.beam_energy]
    check("every magnet has a reference momentum", not unknown,
          f"sections {sorted({s[4:7] for s in supplies if clara._kind(s) != 'cavity'})}"
          if not unknown else f"no energy for {sorted({s[4:7] for s in unknown})}")
    low = clara._momentum("cla_s02_mag_quad_01")
    high = clara._momentum("cla_feh_mag_quad_01")
    check("and it varies along the machine, as it must",
          high > 5 * low, f"S02 {low} MeV/c against FEH {high} MeV/c")
    # The same table the Cheetah model converts with. If these two ever part
    # company, the tuner plans in one set of units and applies in another.
    check("it is the table the model itself uses",
          clara._momentum("cla_s04_mag_quad_01") == utils.beam_energy["s04"],
          "shared with utils.update_cheetah_element")


def test_rf_is_out_of_reach() -> None:
    print("\n3d. RF phases are not the agent's to set")
    spec, fake, env = build()
    phases = [k.name for k in spec.knobs if k.prop == "phase"]
    check("the spec does declare cavity phases", bool(phases),
          f"{len(phases)}: {phases}")
    check("but the environment does not offer them",
          not [p for p in phases if p in env.magnet_names],
          f"{len(env.magnet_names)} knobs, none of them RF")

    # The real exposure was `restore_best`, which re-sets every knob from a
    # snapshot — so a whole episode is the test, not a single call.
    from artifact import Family
    from machines import RealMachine
    from methods import Session, steer

    fake2 = FakeMachine()
    machine = RealMachine(spec, target=TARGET, hardware=fake2)
    steer(Session(machine, Family.load(HERE / "artifacts" / "clara_febe.npz"),
                  spec, budget=25))
    touched = sorted({n for what, n in fake2.log
                      if what == "set" and "cav" in n})
    check("and a whole episode never writes one", not touched,
          f"{len([1 for w, _ in fake2.log if w == 'set'])} magnet writes, "
          f"0 RF" if not touched else f"wrote {touched}")

    # Asking for one anyway is refused, not quietly dropped.
    from dataclasses import replace as _replace

    from rl_framework.config import RLConfig
    from rl_framework.envs.real_env import RealEnv
    try:
        RealEnv(_replace(RLConfig(spec=spec, target=TARGET),
                         controllable_magnets=[phases[0]]),
                hardware=FakeMachine())
        check("asking for RF explicitly is refused", False, "it was accepted")
    except ValueError as error:
        check("asking for RF explicitly is refused", "RF phases" in str(error))


def test_screen_switching() -> None:
    print("\n4. a screen switch moves one mechanism, and only when it must")
    spec, fake, env = build()
    env.reset()
    inserts = fake.calls["INSERT_REAL_SCREEN"]
    env.step({"magnets": {}})
    check("reading the same screen again moves nothing",
          fake.calls["INSERT_REAL_SCREEN"] == inserts,
          f"{inserts} insert(s) for 2 shots on one screen")

    other = next(s for s in env.screen_names if s != env.scored_screen)
    fake.log.clear()
    env.step({"screen": other})
    moves = [entry for entry in fake.log if entry[0] in ("insert", "retract")]
    check("switching retracts the old one before inserting the new",
          moves == [("retract", env.scored_screen), ("insert", other)],
          f"{moves}")
    check("the score holds while the readout is out of the beam",
          env._get_info()["screen"] == other
          and env._get_info()["scored_screen"] == env.scored_screen)


def test_whole_episode() -> None:
    print("\n5. the Refined pipeline drives it end to end")
    from artifact import Family
    from machines import RealMachine
    from methods import Session, steer

    spec = MachineSpec.load(HERE / "machines" / "clara_febe_no_bpms.json")
    family = Family.load(HERE / "artifacts" / "clara_febe.npz")
    fake = FakeMachine()
    machine = RealMachine(spec, target=TARGET, hardware=fake)
    steer(Session(machine, family, spec, budget=25))

    check("an episode completes", machine.shots > 0,
          f"{machine.shots} shots, score {machine.score:.3f}, "
          f"{machine.screen_switches} switch(es)")
    check("the machine-time accounting uses the real switch cost",
          machine.machine_seconds
          == machine.shots + machine.screen_switches * 30.0,
          f"{machine.machine_seconds:.0f} s")
    missed = [name for name in PLACEHOLDERS if not fake.calls[name]]
    # The BPM reader is the exception: this spec has no working BPMs, which is
    # the whole reason it exists, so nothing should be asking them anything.
    expected_quiet = ["GET_REAL_BPM_READING"]
    check("every placeholder the machine needs was reached",
          missed == expected_quiet,
          f"unused: {missed}")


def test_bpms_when_they_work() -> None:
    print("\n6. and the BPM path lights up on a spec that has them")
    spec, fake, env = build("clara_febe.json")
    env.reset()
    check("every BPM is read", fake.calls["GET_REAL_BPM_READING"] >= len(env.bpm_names),
          f"{len(env.bpm_names)} BPMs read")


def test_unpopulated_fails_loudly() -> None:
    print("\n7. an unpopulated placeholder says which one it is")
    from rl_framework.config import RLConfig
    from rl_framework.envs import hardware
    from rl_framework.envs.real_env import RealEnv

    spec = MachineSpec.load(HERE / "machines" / "clara_febe_no_bpms.json")
    try:
        RealEnv(RLConfig(spec=spec, target=TARGET))  # the real, empty module
        check("it refuses to run against empty placeholders", False,
              "it ran, which means something is not going through hardware.py")
    except NotImplementedError as error:
        named = any(name in str(error) for name in PLACEHOLDERS)
        check("it refuses to run against empty placeholders", named,
              str(error).split(" is a placeholder")[0])

    unimplemented = [
        name for name in PLACEHOLDERS
        if not isinstance(getattr(hardware, name, None), type(check))
    ]
    check("every placeholder exists in hardware.py", not unimplemented,
          f"{len(PLACEHOLDERS)} placeholders" if not unimplemented
          else f"missing {unimplemented}")


def main() -> None:
    print("RealEnv, against a fake control system. CLARA is not touched.")
    for test in (test_builds, test_reset_is_read_only, test_coupled_knob,
                 test_corrector_planes_are_separate_supplies,
                 test_settles_before_reading_back, test_rf_is_out_of_reach,
                 test_screen_switching, test_whole_episode,
                 test_bpms_when_they_work, test_unpopulated_fails_loudly,
                 test_clara_driver_offline):
        test()
    print(f"\n{PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
