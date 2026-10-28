from dataclasses import dataclass, field
from typing import Any


@dataclass
class RLConfig:
    # Which machine. A path to a `MachineSpec` JSON, or a `MachineSpec`. There is
    # no default: an accelerator is not a thing to guess at, and the benchmark's
    # own machine is one line away as `RLConfig.clara()`.
    spec: Any = None

    # `lattice_path`, `beam`, `magnet_config` and the three `nominal_*` fields
    # used to live here. They were the lattice file, a Twiss beam and CLARA's
    # magnet ranges — machine facts, in the configuration, which is what
    # `machines/*.json` exists to abolish. Nothing read them: both environments
    # take all of it from the spec.

    # The screen the request is for: where every episode starts with the screen in
    # the beam, and the only screen an episode is scored on. The agent may move the
    # screen elsewhere to look around, but that costs a shot and brings back no new
    # reading of the request — the score holds at the last reading of this screen,
    # as `score.py` measures only at its readout however many screens a tuner used.
    # None takes the spec's `readout`.
    target_screen: str | None = None

    # Episode length before truncation
    max_episode_steps: int = 100

    # ---------------- the simulation ---------------- #
    # `num_particles` through `power_supply_ripple` describe a tracked beam and
    # are `SimEnv` only; the rest — screens, limits, target, reward — is read by
    # both environments.

    # Particles tracked per step, taken from the front of the 200,000 in the
    # distribution file. None uses all of them (slow).
    num_particles: int | None = 10_000

    # Screen images are binned onto this (rows, columns) grid, covering the whole
    # chip whatever its native resolution is. `env.screen_extent(name)` gives the
    # physical size the grid spans.
    screen_shape: tuple[int, int] = (128, 128)
    # Divide each image by its brightest pixel, so observations are in [0, 1].
    normalise_image: bool = True

    # Beam pipe the particles have to stay inside; a particle that leaves it stops
    # counting towards every reading further down the line. None removes the pipe.
    # None takes the spec's `pipe_radius`.
    beam_pipe_radius: float | None = None
    # Fraction of the launched beam a diagnostic needs for its reading to count.
    # None takes the spec's `min_fraction`.
    min_screen_fraction: float | None = None
    # Gaussian noise added to each BPM reading, in metres.
    bpm_noise: float = 0.0

    # How much the machine varies from one episode to the next. Each is a multiplier
    # on the spreads at the top of `sim_env.py`: 0 is the nominal model exactly, 1 is
    # the machine as it realistically differs shot to shot and day to day, and larger
    # values stay a harder version of the same problem rather than a different one.
    # `reset()` draws a new beam and a new survey error from them; the magnets
    # always start at the strengths the lattice file asks for.
    beam_variation: float = 1.0
    misalignment_variation: float = 1.0

    # What the beam is asked for at the observed screen: (mu_x, mu_y, sigma_x,
    # sigma_y) in metres, off the screen's centre. None draws a new request every
    # episode from the two lines below, as `score.sample_target` does; a 4-tuple
    # fixes it. The reward is the error against it in units of the requested spot.
    target: tuple[float, float, float, float] | None = None
    requested_spot: tuple[float, float] = (80e-6, 400e-6)  # m, log-uniform
    requested_offset: float = 2e-3  # m, uniform about the screen centre

    # How that error becomes a reward. "log" keeps a gradient however far the beam
    # is from the request; "linear" is the benchmark's own score, which stops
    # counting at ten requested spot sizes and so is flat across most of the range
    # an untuned machine sits in. `info["benchmark_score"]` is the benchmark's
    # number either way.
    reward_shape: str = "log"

    # Shot-to-shot jitter, as a fraction of `beam_variation`: within an episode the
    # machine holds still, but every step draws its own beam around the episode's,
    # so two identical actions do not give identical readings. Scales with
    # `beam_variation`, so setting that to 0 leaves the environment deterministic.
    beam_jitter: float = 0.1

    # Power supply ripple: how far a magnet's actual field sits from what it was
    # set to, as a fraction of the setting, drawn fresh for every magnet on every
    # step and undone once the shot has been tracked. 0.001 is 0.1%. At 0, the
    # default, nothing is drawn, written or restored at all. Cavity phases are not
    # rippled — RF stability is a different mechanism and a fraction of a phase
    # means nothing.
    power_supply_ripple: float = 0.0

    # How diagnostics are recognised when the spec does not list them and the
    # element types cannot tell them apart — which is the case whenever a lattice
    # models BPMs, screens and aperture markers all as `Screen`, as CLARA's does.
    diagnostic_tags: dict[str, str] = field(default_factory=lambda: {
        "bpm": "dia_bpm", "screen": "dia_scr", "aperture": "sim_aper",
    })

    # Magnets the agent may set, by knob name as the spec declares it. None gives
    # it every knob the spec declares in the modelled stretch.
    controllable_magnets: list[str] | None = None
    # {knob name: [low, high]} overriding the spec's ranges.
    magnet_limits: dict[str, list[float]] | None = None

    # ---------------- selecting a machine ---------------- #

    def machine_spec(self):
        """The `MachineSpec` this configuration names, loaded if it is a path."""
        from machine_spec import MachineSpec

        if self.spec is None:
            raise ValueError(
                "RLConfig has no machine. Pass `spec=` a MachineSpec or a path to "
                "one, or use `RLConfig.clara()` for the machine the benchmark is "
                "scored on. There is no default because an accelerator is not a "
                "thing to guess at."
            )
        if isinstance(self.spec, MachineSpec):
            return self.spec
        return MachineSpec.load(self.spec)

    @classmethod
    def clara(cls, **overrides) -> "RLConfig":
        """CLARA FEBE, the machine `score.py` is scored on.

        Exactly what `RLConfig()` used to give. The values did not change, they
        moved: into `machines/clara_febe.json`, where they are a description of a
        machine rather than constants in an environment that claims to be general.
        """
        from pathlib import Path as _Path

        here = _Path(__file__).resolve().parents[1]  # Refined/
        # An explicit `spec` in the overrides wins, so callers that rebuild a
        # config from an existing one's `__dict__` do not collide with this.
        return cls(**{"spec": str(here / "machines" / "clara_febe.json"),
                      **overrides})
