"""Animate one MBRL_opt episode: the beam arriving on the requested spot.

    python3 -I make_mbrl_episode_animation.py --scan 1234 1248
    python3 -I make_mbrl_episode_animation.py --seed 1241

Runs a single episode of the `MBRL_opt` pipeline against `SimEnv`, keeping the
camera frame of every shot, and renders them as a GIF with the numbered PNGs
alongside. Nothing is interpolated or re-ordered: one frame is one shot of the
machine, in the order the agent took them.

The left panel is the readout camera, zoomed to a window fixed on the *requested*
position so the spot flies into a stationary target rather than the frame chasing
it. The small inset shows where that window sits on the whole chip, which is 22 mm
across and would otherwise render the spot invisible. The right panel is the score
so far, one point per shot.

Colours are `score.py`'s own: purple for what was asked for, orange for what the
camera saw, so these frames read the same way as the plots `score.plot_target`
makes. The frame itself is a single-hue light-to-dark ramp, which keeps it clear of
both. Validated with the dataviz palette checker: CVD ΔE 35 between the two,
comfortably above the 8 floor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Ellipse, Rectangle  # noqa: E402

CHALLENGE = Path(__file__).resolve().parents[2]
for path in (CHALLENGE, CHALLENGE / "RL_framework", CHALLENGE / "MBRL_opt"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# `MBRL_opt` is imported inside the functions that need it rather than here, so
# that another pipeline can import this module for its drawing code — the camera
# panel, the window logic, the score trace — without dragging in an agent it does
# not use. See `make_refined_episode_animation.py`, which renders the `Refined`
# pipeline into exactly the same frames.

# `score.REQUESTED_COLOUR` and `MEASURED_COLOUR`, so these frames and the
# benchmark's own plots say the same thing with the same colours.
REQUESTED = "#7c3aed"
MEASURED = "#f97316"
ROI = "#1e3a8a"  # `score.ROI_COLOUR`, the window box on the whole-chip inset

SURFACE = "#fcfcfb"
INK = "#1c1917"
MUTED = "#78716c"
GRID = "#e7e5e4"

# One hue, light to dark, for intensity. Not a rainbow and not a second identity:
# the frame is magnitude, and the two things with identity are the ellipses.
BEAM = LinearSegmentedColormap.from_list("beam", [SURFACE, "#a8a29e", INK])

# Smallest half-width of the view, in metres. The actual window is whatever holds
# every spot the episode produced, computed once from all the frames and then held
# fixed: the target must not drift about while the beam flies into it, and the
# opening spot is often millimetres across when the request is tens of microns.
MIN_WINDOW = 0.35e-3
# Roughly how many cells to draw across that window. The environment bins its
# camera onto 512 by 512 over the whole 22 mm chip, so a few thousand particles
# inside a 1 mm window land one to a bin and read as speckle rather than as a
# beam. Summing blocks of them is the same data at a sane display resolution.
DISPLAY_CELLS = 110
# How long each shot is held, and how long the last frame lingers.
FRAME_MS = 900
HOLD_LAST = 6


def record(seed: int, config) -> tuple[list[dict], object]:
    """Run one episode, keeping the camera frame and the numbers of every shot."""
    from rl_framework.config import RLConfig

    from MFopt.optics_env import OpticsEnv

    from agent import MBRLAgent
    from screen import measure
    from twin import CheetahTwin

    rl = RLConfig.clara()
    rl.screen_shape = config.screen_shape
    env = OpticsEnv(rl)
    agent = MBRLAgent(config, CheetahTwin.for_env(env, config.twin_particles))

    frames: list[dict] = []
    extent = env.screen_extent()
    stage = {"name": "start"}

    def note(observation, info):
        reading = measure(observation["screen_image"], extent)
        target = np.asarray(observation["target"], dtype=float)
        frames.append(
            {
                "image": np.asarray(observation["screen_image"], dtype=float),
                "target": target,
                "moments": reading.moments if reading.ok else None,
                "on_screen": reading.ok,
                "transmission": float(info["transmission"]),
                "shot": int(info["evaluations"]),
                "score": float(info["benchmark_score"]),
                "mismatch": float(info["scored_mismatch"]),
                "stage": stage["name"],
            }
        )

    original_reset, original_step = env.reset, env.step

    def reset(**kwargs):
        observation, info = original_reset(**kwargs)
        stage["name"] = "start"
        note(observation, info)
        return observation, info

    def step(action):
        observation, reward, terminated, truncated, info = original_step(action)
        note(observation, info)
        return observation, reward, terminated, truncated, info

    env.reset, env.step = reset, step
    # The stage each shot belongs to is known to the agent, not to the environment;
    # the trace it returns names them in order, so they are zipped on afterwards.
    result = agent.run(env, seed=seed)
    for entry, frame in zip(result.trace, frames[1:]):
        frame["stage"] = entry["stage"]
    return frames, result


STAGE_TEXT = {
    "start": "the machine as the day dealt it",
    "optics": "reference optics loaded",
    "steer": "steering on the BPMs",
    "focus": "matching the spot size",
    "restore": "back to the best spot found",
    "place": "placing the spot on the request",
}


def windows_for(frames: list[dict]) -> list[float]:
    """A half-width per frame: wide enough for that shot, and only ever tightening.

    One fixed window cannot do this. The machine opens with a spot millimetres
    across and the request is tens of microns — seventy times smaller — so a view
    that holds the first frame renders the last as a dot, and one that suits the
    last clips the first off the screen entirely.

    So the view zooms, and the zoom is built to be followable: it is centred on the
    requested position for the whole episode, so the target never moves, and each
    window is the largest of what this frame and every later frame needs. That makes
    it monotonically non-increasing — the camera only ever closes in, never pulls
    back or jitters — and guarantees nothing is ever clipped.
    """
    target = frames[0]["target"]
    floor = max(max(target[2], target[3]) * 4, MIN_WINDOW)
    needed = []
    for frame in frames:
        # A frame taken while an upstream screen is in the beam is a different
        # camera: its centroid says nothing about where the readout is looking,
        # and letting it into the schedule would pull every earlier frame out.
        if frame["moments"] is None or frame.get("elsewhere"):
            needed.append(floor)
            continue
        mu_x, mu_y, sigma_x, sigma_y = frame["moments"]
        needed.append(
            max(
                floor,
                1.15 * (abs(mu_x - target[0]) + 2 * sigma_x),
                1.15 * (abs(mu_y - target[1]) + 2 * sigma_y),
            )
        )
    schedule = list(np.maximum.accumulate(needed[::-1])[::-1])

    # The away frames are a different camera and get their own window: one
    # constant, centred on that screen's axis, big enough for every one of them.
    # Constant matters — the whole point of those frames is that the orbit moves
    # between them, and a view that re-centres each shot hides exactly that.
    away = [i for i, f in enumerate(frames) if f.get("elsewhere")]
    if away:
        span = max(
            [floor] + [
                1.15 * max(abs(frames[i]["moments"][0]) + 2 * frames[i]["moments"][2],
                           abs(frames[i]["moments"][1]) + 2 * frames[i]["moments"][3])
                for i in away if frames[i]["moments"] is not None
            ]
        )
        for i in away:
            schedule[i] = span
    return schedule


def _rebinned(image, extent, centre, window):
    """The part of the camera inside the view, summed into displayable cells."""
    rows, columns = image.shape
    x = np.linspace(extent[0], extent[1], columns + 1)[:-1]
    y = np.linspace(extent[2], extent[3], rows + 1)[:-1]
    keep_x = np.flatnonzero(np.abs(x - centre[0]) <= window)
    keep_y = np.flatnonzero(np.abs(y - centre[1]) <= window)
    if not keep_x.size or not keep_y.size:
        return np.zeros((2, 2)), (centre[0] - window, centre[0] + window,
                                  centre[1] - window, centre[1] + window)
    patch = image[np.ix_(keep_y, keep_x)]
    step = max(1, min(patch.shape) // DISPLAY_CELLS)
    if step > 1:  # sum whole blocks; the remainder is dropped from the far edge
        high = (patch.shape[0] // step) * step, (patch.shape[1] // step) * step
        patch = patch[: high[0], : high[1]].reshape(
            high[0] // step, step, high[1] // step, step
        ).sum(axis=(1, 3))
    return patch, (x[keep_x[0]], x[keep_x[-1]], y[keep_y[0]], y[keep_y[-1]])


def draw(frame: dict, frames: list[dict], extent, axes, window: float) -> None:
    """One frame: the camera on the left, the score so far on the right."""
    camera, trace = axes
    camera.clear()
    trace.clear()
    target = frame["target"]
    # While an upstream screen intercepts the beam, the picture is that screen's.
    # The request belongs to the readout, so it is not drawn here and the view
    # centres on what this camera actually sees.
    elsewhere = frame.get("elsewhere")
    extent = frame.get("extent", extent)
    centre = (0.0, 0.0) if elsewhere else target[:2]

    patch, patch_extent = _rebinned(frame["image"], extent, centre, window)
    camera.imshow(
        patch,
        origin="lower",
        extent=[v * 1e3 for v in patch_extent],
        cmap=BEAM,
        aspect="equal",
        interpolation="bilinear",
        vmin=0.0,
        vmax=max(patch.max(), 1.0),
    )
    camera.set_xlim((centre[0] - window) * 1e3, (centre[0] + window) * 1e3)
    camera.set_ylim((centre[1] - window) * 1e3, (centre[1] + window) * 1e3)

    if not elsewhere:
        # What was asked for: one sigma, dashed, always in the middle of the view.
        camera.add_patch(
            Ellipse(
                (centre[0] * 1e3, centre[1] * 1e3),
                2 * target[2] * 1e3,
                2 * target[3] * 1e3,
                fill=False, edgecolor=REQUESTED, lw=2.0, ls="--", zorder=4,
                label="requested",
            )
        )
        # Early on the request is a few pixels across; the cross says where the
        # beam has to end up before the ellipse is big enough to see.
        camera.plot(
            centre[0] * 1e3, centre[1] * 1e3, "+",
            color=REQUESTED, ms=11, mew=1.8, zorder=5,
        )

    if frame["moments"] is not None:
        mu_x, mu_y, sigma_x, sigma_y = frame["moments"]
        camera.add_patch(
            Ellipse(
                (mu_x * 1e3, mu_y * 1e3), 2 * sigma_x * 1e3, 2 * sigma_y * 1e3,
                fill=False, edgecolor=MEASURED, lw=2.0, zorder=4,
                label="measured",
            )
        )
        sizes = f"$\\sigma$  {sigma_x * 1e6:.0f} × {sigma_y * 1e6:.0f} µm"
        if elsewhere:
            sizes += f"      on {elsewhere}, not the readout"
        else:
            sizes += (f"      asked {target[2] * 1e6:.0f}"
                      f" × {target[3] * 1e6:.0f} µm")
    else:
        sizes = "no beam on the screen"

    camera.set_xlabel("x  (mm)", color=MUTED, fontsize=9)
    camera.set_ylabel("y  (mm)", color=MUTED, fontsize=9)
    camera.set_title(
        f"shot {frame['shot']}   ·   {STAGE_TEXT.get(frame['stage'], frame['stage'])}",
        color=INK, fontsize=12, fontweight="bold", loc="left", pad=10,
    )
    camera.text(
        0.0, -0.17,
        f"{sizes}      transmission {frame['transmission']:.0%}"
        f"      view {2 * window * 1e3:.2f} mm across",
        transform=camera.transAxes, color=MUTED, fontsize=9,
    )
    for spine in camera.spines.values():
        spine.set_color(GRID)
    camera.tick_params(colors=MUTED, labelsize=8)
    # Two series, so a legend is always present; the text stays in ink and the
    # swatch beside it carries the identity.
    legend = camera.legend(
        loc="upper left", fontsize=9, frameon=True, facecolor=SURFACE,
        edgecolor=GRID, borderpad=0.5,
    )
    for text in legend.get_texts():
        text.set_color(INK)

    _inset(camera, frame, extent, centre, window)
    _trace(trace, frames, frame)


def _inset(camera, frame, extent, centre, window) -> None:
    """The whole 22 mm chip, with a box where the main view is looking."""
    inset = camera.inset_axes([0.735, 0.715, 0.25, 0.25])
    inset.imshow(
        frame["image"], origin="lower",
        extent=[e * 1e3 for e in (extent[0], extent[1], extent[2], extent[3])],
        cmap=BEAM, aspect="equal", interpolation="nearest",
        vmin=0.0, vmax=max(frame["image"].max(), 1.0),
    )
    inset.add_patch(
        Rectangle(
            ((centre[0] - window) * 1e3, (centre[1] - window) * 1e3),
            2 * window * 1e3, 2 * window * 1e3,
            fill=False, edgecolor=ROI, lw=1.2,
        )
    )
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("whole chip", color=MUTED, fontsize=7, pad=3)
    for spine in inset.spines.values():
        spine.set_color(GRID)


def _trace(trace, frames, frame) -> None:
    """The benchmark score after every shot taken so far. One series, no legend."""
    upto = [f for f in frames if f["shot"] <= frame["shot"]]
    xs = [f["shot"] for f in upto]
    ys = [f["score"] for f in upto]
    trace.plot(xs, ys, "-", color=MUTED, lw=2.0, zorder=2)
    trace.plot(xs[-1:], ys[-1:], "o", color=MEASURED, ms=9, zorder=3,
               markeredgecolor=SURFACE, markeredgewidth=2)
    trace.annotate(
        f"{ys[-1]:.2f}", (xs[-1], ys[-1]), textcoords="offset points",
        xytext=(9, 0), color=INK, fontsize=10, fontweight="bold", va="center",
    )
    trace.set_xlim(-0.4, max(f["shot"] for f in frames) + 1.4)
    # Log, because the episode crosses more than a decade: linear would press
    # everything after the first shot flat against the axis.
    trace.set_yscale("log")
    floor = min(f["score"] for f in frames) * 0.6
    trace.set_ylim(floor, max(f["score"] for f in frames) * 1.6)
    trace.set_xlabel("shots taken", color=MUTED, fontsize=9)
    trace.set_title(
        "benchmark score  (lower is better)",
        color=INK, fontsize=10, fontweight="bold", loc="left", pad=10,
    )
    trace.grid(True, axis="y", color=GRID, lw=0.8)
    trace.set_axisbelow(True)
    for side in ("top", "right"):
        trace.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        trace.spines[side].set_color(GRID)
    trace.tick_params(colors=MUTED, labelsize=8)


def render(frames, extent, out: Path, result, title: str | None = None,
           name: str = "beam_focusing") -> None:
    """The GIF, and the same frames as numbered PNGs.

    `title` and `name` are parameters so a second pipeline can render the same
    way without claiming to be this one.
    """
    out.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    figure.patch.set_facecolor(SURFACE)
    for axis in axes:
        axis.set_facecolor(SURFACE)

    windows = windows_for(frames)
    sequence = list(zip(frames, windows))
    sequence += [sequence[-1]] * HOLD_LAST
    gif = out / f"{name}.gif"
    writer = PillowWriter(fps=1000 / FRAME_MS)
    with writer.saving(figure, str(gif), dpi=110):
        for index, (frame, window) in enumerate(sequence):
            draw(frame, frames, extent, axes, window)
            figure.suptitle(
                title or (
                    f"CLARA · one episode of the MBRL_opt pipeline"
                    f"        final score {result.benchmark_score:.2f}"
                    f" in {result.shots} shots, "
                    f"{result.screen_switches} screen changes"
                ),
                color=INK, fontsize=11, x=0.012, ha="left", y=0.985,
            )
            figure.tight_layout(rect=[0, 0.02, 1, 0.94])
            writer.grab_frame(facecolor=SURFACE)
            if index < len(frames):
                figure.savefig(
                    out / f"frame_{index:02d}.png", dpi=110, facecolor=SURFACE
                )
    plt.close(figure)
    print(f"  {gif}  ({len(frames)} shots, {len(sequence)} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1241)
    parser.add_argument(
        "--scan", type=int, nargs=2, metavar=("FIRST", "LAST"),
        help="report shots and score for a range of seeds, and render nothing",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parents[1] / "mbrl_episode",
    )
    args = parser.parse_args()
    from config import MBRLConfig
    from twin import tune_threads

    tune_threads()
    config = MBRLConfig()

    if args.scan:
        first, last = args.scan
        for seed in range(first, last + 1):
            frames, result = record(seed, config)
            print(
                f"  seed {seed}  shots {result.shots:>2}  "
                f"score {result.benchmark_score:6.3f}  "
                f"stages {[f['stage'] for f in frames]}",
                flush=True,
            )
        return

    print(f"running seed {args.seed}")
    frames, result = record(args.seed, config)
    from rl_framework.config import RLConfig

    from MFopt.optics_env import OpticsEnv

    rl = RLConfig.clara()
    rl.screen_shape = config.screen_shape
    extent = OpticsEnv(rl).screen_extent()
    print(
        f"  {result.shots} shots, score {result.benchmark_score:.3f}, "
        f"{result.screen_switches} screen changes"
    )
    render(frames, extent, args.out, result)


if __name__ == "__main__":
    main()
