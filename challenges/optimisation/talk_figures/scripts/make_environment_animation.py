"""Animate ten episodes of the benchmark: `lattice.plot_overview(beam)`, once per day.

The point of the slide is that no two episodes are the same machine. Each frame is one
case from `score.make_cases` - its own beam, its own survey error, its own set point -
drawn through Cheetah's own overview plot, so the lattice strip stays put and only the
beam moves.

    .venv/bin/python -I talk_figures/scripts/make_environment_animation.py

Run it from `challenges/optimisation` (it imports `score`). Output:
talk_figures/environment/episodes.gif, plus the frames beside it.
"""

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CHALLENGE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHALLENGE))

import score  # noqa: E402

OUT = CHALLENGE / "talk_figures" / "environment"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--particles", type=int, default=2000)
    parser.add_argument("--ms", type=int, default=900, help="milliseconds per frame")
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--max-mm", type=float, default=50.0, help="y-axis cap, mm")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cases = score.make_cases(
        n_cases=args.episodes,
        seed=args.seed,
        num_particles=args.particles,
        misalign=True,
    )
    print(f"{len(cases)} cases")

    # Build every figure first, then give them all the same y range before saving:
    # otherwise matplotlib rescales each frame and the animation reads as the axes
    # moving rather than the beam.
    figures = []
    for index, (beam, _target, misalignments, setpoint) in enumerate(cases):
        segment = score.load_lattice(count_evaluations=False)
        score.apply_misalignments(segment, misalignments)
        score.apply_tune(segment, setpoint)

        figure = plt.figure(figsize=(12, 6.2))
        segment.plot_overview(incoming=beam, fig=figure)
        figure.suptitle(
            f"episode {index + 1} of {len(cases)}   ·   its own beam, "
            f"its own survey error, its own set point",
            fontsize=13,
            y=0.995,
        )
        figures.append(figure)
        print(f"  episode {index + 1}")

    # the x and y panels are the first two axes of the overview
    # ... but capped: one runaway episode would otherwise flatten the other nine into a
    # straight line. The pipe is 17 mm, so past the cap the beam is gone anyway and
    # running off the panel says that more clearly than a rescaled axis.
    reach = max(
        abs(limit)
        for figure in figures
        for axis in figure.axes[:2]
        for limit in axis.get_ylim()
    )
    reach = min(reach, args.max_mm / 1000)
    print(f"common half-range {reach * 1000:.0f} mm")
    frames = []
    for index, figure in enumerate(figures):
        for axis in figure.axes[:2]:
            axis.set_ylim(-reach, reach)
        figure.tight_layout(pad=0.4)
        path = OUT / f"episode_{index:02d}.png"
        # tight bbox trims matplotlib's margin; the frames are padded back to a
        # common size below, since a GIF wants every frame the same shape
        figure.savefig(path, dpi=args.dpi, facecolor="white",
                       bbox_inches="tight", pad_inches=0.06)
        plt.close(figure)
        frames.append(path)

    from PIL import Image

    raw = [Image.open(p).convert("RGB") for p in frames]
    width = max(im.width for im in raw)
    height = max(im.height for im in raw)
    images = []
    for im in raw:
        if im.size != (width, height):
            canvas = Image.new("RGB", (width, height), "white")
            canvas.paste(im, ((width - im.width) // 2, (height - im.height) // 2))
            im = canvas
        images.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
    print(f"frame size {width}x{height}")
    gif = OUT / "episodes.gif"
    images[0].save(
        gif, save_all=True, append_images=images[1:], duration=args.ms, loop=0, optimize=True
    )
    print(f"wrote {gif} ({gif.stat().st_size / 1e6:.1f} MB, {len(images)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
