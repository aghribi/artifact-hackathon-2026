"""How to use the nominal xsuite model.

Builds the xsuite line from the benchmark start (cla_s02_sim_aper_01) to the readout
screen (cla_feh_dia_scr_01) with the lattice-file magnet strengths, tracks the nominal
starting beam (initial_distribution.npz) through it, shows the beam on the readout screen
and plots beta functions and dispersion along the line.

Run: python example_nominal_model.py [--end ELEMENT] [--distribution FILE]
Figures are written to ./figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import xsuite_model as model

FIGURES = Path(__file__).resolve().parent / "figures"
STYLE = {  # set explicitly so the figures look the same on every machine
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "figure.figsize": (4.0, 2.5),
    "figure.constrained_layout.use": True,
    "image.cmap": "inferno",
}


def screen_half_extent(name: str) -> tuple[float, float] | None:
    """Half width and height of a screen's camera chip in metres, from the lattice file."""
    element = next(
        e for e in model.cheetah_segment(end=name).elements if e.name == name
    )
    if not hasattr(element, "pixel_size"):
        return None
    resolution, pixel = (
        np.asarray(element.resolution, dtype=float),
        element.pixel_size.detach().numpy(),
    )
    return tuple(float(v) for v in resolution * pixel / 2)


def plot_screen(final: np.ndarray, name: str, path: Path) -> None:
    """x-y distribution at the end of the line with its projections and the camera chip."""
    x, y = final[:, 0] * 1e3, final[:, 2] * 1e3
    chip = screen_half_extent(name)
    figure, (image_axis, x_axis, y_axis) = plt.subplots(1, 3, figsize=(6.6, 2.2))
    reach = max(4 * x.std(), 4 * y.std(), *(1.05e3 * v for v in chip or ()))
    image_axis.hist2d(x, y, bins=200, range=[(-reach, reach), (-reach, reach)], cmin=1)
    if chip:
        image_axis.add_patch(
            Rectangle(
                (-chip[0] * 1e3, -chip[1] * 1e3),
                2 * chip[0] * 1e3,
                2 * chip[1] * 1e3,
                fill=False,
                linestyle="--",
                color="C0",
                label="camera chip",
            )
        )
        image_axis.legend(loc="lower left", frameon=False)
    image_axis.set(xlabel="x (mm)", ylabel="y (mm)", aspect="equal", title=name)
    for axis, values, label in ((x_axis, x, "x (mm)"), (y_axis, y, "y (mm)")):
        axis.hist(values, bins=200, range=(-reach, reach), histtype="step", color="C0")
        axis.set(xlabel=label, ylabel="particles")
        axis.text(
            0.03,
            0.97,
            f"rms {values.std():.2f} mm\nmean {values.mean():+.2f} mm",
            transform=axis.transAxes,
            va="top",
        )
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_optics(optics: dict, end: str, path: Path) -> None:
    """Beta functions and dispersion along the line."""
    figure, (beta_axis, dispersion_axis) = plt.subplots(
        2, 1, sharex=True, figsize=(6.6, 3.2)
    )
    for plane, colour in (("x", "C0"), ("y", "C1")):
        beta_axis.plot(
            optics["s"],
            optics[f"beta_{plane}"],
            color=colour,
            label=f"$\\beta_{plane}$",
        )
        dispersion_axis.plot(
            optics["s"],
            optics[f"dispersion_{plane}"],
            color=colour,
            label=f"$D_{plane}$",
        )
    beta_axis.set(yscale="log", ylabel=r"$\beta$ (m)")
    dispersion_axis.set(xlabel=f"s from {model.START} (m)", ylabel="D (m)")
    for axis in (beta_axis, dispersion_axis):
        axis.axvline(optics["s"][-1], color="0.5", linestyle=":", linewidth=1)
        axis.legend(loc="upper left", frameon=False)
    beta_axis.set_title(
        f"Optics of the nominal model to {end}, starting from the beam's projected Twiss"
    )
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--end",
        default=model.READOUT,
        help="element to track to (default: the readout screen)",
    )
    parser.add_argument("--distribution", default=str(model.INITIAL_DISTRIBUTION))
    arguments = parser.parse_args()
    plt.rcParams.update(STYLE)
    FIGURES.mkdir(exist_ok=True)

    # 1. The nominal starting beam and the line, nominal (lattice-file) magnet strengths.
    start, energy_ev, metadata = model.load_distribution(arguments.distribution)
    converted = model.build_line(
        energy_ev, lattice_json=model.LATTICE_JSON, end=arguments.end
    )
    print(
        f"{len(start)} particles at {metadata['location']}, {energy_ev / 1e6:.4f} MeV, "
        f"longitudinal plane from {metadata['longitudinal_source']}"
    )

    # 2. Track to the end and look at the beam there.
    final = model.track_forward(converted, start)
    summary = model.beam_summary(final, converted.final_energy_ev)
    print(
        f"at {arguments.end} ({converted.final_energy_ev / 1e6:.2f} MeV): "
        f"sigma {summary['sigma_x'] * 1e3:.3f} / {summary['sigma_y'] * 1e3:.3f} mm, "
        f"mean {summary['mean_x'] * 1e3:+.3f} / {summary['mean_y'] * 1e3:+.3f} mm, "
        f"eps_n {summary['normalised_emittance_x'] * 1e6:.2f} / {summary['normalised_emittance_y'] * 1e6:.2f} um"
    )
    plot_screen(final, arguments.end, FIGURES / "final_screen.png")

    # 3. Twiss functions: test particles tracked from the beam's projected Twiss at the start.
    optics = model.optics_along(converted, model.twiss_from_coordinates(start))
    print(
        f"at {arguments.end}: beta {optics['beta_x'][-1]:.4g} / {optics['beta_y'][-1]:.4g} m, "
        f"D_x {optics['dispersion_x'][-1]:.4f} m"
    )
    plot_optics(optics, arguments.end, FIGURES / "optics.png")
    print(f"figures in {FIGURES}")


if __name__ == "__main__":
    main()
