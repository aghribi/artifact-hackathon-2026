"""Loading and summarising the Monte Carlo results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
UM = 1e6


def load(config: str) -> pd.DataFrame:
    frame = pd.read_json(RESULTS / f"{config}.jsonl", lines=True)
    for column in ("quad_rms_um", "bpm_rms_um", "launch_um"):
        frame[column] = frame[column].round(3)  # 100e-6 * 1e6 is not exactly 100
    return frame


def reference_tables() -> dict:
    """Error-free reference (centroid, design IP optics) per lattice."""
    meta = json.loads((RESULTS / "meta.json").read_text())
    tables = {
        "design": {
            "ip": meta["reference_ip_design"],
            "optics": meta["design_optics_ip"],
        }
    }
    path = RESULTS / "meta_reftune.json"
    if path.exists():
        tables["reftune"] = json.loads(path.read_text())
    return tables


def add_derived(frame: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """Adds IP centroid shifts (um, relative to the error-free beam), growth in percent and
    dispersion changes relative to the design, for the before and after stages."""
    ip, optics = reference["ip"], reference["optics"]
    out = frame.copy()
    for stage in ("before", "after"):
        for plane in ("x", "y"):
            out[f"{stage}_dpos_{plane}"] = (out[f"{stage}_ip_{plane}"] - ip[plane]) * UM
            out[f"{stage}_dang_{plane}"] = (
                out[f"{stage}_ip_p{plane}"] - ip[f"p{plane}"]
            ) * UM
            out[f"{stage}_dD_{plane}"] = (
                out[f"{stage}_D_ip_{plane}"] - optics[f"D_{plane}"]
            )
            out[f"{stage}_dDp_{plane}"] = (
                out[f"{stage}_Dp_ip_{plane}"] - optics[f"Dp_{plane}"]
            )
        out[f"{stage}_dpos"] = np.hypot(out[f"{stage}_dpos_x"], out[f"{stage}_dpos_y"])
        out[f"{stage}_dang"] = np.hypot(out[f"{stage}_dang_x"], out[f"{stage}_dang_y"])
        out[f"{stage}_abs_x_um"] = out[f"{stage}_ip_x"].abs() * UM
        out[f"{stage}_abs_y_um"] = out[f"{stage}_ip_y"].abs() * UM
        out[f"{stage}_bpm_rms"] = (
            np.hypot(out[f"{stage}_bpm_rms_x"], out[f"{stage}_bpm_rms_y"]) * UM
        )
        out[f"{stage}_quad_rms"] = (
            np.hypot(out[f"{stage}_quad_rms_x"], out[f"{stage}_quad_rms_y"]) * UM
        )
        for key in ("sigma_x", "sigma_y", "eps_x", "eps_y"):
            out[f"{stage}_{key}_pct"] = out[f"{stage}_ip_{key}_rel"] * 100
        out[f"{stage}_beatbeat_ip_pct"] = (
            np.maximum(
                out[f"{stage}_beatbeat_ip_x"].abs(), out[f"{stage}_beatbeat_ip_y"].abs()
            )
            * 100
        )
        out[f"{stage}_beatbeat_max_pct"] = (
            np.maximum(
                out[f"{stage}_beatbeat_max_x"].abs(),
                out[f"{stage}_beatbeat_max_y"].abs(),
            )
            * 100
        )
    out["after_size_growth_pct"] = np.maximum(
        out["after_sigma_x_pct"].abs(), out["after_sigma_y_pct"].abs()
    )
    return out


def grouped(frame: pd.DataFrame, by: list[str], column: str) -> pd.DataFrame:
    """Median, 10th and 90th percentile of `column` per group."""
    return frame.groupby(by)[column].agg(
        median="median",
        p90=lambda s: np.nanpercentile(s, 90),
        p10=lambda s: np.nanpercentile(s, 10),
    )


def to_markdown(table: pd.DataFrame, index: bool = True, digits: int = 3) -> str:
    """Pipe table without the tabulate dependency."""
    if index:
        table = table.reset_index()
    table.columns = [
        " ".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in table.columns
    ]
    header = "| " + " | ".join(table.columns) + " |"
    divider = "|" + "|".join("---" for _ in table.columns) + "|"
    lines = [header, divider]
    for _, row in table.iterrows():
        lines.append(
            "| "
            + " | ".join(
                f"{v:.{digits}g}" if isinstance(v, (float, np.floating)) else str(v)
                for v in row
            )
            + " |"
        )
    return "\n".join(lines)
