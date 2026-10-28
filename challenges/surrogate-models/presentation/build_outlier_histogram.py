"""Build the static interactive histogram used by the outlier slide.

The Hopsworks read happens only when this builder is run. The presentation
embeds the generated HTML and CSV snapshot, so viewing the slides does not
contact Hopsworks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Allow this presentation helper to import the shared pipeline modules when
# launched either from the repository root or from presentation/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import feature_target_explorer as explorer
import pallas_pipeline as pp


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data"
FEATURE_PALETTES = {
    "blue": ("#74a9cf", "#0571b0"),
    "red": ("#c99498", "#ef3b3f"),
    "green": ("#66bd63", "#2ca25f"),
    "orange": ("#c49f52", "#f28e0c"),
    "purple": ("#a9a3c9", "#54278f"),
}
FEATURE_COLOUR_CYCLE = ("blue", "red", "green", "orange", "purple")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-rows", type=int, default=50_000,
                        help="Maximum rows saved in the static CSV snapshot.")
    return parser.parse_args()


def stratified_snapshot(frame: pd.DataFrame, row_flags: pd.DataFrame,
                        max_rows: int) -> pd.DataFrame:
    """Keep a reproducible non-outlier/outlier mix for the presentation."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")

    flagged = row_flags["candidate_outlier"]
    outliers = frame.loc[flagged].copy()
    normal = frame.loc[~flagged].copy()
    outlier_count = min(len(outliers), max_rows // 4)
    normal_count = min(len(normal), max_rows - outlier_count)

    selected = pd.concat([
        normal.sample(normal_count, random_state=0),
        outliers.sample(outlier_count, random_state=0),
    ])
    return selected.sample(frac=1, random_state=0).reset_index(drop=True)


def histogram_figure(snapshot: pd.DataFrame, columns: list[str]) -> go.Figure:
    """Create one visible histogram with a dropdown for the other features."""
    populations = {
        "All rows": pd.Series(True, index=snapshot.index),
        "Non-outliers": ~snapshot["candidate_outlier"],
        "Candidate outliers": snapshot["candidate_outlier"],
    }
    first_feature = columns[0]
    bins = {column: explorer._adaptive_xbins(snapshot[column]) for column in columns}
    first_palette = FEATURE_PALETTES[FEATURE_COLOUR_CYCLE[0]]
    figure = go.Figure()

    for index, (population, mask) in enumerate(populations.items()):
        colour = "#8b98a5" if index == 0 else first_palette[index - 1]
        figure.add_trace(go.Histogram(
            x=snapshot.loc[mask, first_feature],
            name=population,
            # "All rows" is represented by the two overlaid population
            # traces, rather than by a third aggregate trace.
            visible=index > 0,
            opacity=0.75,
            marker={"color": colour},
            xbins=bins[first_feature],
        ))

    feature_buttons = []
    for feature in columns:
        feature_buttons.append({
            "label": feature,
            "method": "update",
            "args": [{
                "x": [snapshot.loc[mask, feature].tolist() for mask in populations.values()],
                "xbins": [bins[feature]] * len(populations),
                "marker": [{"color": "#8b98a5"},
                           {"color": FEATURE_PALETTES[FEATURE_COLOUR_CYCLE[
                               columns.index(feature) % len(FEATURE_COLOUR_CYCLE)
                           ]][0]},
                           {"color": FEATURE_PALETTES[FEATURE_COLOUR_CYCLE[
                               columns.index(feature) % len(FEATURE_COLOUR_CYCLE)
                           ]][1]}],
            }, {
                "title": {"text": ""},
                "xaxis": {"title": feature},
            }],
        })

    population_buttons = [
        {
            "label": population,
            "method": "restyle",
            "args": [{
                "visible": (
                    [False, True, True]
                    if index == 0
                    else [False] + [i == index for i in range(1, len(populations))]
                )
            }],
        }
        for index, population in enumerate(populations)
    ]

    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
        xaxis={"gridcolor": "rgba(148,163,184,0.22)"},
        yaxis={"gridcolor": "rgba(148,163,184,0.22)"},
        title={"text": ""},
        barmode="overlay",
        height=470,
        margin={"l": 55, "r": 25, "t": 85, "b": 55},
        xaxis_title=first_feature,
        yaxis_title="rows",
        legend={"orientation": "h", "y": 1.02, "x": 0},
        updatemenus=[{
            "buttons": population_buttons,
            "direction": "down",
            "x": 0,
            "y": 1.18,
            "xanchor": "left",
            "yanchor": "top",
            "showactive": True,
            "type": "dropdown",
        }, {
            "buttons": feature_buttons,
            "direction": "down",
            "x": 0.35,
            "y": 1.18,
            "xanchor": "left",
            "yanchor": "top",
            "showactive": True,
            "type": "dropdown",
        }],
        annotations=[{
            "text": "Outlier flag",
            "x": 0,
            "y": 1.23,
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "xanchor": "left",
        }, {
            "text": "Feature",
            "x": 0.35,
            "y": 1.23,
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "xanchor": "left",
        }],
    )
    return figure


def main() -> None:
    args = parse_args()
    case = pp.CASES["trajectory"]
    project = pp.connect(pp.PROJECT)
    frame = explorer.load_case_rows(project.get_feature_store(), case, all_rows=False)
    columns = explorer.numeric_columns(frame, case)
    row_flags, _ = explorer.flag_outliers(frame, columns)
    snapshot_input = frame[["config", *columns]].copy()
    snapshot_input["candidate_outlier"] = row_flags["candidate_outlier"].to_numpy()
    snapshot = stratified_snapshot(snapshot_input, snapshot_input, args.max_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "trajectory_histogram_snapshot.csv"
    html_path = args.output_dir / "trajectory_histograms.html"
    snapshot.to_csv(csv_path, index=False)
    figure = histogram_figure(snapshot, columns)
    html = pio.to_html(
        figure, full_html=True, include_plotlyjs=True,
        config={"responsive": True, "displaylogo": False},
    )
    html = html.replace(
        "<body>",
        '<body style="margin:0;background:transparent;overflow:hidden;">',
        1,
    )
    html_path.write_text(html, encoding="utf-8")
    print(f"saved {len(snapshot):,} snapshot rows to {csv_path}")
    print(f"saved interactive histogram to {html_path}")


if __name__ == "__main__":
    main()
