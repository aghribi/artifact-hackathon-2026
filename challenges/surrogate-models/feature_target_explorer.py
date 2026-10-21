"""Interactive statistical explorer for one PALLAS surrogate-model case.

The explorer reads the selected case's feature group from Hopsworks and
produces a self-contained Plotly report containing:

* dataset shape, missing-value counts, and descriptive statistics;
* Pearson and Spearman correlation heatmaps;
* interactive feature-vs-feature, corner, histogram, and trajectory views;
* candidate outliers identified with both IQR and robust-z-score rules.

Outliers are reported as candidates only.  This script never deletes rows or
changes Hopsworks data.

Examples::

    .venv/bin/python data_explorer.py --case trajectory
    .venv/bin/python data_explorer.py --case inverse_B --all-rows
    .venv/bin/python feature_target_explorer.py --case direct_B --max-points 5000
    .venv/bin/streamlit run feature_target_explorer.py -- --case trajectory
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import signal

import numpy as np
import pandas as pd

import pallas_pipeline as pp

DEFAULT_MAX_PLOT_ROWS = 50_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(pp.CASES), required=True,
                        help="Case/dataset to explore.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/data_explorer"),
                        help="Root directory for generated reports.")
    parser.add_argument("--all-rows", action="store_true",
                        help="Explore all feature-group rows instead of the case's modeling filter.")
    parser.add_argument("--max-points", type=int, default=5000,
                        help="Maximum rows used in interactive scatter plots.")
    parser.add_argument("--max-plot-rows", type=int, default=DEFAULT_MAX_PLOT_ROWS,
                        help="Maximum rows sent to interactive Plotly figures. Full data is "
                             "still used for statistics and outlier counts.")
    parser.add_argument("--top-outliers", type=int, default=100,
                        help="Number of candidate outlier rows to save in the report.")
    parser.add_argument("--no-open", action="store_true",
                        help="Do not try to open the generated HTML report.")
    return parser.parse_args()


def downsample_for_plots(frame: pd.DataFrame, case: pp.Case, max_rows: int,
                         random_state: int = 0) -> pd.DataFrame:
    """Return a bounded plotting frame without breaking trajectory configs.

    A trajectory is sampled by whole ``config`` groups so a plotted line is not
    silently truncated halfway through a configuration. Non-trajectory cases
    use a reproducible row sample. The original frame and its index are kept so
    the corresponding outlier flags can be selected without recomputation.
    """
    if max_rows <= 0:
        raise ValueError("max_plot_rows must be positive")
    if len(frame) <= max_rows:
        return frame

    if case.positional and "config" in frame:
        group_sizes = frame.groupby("config", sort=False).size()
        configs = group_sizes.sample(frac=1, random_state=random_state)
        selected_configs: list[object] = []
        selected_rows = 0
        for config, group_size in configs.items():
            if not selected_configs and int(group_size) > max_rows:
                # Preserve the hard browser-side cap even when one config is
                # larger than the requested plotting budget.
                return frame[frame["config"] == config].sample(
                    max_rows, random_state=random_state
                )
            if selected_configs and selected_rows + int(group_size) > max_rows:
                continue
            selected_configs.append(config)
            selected_rows += int(group_size)
            if selected_rows >= max_rows:
                break
        if selected_configs:
            return frame[frame["config"].isin(selected_configs)]

    return frame.sample(max_rows, random_state=random_state)


def load_case_rows(fs, case: pp.Case, all_rows: bool) -> pd.DataFrame:
    """Read the case's selected columns from its Hopsworks feature group."""
    feature_group = fs.get_feature_group(case.feature_group, version=1)
    columns = ["config"] + [pp.fs_name(column) for column in case.inputs + case.targets]
    columns = list(dict.fromkeys(columns))
    query = feature_group.select(columns)
    if not all_rows and case.rows is not None:
        query = query.filter(case.rows(feature_group))

    raw = query.read()
    names = {pp.fs_name(column): column
             for column in ["config"] + case.inputs + case.targets}
    return raw.rename(columns=names)[list(names.values())].reset_index(drop=True)


def numeric_columns(frame: pd.DataFrame, case: pp.Case) -> list[str]:
    """Return model features/targets that are numeric, excluding the config ID."""
    selected = list(dict.fromkeys(case.inputs + case.targets))
    return [column for column in selected
            if column in frame and pd.api.types.is_numeric_dtype(frame[column])]


def descriptive_statistics(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    numeric = frame[columns]
    description = numeric.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    description["missing"] = numeric.isna().sum()
    description["missing_pct"] = 100 * description["missing"] / max(len(frame), 1)
    description["unique"] = numeric.nunique(dropna=True)
    return description.reset_index(names="feature")


def correlations(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = frame[columns]
    return numeric.corr(method="pearson"), numeric.corr(method="spearman")


def flag_outliers(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag rows using IQR and robust-z-score rules for each numeric column."""
    flags = pd.DataFrame(False, index=frame.index, columns=columns)
    details: list[dict[str, object]] = []

    for column in columns:
        values = frame[column].astype(float)
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        iqr_flag = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)

        median = values.median()
        mad = (values - median).abs().median()
        if pd.notna(mad) and mad > 0:
            robust_z = (values - median).abs() / (1.4826 * mad)
            robust_flag = robust_z > 3.5
        else:
            robust_z = pd.Series(False, index=frame.index)
            robust_flag = pd.Series(False, index=frame.index)

        column_flags = (iqr_flag | robust_flag).fillna(False)
        flags[column] = column_flags
        details.append({
            "feature": column,
            "iqr_outliers": int(iqr_flag.fillna(False).sum()),
            "robust_z_outliers": int(robust_flag.fillna(False).sum()),
            "combined_outliers": int(column_flags.sum()),
            "q1": q1,
            "q3": q3,
            "median": median,
            "mad": mad,
        })

    row_flags = pd.DataFrame({
        "config": frame["config"].to_numpy() if "config" in frame else frame.index,
        "outlier_count": flags.sum(axis=1).astype(int),
    }, index=frame.index)
    row_flags["outlier_fraction"] = row_flags["outlier_count"] / max(len(columns), 1)
    row_flags["candidate_outlier"] = row_flags["outlier_count"] > 0
    feature_summary = pd.DataFrame(details)
    return row_flags, feature_summary


def _population_masks(row_flags: pd.DataFrame) -> dict[str, pd.Series]:
    """Masks used by the Plotly population selector."""
    outliers = row_flags["candidate_outlier"]
    return {
        "All rows": pd.Series(True, index=row_flags.index),
        "Non-outliers": ~outliers,
        "Candidate outliers": outliers,
    }


def _scatter_selector(frame: pd.DataFrame, columns: list[str], row_flags: pd.DataFrame,
                      max_points: int):
    """Build a linked X/Y scatter plot with an interactive outlier selector."""
    import plotly.graph_objects as go

    sample_index = frame.sample(min(len(frame), max_points), random_state=0).index
    sample = frame.loc[sample_index]
    sample_flags = row_flags.loc[sample_index]
    populations = _population_masks(sample_flags)
    x_name, y_name = columns[:2] if len(columns) > 1 else (columns[0], columns[0])

    def arrays(x_column: str, y_column: str) -> tuple[list, list, list]:
        x_data, y_data, custom_data = [], [], []
        for mask in populations.values():
            selected = sample.loc[mask]
            x_data.append(selected[x_column].tolist())
            y_data.append(selected[y_column].tolist())
            custom_data.append(pd.DataFrame({
                "config": selected["config"].to_numpy(),
                "outlier_count": sample_flags.loc[selected.index, "outlier_count"].to_numpy(),
            }).to_numpy())
        return x_data, y_data, custom_data

    x_data, y_data, custom_data = arrays(x_name, y_name)
    figure = go.Figure()
    population_names = list(populations)
    for index, population in enumerate(population_names):
        figure.add_trace(go.Scattergl(
            x=x_data[index], y=y_data[index], mode="markers", name=population,
            visible=index == 0, marker={"size": 5, "opacity": 0.65},
            customdata=custom_data[index],
            hovertemplate=(f"{x_name}: %{{x:.5g}}<br>{y_name}: %{{y:.5g}}<br>"
                           "config: %{customdata[0]}<br>"
                           "outlier flags: %{customdata[1]}<extra></extra>"),
        ))

    population_buttons = [
        {"label": label, "method": "restyle", "args": [{"visible": [i == index for i in range(3)]}]}
         for index, label in enumerate(population_names)
    ]

    x_buttons = []
    for selected_x in columns:
        next_x, _, next_custom = arrays(selected_x, y_name)
        x_buttons.append({
            "label": selected_x,
            "method": "update",
            "args": [{"x": next_x, "customdata": next_custom},
                     {"xaxis": {"title": selected_x}}],
        })

    y_buttons = []
    for selected_y in columns:
        _, next_y, next_custom = arrays(x_name, selected_y)
        y_buttons.append({
            "label": selected_y,
            "method": "update",
            "args": [{"y": next_y, "customdata": next_custom},
                     {"yaxis": {"title": selected_y}}],
        })

    figure.update_layout(
        title=f"Interactive feature relationship (sampled to {len(sample):,} rows)",
        xaxis_title=x_name, yaxis_title=y_name, height=720,
        updatemenus=[
            {"buttons": population_buttons, "direction": "down", "x": 0.0, "y": 1.18,
             "xanchor": "left", "yanchor": "top", "showactive": True,
             "pad": {"r": 8}, "type": "dropdown"},
            {"buttons": x_buttons, "direction": "down", "x": 0.32, "y": 1.18,
             "xanchor": "left", "yanchor": "top", "showactive": True,
             "pad": {"r": 8}, "type": "dropdown"},
            {"buttons": y_buttons, "direction": "down", "x": 0.66, "y": 1.18,
             "xanchor": "left", "yanchor": "top", "showactive": True,
             "pad": {"r": 8}, "type": "dropdown"},
        ],
        annotations=[
            {"text": "Outlier flag", "x": 0.0, "y": 1.22, "xref": "paper", "yref": "paper",
             "showarrow": False, "xanchor": "left"},
            {"text": "X feature", "x": 0.32, "y": 1.22, "xref": "paper", "yref": "paper",
             "showarrow": False, "xanchor": "left"},
            {"text": "Y feature", "x": 0.66, "y": 1.22, "xref": "paper", "yref": "paper",
             "showarrow": False, "xanchor": "left"},
        ],
    )
    return figure


def _histogram_selector(frame: pd.DataFrame, columns: list[str], row_flags: pd.DataFrame):
    """Build feature-distribution histograms with the same outlier selector."""
    import plotly.graph_objects as go

    populations = _population_masks(row_flags)
    feature_name = columns[0]
    feature_bins = {column: _adaptive_xbins(frame[column]) for column in columns}
    figure = go.Figure()
    for index, population in enumerate(populations):
        selected = frame.loc[populations[population]]
        figure.add_trace(go.Histogram(
            x=selected[feature_name], name=population, visible=index == 0,
            opacity=0.75, xbins=feature_bins[feature_name],
        ))

    feature_buttons = []
    for selected_feature in columns:
        feature_buttons.append({
            "label": selected_feature,
            "method": "update",
            "args": [{"x": [frame.loc[mask, selected_feature].tolist()
                               for mask in populations.values()],
                     "xbins": [feature_bins[selected_feature]] * len(populations)},
                     {"title": f"Distribution: {selected_feature}",
                      "xaxis": {"title": selected_feature}}],
        })
    population_buttons = [
        {"label": label, "method": "restyle", "args": [{"visible": [i == index for i in range(3)]}]}
         for index, label in enumerate(populations)
    ]
    figure.update_layout(
        title=f"Distribution: {feature_name}", barmode="overlay", height=600,
        xaxis_title=feature_name, yaxis_title="rows",
        updatemenus=[
            {"buttons": population_buttons, "direction": "down", "x": 0.0, "y": 1.18,
             "xanchor": "left", "yanchor": "top", "showactive": True,
             "type": "dropdown"},
            {"buttons": feature_buttons, "direction": "down", "x": 0.35, "y": 1.18,
             "xanchor": "left", "yanchor": "top", "showactive": True,
             "type": "dropdown"},
        ],
        annotations=[
            {"text": "Outlier flag", "x": 0.0, "y": 1.22, "xref": "paper", "yref": "paper",
             "showarrow": False, "xanchor": "left"},
            {"text": "Feature", "x": 0.35, "y": 1.22, "xref": "paper", "yref": "paper",
             "showarrow": False, "xanchor": "left"},
        ],
    )
    return figure


def _corner_plot(frame: pd.DataFrame, columns: list[str], row_flags: pd.DataFrame,
                 max_points: int):
    """Build a sampled Plotly corner/scatter-matrix plot."""
    import plotly.express as px

    sample_index = frame.sample(min(len(frame), max_points), random_state=0).index
    sample = frame.loc[sample_index, columns].copy()
    sample["population"] = np.where(
        row_flags.loc[sample_index, "candidate_outlier"], "Candidate outlier", "Non-outlier"
    )
    return px.scatter_matrix(
        sample, dimensions=columns, color="population",
        title=f"Corner plot (sampled to {len(sample):,} rows)",
        height=max(750, 170 * len(columns)),
    )


def _adaptive_xbins(values: pd.Series, max_bins: int = 80) -> dict | None:
    """Return Plotly histogram bins using Freedman-Diaconis widths.

    The same edges are used for every population of a feature, so toggling
    between all rows, non-outliers, and candidate outliers remains comparable.
    """
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size < 2 or np.ptp(clean) == 0:
        return None
    edges = np.histogram_bin_edges(clean, bins="fd")
    if len(edges) < 2 or not np.isfinite(edges).all():
        edges = np.histogram_bin_edges(clean, bins="auto")
    if len(edges) < 2:
        return None
    start, end = float(edges[0]), float(edges[-1])
    size = float(edges[1] - edges[0])
    if size <= 0 or not np.isfinite(size):
        return None
    if len(edges) - 1 > max_bins:
        size = (end - start) / max_bins
    return {"start": start, "end": end, "size": size}


def _histogram_grid(frame: pd.DataFrame, columns: list[str], row_flags: pd.DataFrame):
    """Build an overlaid histogram grid coloured by outlier population."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    histogram_data = frame[columns].copy()
    histogram_data["population"] = np.where(
        row_flags["candidate_outlier"], "Candidate outlier", "Non-outlier"
    )
    population_names = ["Non-outlier", "Candidate outlier"]
    n_columns = min(3, len(columns))
    n_rows = math.ceil(len(columns) / n_columns)
    figure = make_subplots(rows=n_rows, cols=n_columns, subplot_titles=columns)
    for index, feature in enumerate(columns):
        row, col = divmod(index, n_columns)
        bins = _adaptive_xbins(histogram_data[feature])
        for population in population_names:
            selected = histogram_data.loc[histogram_data["population"] == population, feature]
            figure.add_trace(go.Histogram(
                x=selected, name=population, legendgroup=population,
                showlegend=index == 0, opacity=0.7, xbins=bins,
            ), row=row + 1, col=col + 1)
    figure.update_layout(
        title="Feature histograms by population", barmode="overlay",
        height=max(600, 300 * n_rows),
    )
    return figure


def _trajectory_plot(frame: pd.DataFrame, case: pp.Case, row_flags: pd.DataFrame,
                     max_points: int):
    """Build target-vs-position trajectories for a sampled set of configurations."""
    if not case.positional:
        return None
    import plotly.express as px

    config_is_outlier = row_flags.groupby(frame["config"])["candidate_outlier"].any()
    configs = config_is_outlier.index.to_numpy()
    if len(configs) > max_points:
        configs = pd.Series(configs).sample(max_points, random_state=0).to_numpy()
    selected = frame[frame["config"].isin(configs)].copy()
    selected["config"] = selected["config"].astype(str)
    selected = selected.sort_values(["config", "z_mm"])
    trajectory_data = selected.melt(
        id_vars=["config", "z_mm"], var_name="target", value_name="value",
        value_vars=case.targets,
    )
    return px.line(
        trajectory_data, x="z_mm", y="value", color="config", facet_col="target",
        facet_col_wrap=2, title=f"Target trajectories (sampled to {len(configs):,} configs)",
        height=max(650, 320 * ((len(case.targets) + 1) // 2)),
    )


def plotly_figures(frame: pd.DataFrame, columns: list[str], row_flags: pd.DataFrame,
                   pearson: pd.DataFrame, spearman: pd.DataFrame, max_points: int,
                   case: pp.Case) -> dict:
    import plotly.express as px
    import plotly.graph_objects as go

    figures = {}
    missing = frame[columns].isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        missing = pd.Series({"No missing values": 0})
    figures["Missing values"] = px.bar(
        x=missing.index, y=missing.values,
        labels={"x": "feature", "y": "missing rows"},
        title="Missing values by feature",
    )
    figures["Pearson correlation"] = go.Figure(go.Heatmap(
        z=pearson.values, x=pearson.columns, y=pearson.index,
        zmin=-1, zmax=1, colorscale="RdBu", zmid=0,
        colorbar={"title": "r"},
    )).update_layout(title="Pearson correlations")
    figures["Spearman correlation"] = go.Figure(go.Heatmap(
        z=spearman.values, x=spearman.columns, y=spearman.index,
        zmin=-1, zmax=1, colorscale="RdBu", zmid=0,
        colorbar={"title": "rho"},
    )).update_layout(title="Spearman correlations")
    figures["Interactive relationship"] = _scatter_selector(
        frame, columns, row_flags, max_points
    )
    figures["Corner plot"] = _corner_plot(frame, columns, row_flags, max_points)
    figures["Interactive distributions"] = _histogram_selector(frame, columns, row_flags)
    figures["Histogram grid"] = _histogram_grid(frame, columns, row_flags)
    trajectory = _trajectory_plot(frame, case, row_flags, max_points)
    if trajectory is not None:
        figures["Trajectories"] = trajectory
    figures["Outlier counts"] = px.bar(
        row_flags["outlier_count"].value_counts().sort_index().rename_axis("outlier_count")
        .reset_index(name="rows"),
        x="outlier_count", y="rows", title="Rows by number of flagged features",
    )
    return figures


def write_report(path: Path, case_name: str, frame: pd.DataFrame, stats: pd.DataFrame,
                 feature_outliers: pd.DataFrame, figures: dict) -> None:
    import plotly.io as pio

    sections = [
        f"<h1>PALLAS data explorer: {case_name}</h1>",
        "<p>Outliers are statistical candidates, not automatic deletion recommendations. "
        "Check physical validity, simulation diagnostics, and domain constraints before filtering. "
        "Use the <b>Outlier flag</b> controls to switch between all rows, non-outliers, and "
        "candidate outliers.</p>",
        f"<h2>Dataset</h2><p>{len(frame):,} rows, {len(frame.columns):,} columns.</p>",
        "<h2>Descriptive statistics</h2>" + stats.to_html(index=False, float_format="{:.6g}".format),
        "<h2>Feature-level outlier summary</h2>" + feature_outliers.to_html(index=False, float_format="{:.6g}".format),
    ]
    include_js = True
    for title, figure in figures.items():
        sections.append(f"<h2>{title}</h2>" + pio.to_html(
            figure, full_html=False, include_plotlyjs=True if include_js else False,
        ))
        include_js = False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<!doctype html><html><head><meta charset='utf-8'></head><body>"
                    + "\n".join(sections) + "</body></html>", encoding="utf-8")


def analyse_case(fs, case: pp.Case, all_rows: bool):
    """Load and analyse a case once, for either CLI or Streamlit output."""
    frame = load_case_rows(fs, case, all_rows)
    columns = numeric_columns(frame, case)
    if not columns:
        raise ValueError("No numeric inputs or targets found for this case")
    stats = descriptive_statistics(frame, columns)
    pearson, spearman = correlations(frame, columns)
    row_flags, feature_outliers = flag_outliers(frame, columns)
    return frame, columns, stats, pearson, spearman, row_flags, feature_outliers


def _cached_case_analysis(st, case_name: str, all_rows: bool):
    """Cache the expensive Hopsworks read and full-data analysis in Streamlit."""
    @st.cache_data(show_spinner=False, ttl=3600)
    def _read_and_analyse(selected_case: str, include_all_rows: bool):
        project = pp.connect(pp.PROJECT)
        fs = project.get_feature_store()
        return analyse_case(fs, pp.CASES[selected_case], include_all_rows)

    return _read_and_analyse(case_name, all_rows)


def streamlit_app() -> None:
    """Render the interactive browser UI when launched with Streamlit."""
    import streamlit as st

    import plotly.express as px

    st.set_page_config(page_title="PALLAS data explorer", layout="wide")
    st.title("PALLAS data explorer")
    st.caption("Explore feature groups, relationships, missing values, and statistical outliers.")

    with st.sidebar:
        st.header("Dataset controls")
        case_name = st.selectbox("Case", sorted(pp.CASES))
        all_rows = st.checkbox("Include all feature-group rows", value=False)
        max_points = st.slider("Scatter points", 500, 20_000, 5_000, step=500)
        max_plot_rows = st.slider(
            "Rows used in plots", 5_000, 100_000, DEFAULT_MAX_PLOT_ROWS, step=5_000,
            help="Large datasets remain fully available for statistics, while Plotly receives "
                 "a bounded sample to keep the browser responsive.",
        )
        population = st.selectbox(
            "Outlier population",
            ["All rows", "Non-outliers", "Candidate outliers"],
            help="This controls the tables and the focused chart below. The Plotly charts also have their own selector.",
        )
        top_outliers = st.slider("Rows in outlier table", 10, 500, 100, step=10)
        st.divider()
        st.caption("Closing the browser tab does not stop the Streamlit process.")
        if st.button("Stop local server", type="secondary"):
            st.info("Stopping the local Streamlit server...")
            os.kill(os.getpid(), signal.SIGTERM)

    case = pp.CASES[case_name]
    with st.spinner(f"Loading and analysing {case_name} from Hopsworks..."):
        frame, columns, stats, pearson, spearman, row_flags, feature_outliers = _cached_case_analysis(
            st, case_name, all_rows
        )
        plot_frame = downsample_for_plots(frame, case, max_plot_rows)
        plot_flags = row_flags.loc[plot_frame.index]
        figures = plotly_figures(
            plot_frame, columns, plot_flags, pearson, spearman, max_points, case
        )

    if len(plot_frame) < len(frame):
        st.info(
            f"Loaded {len(frame):,} rows. Interactive figures use a reproducible sample of "
            f"{len(plot_frame):,} rows; statistics and outlier counts use all rows."
        )

    selected_mask = {
        "All rows": pd.Series(True, index=frame.index),
        "Non-outliers": ~row_flags["candidate_outlier"],
        "Candidate outliers": row_flags["candidate_outlier"],
    }[population]
    display_frame = frame.copy()
    display_frame["outlier_count"] = row_flags["outlier_count"]
    display_frame["outlier_fraction"] = row_flags["outlier_fraction"]
    display_frame["candidate_outlier"] = row_flags["candidate_outlier"]

    missing_cells = int(frame[columns].isna().sum().sum())
    candidate_count = int(row_flags["candidate_outlier"].sum())
    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows", f"{len(frame):,}")
    metric_columns[1].metric("Numeric features", len(columns))
    metric_columns[2].metric("Candidate outliers", f"{candidate_count:,}")
    metric_columns[3].metric("Missing cells", f"{missing_cells:,}")

    overview_tab, relationships_tab, distributions_tab, outliers_tab, statistics_tab = st.tabs([
        "Overview", "Relationships", "Distributions", "Outliers", "Statistics"
    ])
    with overview_tab:
        st.subheader(f"{case_name}: {case.feature_group}")
        st.write(f"Inputs: `{', '.join(case.inputs)}`")
        st.write(f"Targets: `{', '.join(case.targets)}`")
        st.dataframe(display_frame.loc[selected_mask].head(100), use_container_width=True)
        st.caption("The table follows the selected outlier population. Use the tabs for full summaries.")

    with relationships_tab:
        st.plotly_chart(figures["Pearson correlation"], use_container_width=True)
        st.plotly_chart(figures["Spearman correlation"], use_container_width=True)
        st.plotly_chart(figures["Interactive relationship"], use_container_width=True)
        st.plotly_chart(figures["Corner plot"], use_container_width=True)
        if "Trajectories" in figures:
            st.subheader("Trajectories by configuration")
            st.plotly_chart(figures["Trajectories"], use_container_width=True)

    with distributions_tab:
        st.plotly_chart(figures["Interactive distributions"], use_container_width=True)
        st.plotly_chart(figures["Histogram grid"], use_container_width=True)
        st.plotly_chart(figures["Missing values"], use_container_width=True)

    with outliers_tab:
        st.warning("Statistical outliers are candidates for review, not automatic discard decisions.")
        st.plotly_chart(figures["Outlier counts"], use_container_width=True)
        outlier_rows = display_frame.loc[row_flags["candidate_outlier"]].sort_values(
            ["outlier_count", "outlier_fraction"], ascending=False
        ).head(top_outliers)
        st.dataframe(outlier_rows, use_container_width=True)
        st.download_button(
            "Download candidate outlier rows",
            outlier_rows.to_csv(index=False),
            file_name=f"{case_name}_candidate_outliers.csv",
            mime="text/csv",
        )
        st.dataframe(feature_outliers, use_container_width=True)

    with statistics_tab:
        st.subheader("Descriptive statistics")
        st.dataframe(stats, use_container_width=True)
        st.download_button(
            "Download descriptive statistics",
            stats.to_csv(index=False),
            file_name=f"{case_name}_descriptive_statistics.csv",
            mime="text/csv",
        )
        st.subheader("Selected population distribution")
        selected_long = display_frame.loc[selected_mask, columns].sample(
            min(int(selected_mask.sum()), max_plot_rows), random_state=0
        ).melt(
            var_name="feature", value_name="value"
        )
        st.plotly_chart(px.box(selected_long, x="feature", y="value", points=False),
                        use_container_width=True)


def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    case = pp.CASES[args.case]
    project = pp.connect(pp.PROJECT)
    fs = project.get_feature_store()
    frame = load_case_rows(fs, case, args.all_rows)
    columns = numeric_columns(frame, case)
    if not columns:
        raise ValueError(f"No numeric inputs or targets found for case {args.case}")

    stats = descriptive_statistics(frame, columns)
    pearson, spearman = correlations(frame, columns)
    row_flags, feature_outliers = flag_outliers(frame, columns)
    plot_frame = downsample_for_plots(frame, case, args.max_plot_rows)
    plot_flags = row_flags.loc[plot_frame.index]
    figures = plotly_figures(
        plot_frame, columns, plot_flags, pearson, spearman, args.max_points, case
    )

    output_dir = args.output_dir / args.case
    output_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_dir / "descriptive_statistics.csv", index=False)
    pearson.to_csv(output_dir / "correlations_pearson.csv")
    spearman.to_csv(output_dir / "correlations_spearman.csv")
    feature_outliers.to_csv(output_dir / "outlier_feature_summary.csv", index=False)
    row_flags[row_flags["candidate_outlier"]].sort_values(
        ["outlier_count", "outlier_fraction"], ascending=False
    ).head(args.top_outliers).to_csv(output_dir / "candidate_outlier_rows.csv", index=False)
    write_report(output_dir / "report.html", args.case, frame, stats, feature_outliers, figures)

    print(f"explored {args.case}: {len(frame):,} rows, {len(columns)} numeric features")
    if len(plot_frame) < len(frame):
        print(f"interactive figures: sampled {len(plot_frame):,} rows for plotting")
    print(f"candidate outlier rows: {int(row_flags['candidate_outlier'].sum()):,}")
    print(f"report: {output_dir / 'report.html'}")

    if not args.no_open:
        import webbrowser
        webbrowser.open((output_dir / "report.html").resolve().as_uri())


if __name__ == "__main__":
    if _running_in_streamlit():
        streamlit_app()
    else:
        main()
