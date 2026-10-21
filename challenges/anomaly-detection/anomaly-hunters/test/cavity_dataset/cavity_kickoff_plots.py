"""Plots for the held-out kickoff models; separate from model/cache code."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LABELS = {"waveform": "Waveforms + cavity", "parameters": "Parameters + cavity", "combined": "Both"}


def plot_weak_comparison(result):
    table = result["weak_summary"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, std, title in zip(axes, ("mean_event_auc", "mean_event_AP"),
                                    ("std_event_auc", "std_event_AP"), ("Capture ROC-AUC", "Capture average precision")):
        for i, model in enumerate(("Forest", "Small tree")):
            part = table.query("model == @model").set_index("scope").loc[list(LABELS)]
            ax.bar(np.arange(3) + (i-.5)*.34, part[metric], width=.34, yerr=part[std], capsize=4, label=model)
        ax.set_xticks(range(3), LABELS.values(), rotation=10)
        ax.set(ylim=(0, 1.08), title=title, ylabel="Mean of five held-out capture folds")
        ax.legend()
        ax.grid(axis="y", alpha=.2)
    axes[0].axhline(.5, color="gray", linestyle="--", linewidth=.8)
    fig.suptitle("Weak label: DD=0, final PM=1; imputation and encoding fitted within each training fold")
    fig.tight_layout()
    return fig


def plot_clusters(result):
    table = result["cluster_memberships"]
    representations = table.representation.drop_duplicates()
    causes = sorted(table.trip_cause.unique())
    palette = {name: plt.get_cmap("tab20")(i) for i, name in enumerate(causes)}
    palette["Unlisted trip"] = "black"
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for i, representation in enumerate(representations):
        part = table.query("representation == @representation")
        for cluster, group in part.groupby("cluster"):
            label = "Noise" if cluster < 0 else f"C{cluster}"
            axes[i, 0].scatter(group.MDS1, group.MDS2, s=38, label=f"{label} (n={len(group)})")
        for cause, group in part.groupby("trip_cause"):
            axes[i, 1].scatter(group.MDS1, group.MDS2, s=38, label=cause, color=palette[cause],
                               marker="x" if cause == "Unlisted trip" else "o")
        for year, group in part.groupby("event_year"):
            axes[i, 2].scatter(group.MDS1, group.MDS2, s=38, label=str(year))
        for j, title in enumerate(("Clusters", "Recorded systems (annotation only)", "Year (diagnostic only)")):
            axes[i, j].set(title=f"{representation}\n{title}", xlabel="MDS1", ylabel="MDS2")
            axes[i, j].grid(alpha=.2)
            if j != 1:
                axes[i, j].legend(fontsize=8)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("Unsupervised trip signatures: one point = one capture; fixed 80/20 block weights")
    fig.tight_layout(rect=(0, .1, 1, .96))
    return fig


def plot_weak_importance(result, n=18):
    table = result["weak_global_explanations"].head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(table.feature, table.mean_abs_probability_contribution, color="#487eaa")
    ax.set(xlabel="Mean absolute contribution to PM score (held-out data; equal capture weight)",
           title="Which inputs the combined forest uses")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    return fig


def choose_examples(result):
    cases = result["weak_local_cases"]
    selected = [cases.query("label == 1").sort_values("score", ascending=False).iloc[0],
                cases.query("label == 0").sort_values("score", ascending=False).iloc[0],
                cases.query("label == 1").sort_values("score").iloc[0]]
    return pd.DataFrame(selected).reset_index(drop=True)


def plot_local_waveform(result, metadata, waves, dataset_index):
    case = result["weak_local_cases"].query("dataset_index == @dataset_index").iloc[0]
    contribution = result["weak_local_contributions"].query("dataset_index == @dataset_index").copy()
    contribution = contribution.sort_values("contribution")
    row = metadata.index[metadata.dataset_index.eq(dataset_index)][0]
    info = metadata.loc[row]
    fig = plt.figure(figsize=(17, 7))
    grid = fig.add_gridspec(2, 3, width_ratios=(1.6, 1, 1))
    bar = fig.add_subplot(grid[:, 0])
    bar.barh(contribution.feature, contribution.contribution,
             color=["#da793e" if n > 0 else "#4f81b4" for n in contribution.contribution])
    bar.axvline(0, color="gray", linewidth=.8)
    bar.set(xlabel="Contribution to PM score", title=f"Held-out score={case.score:.3f}; base={case.base_score:.3f}")
    bar.tick_params(axis="y", labelsize=8)
    for k, signal in enumerate(("field_amp", "field_phase", "fwd_amp", "refl_amp")):
        ax = fig.add_subplot(grid[k//2, 1+k%2])
        ax.plot(np.arange(1, 65), waves[signal][row], color="#354d66")
        for item in contribution.itertuples():
            name = item.feature
            if ".block" not in name:
                continue
            corresponding = "field_phase" if "field_phase_" in name else name.split(".")[1]
            if corresponding == signal:
                block = int(name.rsplit("block", 1)[1])
                ax.axvspan((block-1)*4+.5, block*4+.5, alpha=.12,
                           color="#da793e" if item.contribution > 0 else "#4f81b4")
        ax.set(title=signal, xlabel="Original waveform bin (1–64)")
        ax.grid(alpha=.2)
    fig.suptitle(f"capture={info.capture} | cavity={info.cavity} | weak label={info.label}\n"
                 f"mode={info.beam_mode}, beam={info.beam_present}, trig_delay={info.trig_delay:.6f}; "
                 "shown waveform has the highest PM score in this capture")
    fig.tight_layout(rect=(0, 0, 1, .91))
    return fig


def plot_cause_metrics(result):
    table = result["cause_metrics"]
    systems = table.system.drop_duplicates().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, value, title in zip(axes, ("mean_fold_auc", "mean_fold_AP"), ("ROC-AUC", "Average precision")):
        matrix = table.pivot(index="system", columns="scope", values=value).reindex(index=systems, columns=list(LABELS))
        ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(3), ["Waveform changes", "Parameters", "Both"], rotation=15)
        ax.set_yticks(range(len(matrix)), matrix.index)
        for i in range(len(matrix)):
            for j in range(3):
                n = matrix.iloc[i, j]
                ax.text(j, i, f"{n:.3f}", ha="center", va="center", color="black" if n > .7 else "white")
        ax.set_title(title + " (mean of 3 held-out folds)")
    fig.suptitle("Recorded-system association: one-vs-system, only annotated captures; no unlisted negatives")
    fig.tight_layout()
    return fig


def plot_cause_permutation(result):
    table = result["cause_permutation"].groupby(["system", "block"]).mean_auc_drop.mean().unstack("block")
    fig, ax = plt.subplots(figsize=(15, 5))
    limit = max(abs(table.to_numpy()).max(), .05)
    image = ax.imshow(table, cmap="coolwarm", aspect="auto", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(table)), table.index)
    for i in range(len(table)):
        for j in range(len(table.columns)):
            ax.text(j, i, f"{table.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Combined cause forest: held-out event-block permutation\nPositive = AUC decreases when the block is shuffled; marginal reliance, not a causal effect")
    fig.colorbar(image, ax=ax, label="Mean AUC decrease (3 folds × 3 repeats)")
    fig.tight_layout()
    return fig


def plot_cause_example(result, system="MEBT buncher"):
    local = result["cause_local_contributions"].query("system == @system")
    case = local.query("label == 1").sort_values("score", ascending=False).iloc[0]
    table = local.query("event_id == @case.event_id").sort_values("contribution")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.barh(table.feature, table.contribution,
             color=["#da793e" if c > 0 else "#4f81b4" for c in table.contribution])
    ax.axvline(0, color="gray", linewidth=.8)
    ax.set(xlabel="Contribution to recorded-system model score",
           title=f"Held-out event {case.event_id}: recorded {system}\nscore={case.score:.3f}; base={case.base_score:.3f}; cavity changes use fixed physical unit scales")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig


def plot_small_tree(bundle):
    from sklearn.tree import plot_tree
    model = bundle["weak_models"]["Small tree"]
    fig, ax = plt.subplots(figsize=(23, 8))
    plot_tree(model.model, feature_names=list(model.feature_names), class_names=["DD weak 0", "PM weak 1"],
               proportion=True, filled=True, precision=3, fontsize=7, ax=ax)
    ax.set_title("Depth-3 combined model fitted on all data for inspection; predictive scores are reported separately on held-out captures")
    fig.tight_layout()
    return fig


def plot_normal_trip_densities(result):
    """Separate all-point densities so Trip markers cannot obscure Normal."""
    from matplotlib.colors import LogNorm
    table = result["memberships"]
    representations = table.representation.drop_duplicates().tolist()
    fig, axes = plt.subplots(len(representations), 2, figsize=(15, 9), squeeze=False, constrained_layout=True)
    images = []
    maximum = 1
    for i, view_name in enumerate(representations):
        view = table.query("representation == @view_name")
        x_min, x_max = view.PC1.min(), view.PC1.max()
        y_min, y_max = view.PC2.min(), view.PC2.max()
        dx, dy = max((x_max-x_min)*.01, 1e-6), max((y_max-y_min)*.01, 1e-6)
        extent = (x_min-dx, x_max+dx, y_min-dy, y_max+dy)
        for j, (label, name) in enumerate(((0, "Normal (DD)"), (1, "Trip (PM last)"))):
            part = view.loc[view.label.eq(label)]
            image = axes[i, j].hexbin(part.PC1, part.PC2, gridsize=80, mincnt=1,
                                       extent=extent, cmap="viridis", rasterized=True)
            counts = image.get_array()
            if int(counts.sum()) != len(part):
                raise AssertionError("Per-label density did not count every point")
            maximum = max(maximum, counts.max())
            images.append(image)
            axes[i, j].set(title=f"{view_name}\n{name}: all {len(part):,} points", xlabel="PC1", ylabel="PC2",
                            xlim=extent[:2], ylim=extent[2:])
    norm = LogNorm(vmin=1, vmax=maximum)
    for image in images:
        image.set_norm(norm)
    fig.colorbar(images[0], ax=axes.ravel().tolist(), label="Points per hexagon (shared logarithmic color scale)", shrink=.75)
    fig.suptitle("Normal and Trip shown separately; identical coordinates and binning within each row; no subsampling")
    return fig
