"""Build an exploratory One-Class SVM map for one PV, without changing its data.

Usage:
    python waveform_svm_map.py --pv 'MBL-090RFC:RFS-DIG-101:Dwn0-Cmp0SDS-PMData'

Dependencies: numpy, pandas, scikit-learn, matplotlib.
Maps fit an RBF One-Class SVM in a two-component PCA representation. Separate
models are fitted for waveform length, beam mode and beam length. A negative
score indicates an unusual candidate in this representation, not a fault label.
"""

import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "anomalyhunter-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from threadpoolctl import threadpool_limits


DEFAULT_PV = "MBL-090RFC:RFS-DIG-101:Dwn0-Cmp0SDS-PMData"
BLUE, RED = "#2563a6", "#e15449"


def block_features(array, bins=128):
    """Summarize standardized shape while retaining short peaks and troughs."""
    array = np.asarray(array, dtype=np.float64)
    std = array.std(ddof=0)
    normalized = (array - array.mean()) / std if std > 1e-12 else np.zeros_like(array)
    blocks = np.array_split(normalized, min(bins, array.size))
    return np.asarray([[b.mean(), b.min(), b.max()] for b in blocks]).ravel()


def prepare_features(arrays, bins=128):
    """Give shape and amplitude blocks equal total variance before PCA."""
    shape = np.vstack([block_features(a, bins) for a in arrays])
    shape -= shape.mean(axis=0)
    shape_scale = np.sqrt(np.sum(np.var(shape, axis=0)))
    if shape_scale > 1e-12:
        shape /= shape_scale
    amplitude = np.asarray([
        [np.std(a, dtype=np.float64), np.max(a), np.mean(a, dtype=np.float64)]
        for a in arrays
    ])
    amplitude = StandardScaler().fit_transform(amplitude) / np.sqrt(3)
    return np.column_stack([shape, amplitude])


def waveform_preview(array, bins=600):
    """Display-only envelope; all original samples contribute to its extrema."""
    edges = np.linspace(0, array.size, min(bins, array.size) + 1).astype(int)
    blocks = [array[start:end] for start, end in zip(edges[:-1], edges[1:])]
    def compact(values):
        return [float(f"{v:.7g}") for v in values]
    return dict(
        x=compact((edges[:-1] + edges[1:] - 1) / 2),
        mean=compact([b.mean(dtype=np.float64) for b in blocks]),
        low=compact([b.min() for b in blocks]),
        high=compact([b.max() for b in blocks]),
    )


def draw_map(ax, group, annotate=False):
    points, labels, model = group["points"], group["labels"], group["model"]
    span = np.ptp(points, axis=0)
    margin = np.maximum(span * 0.18, max(span.max(), 0.1) * 0.08)
    low, high = points.min(axis=0) - margin, points.max(axis=0) + margin
    xx, yy = np.meshgrid(np.linspace(low[0], high[0], 180),
                         np.linspace(low[1], high[1], 180))
    zz = model.decision_function(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    levels = sorted(set([min(float(zz.min()), -1e-8), 0.0, max(float(zz.max()), 1e-8)]))
    ax.contourf(xx, yy, zz, levels=levels, colors=["#fff1ed", "#eef6fd"], alpha=0.95)
    if zz.min() < 0 < zz.max():
        ax.contour(xx, yy, zz, levels=[0], colors=["#475569"], linewidths=1.25)
    for value, color, title, gid in (
        (1, BLUE, "Usual region", "usual_points"),
        (-1, RED, "Unusual candidate", "candidate_points"),
    ):
        mask = labels == value
        artist = ax.scatter(points[mask, 0], points[mask, 1], s=38, color=color,
                            edgecolors="white", linewidths=0.55, alpha=0.86,
                            zorder=3, label=f"{title} ({mask.sum()})")
        artist.set_gid(gid)
    if annotate:
        for index in np.argsort(group["scores"])[:min(3, len(points))]:
            ax.annotate(str(group["rows"].iloc[index].cycle_id), points[index],
                        xytext=(5, 8), textcoords="offset points", fontsize=8,
                        color="#9a342b", zorder=5)
    variance = group["variance"]
    ax.set_xlabel(f"PC1 ({variance[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({variance[1]:.1%} variance)")
    ax.set_title(f"{group['length']:,} samples | {group['mode']} | beam_len={group['beam_len']}\n"
                 f"{len(points)} cycles | PCA retains {sum(variance):.1%}", fontsize=11, pad=12)
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.grid(alpha=0.18)
    ax.legend(loc="best", framealpha=0.95, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def interactive_svg(fig, group):
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    root = ET.fromstring(buffer.getvalue())
    root.attrib.pop("width", None)
    root.attrib.pop("height", None)
    root.set("role", "img")
    root.set("aria-label", "One-Class SVM classification map; select a point to inspect its waveform")
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    ns = "{http://www.w3.org/2000/svg}"
    for gid, label in (("usual_points", 1), ("candidate_points", -1)):
        parent = next((el for el in root.iter() if el.get("id") == gid), None)
        uses = list(parent.iter(ns + "use")) if parent is not None else []
        row_ids = np.flatnonzero(group["labels"] == label)
        if len(uses) != len(row_ids):
            raise RuntimeError("SVG point mapping does not match the classification rows")
        for element, row_id in zip(uses, row_ids):
            element.set("data-row", str(int(row_id)))
            element.set("tabindex", "0")
            element.set("role", "button")
            element.set("aria-label", f"cycle_id {group['rows'].iloc[row_id].cycle_id}")
            title = ET.SubElement(element, ns + "title")
            title.text = f"cycle_id={group['rows'].iloc[row_id].cycle_id} | score={group['scores'][row_id]:.6g}"
    return ET.tostring(root, encoding="unicode")


def build_svm_map(input_path="dd_all_cycles.pkl", pv=DEFAULT_PV,
                  output_dir="waveform_svm_maps", nu=0.05, bins=128, minimum=8):
    if not 0 < nu < 1:
        raise ValueError("nu must be between zero and one")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading {input_path}", flush=True)
    data = pd.read_pickle(input_path)
    selected = data.loc[data.pv.eq(pv)].copy()
    if selected.empty:
        raise ValueError(f"PV not found: {pv}")
    if "original_row" not in selected:
        selected["original_row"] = selected.index
    del data
    gc.collect()
    selected = selected.sort_values(["cycle_id", "data_ts", "original_row"])
    signatures, valid, seen = [], [], set()
    duplicates = 0
    for _, row in selected.iterrows():
        array = row.data_array
        ok = (isinstance(array, np.ndarray) and array.ndim == 1
              and array.size > 0 and np.isfinite(array).all())
        valid.append(ok)
        signature = None
        if ok:
            signature = (int(row.cycle_id), str(row.data_ts), array.dtype.str,
                         array.shape, hashlib.sha256(array.tobytes()).hexdigest())
        duplicate = signature is not None and signature in seen
        signatures.append(duplicate)
        duplicates += int(duplicate)
        if signature is not None:
            seen.add(signature)
    selected["status"] = np.where(valid, "pending", "empty_or_nonfinite")
    selected.loc[signatures, "status"] = "duplicate_capture"
    for column in ("PC1", "PC2", "svm_score", "svm_label"):
        selected[column] = np.nan
    selected["condition"] = ""

    modeled, summaries = [], []
    usable = selected[selected.status.eq("pending")]
    for (length, mode, beam_len), rows in usable.groupby(
            ["array_size", "beam_mode", "beam_len"], dropna=False):
        rows = rows.copy()
        condition = f"{int(length)} / {mode} / beam_len={int(beam_len)}"
        selected.loc[rows.index, "condition"] = condition
        summary = dict(length=int(length), mode=str(mode), beam_len=int(beam_len),
                       rows=len(rows), condition=condition)
        if len(rows) < minimum:
            selected.loc[rows.index, "status"] = "insufficient_examples"
            summary["status"] = "insufficient_examples"
            summaries.append(summary)
            continue
        print(f"Fitting {condition}: {len(rows)} cycles", flush=True)
        features = prepare_features(rows.data_array.tolist(), bins)
        if np.linalg.norm(features - features.mean(axis=0)) < 1e-10:
            selected.loc[rows.index, "status"] = "identical_features"
            summary["status"] = "identical_features"
            summaries.append(summary)
            continue
        with threadpool_limits(limits=2):
            pca = PCA(n_components=2, svd_solver="full")
            points = pca.fit_transform(features)
            model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu, tol=1e-6).fit(points)
        scores = model.decision_function(points)
        labels = model.predict(points)
        selected.loc[rows.index, ["PC1", "PC2", "svm_score", "svm_label"]] = np.column_stack(
            [points, scores, labels])
        selected.loc[rows.index, "status"] = np.where(labels == 1, "usual_region", "unusual_candidate")
        variance = pca.explained_variance_ratio_.tolist()
        summary.update(status="modeled", candidates=int((labels == -1).sum()),
                       pca_variance=variance, gamma=float(model._gamma), nu=nu)
        summaries.append(summary)
        modeled.append({**summary, "points": points, "labels": labels, "scores": scores,
                        "model": model, "rows": rows, "variance": variance})
    modeled.sort(key=lambda group: group["rows"].shape[0], reverse=True)
    if not modeled:
        raise ValueError("No group has enough valid and varying captures for an SVM map")

    csv_columns = ["original_row", "pv", "cycle_id", "cycle_ts", "data_ts", "array_size",
                   "beam_mode", "beam_len", "condition", "status", "PC1", "PC2",
                   "svm_score", "svm_label", "data_array_std", "data_array_max", "data_array_mean",
                   "source_archive", "source_row"]
    selected[[c for c in csv_columns if c in selected]].to_csv(output_dir / "classification.csv", index=False)
    report = dict(pv=pv, input_path=str(Path(input_path).resolve()), rows=len(selected),
                  invalid=int((selected.status == "empty_or_nonfinite").sum()), duplicates=duplicates,
                  nu=nu, bins=bins, minimum_examples=minimum,
                  method="shape + amplitude -> PCA(2) -> RBF One-Class SVM",
                  interpretation="Exploratory unusual candidates, not fault labels; models are fitted to these same captures.",
                  groups=summaries)
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "axes.facecolor": "#ffffff", "figure.facecolor": "#ffffff"})
    payload_groups = []
    for group_index, group in enumerate(modeled):
        fig, ax = plt.subplots(figsize=(9, 6.5), layout="constrained")
        fig.suptitle(pv, fontsize=10, color="#475569")
        draw_map(ax, group)
        svg = interactive_svg(fig, group)
        name = f"map_{group['length']}_{re.sub(r'[^A-Za-z0-9_-]', '_', group['mode'])}_{group['beam_len']}"
        fig.savefig(output_dir / f"{name}.png", dpi=170)
        (output_dir / f"{name}.svg").write_text(svg)
        if group_index == 0:
            fig.savefig(output_dir / "classification_map.png", dpi=190)
        plt.close(fig)
        # The medoid is an actual usual-region waveform closest to its median point.
        usual = np.flatnonzero(group["labels"] == 1)
        candidates = usual if usual.size else np.arange(len(group["points"]))
        center = np.median(group["points"][candidates], axis=0)
        representative = int(candidates[np.argmin(np.linalg.norm(group["points"][candidates] - center, axis=1))])
        records = []
        for index, (_, row) in enumerate(group["rows"].iterrows()):
            records.append(dict(cycle_id=str(int(row.cycle_id)), data_ts=str(row.data_ts),
                                original_row=int(row.original_row), score=float(group["scores"][index]),
                                label=int(group["labels"][index]),
                                std=float(np.std(row.data_array, dtype=np.float64)),
                                maximum=float(np.max(row.data_array)),
                                mean=float(np.mean(row.data_array, dtype=np.float64)),
                                waveform=waveform_preview(row.data_array)))
        payload_groups.append(dict(condition=group["condition"], length=group["length"],
                                   mode=group["mode"], beam_len=group["beam_len"],
                                   variance=group["variance"], candidates=group["candidates"],
                                   representative=representative, svg=svg, records=records))

    fig, axes = plt.subplots(int(np.ceil(len(modeled) / 2)), 2,
                             figsize=(15, 5.5 * int(np.ceil(len(modeled) / 2))), squeeze=False,
                             layout="constrained")
    for ax, group in zip(axes.flat, modeled):
        draw_map(ax, group, annotate=True)
    for ax in list(axes.flat)[len(modeled):]:
        ax.axis("off")
        ax.text(0.05, 0.8, f"PV: {pv}\n\nOne-Class SVM fitted in a 2D PCA representation\n"
                f"nu={nu}; blue = usual region; red = unusual candidate\n"
                f"{report['invalid']} empty/nonfinite captures excluded\n"
                "Groups with fewer than 8 captures are not modeled.\n"
                "Candidates are not confirmed faults.", transform=ax.transAxes,
                fontsize=11, va="top", linespacing=1.8)
    fig.suptitle(pv + "\nShape and amplitude | separate acquisition conditions", fontsize=13)
    fig.savefig(output_dir / "all_conditions.png", dpi=160)
    plt.close(fig)

    payload = dict(**report, maps=payload_groups)
    template = Path(__file__).with_name("waveform_svm_map.html").read_text()
    html = template.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/"))
    (output_dir / "index.html").write_text(html)
    print(f"Saved {len(modeled)} maps to {output_dir.resolve()}", flush=True)
    print(selected.status.value_counts().to_string(), flush=True)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dd_all_cycles.pkl", dest="input_path")
    parser.add_argument("--pv", default=DEFAULT_PV)
    parser.add_argument("--output", default="waveform_svm_maps", dest="output_dir")
    parser.add_argument("--nu", type=float, default=0.05)
    parser.add_argument("--bins", type=int, default=128)
    args = parser.parse_args()
    build_svm_map(**vars(args))
