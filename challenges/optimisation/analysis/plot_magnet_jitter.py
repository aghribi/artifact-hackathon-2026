"""Figure: the magnet set/readback jitter term.

Panel A  distribution of the jitter (deviation of a reading from the median
         reading at the same setpoint), as an ECDF per magnet family
Panel B  per-magnet jitter against the fixed calibration offset, with the
         magnets whose jitter is exactly zero on their own strip
"""
import csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

CSV, OUT, MODE = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "light")

T = dict(
    light=dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
               grid="#e1e0d9", axis="#c3c2b7", s=("#2a78d6", "#eb6834", "#1baf7a")),
    dark=dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
              grid="#2c2c2a", axis="#383835", s=("#3987e5", "#d95926", "#199e70")),
)[MODE]

SENT = 1e4
BAD = {"CLA-LRG1-MAG-SOL-01", "CLA-LRG1-MAG-BSOL-01", "CLA-FEH-MAG-HCOR-08", "CLA-FEH-MAG-VCOR-08"}
MIRROR = {"CLA-FEH-MAG-VCOR-01", "CLA-FEH-MAG-VCOR-02", "CLA-FEH-MAG-VCOR-03", "CLA-FEH-MAG-VCOR-05",
          "CLA-FEH-MAG-VCOR-06", "CLA-FEH-MAG-HCOR-01", "CLA-FEH-MAG-HCOR-02", "CLA-FEH-MAG-HCOR-03",
          "CLA-FEH-MAG-HCOR-04", "CLA-FEH-MAG-HCOR-05", "CLA-FEH-MAG-HCOR-06", "CLA-S01-MAG-HCOR-01",
          "CLA-S01-MAG-HCOR-02", "CLA-S01-MAG-VCOR-01", "CLA-S01-MAG-VCOR-02"}
GROUP = {"QUAD": "Quadrupole", "HCOR": "Corrector", "VCOR": "Corrector",
         "DIP": "Dipole / solenoid", "SOL": "Dipole / solenoid", "BSOL": "Dipole / solenoid"}
ORDER = ["Quadrupole", "Corrector", "Dipole / solenoid"]

with open(CSV, newline="") as fh:
    rdr = csv.reader(fh); header = next(rdr); rows = list(rdr)
cols = {c: i for i, c in enumerate(header)}
o = np.argsort(np.array([np.datetime64(r[cols["timestamp"]].rstrip("Z")) for r in rows]))

def col(name):
    i = cols[name]
    a = np.array([float(r[i]) if r[i] not in ("", "nan", "NaN") else np.nan for r in rows])[o]
    a[np.abs(a) > SENT] = np.nan
    return a

mags = sorted({c.rsplit(":", 1)[0] for c in header if c.endswith(":SETI")}
              & {c.rsplit(":", 1)[0] for c in header if c.endswith(":READI")})

resid_by_group = {g: [] for g in ORDER}
per = []
for m in mags:
    if m in BAD or m in MIRROR:
        continue
    s, r = col(m + ":SETI"), col(m + ":READI")
    fin = np.isfinite(s) & np.isfinite(r); s, r = s[fin], r[fin]
    zero = np.abs(s) < 1e-6
    off = (~zero) & (np.abs(r) < 0.05 * np.abs(s))
    on = (~zero) & (~off)
    if on.sum() < 100:
        continue
    s_on, r_on = s[on], r[on]
    su, inv = np.unique(np.round(s_on, 5), return_inverse=True)
    res, ng = [], 0
    for k in range(len(su)):
        sel = inv == k
        if sel.sum() < 10:
            continue
        res.append(r_on[sel] - np.median(r_on[sel])); ng += 1
    if ng < 2:
        continue
    res = np.concatenate(res)
    g = GROUP[m.split("-")[3]]
    resid_by_group[g].append(res)
    per.append(dict(name=m, group=g, jit=float(np.percentile(np.abs(res), 95)),
                    offset=abs(float(np.median(r_on - s_on)))))

for g in ORDER:
    resid_by_group[g] = np.concatenate(resid_by_group[g])
allres = np.concatenate([resid_by_group[g] for g in ORDER])
FLOOR = 1e-4

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "figure.facecolor": T["surface"], "axes.facecolor": T["surface"],
    "savefig.facecolor": T["surface"], "text.color": T["ink"],
    "axes.labelcolor": T["ink2"], "xtick.color": T["muted"], "ytick.color": T["muted"],
})
fig = plt.figure(figsize=(12.0, 5.3))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.13], hspace=0.08, wspace=0.24,
                      left=0.062, right=0.985, top=0.775, bottom=0.115)
axA = fig.add_subplot(gs[:, 0])
axB = fig.add_subplot(gs[0, 1])
axZ = fig.add_subplot(gs[1, 1], sharex=axB)

def clean(ax, left=True, bottom=True):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp, on_ in (("left", left), ("bottom", bottom)):
        ax.spines[sp].set_visible(on_)
        ax.spines[sp].set_color(T["axis"]); ax.spines[sp].set_linewidth(1.0)
    ax.tick_params(length=3, width=1.0)

amp = FuncFormatter(lambda v, p: "%g" % v)

# ------------------------------------------------- Panel A : ECDF of |jitter|
clean(axA)
floors = {}
for g, c in zip(ORDER, T["s"]):
    a = np.abs(resid_by_group[g])
    f0 = float(np.mean(a == 0)); floors[g] = f0
    nz = np.sort(a[a > 0])
    y = f0 + (1 - f0) * np.arange(1, nz.size + 1) / nz.size
    x = np.clip(nz, FLOOR, None)
    axA.plot(x, y, lw=2, color=c, solid_capstyle="round", zorder=4)
    axA.plot([FLOOR * 1.02], [f0], marker="o", ms=7, color=c, zorder=5,
             markeredgecolor=T["surface"], markeredgewidth=1.4)
    axA.text(FLOOR * 1.22, f0 + 0.022, f"{100*f0:.0f}%", color=c, fontsize=8.5,
             fontweight="bold", ha="left", va="bottom", zorder=6)

axA.axvline(0.01, color=T["axis"], lw=1, ls=(0, (4, 3)), zorder=1)
axA.text(0.0118, 0.42, "0.01 A\nreadback\nquantum", color=T["muted"], fontsize=7.5,
         ha="left", va="center", linespacing=1.35)

for g, c, (lx, ly, ha) in zip(
        ORDER, T["s"],
        [(0.016, 0.885, "left"), (2.2e-4, 0.335, "left"), (0.030, 0.755, "left")]):
    axA.text(lx, ly, g, color=c, fontsize=9.5, fontweight="bold", ha=ha, va="center",
             zorder=6)

axA.annotate("readings that repeat bit-for-bit:\nthe jitter is exactly zero",
             xy=(FLOOR * 1.12, 0.455), xytext=(3.0e-4, 0.165),
             color=T["ink2"], fontsize=8, linespacing=1.4, ha="left",
             arrowprops=dict(arrowstyle="-", color=T["axis"], lw=1,
                             connectionstyle="arc3,rad=0.25", shrinkA=2, shrinkB=4))
axA.set_xscale("log")
axA.set_xlim(FLOOR, 20)
axA.set_ylim(0, 1.005)
axA.set_xlabel("|jitter|  (A)  —  a reading's deviation from the median reading at that same setpoint")
axA.set_ylabel("fraction of readings at or below")
axA.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0%}"))
axA.xaxis.set_major_locator(LogLocator(base=10, numticks=7))
axA.xaxis.set_major_formatter(amp)
axA.grid(axis="y", color=T["grid"], lw=0.8, zorder=0)
axA.set_axisbelow(True)
axA.set_title("A.  The jitter is quantisation-limited", loc="left",
              fontsize=10.5, fontweight="bold", color=T["ink"], pad=8)
aa = np.abs(allres)
axA.text(0.995, 0.035,
         f"all magnets:  p90 {np.percentile(aa,90):.3g} A   ·   p99 {np.percentile(aa,99):.3g} A"
         f"   ·   p99.9 {np.percentile(aa,99.9):.3g} A",
         transform=axA.transAxes, ha="right", va="bottom", color=T["muted"], fontsize=7.5)

# --------------------------------- Panel B : jitter vs offset (+ zero strip)
XL = (2e-4, 0.5)
YL = (1.5e-3, 0.3)
clean(axB, bottom=False)
axB.plot([YL[0], YL[1]], [YL[0], YL[1]], color=T["axis"], lw=1, ls=(0, (4, 3)), zorder=1)
axB.text(0.175, 0.205, "equal", color=T["muted"], fontsize=7.5, rotation=33,
         ha="center", va="bottom", zorder=2)
nz_counts, z_counts = {}, {}
rng = np.random.default_rng(0)
for g, c in zip(ORDER, T["s"]):
    pts = [d for d in per if d["group"] == g]
    nzp = [d for d in pts if d["jit"] > 0]
    zp = [d for d in pts if d["jit"] == 0]
    nz_counts[g], z_counts[g] = len(nzp), len(zp)
    if nzp:
        axB.scatter(np.clip([d["offset"] for d in nzp], XL[0] * 1.1, None),
                    [d["jit"] for d in nzp], s=48, color=c, alpha=0.85,
                    linewidths=1.6, edgecolors=T["surface"], zorder=3)
    if zp:
        axZ.scatter(np.clip([d["offset"] for d in zp], XL[0] * 1.1, None),
                    rng.uniform(-0.3, 0.3, len(zp)), s=48, color=c, alpha=0.85,
                    linewidths=1.6, edgecolors=T["surface"], zorder=3)
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xlim(*XL); axB.set_ylim(*YL)
axB.set_ylabel("jitter   p95 |deviation|   (A)")
axB.yaxis.set_major_formatter(amp)
axB.grid(color=T["grid"], lw=0.8, zorder=0)
axB.set_axisbelow(True)
axB.tick_params(axis="x", which="both", labelbottom=False, bottom=False)
axB.set_title("B.  Jitter is flat at ~1 quantum, whatever the offset", loc="left",
              fontsize=10.5, fontweight="bold", color=T["ink"], pad=8)

handles = [plt.Line2D([], [], marker="o", ls="", ms=7.5, color=c,
                      markeredgecolor=T["surface"], markeredgewidth=1.4,
                      label=f"{g}  (n={nz_counts[g]+z_counts[g]})")
           for g, c in zip(ORDER, T["s"])]
leg = axB.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5,
                 handletextpad=0.4, borderpad=0.1, labelspacing=0.35)
for t in leg.get_texts():
    t.set_color(T["ink2"])

clean(axZ, left=False)
axZ.set_xscale("log"); axZ.set_xlim(*XL); axZ.set_ylim(-1, 1)
axZ.set_yticks([])
axZ.set_xlabel("fixed calibration offset   |median(READI − SETI)|   (A)")
axZ.xaxis.set_major_formatter(amp)
axZ.grid(axis="x", color=T["grid"], lw=0.8, zorder=0)
axZ.set_axisbelow(True)
nz_tot = sum(z_counts.values())
axZ.text(-0.012, 0.5, f"jitter = 0\n({nz_tot} of {len(per)})", transform=axZ.transAxes,
         ha="right", va="center", color=T["ink2"], fontsize=8, linespacing=1.35,
         fontweight="bold")
# axis break marker between the log panel and the zero strip
for xpos in (0.0, 1.0):
    axZ.plot([xpos, xpos], [1.0, 1.0], transform=axZ.transAxes, clip_on=False)
axZ.annotate("", xy=(0.0, 1.0), xytext=(1.0, 1.0), xycoords="axes fraction",
             arrowprops=dict(arrowstyle="-", color=T["axis"], lw=1,
                             linestyle=(0, (2, 2))))

fig.text(0.062, 0.945, "CLARA magnet set/readback jitter",
         fontsize=13.5, fontweight="bold", color=T["ink"], ha="left")
fig.text(0.062, 0.888,
         f"{len(per)} power supplies · {allres.size:,} repeat readings at an unchanged setpoint · "
         "2026-03-02 to 2026-04-27",
         fontsize=8.5, color=T["ink2"], ha="left")
fig.text(0.062, 0.845,
         "Archiver dropout sentinels, supply-off rows and the 15 PVs that mirror their setpoint are excluded.",
         fontsize=8, color=T["muted"], ha="left")

fig.savefig(OUT, dpi=200)
print(f"wrote {OUT}  ({len(per)} magnets, {allres.size} residuals)")
print(f"  zero-jitter magnets: {nz_tot}/{len(per)}  " +
      ", ".join(f"{g}: {z_counts[g]}/{z_counts[g]+nz_counts[g]}" for g in ORDER))
print(f"  group zero-fractions: " + ", ".join(f"{g} {100*floors[g]:.1f}%" for g in ORDER))
