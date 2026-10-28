"""Talk figures: CLARA line geometry with apertures, and transmission along s for the
12x12 matched-optics nodes. Run from challenges/optimisation after track_along_s.py:
    python talk_figures/scripts/make_figures.py talk_figures
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon, Rectangle

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "RL_framework"), str(ROOT)]
from rl_framework.config import RLConfig
from rl_framework.envs.sim_env import SimEnv
import cheetah

OUT = Path(sys.argv[1])
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.labelcolor": "#222", "xtick.color": "#444",
                     "ytick.color": "#444", "grid.color": "#e5e5e5", "savefig.dpi": 200,
                     "font.family": "DejaVu Sans"})
BLUE = ["#86b6ef", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE)
C = dict(quad="#eb6834", dip="#2a78d6", cav="#eda100", tdc="#6250d6", sext="#008300",
         cor="#1baf7a", scr="#9a9a9a", ink="#222222", muted="#6b6b6b", red="#e34948")

# ---------------- lattice ----------------
env = SimEnv(RLConfig.clara(backend="cheetah", num_particles=1000, apertures=True))
els = env.segment.elements
L = np.array([float(e.length) for e in els]); s1 = np.cumsum(L); s0 = s1 - L

def section(name):
    t = name.split("_")
    sec = (t[1] if t[0] == "cla" else t[0]).upper()
    return "FEH" if sec == "FEH1" else sec
secs = []
for e, a, b in zip(els, s0, s1):
    sec = section(e.name)
    if sec in ("DRIFT", "FMB1", "FEBE"):
        sec = secs[-1][0] if secs else sec
    if secs and secs[-1][0] == sec: secs[-1][2] = b
    else: secs.append([sec, a, b])

ap_s, ap_x, ap_y, ap_shape, ap_name = [], [], [], [], []
for e, a in zip(els, s0):
    if isinstance(e, cheetah.Aperture):
        ap_s.append(a); ap_x.append(float(e.x_max)); ap_y.append(float(e.y_max))
        ap_shape.append(e.shape); ap_name.append(e.name)
ap_s, ap_x, ap_y = map(np.array, (ap_s, ap_x, ap_y))

def kind(e):
    if isinstance(e, cheetah.Quadrupole): return "quad"
    if isinstance(e, cheetah.Dipole): return "dip"
    if isinstance(e, cheetah.TransverseDeflectingCavity): return "tdc"
    if isinstance(e, cheetah.Cavity): return "cav"
    if isinstance(e, cheetah.Sextupole): return "sext"
    if isinstance(e, cheetah.CombinedCorrector): return "cor"
    if isinstance(e, cheetah.Screen) and "_scr_" in e.name: return "scr"
    return None

def lattice_strip(ax, label=True):
    ax.axhline(0, color="#bbb", lw=0.8, zorder=0)
    for e, a, b in zip(els, s0, s1):
        k = kind(e)
        if k is None: continue
        w = max(b - a, 0.12)
        if k == "quad":
            sgn = np.sign(float(e.k1)) or 1
            ax.add_patch(Rectangle((a, 0), w, 0.8 * sgn, color=C[k], lw=0))
        elif k == "dip": ax.add_patch(Rectangle((a, -0.5), w, 1.0, color=C[k], lw=0))
        elif k in ("cav", "tdc"): ax.add_patch(Rectangle((a, -0.35), w, 0.7, color=C[k], lw=0))
        elif k == "sext": ax.add_patch(Rectangle((a, -0.6), w, 1.2, color=C[k], lw=0))
        elif k == "cor": ax.add_patch(Rectangle((a, -0.25), 0.08, 0.5, color=C[k], lw=0))
        elif k == "scr": ax.plot([a, a], [-0.9, 0.9], color=C[k], lw=0.6)
    ax.set_ylim(-1.1, 1.1); ax.set_yticks([])
    for sp in ("left", "bottom"): ax.spines[sp].set_visible(False)
    ax.tick_params(axis="x", length=0)
    if label:
        for i, (n, a, b) in enumerate(secs):
            if b - a > 0.6:
                ax.text((a + b) / 2, 1.25 + 0.45 * (i % 2), n, ha="center", va="bottom",
                        fontsize=8, color=C["muted"], clip_on=False)

def shade_sections(ax):
    for i, (n, a, b) in enumerate(secs):
        if i % 2: ax.axvspan(a, b, color="#f4f4f2", lw=0, zorder=0)

def legend_elements(ax, loc="upper left", ncol=7):
    h = [Rectangle((0, 0), 1, 1, color=C[k]) for k in ("quad", "dip", "cav", "tdc", "sext", "cor")]
    h.append(plt.Line2D([], [], color=C["scr"], lw=1))
    ax.legend(h, ["quadrupole (up: focusing in x)", "dipole", "RF cavity", "TDC", "sextupole",
                  "corrector", "screen"], loc=loc, ncol=ncol, fontsize=8, frameon=False,
              handlelength=1.2, columnspacing=1.0, bbox_to_anchor=(0, -0.05))

# floor plan: integrate bending angles
X, Z, TH = [0.0], [0.0], [0.0]
th = 0.0
pts = []
for e, l in zip(els, L):
    ang = float(e.angle) if isinstance(e, cheetah.Dipole) else 0.0
    nseg = 8 if ang else 1
    for _ in range(nseg):
        th_mid = th + ang / nseg / 2
        Z.append(Z[-1] + l / nseg * np.cos(th_mid)); X.append(X[-1] + l / nseg * np.sin(th_mid))
        th += ang / nseg
    TH.append(th)
Z, X = np.array(Z), np.array(X)
node_s = np.concatenate([[0], np.repeat(s1, 1)])

def floor_xy(s):
    # interpolate along the polyline by path length
    seglen = np.hypot(np.diff(Z), np.diff(X)); cs = np.concatenate([[0], np.cumsum(seglen)])
    return np.interp(s, cs, Z), np.interp(s, cs, X), cs

def figure_geometry(nominal=None):
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(5, 1, height_ratios=[2.2, 0.45, 0.55, 2.0, 2.0], hspace=0.12)
    axf = fig.add_subplot(gs[0]); axl = fig.add_subplot(gs[2])
    axx = fig.add_subplot(gs[3], sharex=axl); axy = fig.add_subplot(gs[4], sharex=axl)
    # floor plan
    z, x, cs = floor_xy(np.array([0]))
    axf.plot(Z, X, color="#bbb", lw=1.2, zorder=1)
    for e, a, b in zip(els, s0, s1):
        k = kind(e)
        if k is None or k == "scr": continue
        sa = np.linspace(a, max(b, a + 0.1), 6)
        zz, xx, _ = floor_xy(sa)
        width = {"quad": 0.12, "dip": 0.22, "cav": 0.16, "tdc": 0.16, "sext": 0.1, "cor": 0.06}[k]
        axf.plot(zz, xx, color=C[k], lw=width * 40, solid_capstyle="butt", zorder=3)
    for i, (n, a, b) in enumerate(secs):
        if b - a > 0.6:
            zz, xx, _ = floor_xy(np.array([(a + b) / 2]))
            axf.annotate(n, (zz[0], xx[0]), xytext=(0, 14 if i % 2 else -18),
                         textcoords="offset points", ha="center", fontsize=8, color=C["muted"])
    axf.set_xlabel("Z [m]"); axf.set_ylabel("X [m]")
    axf.set_title("CLARA FE line, S02 to FEH screen 1 (VBC at 118.5 mrad, FEA dogleg 4 x 14 deg)",
                  loc="left", fontsize=12, color=C["ink"])
    axf.margins(y=0.25)
    legend_elements(axf, loc="lower left", ncol=2)
    axf.get_legend().set_bbox_to_anchor((0, 0.0))

    lattice_strip(axl, label=False)
    plt.setp(axl.get_xticklabels(), visible=False)
    for ax, ap, lab, idx in ((axx, ap_x, "x", 1), (axy, ap_y, "y", 2)):
        shade_sections(ax)
        a = np.minimum(ap * 1e3, 40)
        se = np.append(ap_s, s1[-1]); ae = np.append(a, a[-1])
        ax.fill_between(se, ae, 45, step="post", color="#8c8c8c", alpha=0.9, lw=0)
        ax.fill_between(se, -ae, -45, step="post", color="#8c8c8c", alpha=0.9, lw=0)
        if nominal is not None:
            s_nom, sig = nominal
            ax.fill_between(s_nom, -3 * sig * 1e3, 3 * sig * 1e3, color=C["dip"], alpha=0.35, lw=0,
                            label="nominal beam, ±3σ (survivors)")
            ax.plot(s_nom, 3 * sig * 1e3, color=C["dip"], lw=1); ax.plot(s_nom, -3 * sig * 1e3, color=C["dip"], lw=1)
            ax.legend(loc="lower left", fontsize=8, frameon=False)
        ax.set_ylim(-42, 42); ax.set_ylabel(f"{lab} half-aperture [mm]")
        ax.grid(axis="y", lw=0.5)
        for sec in secs: pass
    # annotate the tight spots
    for nm, txt in (("cla_l4h_sim_aper_01_pipe", "L4H 4.1 mm"), ("cla_s07_sim_tdc_01_aper_pipe", "TDC 5 mm"),
                    ("cla_s02_dia_ecol_01_col_in", "collimator 10 mm")):
        i = ap_name.index(nm)
        for ax in (axx, axy):
            ax.annotate(txt, (ap_s[i], ap_y[i] * 1e3 if ax is axy else min(ap_x[i] * 1e3, 40)),
                        xytext=(0, 22), textcoords="offset points", ha="left" if ap_s[i] < 5 else "center", fontsize=8,
                        color=C["red"], arrowprops=dict(arrowstyle="-", color=C["red"], lw=0.8))
    axx.text(37.6, 30, "x unlimited\n(S07 to FEA)", fontsize=8, color=C["muted"])
    plt.setp(axx.get_xticklabels(), visible=False)
    axy.set_xlabel("s [m]"); axy.set_xlim(0, s1[-1])
    for i, (n, a, b) in enumerate(secs):
        if b - a > 0.6:
            axl.text((a + b) / 2, 1.2, n, ha="center", va="bottom", fontsize=7.5, color=C["muted"])
    return fig

# ---------------- tracking results ----------------
on = np.load(OUT / "data/along_s_vbc_on.npz"); off = np.load(OUT / "data/along_s_vbc_off.npz")
s = on["s"]
grid = np.load(ROOT / "optics_matching/results/validated_grid.npz")
sizes = grid["spot_sizes_m"]; n = len(sizes)

def endT(d):
    T = np.full(n * n, np.nan); T[d["nodes"]] = d["T"][:, -1]; return T.reshape(n, n)

geo = OUT / "geometry"; geo.mkdir(parents=True, exist_ok=True)
fig = figure_geometry(); fig.savefig(geo / "line_geometry_apertures.png", bbox_inches="tight"); plt.close(fig)
fig = figure_geometry(nominal=None)
plt.close(fig)
# with nominal beam envelope in x and y
fig = figure_geometry(nominal=(s, on["nominal"][1]))
ax_y = fig.axes[3]
for c in list(ax_y.collections): pass
plt.close(fig)

def geometry_with_beam():
    fig = figure_geometry()
    axx, axy = fig.axes[2], fig.axes[3]
    for ax, sig in ((axx, on["nominal"][1]), (axy, on["nominal"][2])):
        ax.fill_between(s, -3 * sig * 1e3, 3 * sig * 1e3, color=C["dip"], alpha=0.35, lw=0,
                        label="nominal optics beam, ±3σ of survivors (VBC on)")
        ax.plot(s, 3 * sig * 1e3, color=C["dip"], lw=0.9); ax.plot(s, -3 * sig * 1e3, color=C["dip"], lw=0.9)
        ax.legend(loc="lower left", fontsize=8, frameon=False)
    return fig
fig = geometry_with_beam(); fig.savefig(geo / "line_geometry_apertures_nominal_beam.png", bbox_inches="tight"); plt.close(fig)

# ---- transmission along s, all nodes ----
tr = OUT / "transmission_along_s"; tr.mkdir(parents=True, exist_ok=True)
norm = Normalize(0, 1)
def along_s(d, ref_off, title, fname, zoom=(57.0, None)):
    fig = plt.figure(figsize=(15, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.55, 4], width_ratios=[3.2, 1], hspace=0.05, wspace=0.08)
    axl = fig.add_subplot(gs[0, 0]); ax = fig.add_subplot(gs[1, 0], sharex=axl)
    axz = fig.add_subplot(gs[1, 1], sharey=ax)
    lattice_strip(axl); plt.setp(axl.get_xticklabels(), visible=False)
    order = np.argsort(d["T"][:, -1])[::-1]
    for a in (ax, axz):
        shade_sections(a)
        for k in order:
            a.plot(s, d["T"][k] * 100, color=CMAP(norm(d["T"][k, -1])), lw=0.9, alpha=0.85)
        a.plot(s, d["nominal"][0] * 100, color=C["ink"], lw=2.2,
               label=f"nominal optics, {'VBC on' if float(d['vbc']) else 'VBC off'} ({d['nominal'][0][-1]*100:.1f}%)")
        if ref_off is not None:
            a.plot(s, ref_off["nominal"][0] * 100, color=C["ink"], lw=1.8, ls="--",
                   label=f"nominal optics, VBC off ({ref_off['nominal'][0][-1]*100:.1f}%)")
        a.grid(axis="y", lw=0.5)
    ax.set_ylim(0, 102); ax.set_xlim(0, s[-1]); ax.set_xlabel("s [m]"); ax.set_ylabel("transmission [%]")
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    ax.axvspan(zoom[0], s[-1], color=C["red"], alpha=0.06, lw=0)
    axz.set_xlim(zoom[0], s[-1]); axz.set_xlabel("s [m] (zoom: FEA exit to FEH screen)")
    plt.setp(axz.get_yticklabels(), visible=False)
    for nm, x0, x1 in secs:
        if x1 > zoom[0] and nm in ("FEA", "FEH"):
            axz.text(max(x0, zoom[0]) + 0.1, 101, nm, fontsize=8, color=C["muted"], va="bottom")
    axz.annotate("node losses here: 17 mm pipe\nat shutter, FEH wall, FMB1 mirror", (57.3, 8), fontsize=8, color=C["muted"])
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 100), cmap=CMAP)
    cb = fig.colorbar(sm, ax=[axl, ax, axz], pad=0.01, fraction=0.02)
    cb.set_label("node: final transmission [%]")
    axl.set_title(title, loc="left", fontsize=12, color=C["ink"], pad=26)
    fig.savefig(tr / fname, bbox_inches="tight"); plt.close(fig)

along_s(on, off, "Transmission along the line, 144 matched-optics nodes (VBC on, 2nd-order Cheetah)", "transmission_along_s_vbc_on.png")
along_s(off, None, "Transmission along the line, 144 matched-optics nodes (VBC off, 2nd-order Cheetah)", "transmission_along_s_vbc_off.png")

# ---- 12x12 small multiples ----
def small_multiples(d, ref, title, fname):
    T = d["T"]; idx = {k: i for i, k in enumerate(d["nodes"])}
    fig, axs = plt.subplots(n, n, figsize=(16, 15), sharex=True, sharey=True)
    for i in range(n):          # sigma_x index -> column
        for j in range(n):      # sigma_y index -> row (bottom = small)
            ax = axs[n - 1 - j, i]
            k = idx.get(i * n + j)
            ax.plot(s, ref["nominal"][0] * 100, color="#bbbbbb", lw=0.8)
            if k is not None:
                t = T[k]
                ax.fill_between(s, 0, t * 100, color=CMAP(norm(t[-1])), lw=0, alpha=0.9)
                ax.text(0.96, 0.08, f"{t[-1]*100:.0f}", transform=ax.transAxes, ha="right",
                        fontsize=9, color="white" if t[-1] > 0.45 else C["ink"], weight="bold")
            ax.set_ylim(0, 100); ax.set_xlim(0, s[-1])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            if j == 0: ax.set_xlabel(f"{sizes[i]*1e3:.2g}", fontsize=9)
            if i == 0: ax.set_ylabel(f"{sizes[j]*1e3:.2g}", fontsize=9, rotation=0, ha="right", va="center")
    fig.supxlabel("target σx at FEH screen [mm]  (each panel: transmission 0–100% vs s, 0–64 m; grey = nominal optics)", fontsize=11)
    fig.supylabel("target σy [mm]", fontsize=11)
    fig.suptitle(title, x=0.05, ha="left", fontsize=13)
    fig.subplots_adjust(wspace=0.06, hspace=0.06, left=0.07, bottom=0.05, top=0.95)
    fig.savefig(tr / fname, bbox_inches="tight"); plt.close(fig)

small_multiples(on, on, "Transmission along s for each node of the 12×12 grid (VBC on); number = final %", "transmission_along_s_grid_vbc_on.png")
small_multiples(off, off, "Transmission along s for each node of the 12×12 grid (VBC off); number = final %", "transmission_along_s_grid_vbc_off.png")

# ---- where the beam is lost: losses per section ----
def losses_by_section(d):
    lost = -np.diff(np.concatenate([np.ones((len(d["T"]), 1)), d["T"]], axis=1), axis=1)
    nom = -np.diff(np.concatenate([[1.0], d["nominal"][0]]))
    sec_of = np.empty(len(s), dtype=object)
    for nme, a, b in secs: sec_of[(s > a - 1e-9) & (s <= b + 1e-9)] = nme
    names = [x[0] for x in secs]
    uniq = list(dict.fromkeys(names))
    per = np.array([[lost[k][sec_of == u].sum() for u in uniq] for k in range(len(lost))])
    pn = np.array([nom[sec_of == u].sum() for u in uniq])
    return uniq, per, pn
uniq, per, pn = losses_by_section(on)
keep = [i for i, u in enumerate(uniq) if per[:, i].max() > 1e-3 or pn[i] > 1e-3]
fig, ax = plt.subplots(figsize=(11, 4.6))
xs = np.arange(len(keep))
med = np.median(per[:, keep], axis=0) * 100
lo, hi = np.percentile(per[:, keep], [10, 90], axis=0) * 100
ax.bar(xs - 0.2, pn[keep] * 100, 0.38, color=C["ink"], label="nominal optics")
ax.bar(xs + 0.2, med, 0.38, color=C["dip"], label="144 nodes: median (bar), 10–90% (line)")
ax.vlines(xs + 0.2, lo, hi, color=C["dip"], lw=1.5)
ax.set_xticks(xs); ax.set_xticklabels([uniq[i] for i in keep])
ax.set_ylabel("beam lost in section [% of bunch]"); ax.grid(axis="y", lw=0.5)
ax.legend(frameon=False); ax.set_title("Where the beam is lost (VBC on, 2nd order)", loc="left")
fig.savefig(tr / "losses_by_section_vbc_on.png", bbox_inches="tight"); plt.close(fig)

# ---- summary numbers ----
Ton, Toff = endT(on), endT(off)
summ = dict(
    nominal_vbc_on=float(on["nominal"][0][-1]), nominal_vbc_off=float(off["nominal"][0][-1]),
    nodes_vbc_on=dict(min=float(np.nanmin(Ton)), median=float(np.nanmedian(Ton)), max=float(np.nanmax(Ton)),
                      n_above_90=int((Ton > 0.9).sum()), n_above_50=int((Ton > 0.5).sum())),
    nodes_vbc_off=dict(min=float(np.nanmin(Toff)), median=float(np.nanmedian(Toff)), max=float(np.nanmax(Toff)),
                       n_above_90=int((Toff > 0.9).sum()), n_above_50=int((Toff > 0.5).sum())),
    losses_by_section_nominal_vbc_on={u: float(p) for u, p in zip(uniq, pn) if p > 1e-4},
    losses_by_section_nodes_median_vbc_on={u: float(np.median(per[:, i])) for i, u in enumerate(uniq) if np.median(per[:, i]) > 1e-4},
    worst_nodes_vbc_on=[dict(sx_mm=float(sizes[i]*1e3), sy_mm=float(sizes[j]*1e3), T=float(Ton[i, j]))
                        for i, j in zip(*np.unravel_index(np.argsort(Ton, axis=None)[:5], Ton.shape))],
    tightest_apertures_mm=sorted({(nm, round(x*1e3, 2), round(y*1e3, 2)) for nm, x, y in zip(ap_name, ap_x, ap_y) if min(x, y) < 0.0099}),
    vbc_rad_on=float(on["vbc"]), vbc_rad_off=float(off["vbc"]), order=str(on["order"]), line_length_m=float(s[-1]),
)
(OUT / "data/summary_along_s.json").write_text(json.dumps(summ, indent=2))
print(json.dumps(summ, indent=2))
