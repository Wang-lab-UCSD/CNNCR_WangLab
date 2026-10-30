"""
Figure 4 — Core motifs mark functional variants, are broadly occupied, generalize
to a held-out cell line, and predict cell-type-specific expression.

Individual subplots only (the user assembles the montage in Illustrator). Every panel
uses the SAME Nature house style as make_fig3_panels.py:
  palette  Core = orange #eb6834, JASPAR = blue #5b8db8, Control/background = grey #b8b6ae
  violins + jittered individual points + inner IQR box + white median marker
  exact Mann-Whitney / Wilcoxon P-values PRINTED on-panel (no significance stars)

Panel map (new layout, 2026-07-17):
  4a  wide   eQTL: causal-SNP overlap COUNT per biosample, core vs control (16 biosamples)
  4b  square eQTL: core-vs-control overlap-RATIO paired scatter (16 biosamples)
  (4c = IGV genome-browser locus — user-supplied screenshot, not generated here)
  4d  ChIP occupancy, two sub-panels: (i) pooled core vs control %, (ii) per-cell-line medians
  4e  cross-cell-type Pearson r by variance tercile (motif vs naive)
  4f  HEK293 held-out: 116-line test-Spearman distribution + HEK star, per model class
  4g  marker genes (8, 2x4): actual vs motif-model expression across 116 biosamples

Data sources:
  4a/4b enformer_snp_overlap/results/SNP_overlap.csv + wilcoxon_stats.csv
  4d    chipseq_enrichment/results/enrichment_summary.csv + results/enrichment/{cell}_enrichment.csv
  4e    proof1_cell_type_specificity/results/per_gene_correlations_zscored.csv
  4f    posthoc_analysis/results/linear_rf/correlations.csv + results/cnn/*_universal_test_S.npy
        HEK held-out values: Linear 0.6592, RF 0.6664, CNN 0.7280
  4g    training_y/{bssid}.y.npy + posthoc_analysis/results/count_matrices_universal/{bssid}_count_matrix.npy
        (5-fold RidgeCV recompute, identical to proof1 01_gene_wise_correlation.py)
"""
import os, glob
import glob as _glob
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon, pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Single font name (not "sans-serif"): matplotlib writes the resolved name into the
# SVG, and a generic family makes PowerPoint substitute. Arial is absent on this box
# so PDF/PNG rasterize with the DejaVu fallback, but the SVG asks for real Arial.
# Register the Liberation Sans copies shipped in figures/fonts/ so the panels do not depend on a
# system Arial. Many system "Arial" builds lack the Unicode superscript glyphs (U+207B, U+2070), which
# is why exponents are drawn with mathtext below rather than with superscript characters.
from matplotlib import font_manager as _fm
# The data root: the work directory you created with bin/setup_work_dir.py and unpacked the
# GitHub and Zenodo data into. `source env.sh` exports it, or set it yourself (see PATHS.md).
ENHANCER_ROOT = os.environ.get("ENHANCER_ROOT")
if not ENHANCER_ROOT:
    raise SystemExit("set ENHANCER_ROOT to your work directory: source env.sh (see PATHS.md)")

_FONTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")
for _f in _glob.glob(os.path.join(_FONTS, "LiberationSans-*.ttf")): _fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
                     "mathtext.default": "regular", "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})

SD  = os.path.dirname(os.path.abspath(__file__))
EP  = ENHANCER_ROOT
MA  = f"{EP}/multi_agent"
SNP = f"{MA}/enformer_snp_overlap/results/SNP_overlap.csv"
WLX = f"{MA}/enformer_snp_overlap/results/wilcoxon_stats.csv"
CHIP_SUM = f"{MA}/chipseq_enrichment/results/enrichment_summary.csv"   # single-control version (superseded)
CHIP_DIR = f"{MA}/chipseq_enrichment/results/enrichment"               # single-control version (superseded)
# TEN control sets (2026-09-05): 4d now uses the same ten old-pool control sets as 4a/4b,
# scored from one pool-wide FIMO scan by chipseq_enrichment/10_oldpool_10sets.py. (i) pools the
# 141 distinct control motifs (each once); (ii) shows the mean over the ten sets of the per-set
# median with an s.d. error bar, as 4a does.
OP10 = f"{MA}/chipseq_enrichment/results/oldpool10"
CHIP_PM10, CHIP_PC10, CHIP_ST10 = (f"{OP10}/fig4d_permotif.csv", f"{OP10}/fig4d_per_cell.csv",
                                   f"{OP10}/fig4d_stats.csv")
# Avsec-standardized correlations + biosample-label permutation null (04_final_specificity.py).
# The old per_gene_correlations.csv used raw expression and a gene-mean "naive"
# baseline that is negative by construction; both were replaced 2026-08-12.
PGC = f"{MA}/proof1_cell_type_specificity/results/per_gene_correlations_zscored.csv"
LRF = f"{MA}/posthoc_analysis/results/linear_rf/correlations.csv"
CNN_DIR = f"{MA}/posthoc_analysis/results/cnn"

CORE, JASP, CTRL = "#eb6834", "#5b8db8", "#b8b6ae"
# one y range for both halves of panel e (HEK293 and RA/OA) so the two
# external validations are read off the same scale
EXT_YLIM = (0.42, 0.88)
INK, SUB, MUT, GRID, BASE = "#0b0b0b", "#52514e", "#8f8d88", "#e8e7e1", "#c3c2b7"
RNG = np.random.default_rng(7)

# ── p-value formatting (identical to fig3) ──────────────────────────────────────
def fmtp(p):
    """Exponents as mathtext ($10^{-5}$), not Unicode superscripts: a system Arial without
    U+207B renders those as empty boxes (the numbers stay right, the label does not)."""
    if p <= 0 or p < 1e-300: return "P < $10^{-300}$"
    if p >= 0.05:  return f"P = {p:.2f}"
    if p >= 1e-3:  return f"P = {p:.3f}".rstrip("0")
    e = int(np.floor(np.log10(p))); m = p / 10**e
    return f"P = {m:.1f}\u00d7$10^{{{e}}}$"

def chrome(ax, ylab):
    ax.set_ylabel(ylab, fontsize=6, color=SUB)
    ax.tick_params(axis="y", labelsize=5.5, colors=MUT, length=0)
    ax.tick_params(axis="x", labelsize=5.5, colors=INK, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(BASE); ax.spines[s].set_linewidth(0.9)
def title(ax, t): ax.set_title(t, fontsize=7, color=INK, fontweight="bold", pad=6)
def bracket(ax, x1, x2, y, dy, txt):
    ax.plot([x1, x1, x2, x2], [y, y+dy, y+dy, y], lw=0.9, color=MUT, clip_on=False, zorder=8)
    ax.text((x1+x2)/2, y+dy, " " + txt, ha="center", va="bottom", fontsize=5,
            color=INK, zorder=8)
def strip(ax, i, vals, color, maxpts=220):
    v = vals if len(vals) <= maxpts else RNG.choice(vals, maxpts, replace=False)
    jit = RNG.normal(0, 0.055, size=len(v)); jit = np.clip(jit, -0.17, 0.17)
    ax.scatter(i + jit, v, s=6.5, color=color, alpha=0.45, edgecolor="white",
               lw=0.15, zorder=4, rasterized=True)
def innerbox(ax, i, vals):
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    ax.plot([i, i], [q1, q3], color=INK, lw=4.0, solid_capstyle="round", zorder=6, alpha=0.85)
    ax.scatter(i, med, s=26, facecolor="white", edgecolor=INK, lw=1.3, zorder=7)
def draw_violins(ax, data, cols, labels, ylab, ttl, comps, maxpts=220):
    pos = list(range(len(data)))
    parts = ax.violinplot(data, positions=pos, widths=0.84, showextrema=False)
    for b, c in zip(parts["bodies"], cols):
        b.set_facecolor(c); b.set_edgecolor(c); b.set_alpha(0.28); b.set_linewidth(1.0); b.set_zorder(2)
    for i, (d, c) in enumerate(zip(data, cols)):
        strip(ax, i, d, c, maxpts); innerbox(ax, i, d)
    ax.set_xticks(pos); ax.set_xticklabels(labels)
    chrome(ax, ylab); title(ax, ttl)
    allv = np.concatenate(data); lo, hi = allv.min(), allv.max(); span = hi - lo
    dy = span * 0.028; ytop = hi + span * 0.07
    for (i, j, p) in comps:
        bracket(ax, i, j, ytop, dy, fmtp(p)); ytop += span * 0.145
    ax.set_xlim(-0.6, len(data)-0.4); ax.set_ylim(lo - span*0.05, ytop + span*0.02)

def save(fig, name):
    # NO bbox_inches="tight": it re-crops to the drawn content and returns some
    # other size than figsize, so panels no longer match their slot on the
    # assembled sheet and get rescaled, which drags the type under 5 pt.
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{SD}/{name}.{ext}", dpi=600, transparent=True)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 4a — eQTL causal-SNP overlap count per biosample: core vs control (WIDE)
# ══════════════════════════════════════════════════════════════════════════════
def panel_4a():
    df = pd.read_csv(SNP).copy()
    df = df[df["num_pos_SNP"] >= 500].copy()          # 16 biosamples with enough causal variants
    df = df.sort_values("SNP_in_universal_pos", ascending=False).reset_index(drop=True)
    core = df["SNP_in_universal_pos"].values.astype(float)
    ctrl = df["SNP_in_control_pos_mean"].values.astype(float)
    ctrl_sd = df["SNP_in_control_pos_std"].values.astype(float)
    _AB = {"Esophagus_Gastroesophageal_Junction": "Esophagus GJ",
           "Skin_Sun_Exposed_Lower_leg": "Skin (sun exp.)",
           "Adipose_Visceral_Omentum": "Adipose visceral",
           "Adipose_Subcutaneous": "Adipose subcut.",
           "Breast_Mammary_Tissue": "Breast mammary",
           "Esophagus_Mucosa": "Esophagus mucosa",
           "Colon_Transverse": "Colon transverse",
           "Colon_Sigmoid": "Colon sigmoid",
           "Adrenal_Gland": "Adrenal gland",
           "Nerve_Tibial": "Nerve tibial", "Artery_Aorta": "Artery aorta"}
    labels = [_AB.get(t, t.replace("_", " ")) for t in df["tissue"]]
    n = len(df); x = np.arange(n); w = 0.40
    p = wilcoxon(core, ctrl, alternative="greater").pvalue     # paired core > control

    fig, ax = plt.subplots(figsize=(4.30, 1.95))
    ax.bar(x - w/2, core, w, color=CORE, edgecolor="white", lw=0.3, zorder=3, label="Core (43)")
    ax.bar(x + w/2, ctrl, w, color=CTRL, edgecolor="white", lw=0.3, zorder=3, label="Control (43)")
    ax.errorbar(x + w/2, ctrl, yerr=ctrl_sd, fmt="none", ecolor=MUT, elinewidth=0.7,
                capsize=1.5, capthick=0.7, zorder=4)
    chrome(ax, "Causal eQTLs in motif sites")
    title(ax, "Causal-variant overlap across biosamples")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=5.5)
    ax.set_xlim(-0.7, n-0.3)
    ax.legend(fontsize=5, frameon=False, loc="upper right", handlelength=1.0,
              labelcolor=INK, borderaxespad=0.2, handletextpad=0.5)
    ax.text(0.55, 0.90, f"core > control · {fmtp(p)}\n(paired Wilcoxon, n = {n} biosamples)",
            transform=ax.transAxes, ha="center", va="top", fontsize=5,
            color=INK)
    fig.subplots_adjust(left=0.115, right=0.995, top=0.865, bottom=0.44)
    save(fig, "fig4a_eqtl_count")
    print(f"4a: core median {np.median(core):.0f} vs control {np.median(ctrl):.0f}, {fmtp(p)}")

# ══════════════════════════════════════════════════════════════════════════════
# 4b — eQTL overlap-ratio paired scatter: core vs control (SQUARE)
# ══════════════════════════════════════════════════════════════════════════════
def panel_4b():
    df = pd.read_csv(SNP).copy()
    df = df[df["num_pos_SNP"] >= 500].copy()          # 16 biosamples with enough causal variants
    uni = (df["SNP_in_universal_pos"] / df["total_universal"]).values.astype(float) * 1e3   # per-mille
    # per-biosample control ratio = mean over the 10 control sets
    setcols = [(f"ctrl_pos_set_{s}", f"ctrl_total_set_{s}") for s in range(10)]
    setcols = [(p, t) for p, t in setcols if p in df.columns and t in df.columns]
    ctrl_mat = np.column_stack([df[p].values / df[t].values for p, t in setcols]) * 1e3
    ctrl_ratio = ctrl_mat.mean(axis=1)
    ctrl_sd = ctrl_mat.std(axis=1, ddof=1)           # s.d. over the ten control sets (as 4a)
    p = wilcoxon(uni, ctrl_ratio, alternative="greater").pvalue
    n = len(uni); nabove = int((uni > ctrl_ratio).sum())
    lo = 0.0; hi = max(uni.max(), (ctrl_ratio + ctrl_sd).max()) * 1.10

    fig, ax = plt.subplots(figsize=(2.17, 1.95))
    ax.fill_between([lo, hi], [lo, hi], hi, color=CORE, alpha=0.06, zorder=1)   # core>control field
    ax.plot([lo, hi], [lo, hi], color=MUT, lw=0.9, ls="--", zorder=2)
    # horizontal bar = s.d. of the control ratio over the ten sets, behind the markers
    ax.errorbar(ctrl_ratio, uni, xerr=ctrl_sd, fmt="none", ecolor="#6f6d68", elinewidth=0.5,
                capsize=1.1, capthick=0.5, alpha=0.9, zorder=3)
    ax.scatter(ctrl_ratio, uni, s=22, color=CORE, edgecolor="white", lw=0.4, alpha=0.9, zorder=4)
    ax.set_xlabel("Control overlap ratio (×10\u207b\u00b3)\nmean ± s.d. of ten sets", fontsize=6, color=SUB)
    chrome(ax, "Core overlap ratio (×10\u207b\u00b3)")
    title(ax, "Causal-variant enrichment")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
    ax.text(0.955, 0.06, f"{nabove} / {n} biosamples\ncore > control\n{fmtp(p)}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5,
            color=INK)
    fig.subplots_adjust(left=0.215, right=0.985, top=0.865, bottom=0.235)
    save(fig, "fig4b_eqtl_ratio")
    print(f"4b: {nabove}/{n} core>control, {fmtp(p)}")

# ══════════════════════════════════════════════════════════════════════════════
# 4d — ChIP-seq occupancy: pooled (i) + per-cell-line medians (ii)
# ══════════════════════════════════════════════════════════════════════════════
CELLS = ["HepG2", "K562", "GM12878", "MCF.7", "A549"]
CELL_LABEL = {"HepG2": "HepG2", "K562": "K562", "GM12878": "GM12878", "MCF.7": "MCF-7", "A549": "A549"}
def panel_4d():
    s = pd.read_csv(CHIP_PM10); st = pd.read_csv(CHIP_ST10).set_index("set")
    core = s[s.group == "core"]["mean_pct"].dropna().values
    ctrl = s[s.group == "control"]["mean_pct"].dropna().values
    p_pool = mannwhitneyu(core, ctrl, alternative="greater").pvalue
    ps = st.loc[[str(i) for i in range(10)], "p"].astype(float)
    print(f"4d: per-set P range {ps.min():.1e}-{ps.max():.1e}, {(ps < 0.05).sum()}/10 sets P < 0.05; "
          f"per-set fold {st.loc[[str(i) for i in range(10)], 'fold'].astype(float).min():.2f}-"
          f"{st.loc[[str(i) for i in range(10)], 'fold'].astype(float).max():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(3.15, 1.85), gridspec_kw={"width_ratios": [1.0, 1.5]})
    # (i) pooled: 43 core vs the 141 distinct control motifs of the ten sets
    ax = axes[0]
    draw_violins(ax, [core, ctrl], [CORE, CTRL], ["Core", "Control"],
                 "% of TF peaks with motif", "Pooled", [(0, 1, p_pool)], maxpts=60)
    # (ii) per-cell-line median %: core vs mean over ten control sets (bar = s.d. over sets)
    ax = axes[1]
    pc = pd.read_csv(CHIP_PC10).set_index("cell").loc[CELLS]
    med_core, med_ctrl, sd_ctrl = pc["core_median"].values, pc["ctrl_mean"].values, pc["ctrl_sd"].values
    xc = np.arange(len(CELLS)); w = 0.38
    ax.bar(xc - w/2, med_core, w, color=CORE, edgecolor="white", lw=0.3, zorder=3, label="Core")
    ax.bar(xc + w/2, med_ctrl, w, color=CTRL, edgecolor="white", lw=0.3, zorder=3, label="Control")
    ax.errorbar(xc + w/2, med_ctrl, yerr=sd_ctrl, fmt="none", ecolor=MUT, elinewidth=0.7,
                capsize=1.5, capthick=0.7, zorder=4)
    for k, (a, b, e) in enumerate(zip(med_core, med_ctrl, sd_ctrl)):
        ax.text(xc[k], max(a, b + e) + 0.15, f"{a/b:.1f}×", ha="center", va="bottom",
                fontsize=5, color=INK)
    print("4d per cell line: " + ", ".join(f"{c} {a/b:.2f}x (sets {pc.loc[c,'ctrl_min']:.2f}-{pc.loc[c,'ctrl_max']:.2f}%)"
                                          for c, a, b in zip(CELLS, med_core, med_ctrl)))
    chrome(ax, "median % of TF peaks"); title(ax, "Per cell line")
    ax.set_xticks(xc); ax.set_xticklabels([CELL_LABEL[c] for c in CELLS], rotation=30, ha="right", fontsize=5.5)
    ax.set_xlim(-0.6, len(CELLS)-0.4)
    ax.set_ylim(0, max(med_core) * 1.34)          # headroom so fold labels stay clear
    # legend below the axis (under the rotated cell-line labels) — inside the axes
    # it overlapped the A549 bars
    fig.legend(handles=[Patch(facecolor=CORE, label="Core"), Patch(facecolor=CTRL, label="Control")],
               fontsize=5, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.72, -0.005), columnspacing=0.8,
               labelcolor=INK, handletextpad=0.5, handlelength=1.0)
    fig.subplots_adjust(left=0.100, right=0.99, top=0.865, bottom=0.245, wspace=0.38)
    # (i) needs no room for rotated tick labels, so extend it down to bottom=0.165,
    # the baseline of the panel-e figures — its axis then bottom-aligns with them
    # (and with (ii)'s label block) in the assembled montage
    p0 = axes[0].get_position()
    axes[0].set_position([p0.x0, 0.165, p0.width, 0.865 - 0.165])
    save(fig, "fig4d_chipseq_occupancy")
    print(f"4d: pooled core {np.median(core):.2f}% vs ctrl {np.median(ctrl):.2f}% "
          f"({np.median(core)/np.median(ctrl):.2f}x), {fmtp(p_pool)}")

# ══════════════════════════════════════════════════════════════════════════════
# 4e — cross-cell-type Pearson r by variance tercile (motif vs permutation null)
# ══════════════════════════════════════════════════════════════════════════════
def panel_4e():
    """Left: all 116 biosamples, core motifs against the permutation null.
    Right: the 59 biosamples Enformer covers, core motifs against Enformer,
    both scored on identical genes and biosamples. The two halves are separate
    experiments on different subsets and are deliberately drawn apart."""
    df = pd.read_csv(PGC)
    e59 = pd.read_csv(f"{MA}/proof1_cell_type_specificity/results/per_gene_enformer59.csv")
    e59 = e59[e59.valid]
    order = ["High", "Medium", "Low"]
    fig, axes = plt.subplots(1, 2, figsize=(3.94, 1.85), gridspec_kw=dict(wspace=0.42))
    xs = np.arange(len(order)); w = 0.46

    def pair(ax, series, cols, labels, note):
        for k, terc in enumerate(order):
            vals = [s[k] for s in series]
            for xi, v, col in zip((k - w/2, k + w/2), vals, cols):
                bp = ax.boxplot([v], positions=[xi], widths=w*0.85, showfliers=False,
                                patch_artist=True, zorder=3)
                bp["boxes"][0].set(facecolor=col, edgecolor=col, alpha=0.35, linewidth=0.9)
                for e in ("whiskers", "caps"):
                    for ln in bp[e]:
                        ln.set(color=col, linewidth=0.8)
                bp["medians"][0].set(color=INK, linewidth=1.4)
                # median printed for BOTH comparison groups, not just the first
                ax.text(xi, 1.03, f"{np.median(v):.2f}", ha="center", va="bottom",
                        fontsize=5, color=INK)
        ax.axhline(0, color=MUT, lw=0.8, ls=":", zorder=1)
        chrome(ax, "cross-cell-type r" if ax is axes[0] else "")
        title(ax, note)
        ax.set_xticks(xs)
        ax.set_xticklabels(["High", "Med.", "Low"], fontsize=TK if "TK" in globals() else 5.5)
        ax.set_xlabel("variance tercile", fontsize=6, color=SUB, labelpad=1.5)
        ax.set_xlim(-0.6, len(order)-0.4); ax.set_ylim(-0.75, 1.20)
        ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(cols, labels)],
                  fontsize=5, frameon=False, loc="lower left", handlelength=1.0,
                  labelcolor=INK, borderaxespad=0.2, handletextpad=0.5)

    pair(axes[0],
         [[df[df.variance_tercile == t].pearson_r_motif.dropna().values for t in order],
          [df[df.variance_tercile == t].pearson_r_permutation_null.dropna().values for t in order]],
         [CORE, CTRL], ["43 core motifs", "Permutation null"], "Cell-type specificity")
    pair(axes[1],
         [[e59[e59.tercile == t].r_core43.dropna().values for t in order],
          [e59[e59.tercile == t].r_enformer.dropna().values for t in order]],
         [CORE, JASP], ["43 core motifs", "Enformer"],
         "Versus Enformer")
    fig.subplots_adjust(left=0.115, right=0.99, top=0.865, bottom=0.215, wspace=0.34)
    save(fig, "fig4e_crosscell_tercile")
    print("4e left  (116): ", [round(df[df.variance_tercile==t].pearson_r_motif.median(), 3) for t in order])
    print("4e right  (59): core",
          [round(e59[e59.tercile==t].r_core43.median(), 3) for t in order],
          "| enformer", [round(e59[e59.tercile==t].r_enformer.median(), 3) for t in order])

# ══════════════════════════════════════════════════════════════════════════════
# 4e (right) — RA/OA synoviocytes: the second external validation, drawn as the
# twin of the HEK293 panel so the two read as one lettered panel
# ══════════════════════════════════════════════════════════════════════════════
RAOA = f"{MA}/raoa_external_prediction/results"


def panel_4raoa():
    import glob
    lrf = pd.read_csv(LRF)
    pat = pd.read_csv(f"{RAOA}/linear_rf_per_patient.csv")
    cnn_ep = np.array([float(np.load(f)) for f in
                       sorted(glob.glob(f"{CNN_DIR}/*_universal_test_S.npy"))])
    # All 20 patients live in the tuning summary, not in results/cnn/*.npy (which
    # holds a single patient). Column "tuned" is the one the published RA/OA
    # figure uses. NB the neighbouring "testmax" column is known-edited - do not use it.
    cnn_ra = pd.read_csv(f"{RAOA}/cnn/tune/PANEL_SUMMARY.csv")["tuned"].values
    dists = [("Linear", lrf["linear_universal_test"].dropna().values, pat.linear_test.values),
             ("RF", lrf["rf_universal_test"].dropna().values, pat.rf_test.values),
             ("CNN-CR", cnn_ep, cnn_ra)]
    fig, ax = plt.subplots(figsize=(1.73, 1.85))
    for i, (name, ep, ra) in enumerate(dists):
        if len(ep):
            parts = ax.violinplot([ep], positions=[i], widths=0.62, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor(JASP); b.set_edgecolor(JASP); b.set_alpha(0.28)
                b.set_linewidth(1.0); b.set_zorder(2)
            # EpiMap drawn exactly as in the HEK panel: blue jittered biosamples
            # + black IQR bar + white median dot, so the two halves of e read uniformly
            jit = RNG.normal(0, 0.045, size=len(ep)); jit = np.clip(jit, -0.13, 0.13)
            ax.scatter(i + jit, ep, s=6, color=JASP, alpha=0.40, edgecolor="none",
                       zorder=3, rasterized=True)
            q1, med_ep, q3 = np.percentile(ep, [25, 50, 75])
            ax.plot([i, i], [q1, q3], color=INK, lw=3.2, solid_capstyle="round",
                    zorder=5, alpha=0.85)
            ax.scatter(i, med_ep, s=20, facecolor="white", edgecolor=INK, lw=1.2, zorder=6)
        if len(ra):
            j = RNG.normal(0, 0.045, size=len(ra)); j = np.clip(j, -0.12, 0.12)
            ax.scatter(i + 0.32 + j, ra, s=5, color=CORE, alpha=0.55,
                       edgecolor="none", zorder=4)
            ax.plot([i + 0.20, i + 0.44], [np.median(ra)] * 2, color=CORE, lw=1.4, zorder=7)
            ax.text(i + 0.32, np.median(ra) - 0.030, f"{np.median(ra):.2f}", ha="center",
                    va="top", fontsize=5, color=CORE)
        print(f"   raoa {name}: EpiMap {np.median(ep):.3f}  RA/OA {np.median(ra):.3f}  "
              f"delta {np.median(ra)-np.median(ep):+.3f}  (n={len(ra)})")
    chrome(ax, "held-out test Spearman \u03c1"); title(ax, "RA/OA patients")
    ax.set_xticks(range(len(dists)))
    ax.set_xticklabels([n for n, _, _ in dists], fontsize=5.5)
    ax.set_xlim(-0.55, len(dists) - 0.25); ax.set_ylim(*EXT_YLIM)
    ax.text(0.02, 0.095, "\u25cf 20 patients",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=5, color=CORE)
    ax.text(0.02, 0.02, "\u25cf 116 EpiMap data",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=5, color=JASP)
    fig.subplots_adjust(left=0.255, right=0.985, top=0.865, bottom=0.165)
    save(fig, "fig4e2_raoa")

# ══════════════════════════════════════════════════════════════════════════════
# 4f — HEK293 held-out: 116-line test-Spearman distribution + HEK star per model
# ══════════════════════════════════════════════════════════════════════════════
HEK = {"Linear": 0.6592, "Random forest": 0.6664, "CNN-CR": 0.7280}   # published values
_HK = f"{MA}/hek_universal_prediction/results"
if os.path.exists(f"{_HK}/hek_linear_rf.npz") and os.path.exists(f"{_HK}/cnn/hek_universal_test_S.npy"):   # notebook 09 results, when present
    _z = np.load(f"{_HK}/hek_linear_rf.npz"); HEK = {"Linear": float(_z["linear_test"]), "Random forest": float(_z["rf_test"]), "CNN-CR": float(np.load(f"{_HK}/cnn/hek_universal_test_S.npy"))}
def panel_4f():
    lrf = pd.read_csv(LRF)
    lin = lrf["linear_universal_test"].dropna().values
    rf  = lrf["rf_universal_test"].dropna().values
    cnn = np.array([float(np.load(f)) for f in sorted(glob.glob(f"{CNN_DIR}/*_universal_test_S.npy"))])
    dists = [("Linear", lin), ("Random forest", rf), ("CNN-CR", cnn)]

    fig, ax = plt.subplots(figsize=(1.73, 1.85))
    for i, (name, arr) in enumerate(dists):
        parts = ax.violinplot([arr], positions=[i], widths=0.66, showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(JASP); b.set_edgecolor(JASP); b.set_alpha(0.28); b.set_linewidth(1.0); b.set_zorder(2)
        jit = RNG.normal(0, 0.045, size=len(arr)); jit = np.clip(jit, -0.13, 0.13)
        ax.scatter(i + jit, arr, s=6, color=JASP, alpha=0.40, edgecolor="none", zorder=3, rasterized=True)
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        ax.plot([i, i], [q1, q3], color=INK, lw=3.2, solid_capstyle="round", zorder=5, alpha=0.85)
        ax.scatter(i, med, s=20, facecolor="white", edgecolor=INK, lw=1.2, zorder=6)
        hek = HEK[name]; pct = (arr < hek).mean() * 100
        ax.scatter(i, hek, marker="*", s=130, color=CORE, edgecolor="white", lw=0.8, zorder=8)
        ax.text(i, hek + 0.034, f"{hek:.2f}", ha="center", va="bottom",
                fontsize=5, color=CORE)
        # percentile is stated in the text and legend; on a 48 mm panel it lands
        # on the star and the IQR bar, so it is not repeated here
    chrome(ax, "held-out test Spearman ρ"); title(ax, "Held-out HEK293")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Linear", "RF", "CNN-CR"], fontsize=5.5)
    ax.set_xlim(-0.60, 2.60); ax.set_ylim(*EXT_YLIM)
    ax.text(0.02, 0.095, "★ HEK293",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=5, color=CORE)
    ax.text(0.02, 0.02, "● 116 EpiMap data",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=5, color=JASP)
    fig.subplots_adjust(left=0.255, right=0.985, top=0.865, bottom=0.165)
    save(fig, "fig4f_hek_heldout")
    print("4f: CNN median", round(float(np.median(cnn)), 3), "HEK", HEK["CNN-CR"])

# ══════════════════════════════════════════════════════════════════════════════
# 4g — marker genes (8, 2x4): actual vs motif-model across 116 biosamples
# ══════════════════════════════════════════════════════════════════════════════
_ZR = None   # panel g (LIN28A trace) is outside the repository scope

MARKERS = [  # (symbol, gene_idx, tissue, expected-high keyword list)
    ("POU5F1", 16898, "Pluripotency (OCT4)",
        ["h1-hesc", "h7-hesc", "hues64", "ucsf-4", "gm23248", "mesenchymal_stem_cell",
         "mesendoderm", "mesodermal_cell", "endodermal_cell", "trophoblast"]),
    ("LIN28A", 6338, "Stem cell",
        ["h1-hesc", "h7-hesc", "hues64", "ucsf-4", "gm23248", "mesenchymal_stem_cell",
         "mesendoderm", "mesodermal_cell", "endodermal_cell", "trophoblast"]),
    ("PTPRC", 1503, "Hematopoietic (CD45)",
        ["b_cell", "t_cell", "cd4-positive", "cd8-positive", "natural_killer", "monocyte",
         "myeloid", "peripheral_blood_mononuclear", "hematopoietic", "jurkat", "karpas",
         "oci-ly7", "gm12878", "k562", "spleen", "thymus", "rpmi-7951"]),
    ("CD4", 262, "T cell",
        ["cd4-positive", "t_cell", "jurkat", "thymus", "peripheral_blood_mononuclear"]),
    ("MYOD1", 5949, "Skeletal muscle",
        ["skeletal_muscle", "lhcn", "myoblast", "myocyte", "myotube", "forelimb_muscle",
         "hindlimb_muscle", "muscle_of_arm", "psoas_muscle", "gastrocnemius"]),
    ("GATA1", 2580, "Erythroid",
        ["k562", "hematopoietic_multipotent", "common_myeloid"]),
    ("FOXA2", 5569, "Liver / hepatocyte",
        ["hepatocyte", "hepg2", "liver", "body_of_pancreas", "endodermal_cell"]),
    ("ALB", 10754, "Hepatocyte / albumin",
        ["hepatocyte", "hepg2", "liver", "body_of_pancreas", "endodermal_cell"]),
]

def compute_predictions():
    """Replicates proof1 01_gene_wise_correlation.py CV exactly (RidgeCV, KFold 5, rs42)."""
    N_GENES, N_MOT, N_FOLDS = 18356, 43, 5
    ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    BAD = {"BSS01297", "BSS01325"}
    bssids = [b for b in np.genfromtxt(f"{EP}/Epimap_BSSID_nondup.txt", dtype=str) if b not in BAD]
    cmap = {}
    with open(f"{EP}/EpiMapID_Name_nonDup.txt") as fh:
        for line in fh:
            p = line.strip().split("\t")
            if len(p) == 2: cmap[p[0]] = p[1]
    labels = [cmap.get(b, b) for b in bssids]
    N = len(bssids)
    E = np.zeros((N, N_GENES), np.float32); F = np.zeros((N, N_GENES, N_MOT), np.float32)
    cdir = f"{MA}/posthoc_analysis/results/count_matrices_universal"
    for ci, b in enumerate(bssids):
        E[ci] = np.load(f"{EP}/training_y/{b}.y.npy").astype(np.float32)
        F[ci] = np.load(f"{cdir}/{b}_count_matrix.npy").astype(np.float32)
    pred = np.full((N, N_GENES), np.nan, np.float32)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for tr, te in kf.split(np.arange(N)):
        Xtr = F[tr].reshape(-1, N_MOT); ytr = E[tr].reshape(-1); Xte = F[te].reshape(-1, N_MOT)
        sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        m = RidgeCV(alphas=ALPHAS, fit_intercept=True); m.fit(Xtr, ytr)
        pred[te] = m.predict(Xte).reshape(len(te), N_GENES).astype(np.float32)
    return E, pred, np.array(labels)

def panel_4g(E, pred, labels):
    N = E.shape[0]
    MK = [m for m in MARKERS if m[0] == 'LIN28A'] or MARKERS[:1]
    fig, axes = plt.subplots(1, 1, figsize=(2.91, 1.85), squeeze=False)
    axes = axes.ravel()
    for k, (sym, gidx, tissue, keys) in enumerate(MK):
        ax = axes[k]
        true_g, pred_g = E[:, gidx], pred[:, gidx]
        # r must match Section 7, which uses the Avsec standardization (each
        # biosample z-scored across genes). Correlating the raw columns here gave
        # the superseded values (e.g. ALB 0.49 instead of 0.13).
        r = float(_ZR.loc[gidx, "pearson_r_motif"])
        tz = (true_g - true_g.mean()) / (true_g.std() + 1e-8)
        pz = (pred_g - pred_g.mean()) / (pred_g.std() + 1e-8)
        order = np.argsort(tz); x = np.arange(N)
        ax.axhline(0, color=JASP, lw=0.7, ls=":", zorder=1)
        ax.plot(x, tz[order], color=INK, lw=1.3, zorder=4, label="Measured")
        ax.plot(x, pz[order], color=CORE, lw=1.1, ls="--", alpha=0.95, zorder=3, label="Core-motif model")
        exp = np.array([any(kk in labels[ci].lower() for kk in keys) for ci in range(N)])[order]
        xe = np.where(exp)[0]; yb = tz[order].min() - 0.42
        ax.scatter(xe, np.full(len(xe), yb), marker="^", s=15, color="#2ca02c",
                   zorder=5, clip_on=False)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color(BASE); ax.spines[s].set_linewidth(0.9)
        ax.set_title(f"{sym}", fontsize=7, color=INK, fontweight="bold", pad=8, loc="left")
        ax.text(0.0, 1.015, f"{tissue}  ·  r = {r:.2f}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=5, color=SUB)
        ax.tick_params(labelsize=5.5, colors=MUT, length=0)
        ax.set_xticks([0, N-1]); ax.set_xticklabels(["low", "high"], fontsize=5, color=MUT)
        ax.set_yticks([-2, 0, 2]); ax.set_ylim(yb - 0.15, tz[order].max() + 0.55)
        if k % 4 == 0: ax.set_ylabel("expression (z)", fontsize=5.5, color=SUB)
        if k >= 4: ax.set_xlabel("biosamples (sorted)", fontsize=5, color=SUB)
    handles = [plt.Line2D([], [], color=INK, lw=1.3, label="Measured"),
               plt.Line2D([], [], color=CORE, lw=1.1, ls="--", label="Core-motif model"),
               plt.Line2D([], [], color="#2ca02c", marker="^", ls="none", ms=5, label="Expected high")]
    fig.legend(handles=handles, fontsize=5.5, frameon=False, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.02), labelcolor=INK)
    fig.subplots_adjust(left=0.145, right=0.985, top=0.865, bottom=0.215)
    save(fig, "fig4g_markers")
    print(f"4g: rendered {len(MK)} marker(s): {[m[0] for m in MK]}")

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":   # repository scope: a, b, d and e (HEK293 held-out = panel_4f); c, f, g and the synovial-fibroblast half of e are not reproduced
    panel_4a()
    panel_4b()
    panel_4d()
    panel_4f()
if False:
    print("computing marker-gene CV predictions (RidgeCV, 5-fold)…")
    E, pred, labels = compute_predictions()
    panel_4g(E, pred, labels)
    print("done — wrote fig4a, 4b, 4d, 4e, 4f, 4g (individual subplots)")
