"""
Analysis: DBD-level protein conservation via gnomAD regional missense constraint

For each TF (universal regulators vs all other JASPAR human TFs):
  1. Parse UniProt flat file → gene name → DBD amino acid intervals
     Feature types used: DNA_BIND, ZN_FING, DOMAIN (DNA-binding domains only)
  2. Parse gnomAD v2.1.1 regional missense constraint → gene → (start_aa, stop_aa, oe)
  3. For each TF, find gnomAD windows overlapping the DBD range
     → weighted-mean oe_mis over DBD (weight = overlap length in AA)
  4. Mann-Whitney U: universal TFs vs background TFs
  5. Violin plot + CSV

Run: python 02_dbd_conservation.py
"""

import os
import re
import gzip
import json
import numpy as np
import pandas as pd
import matplotlib
# The data root: the work directory you created with bin/setup_work_dir.py and unpacked the
# GitHub and Zenodo data into. `source env.sh` exports it, or set it yourself (see PATHS.md).
ENHANCER_ROOT = os.environ.get("ENHANCER_ROOT")
if not ENHANCER_ROOT:
    raise SystemExit("set ENHANCER_ROOT to your work directory: source env.sh (see PATHS.md)")

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── paths ──────────────────────────────────────────────────────────────────────
JASPAR_ASSIGNMENTS = "../posthoc_analysis/results/step06_JASPAR_assignments.json"
TF_INFO            = ENHANCER_ROOT + "/motif_analysis/TF_Information.txt"
UNIPROT_DAT        = "uniprot_sprot_human.dat.gz"
GNOMAD_RMC         = "gnomAD_v2.1.1_transcripts_with_rmc.tsv"
OUT_DIR            = "results"

os.makedirs(OUT_DIR, exist_ok=True)

# DBD feature types to extract from UniProt
DBD_FEATURE_TYPES = {"DNA_BIND", "ZN_FING"}

# DOMAIN entries whose notes indicate a DNA-binding domain
DBD_DOMAIN_KEYWORDS = {
    "homeodomain", "bhlh", "bzip", "ets", "forkhead", "fork head",
    "runt", "stat", "t-box", "tbox", "p53", "myb", "gata",
    "nuclear receptor", "dna-binding", "dna binding", "rel homology",
    "winged-helix", "paired box", "pou", "high mobility group",
    "hmg", "sand", "cut", "grainyhead", "nf-1",
}


# ── 1. parse UniProt flat file ─────────────────────────────────────────────────

def parse_uniprot_dbd(dat_path):
    """
    Returns dict: gene_name_upper → list of (start_aa, end_aa) 1-based inclusive.
    Collects DNA_BIND, ZN_FING, and DBD-annotated DOMAIN features.
    """
    print("Parsing UniProt flat file...")
    dbd_map = {}

    opener = gzip.open if dat_path.endswith(".gz") else open
    with opener(dat_path, "rt", encoding="latin-1") as fh:
        gene_names = []
        features   = []
        in_ft      = False
        cur_ftype  = None
        cur_start  = None
        cur_end    = None
        cur_note   = ""

        def flush_feature():
            nonlocal cur_ftype, cur_start, cur_end, cur_note
            if cur_ftype and cur_start and cur_end:
                keep = cur_ftype in DBD_FEATURE_TYPES
                if not keep and cur_ftype == "DOMAIN":
                    keep = any(kw in cur_note.lower() for kw in DBD_DOMAIN_KEYWORDS)
                if keep:
                    features.append((cur_start, cur_end))
            cur_ftype = cur_start = cur_end = None
            cur_note = ""

        for line in fh:
            tag = line[:2]

            if tag == "GN":
                # GN   Name=SP1; Synonyms=TSFP1;
                m = re.search(r"Name=([^;{]+)", line)
                if m:
                    gene_names.append(m.group(1).strip().upper())
                # also capture ORFNames / ordered locus names as fallback
                for m2 in re.finditer(r"(?:ORFNames|OrderedLocusNames)=([^;{]+)", line):
                    for g in m2.group(1).split(","):
                        gene_names.append(g.strip().upper())

            elif tag == "FT":
                rest = line[5:]
                # new feature line: starts with non-space in col 5
                if rest and rest[0] != " ":
                    flush_feature()
                    parts   = rest.split()
                    ftype   = parts[0]
                    cur_ftype = ftype
                    # position e.g. "626..650" or "626" or "?..650"
                    if len(parts) > 1:
                        pos_str = parts[1]
                        m = re.findall(r"\d+", pos_str)
                        if len(m) >= 2:
                            cur_start, cur_end = int(m[0]), int(m[1])
                        elif len(m) == 1:
                            cur_start = cur_end = int(m[0])
                else:
                    # continuation: collect /note content
                    note_m = re.search(r'/note="([^"]*)"?', rest)
                    if note_m:
                        cur_note += note_m.group(1).lower()

            elif tag == "//":
                flush_feature()
                if gene_names and features:
                    for gn in gene_names:
                        if gn not in dbd_map:
                            dbd_map[gn] = []
                        dbd_map[gn].extend(features)
                # reset entry
                gene_names = []
                features   = []
                cur_ftype = cur_start = cur_end = None
                cur_note = ""

    print(f"  UniProt entries with DBD annotation: {len(dbd_map)}")
    return dbd_map


# ── 2. parse gnomAD regional missense constraint ───────────────────────────────

def parse_aa(s):
    """'Arg97' → 97, 'Met1' → 1"""
    m = re.search(r"\d+", str(s))
    return int(m.group()) if m else None


def parse_gnomad_rmc(tsv_path):
    """
    Returns dict: gene_name_upper → list of (start_aa, stop_aa, oe).
    oe = observed/expected missense ratio (lower = more constrained).
    """
    print("Parsing gnomAD regional missense constraint...")
    df = pd.read_csv(tsv_path, sep="\t")
    df["start_aa_int"] = df["start_aa"].apply(parse_aa)
    df["stop_aa_int"]  = df["stop_aa"].apply(parse_aa)
    df = df.dropna(subset=["start_aa_int", "stop_aa_int", "oe"])
    df["gene_upper"] = df["gene_name"].str.upper()

    rmc_map = {}
    for gene, grp in df.groupby("gene_upper"):
        windows = list(zip(
            grp["start_aa_int"].astype(int),
            grp["stop_aa_int"].astype(int),
            grp["oe"].astype(float),
        ))
        rmc_map[gene] = windows

    print(f"  Genes in gnomAD RMC: {len(rmc_map)}")
    return rmc_map


# ── 3. compute weighted-mean oe over DBD ───────────────────────────────────────

def dbd_oe(dbd_intervals, rmc_windows):
    """
    For a TF with DBD intervals [(s,e)...] and gnomAD windows [(ws,we,oe)...],
    return weighted-mean oe weighted by overlap length in AA.
    Returns np.nan if no overlap found.
    """
    total_weight = 0.0
    total_oe     = 0.0

    for dbd_s, dbd_e in dbd_intervals:
        for win_s, win_e, oe in rmc_windows:
            overlap = min(dbd_e, win_e) - max(dbd_s, win_s)
            if overlap > 0:
                total_weight += overlap
                total_oe     += oe * overlap

    if total_weight == 0:
        return np.nan
    return total_oe / total_weight


# ── 4. load TF sets ────────────────────────────────────────────────────────────
print("Loading TF sets...")
with open(JASPAR_ASSIGNMENTS) as fh:
    assignments = json.load(fh)

universal_upper = {t.upper() for v in assignments.values() for t in v}

tf_info = pd.read_csv(TF_INFO, sep="\t")
tf_info = tf_info[tf_info["TF_Species"] == "Homo_sapiens"].copy()
tf_info["TF_Name_upper"] = tf_info["TF_Name"].str.upper()
tf_dedup = tf_info.drop_duplicates("TF_Name_upper")[["TF_Name", "TF_Name_upper"]].copy()
tf_dedup["is_universal"] = tf_dedup["TF_Name_upper"].isin(universal_upper)
print(f"JASPAR human TFs: {len(tf_dedup)}  "
      f"(universal={tf_dedup['is_universal'].sum()}, "
      f"background={(~tf_dedup['is_universal']).sum()})")

# ── 5. parse data files ────────────────────────────────────────────────────────
dbd_map = parse_uniprot_dbd(UNIPROT_DAT)
rmc_map = parse_gnomad_rmc(GNOMAD_RMC)

# ── 6. compute per-TF DBD oe ──────────────────────────────────────────────────
print("\nComputing DBD oe per TF...")
records = []
for row in tf_dedup.itertuples(index=False):
    name  = row.TF_Name_upper
    group = "universal" if row.is_universal else "background"

    dbd = dbd_map.get(name)
    rmc = rmc_map.get(name)

    if dbd is None:
        note = "no_uniprot_dbd"
    elif rmc is None:
        note = "no_gnomad_rmc"
    else:
        score = dbd_oe(dbd, rmc)
        note  = "ok" if not np.isnan(score) else "no_overlap"
        records.append({"tf": row.TF_Name, "group": group,
                        "dbd_oe": score if note == "ok" else np.nan,
                        "note": note})
        continue

    records.append({"tf": row.TF_Name, "group": group,
                    "dbd_oe": np.nan, "note": note})

results_df = pd.DataFrame(records)
results_df.to_csv(f"{OUT_DIR}/dbd_oe_scores.csv", index=False)
print(f"Saved → {OUT_DIR}/dbd_oe_scores.csv")

# ── 7. coverage summary ────────────────────────────────────────────────────────
print("\n--- Coverage ---")
for note, grp in results_df.groupby("note"):
    uni  = (grp["group"] == "universal").sum()
    bg   = (grp["group"] == "background").sum()
    print(f"  {note}: {len(grp)} total (universal={uni}, background={bg})")

scored   = results_df[results_df["dbd_oe"].notna()]
uni_vals = scored[scored["group"] == "universal"]["dbd_oe"].values
bg_vals  = scored[scored["group"] == "background"]["dbd_oe"].values
print(f"\nScored: universal={len(uni_vals)}, background={len(bg_vals)}")

# ── 8. statistics ─────────────────────────────────────────────────────────────
stat, pval = stats.mannwhitneyu(uni_vals, bg_vals, alternative="two-sided")
print(f"Mann-Whitney U={stat:.0f}, p={pval:.4g}")
print(f"Universal median DBD oe: {np.median(uni_vals):.4f}")
print(f"Background median DBD oe: {np.median(bg_vals):.4f}")

stats_row = pd.DataFrame([{
    "metric":            "gnomAD RMC oe_mis at DBD",
    "n_universal":       len(uni_vals),
    "median_universal":  np.median(uni_vals),
    "n_background":      len(bg_vals),
    "median_background": np.median(bg_vals),
    "mannwhitney_U":     stat,
    "p_value":           pval,
}])
stats_row.to_csv(f"{OUT_DIR}/dbd_constraint_comparison.csv", index=False)
print(f"Saved → {OUT_DIR}/dbd_constraint_comparison.csv")

# ── 9. violin plot ────────────────────────────────────────────────────────────
COLORS = {"Universal TFs": "#e07b54", "All other TFs": "#5b8db8"}

fig, ax = plt.subplots(figsize=(5, 5))
parts = ax.violinplot(
    [uni_vals, bg_vals], positions=[0, 1],
    showmedians=True, showextrema=True,
)
for pc, color in zip(parts["bodies"], list(COLORS.values())):
    pc.set_facecolor(color)
    pc.set_alpha(0.75)
for key in ("cmedians", "cmaxes", "cmins", "cbars"):
    parts[key].set_color("black")
    parts[key].set_linewidth(1.2)

p_str = f"p = {pval:.3g}" if pval >= 0.001 else "p < 0.001"
ax.set_title(f"gnomAD Regional Missense oe at DNA Binding Domain\n{p_str}", fontsize=11)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Universal TFs", "All other TFs"], fontsize=10)
ax.set_ylabel("Weighted mean oe (DBD)\n← more constrained", fontsize=10)

patches = [mpatches.Patch(color=c, label=l) for l, c in COLORS.items()]
ax.legend(handles=patches, fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/dbd_oe_violin.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/dbd_oe_violin.png", dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_DIR}/dbd_oe_violin.pdf")
print("\nDone.")
