"""
Shared configuration for the locus-resolved ChIP-seq reader-switching pipeline.
Single source of truth for every path, cell line, and threshold. Import as:
    from lib.config import *
"""
import os, re

# The data root: the work directory you created with bin/setup_work_dir.py and unpacked the
# GitHub and Zenodo data into. `source env.sh` exports it, or set it yourself (see PATHS.md).
ENHANCER_ROOT = os.environ.get("ENHANCER_ROOT")
if not ENHANCER_ROOT:
    raise SystemExit("set ENHANCER_ROOT to your work directory: source env.sh (see PATHS.md)")

ROOT     = ENHANCER_ROOT
CHIP     = f"{ROOT}/multi_agent/chipseq_enrichment"
HERE     = f"{CHIP}/reader_switch"
OUT      = f"{HERE}/out"

# --- primary inputs (nothing here is produced by this pipeline) ---
ENCODE4  = ENHANCER_ROOT + "/encode4_tf_chipseq"                       # raw narrowPeak per TF
PROMOTER = ENHANCER_ROOT + "/genome_annotation/hg38_promoters_cage_corrected.bed"
HG38     = ENHANCER_ROOT + "/genomes/hg38.fa"
BEDTOOLS = "bedtools"
GENES_TXT = f"{ROOT}/gene_ENSG.txt"
JASPAR   = f"{ROOT}/multi_agent/posthoc_analysis/results/step06_JASPAR_assignments.json"
UNIV_IDS = f"{CHIP}/data/universal_motif_ids.txt"                       # 43 core motif ids

# promoter FIMO already run for the linear/RF models (reused, NOT rerun)
EPIMAP_FIMO  = f"{ROOT}/multi_agent/posthoc_analysis/results/fimo_universal"   # {bssid}_fimo.txt
EPIMAP_COORD = f"{ROOT}/data_generated"                                        # {bssid}.coord.exp.df.final.csv
HEK_FIMO     = f"{ROOT}/multi_agent/hek_universal_prediction/fimo/HEK293_fimo.txt"
HEK_WINDOWS  = f"{ROOT}/multi_agent/hek_universal_prediction/regions/hek293_windows.bed"

# --- the 5 cell lines (Fig 5b) ---
CELLS  = ["K562", "HepG2", "A549", "GM12878", "HEK293"]
BSSID  = {"K562": "BSS00762", "HepG2": "BSS00558", "GM12878": "BSS00439",
          "A549": "BSS00007"}                                           # HEK293 is not in EpiMap

# --- thresholds (fixed by the user) ---
SIG_PCTL = 90       # keep peaks >= this WITHIN-TF signalValue percentile (top 10%)
FIMO_P   = 1e-4     # motif-instance threshold, unified across lines (HEK was scanned looser)
MIN_GENES = 20      # a reader must be top at >= this many genes in a line to be "dominant"

# --- palette (project standard) ---
INK, SUB, MUT = "#0b0b0b", "#52514e", "#7d7b76"
CELL_COL = {"K562": "#4878a8", "HepG2": "#5f9a6a", "A549": "#c2884a",
            "GM12878": "#8a72a8", "HEK293": "#b06a72"}

os.makedirs(OUT, exist_ok=True)
_NORM = re.compile(r"[-_.]")
def norm(s): return _NORM.sub("", str(s)).upper()
