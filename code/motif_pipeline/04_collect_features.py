"""
Step 4 — Collect All Cell-Line Representative Motifs

After all per-cell-line jobs (step 3) have finished, this script:
  - Reads each cell line's output: results/cell_motifs/{bssid}/{bssid}_cell_motifs.meme
  - Renames every motif as {bssid}_feature_{group_id} so that motifs can be
    traced back to their cell line of origin in downstream steps
  - Combines all cell lines into a single MEME file

Output: results/all_feature_motifs.meme

Run: python 04_collect_features.py

Then submit step 5 (sbatch 05_self_tomtom.sh) to run the final cross-cell-line
TOMTOM comparison before proceeding to step 6.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from utils import load_bssids, parse_meme_file, write_meme_file

# ── configuration ─────────────────────────────────────────────────────────────
BSSID_FILE  = "/stg3/data1/sam/enhancer_prediction/Epimap_BSSID_nondup.txt"
CELL_DIR    = os.path.join(SCRIPT_DIR, "results", "cell_motifs")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "results", "all_feature_motifs.meme")

# ── main ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

bssids  = load_bssids(BSSID_FILE)
all_motifs, all_names = [], []
missing = []

for bssid in bssids:
    meme_file = os.path.join(CELL_DIR, bssid, f"{bssid}_cell_motifs.meme")
    if not os.path.exists(meme_file):
        missing.append(bssid)
        continue
    motifs, names = parse_meme_file(meme_file)
    all_motifs.extend(motifs)
    # Prefix with cell line ID so groups can be attributed to their source later
    all_names.extend(f"{bssid}_feature_{n}" for n in names)

if missing:
    print(f"[04] WARNING: {len(missing)} cell lines are missing cell_motifs.meme — "
          f"check step 3 logs in jobs/logs/")
    for b in missing[:10]:
        print(f"       {b}")
    if len(missing) > 10:
        print(f"       … and {len(missing) - 10} more")

write_meme_file(all_motifs, OUTPUT_FILE, all_names)
n_done = len(bssids) - len(missing)
print(f"[04] {len(all_motifs)} total motifs from {n_done}/{len(bssids)} cell lines")
print(f"     → {OUTPUT_FILE}")
print()
print("Next step: sbatch 05_self_tomtom.sh")
print("Then run:  python 06_universal_regulators.py")
