"""
Step 1 — IC Filtering and Weight-Based Selection

For every cell line, reads CNN-trained motifs from 10 independent training
iterations (2 560 motifs each = 25 600 total) and produces a filtered set:

  1. Drop motifs with total information content <= IC_THRESHOLD (default 10 bits).
  2. From the survivors, keep the top MOTIFS_PER_ITER ranked by filter weight.

The output MEME file assigns each motif a globally unique integer ID:
  iteration 0 → IDs 0 … 2559
  iteration 1 → IDs 2560 … 5119  …etc.

Output per cell line:
  results/cell_motifs/{bssid}/{bssid}_ICfiltered.meme   (≤ 7 500 motifs)

Run: python 01_ic_filter.py
"""

import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from utils import load_bssids, parse_meme_file, write_meme_file, information_content

# ── configuration ─────────────────────────────────────────────────────────────
CNN_DIR         = "/stg3/data1/sam/enhancer_prediction/cell_type_specific_CNN"
BSSID_FILE      = "/stg3/data1/sam/enhancer_prediction/Epimap_BSSID_nondup.txt"
OUTPUT_DIR      = os.path.join(SCRIPT_DIR, "results", "cell_motifs")
MODEL_REMARK    = "256MOTIF_pytorch"
ANALYSIS_TYPE   = "pro"
N_ITERATIONS    = 10
MOTIFS_PER_ITER = 750    # top motifs to keep per iteration (after IC filter)
IC_THRESHOLD    = 10.0   # minimum total information content

# ── main ──────────────────────────────────────────────────────────────────────
bssids = load_bssids(BSSID_FILE)
bssids = sys.argv[1:] or list(bssids)   # biosample(s) on the command line; default: all 116

for bssid in bssids:
    print(f"[01] IC filter: {bssid}")
    cell_cnn_dir = os.path.join(CNN_DIR, bssid)
    cell_out_dir = os.path.join(OUTPUT_DIR, bssid)
    os.makedirs(cell_out_dir, exist_ok=True)

    all_motifs, all_names = [], []

    for it in range(N_ITERATIONS):
        stem         = f"{bssid}_it_{it}_{MODEL_REMARK}_{it}_{ANALYSIS_TYPE}"
        meme_file    = os.path.join(cell_cnn_dir, f"{stem}_motifs_all_256.meme")
        weights_file = os.path.join(cell_cnn_dir, f"{stem}_motifs_weights_all_256.npy")

        motifs, _  = parse_meme_file(meme_file)
        weights    = np.load(weights_file)

        if len(motifs) != 2560:
            print(f"  WARNING: expected 2560 motifs, got {len(motifs)} — {meme_file}")

        # Assign globally unique sequential IDs across all iterations
        global_ids = list(range(it * 2560, it * 2560 + len(motifs)))

        # Step 1: keep motifs above IC threshold
        ic_pass = [i for i, m in enumerate(motifs) if information_content(m) > IC_THRESHOLD]
        if not ic_pass:
            continue

        # Step 2: rank survivors by filter weight, take top MOTIFS_PER_ITER
        ic_weights = weights[ic_pass]
        top_k      = np.argsort(ic_weights)[::-1][: min(MOTIFS_PER_ITER, len(ic_pass))]
        kept       = [ic_pass[j] for j in top_k]

        all_motifs.extend(motifs[i] for i in kept)
        all_names.extend(global_ids[i] for i in kept)

    out_meme = os.path.join(cell_out_dir, f"{bssid}_ICfiltered.meme")
    write_meme_file(all_motifs, out_meme, all_names)
    print(f"       {len(all_motifs)} motifs → {out_meme}")

print("[01] Done.")
