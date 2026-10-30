"""
For each of the 5 cell lines:
  1. Load TSS ±10kb promoter regions (hg38_promoters_cage_corrected.bed)
  2. Read every TF's narrowPeak BED → extract summit ±150 bp windows
  3. Keep only summit windows that overlap a TSS ±10kb region
  4. Write a deduplicated union BED (input to bedtools getfasta)
  5. Pickle a dict {tf_name: set(window_keys)} for the enrichment step

Outputs per cell line (in data/{cell}/):
  unique_summits.bed   - deduplicated ±150 bp summit windows (TSS-proximal only)
  tf_peak_sets.pkl     - {tf: set of "chr:start-end" keys}
  tf_names.txt         - ordered list of TF names
"""

import os
import sys
import pickle
import bisect

# The data root: the work directory you created with bin/setup_work_dir.py and unpacked the
# GitHub and Zenodo data into. `source env.sh` exports it, or set it yourself (see PATHS.md).
ENHANCER_ROOT = os.environ.get("ENHANCER_ROOT")
if not ENHANCER_ROOT:
    raise SystemExit("set ENHANCER_ROOT to your work directory: source env.sh (see PATHS.md)")

ENCODE4_DIR  = ENHANCER_ROOT + "/encode4_tf_chipseq"
WORK_DIR     = ENHANCER_ROOT + "/multi_agent/chipseq_enrichment"
PROMOTER_BED = ENHANCER_ROOT + "/genome_annotation/hg38_promoters_cage_corrected.bed"

# Default = original 5 lines; pass cell line(s) as CLI args to run a subset,
# e.g.  python 01_prepare_peaks.py HEK293   (identical protocol, new line).
CELL_LINES = sys.argv[1:] if len(sys.argv) > 1 else ["HepG2", "K562", "GM12878", "MCF.7", "A549"]
HALF_WIN   = 150   # bp either side of ChIP-seq summit

os.chdir(WORK_DIR)

# ── Load TSS ±10kb promoter regions ──────────────────────────────────────────
# Build per-chrom sorted list of (start, end) for fast overlap lookup
print("Loading promoter regions ...", flush=True)
promoters = {}   # chrom → sorted list of (start, end)
with open(PROMOTER_BED) as fh:
    for line in fh:
        cols = line.strip().split("\t")
        if len(cols) < 3:
            continue
        chrom, start, end = cols[0], int(cols[1]), int(cols[2])
        promoters.setdefault(chrom, []).append((start, end))

for chrom in promoters:
    promoters[chrom].sort()

promo_starts = {chrom: [r[0] for r in regions]
                for chrom, regions in promoters.items()}

print(f"  Loaded {sum(len(v) for v in promoters.values())} promoter regions "
      f"across {len(promoters)} chromosomes")


def overlaps_promoter(chrom, win_s, win_e):
    """Return True if (win_s, win_e) overlaps any TSS ±10kb region on chrom."""
    if chrom not in promoters:
        return False
    regions = promoters[chrom]
    starts  = promo_starts[chrom]
    # Find rightmost promoter whose start <= win_e
    idx = bisect.bisect_right(starts, win_e) - 1
    # Check a small window of candidates around idx
    for i in range(max(0, idx - 1), min(len(regions), idx + 2)):
        ps, pe = regions[i]
        if win_s < pe and win_e > ps:   # overlap condition
            return True
    return False


# ── Process each cell line ────────────────────────────────────────────────────
for cell in CELL_LINES:
    print(f"\n=== {cell} ===", flush=True)
    cell_dir = os.path.join(ENCODE4_DIR, cell)
    out_dir  = os.path.join(WORK_DIR, "data", cell)
    os.makedirs(out_dir, exist_ok=True)

    tf_peak_sets  = {}   # tf_name → set of window keys
    all_windows   = {}   # key → (chrom, start, end) — deduplicated union
    total_raw     = 0
    total_kept    = 0

    for tf_folder in sorted(os.listdir(cell_dir)):
        tf_path = os.path.join(cell_dir, tf_folder)
        if not os.path.isdir(tf_path):
            continue
        beds = [f for f in os.listdir(tf_path) if f.endswith(".bed")]
        if not beds:
            continue

        tf_name  = tf_folder.replace("_human", "")
        bed_file = os.path.join(tf_path, beds[0])

        peaks = set()
        with open(bed_file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                cols = line.split("\t")
                if len(cols) < 10:
                    continue
                chrom  = cols[0]
                start  = int(cols[1])
                offset = int(cols[9])
                summit = start + offset
                win_s  = max(0, summit - HALF_WIN)
                win_e  = summit + HALF_WIN
                total_raw += 1

                if not overlaps_promoter(chrom, win_s, win_e):
                    continue

                key = f"{chrom}:{win_s}-{win_e}"
                peaks.add(key)
                all_windows[key] = (chrom, win_s, win_e)
                total_kept += 1

        if peaks:
            tf_peak_sets[tf_name] = peaks

    # Write deduplicated union BED
    bed_path = os.path.join(out_dir, "unique_summits.bed")
    with open(bed_path, "w") as fh:
        for key, (chrom, s, e) in sorted(all_windows.items(),
                                          key=lambda x: (x[1][0], x[1][1])):
            fh.write(f"{chrom}\t{s}\t{e}\t{key}\n")

    # Pickle TF→peaks mapping
    with open(os.path.join(out_dir, "tf_peak_sets.pkl"), "wb") as fh:
        pickle.dump(tf_peak_sets, fh)

    # TF name list
    with open(os.path.join(out_dir, "tf_names.txt"), "w") as fh:
        fh.write("\n".join(sorted(tf_peak_sets.keys())) + "\n")

    pct = 100 * total_kept / total_raw if total_raw else 0
    print(f"  Raw peaks:    {total_raw:>10,}")
    print(f"  TSS-proximal: {total_kept:>10,}  ({pct:.1f}% retained)")
    print(f"  Unique windows: {len(all_windows):,}")
    print(f"  TFs with ≥1 TSS peak: {len(tf_peak_sets)}")

print("\nDone. Submit FIMO jobs next: bash submit_fimo_jobs.sh")
