"""
Generate 10 independent control motif sets by sampling 43 motifs each
from the pre-verified candidate pool (control_candidate_pool.txt).

All 149 candidates in the pool have already passed:
  1. Not matched to any universal regulator (TOMTOM)
  2. Not from the same TF family (first 3 chars)
  3. Not PWM-similar to any universal regulator (verified TOMTOM)

So any sample of 43 from this pool is a valid control set.

Outputs:
  results/control_motifs_set_{0-9}.meme
  results/fimo_jobs_multi/{set_i}/{bssid}.sh   (only for 26 SNP-overlap cell lines)
  results/fimo_multi/{set_i}/{bssid}_fimo.txt  (FIMO outputs go here)
  results/fimo_jobs_multi/submit_all.sh

Run:
  python generate_10_controls.py
  bash results/fimo_jobs_multi/submit_all.sh
"""

import os
import re
import random
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

JASPAR_MEME    = "/stg3/data1/sam/enhancer_prediction/JASPAR2022_CORE_vertebrates_non_redundant_unique_names.meme"
CANDIDATE_FILE = "results/control_candidate_pool.txt"
FASTA_DIR      = "/new-stg/home/sam2/training_sequences_unique_fasta_noN_pro"
BASE_DIR       = "/stg3/data1/sam/enhancer_prediction/multi_agent/posthoc_analysis"
N_SETS         = 10
N_MOTIFS       = 43   # must match universal regulator count

# Only these 26 cell lines are needed for the SNP overlap analysis
TARGET_BSSIDS = [
    "BSS00043", "BSS01393", "BSS00045", "BSS00079", "BSS00088",
    "BSS01377", "BSS01126", "BSS01614", "BSS01209", "BSS00439",
    "BSS01542", "BSS01848", "BSS00381", "BSS00316", "BSS00558",
    "BSS01190", "BSS01840", "BSS01399", "BSS01406", "BSS01456",
    "BSS01181", "BSS01588", "BSS01715", "BSS01831", "BSS01884",
    "BSS01887",
]


# ── MEME I/O ─────────────────────────────────────────────────────────────────

def parse_meme(path):
    motifs, names = [], []
    cur, cur_name, reading = [], None, False
    with open(path) as fh:
        for line in fh:
            if line.startswith("MOTIF"):
                if cur:
                    motifs.append(np.array(cur)); names.append(cur_name); cur = []
                parts = line.strip().split()
                cur_name = parts[1] if len(parts) >= 2 else f"motif_{len(motifs)}"
                reading = False
            elif line.startswith("letter-probability matrix"):
                reading = True; cur = []
            elif reading and line.strip():
                try:
                    cur.append([float(x) for x in line.strip().split()])
                except ValueError:
                    reading = False
    if cur:
        motifs.append(np.array(cur)); names.append(cur_name)
    return motifs, names


def write_meme(motifs, names, path):
    with open(path, "w") as fh:
        fh.write("MEME version 4.1.1\n\nALPHABET= ACGT\n\nstrands: + -\n\n")
        for name, m in zip(names, motifs):
            fh.write(f"MOTIF {name} {name}\n")
            fh.write(f"letter-probability matrix: alength=4 w={m.shape[0]} nsites=1 E=0.0e+00\n")
            for row in m:
                fh.write("\t".join(str(x) for x in row) + "\n")
            fh.write("\n")


# ── Load candidates ──────────────────────────────────────────────────────────

candidates = []
with open(CANDIDATE_FILE) as fh:
    for line in fh:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("\t")[0]
        candidates.append(name)

print(f"Candidate pool: {len(candidates)} motifs")
assert len(candidates) >= N_MOTIFS, f"Pool too small: {len(candidates)} < {N_MOTIFS}"

jaspar_pwms, jaspar_names = parse_meme(JASPAR_MEME)
jaspar_pwm_map = {n: p for n, p in zip(jaspar_names, jaspar_pwms)}

# ── Generate 10 control sets ─────────────────────────────────────────────────

os.makedirs("results", exist_ok=True)
all_selected = []

for i in range(N_SETS):
    rng = random.Random(i * 137 + 42)   # deterministic but different per set
    selected = rng.sample(candidates, N_MOTIFS)
    all_selected.append(selected)

    meme_path = f"results/control_motifs_set_{i}.meme"
    write_meme([jaspar_pwm_map[n] for n in selected], selected, meme_path)
    print(f"Set {i}: {selected[:3]} ... → {meme_path}")

# Log which motifs were chosen per set
with open("results/control_sets_summary.txt", "w") as fh:
    fh.write(f"# 10 control sets, {N_MOTIFS} motifs each from pool of {len(candidates)}\n")
    for i, sel in enumerate(all_selected):
        fh.write(f"\n## Set {i}  (seed={i * 137 + 42})\n")
        for n in sel:
            fh.write(f"{n}\n")
print("Set summary → results/control_sets_summary.txt")

# ── Generate FIMO SLURM jobs ─────────────────────────────────────────────────

submit_lines = ["#!/bin/bash"]

for i in range(N_SETS):
    job_dir = f"{BASE_DIR}/results/fimo_jobs_multi/set_{i}"
    out_dir = f"{BASE_DIR}/results/fimo_multi/set_{i}"
    meme_path = f"{BASE_DIR}/results/control_motifs_set_{i}.meme"
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    submit_lines.append(f"\n# Control set {i}")
    for bssid in TARGET_BSSIDS:
        script = f"""\
#!/bin/bash -l
#SBATCH --job-name="fimo_c{i}_{bssid}"
#SBATCH --output="{job_dir}/{bssid}.out"
#SBATCH --error="{job_dir}/{bssid}.err"

source /stg3/data1/sam/anaconda3/etc/profile.d/conda.sh
conda activate enhancer

fimo --no-qvalue --text \\
    {meme_path} \\
    {FASTA_DIR}/{bssid}.fasta \\
    > {out_dir}/{bssid}_fimo.txt
"""
        path = f"{job_dir}/{bssid}.sh"
        with open(path, "w") as fh:
            fh.write(script)
        submit_lines.append(f"sbatch {path}")

submit_path = f"{BASE_DIR}/results/fimo_jobs_multi/submit_all.sh"
with open(submit_path, "w") as fh:
    fh.write("\n".join(submit_lines) + "\n")

total_jobs = N_SETS * len(TARGET_BSSIDS)
print(f"\nGenerated {total_jobs} FIMO jobs ({N_SETS} sets × {len(TARGET_BSSIDS)} cell lines)")
print(f"Submit: bash {submit_path}")
print(f"\nOnce FIMO jobs complete, re-run snp_overlap_universal.py")
