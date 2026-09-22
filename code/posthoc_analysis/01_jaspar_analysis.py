"""
Step 01 — JASPAR motif analysis for step-06 universal regulators.

1. Parses TOMTOM results (step06 motifs vs JASPAR) to assign TF identities to
   each universal regulator → saved as step06_JASPAR_assignments.json.

2. Builds a random control MEME set (same number of motifs as step06 output)
   drawn from JASPAR motifs that are:
     (a) not directly matched to any universal regulator,
     (b) not from the same TF family (first 3 characters of name),
     (c) not PWM-similar to any universal regulator (verified by a second
         TOMTOM run between candidates and step06 output).

Run: python 01_jaspar_analysis.py
Requires: results/step06_jaspar_tomtom.txt (produced by 01_jaspar_tomtom.sh)
Outputs:  results/step06_JASPAR_assignments.json
          results/control_motifs.meme
"""

import os
import re
import sys
import json
import random
import subprocess
import tempfile
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
random.seed(42)
np.random.seed(42)

STEP06_MEME  = "/stg3/data1/sam/enhancer_prediction/multi_agent/motif_pipeline/results/universal_regulator_motifs.meme"
JASPAR_MEME  = "/stg3/data1/sam/enhancer_prediction/JASPAR2022_CORE_vertebrates_non_redundant_unique_names.meme"
TOMTOM_FILE  = "results/step06_jaspar_tomtom.txt"
ASSIGN_JSON      = "results/step06_JASPAR_assignments.json"
CONTROL_MEME     = "results/control_motifs.meme"
CANDIDATE_LIST   = "results/control_candidate_pool.txt"

# TOMTOM thresholds used for both the annotation pass and the verification pass
E_THRESH = 1.0
P_THRESH = 0.01


# ── MEME I/O ──────────────────────────────────────────────────────────────────

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


# ── TOMTOM helpers ─────────────────────────────────────────────────────────────

def read_tomtom(path, e_thresh=E_THRESH, p_thresh=P_THRESH):
    df = pd.read_csv(path, sep="\t", comment="#")
    df["Optimal_offset"] = df["Optimal_offset"].astype(int)
    return df[(df["E-value"] <= e_thresh) & (df["p-value"] <= p_thresh)]


def run_tomtom_inline(query_meme, target_meme):
    """Run TOMTOM and return results as a filtered DataFrame."""
    cmd = [
        "tomtom", "-no-ssc", "-verbosity", "1", "-text",
        "-dist", "pearson", "-min-overlap", "7",
        "-evalue", "-thresh", str(E_THRESH),
        query_meme, target_meme,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = [l for l in result.stdout.splitlines() if not l.startswith("#") and l.strip()]
    if len(lines) < 2:
        return pd.DataFrame()
    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(lines)), sep="\t")
    df["Optimal_offset"] = df["Optimal_offset"].astype(int)
    return df[(df["E-value"] <= E_THRESH) & (df["p-value"] <= P_THRESH)]


# ── Main ───────────────────────────────────────────────────────────────────────

for path in (STEP06_MEME, TOMTOM_FILE):
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found.")

step06_pwms, step06_names = parse_meme(STEP06_MEME)
N_MOTIFS = len(step06_names)
print(f"Step06 universal regulators: {N_MOTIFS} motifs")

jaspar_pwms, jaspar_names = parse_meme(JASPAR_MEME)
jaspar_pwm_map = {n: p for n, p in zip(jaspar_names, jaspar_pwms)}
print(f"JASPAR database: {len(jaspar_names)} motifs")

# ── 1. TF assignments ─────────────────────────────────────────────────────────
tomtom = read_tomtom(TOMTOM_FILE)
regulator_to_tfs = {}
for qid, grp in tomtom.groupby("Query_ID", sort=False):
    tfs = (grp["Target_ID"].dropna().drop_duplicates()
           .apply(lambda x: re.sub(r"_\d+$", "", x)).tolist())
    regulator_to_tfs[qid] = tfs

print("\nUniversal regulator → JASPAR TF assignments:")
for reg, tfs in sorted(regulator_to_tfs.items()):
    print(f"  {reg}: {tfs}")

with open(ASSIGN_JSON, "w") as fh:
    json.dump(regulator_to_tfs, fh, indent=2)
print(f"\nAssignments saved → {ASSIGN_JSON}")

# ── 2. Build exclusion sets ───────────────────────────────────────────────────
matched_names   = set(tomtom["Target_ID"].dropna().unique())
matched_families = {re.sub(r"_\d+$", "", n)[:3].upper() for n in matched_names}
print(f"\nDirect JASPAR matches: {len(matched_names)}")
print(f"Excluded families (first 3 chars): {sorted(matched_families)}")

candidates = [
    n for n in jaspar_names
    if n not in matched_names
    and re.sub(r"_\d+$", "", n)[:3].upper() not in matched_families
]
print(f"Candidate pool after family exclusion: {len(candidates)}")

if len(candidates) < N_MOTIFS:
    sys.exit(f"ERROR: only {len(candidates)} candidates remain; need {N_MOTIFS}.")

# ── 3. Sample + verify by PWM similarity ──────────────────────────────────────
# Iteratively sample N_MOTIFS candidates; run TOMTOM to check if any are
# similar to the step06 universal regulators. Remove any that match and resample.

excluded_by_similarity = set()
MAX_ROUNDS = 20

for rnd in range(MAX_ROUNDS):
    pool = [n for n in candidates if n not in excluded_by_similarity]
    if len(pool) < N_MOTIFS:
        sys.exit(f"ERROR: pool too small after similarity filtering ({len(pool)} < {N_MOTIFS}).")

    selected = random.sample(pool, N_MOTIFS)

    # Write selected candidates to a temp MEME file
    with tempfile.NamedTemporaryFile(suffix=".meme", mode="w", delete=False) as tmp:
        tmp_path = tmp.name
    write_meme([jaspar_pwm_map[n] for n in selected], selected, tmp_path)

    # TOMTOM: step06 output (query) vs candidate set (target)
    try:
        sim_df = run_tomtom_inline(STEP06_MEME, tmp_path)
    except subprocess.CalledProcessError as e:
        print(f"  TOMTOM failed: {e.stderr[:200]}")
        sim_df = pd.DataFrame()
    finally:
        os.unlink(tmp_path)

    if sim_df.empty:
        print(f"Round {rnd+1}: no PWM similarity found — control set is clean.")
        break

    similar_targets = set(sim_df["Target_ID"].dropna().unique())
    print(f"Round {rnd+1}: {len(similar_targets)} candidates similar to universal "
          f"regulators — removing and resampling.")
    excluded_by_similarity |= similar_targets
else:
    sys.exit("ERROR: could not find a clean control set after max rounds.")

control_pwms  = [jaspar_pwm_map[n] for n in selected]
control_names = list(selected)

# ── save full candidate pool (all motifs passing name + PWM similarity filters)
final_pool = [n for n in candidates if n not in excluded_by_similarity]
with open(CANDIDATE_LIST, "w") as fh:
    fh.write(f"# Full candidate control pool: {len(final_pool)} JASPAR motifs\n")
    fh.write("# Filters applied:\n")
    fh.write("#   1. Not directly matched to any universal regulator (TOMTOM)\n")
    fh.write("#   2. Not from the same TF family (first 3 chars of name)\n")
    fh.write("#   3. Not PWM-similar to any universal regulator (verified TOMTOM)\n")
    fh.write(f"# {len(selected)} randomly selected as control set (saved in control_motifs.meme)\n#\n")
    for n in sorted(final_pool):
        selected_flag = "SELECTED" if n in selected else ""
        fh.write(f"{n}\t{selected_flag}\n")
print(f"Full candidate pool ({len(final_pool)} motifs) saved → {CANDIDATE_LIST}")

print(f"\nSelected {len(selected)} control motifs (verified non-similar):")
for n in control_names:
    print(f"  {n}")

write_meme(control_pwms, control_names, CONTROL_MEME)
print(f"\nControl set written → {CONTROL_MEME}")
print(f"\nNext steps:")
print(f"  python generate_10_controls.py   # the ten control sets (notebook 04)")
print(f"  (submit all FIMO jobs)")
pass
