#!/bin/bash -l
#SBATCH --job-name="final_tomtom"
#SBATCH --output="jobs/logs/final_tomtom.out"
#SBATCH --error="jobs/logs/final_tomtom.err"

# Step 5 — Cross-Cell-Line Self-TOMTOM
#
# Compares all cell-line representative motifs (from step 4) against themselves.
# This identifies motifs that are similar across multiple cell lines, which is
# the basis for discovering universal regulators in step 6.
#
# Input:  results/all_feature_motifs.meme   (produced by 04_collect_features.py)
# Output: results/all_feature_self_tomtom.txt
#
# Submit: sbatch 05_self_tomtom.sh
# Then:   python 06_universal_regulators.py

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cnncr

# Run from the motif_pipeline directory regardless of where sbatch is called from
cd /stg3/data1/sam/enhancer_prediction/multi_agent/motif_pipeline/

tomtom \
    -no-ssc -verbosity 2 -text \
    -dist pearson -min-overlap 7 \
    -evalue -thresh 1 \
    results/all_feature_motifs.meme \
    results/all_feature_motifs.meme \
    > results/all_feature_self_tomtom.txt

echo "Step 5 done. Now run: python 06_universal_regulators.py"
