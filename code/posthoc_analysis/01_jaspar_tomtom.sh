#!/bin/bash -l
#SBATCH --job-name="ph_tomtom"
#SBATCH --output="ph_tomtom.out"
#SBATCH --error="ph_tomtom.err"

# Run TOMTOM: step06 universal regulator motifs vs full JASPAR database.
# Output is parsed by 01_jaspar_analysis.py to assign TF identities and
# generate a matched-excluded random control set.
#
# Submit: sbatch 01_jaspar_tomtom.sh
# Then:   python 01_jaspar_analysis.py

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cnncr
cd /stg3/data1/sam/enhancer_prediction/multi_agent/posthoc_analysis/

STEP06_MEME="/stg3/data1/sam/enhancer_prediction/multi_agent/motif_pipeline/results/universal_regulator_motifs.meme"
JASPAR_MEME="/stg3/data1/sam/enhancer_prediction/JASPAR2022_CORE_vertebrates_non_redundant_unique_names.meme"

tomtom -no-ssc -verbosity 2 -text -dist pearson -min-overlap 7 -evalue -thresh 1 \
    "$STEP06_MEME" "$JASPAR_MEME" \
    > results/step06_jaspar_tomtom.txt

echo "JASPAR TOMTOM done → results/step06_jaspar_tomtom.txt"
echo "Next: python 01_jaspar_analysis.py"
