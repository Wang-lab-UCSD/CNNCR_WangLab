#!/usr/bin/env python
"""Create the work directory the notebooks use and point the archived scripts at it.

    python bin/setup_work_dir.py --enhancer-root /abs/path/work

Creates $ENHANCER_ROOT, links the repository's `code/` as $ENHANCER_ROOT/multi_agent and the small inputs shipped in `data/` and `figures/`
at the paths the archived scripts expect (see PATHS.md), then runs bin/rewrite_paths.py. Rerun after moving the repository or the work directory.
Zenodo archives are unpacked into $ENHANCER_ROOT afterwards (tar -xzf <archive> -C $ENHANCER_ROOT); ENCODE peaks: bin/download_encode_peaks.py."""
import os, argparse, subprocess, sys
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--enhancer-root", required=True); a = ap.parse_args()
W = os.path.abspath(a.enhancer_root); os.makedirs(W, exist_ok=True)
def ln(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst): os.remove(dst) if not os.path.isdir(dst) or os.path.islink(dst) else None
    if not os.path.exists(dst): os.symlink(src, dst)
ln(f"{here}/code", f"{W}/multi_agent")
for f in ("Epimap_BSSID_nondup.txt", "EpiMapID_Name_nonDup.txt", "EpiMapID_Name.txt", "mnemonic_mapping.tsv"): ln(f"{here}/data/biosamples/{f}", f"{W}/{f}")
ln(f"{here}/data/biosamples/en_models_to_BSSID.csv", f"{W}/geuvadis/predixscan/en_models_to_BSSID.csv")
for f in ("gene_ENSG.txt", "gene_ENSG_name.txt"): ln(f"{here}/data/genes/{f}", f"{W}/{f}")
ln(f"{here}/data/genes/hg38_promoters_cage_corrected.bed", f"{W}/genome_annotation/hg38_promoters_cage_corrected.bed")
ln(f"{here}/data/genes/CAGE_TSS_hg38.bed", f"{W}/IGV_tracks/CAGE_TSS_hg38.bed")
ln(f"{here}/data/motifs/JASPAR2022_CORE_vertebrates_non_redundant_unique_names.meme", f"{W}/JASPAR2022_CORE_vertebrates_non_redundant_unique_names.meme")
ln(f"{here}/data/motifs/CISBP_TF_Information.txt", f"{W}/motif_analysis/TF_Information.txt")
ln(f"{here}/data/motifs/CISBP_human_motifs.meme", f"{W}/motif_analysis/CISBP_motifs.meme")
for d in ("fig1", "fig3", "fig4", "fonts"): ln(f"{here}/figures/{d}", f"{W}/paper_figures/{d}")
os.makedirs(f"{W}/paper_figures/fig5", exist_ok=True)      # the Fig. 5 sheet is copied here by 09k4_fig5_defg.py
subprocess.run([sys.executable, f"{here}/bin/rewrite_paths.py", "--enhancer-root", W, "--seq-root", f"{W}/sequences"], check=True)
print(f"work directory ready: {W}\nnext: unpack the Zenodo archives you need into {W} (external_data/ZENODO_MANIFEST.md) and, for notebooks 08/10/11, run bin/download_encode_peaks.py --dest {W}/encode4_tf_chipseq")
