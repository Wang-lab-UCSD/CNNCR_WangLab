#!/usr/bin/env python
"""Create the work directory the notebooks use and point the archived scripts at it.

    python bin/setup_work_dir.py --enhancer-root /abs/path/work

Creates $ENHANCER_ROOT, links the repository's `code/` as $ENHANCER_ROOT/multi_agent and the small inputs shipped in `data/` and `figures/`
at the paths the scripts expect (see PATHS.md), and writes env.sh and .roots.json, which tell the scripts and the notebooks where that
directory is. Rerun after moving the repository or the work directory.
Zenodo archives are unpacked into $ENHANCER_ROOT afterwards (tar --keep-directory-symlink -xzf <archive> -C $ENHANCER_ROOT); ENCODE peaks: bin/download_encode_peaks.py."""
import os, argparse, json, sys, shutil
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--enhancer-root", required=True)
ap.add_argument("--repair-multi-agent", action="store_true",
                help="if $ENHANCER_ROOT/multi_agent is a real directory (an archive was unpacked without "
                     "--keep-directory-symlink), move its contents into the repository's code/ and restore the symlink")
a = ap.parse_args()
W = os.path.abspath(a.enhancer_root); os.makedirs(W, exist_ok=True)
def ln(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst): os.remove(dst) if not os.path.isdir(dst) or os.path.islink(dst) else None
    if not os.path.exists(dst): os.symlink(src, dst)

def merge_into(src_dir, dst_dir):
    """Move every file of src_dir into dst_dir, keeping the directory tree. Returns the number of files moved."""
    moved = 0
    for root, _, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        out = dst_dir if rel == "." else os.path.join(dst_dir, rel)
        os.makedirs(out, exist_ok=True)
        for f in files:
            shutil.move(os.path.join(root, f), os.path.join(out, f)); moved += 1
    return moved

# GNU tar replaces the multi_agent -> code/ symlink with a real directory unless it is given
# --keep-directory-symlink, which strands every archived file outside code/ (see README).
MA = f"{W}/multi_agent"
if os.path.isdir(MA) and not os.path.islink(MA):
    if a.repair_multi_agent:
        n = merge_into(MA, f"{here}/code"); shutil.rmtree(MA)
        print(f"repaired {MA}: moved {n} files into {here}/code and restored the symlink")
    else:
        sys.exit(f"""ERROR: {MA} is a real directory, not the symlink to {here}/code.
A Zenodo archive was unpacked without --keep-directory-symlink, so its files never reached code/
and notebooks 06, 08, 10 and 11 will report their inputs as missing. Fix it with

    python bin/setup_work_dir.py --enhancer-root {W} --repair-multi-agent

which moves those files into code/ and restores the symlink, and from now on unpack with
    tar --keep-directory-symlink -xzf <archive> -C {W}""")
ln(f"{here}/code", MA)
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
# The scripts read their data root from the environment; the notebooks read it from .roots.json.
json.dump({"enhancer_root": W, "seq_root": f"{W}/sequences"}, open(f"{here}/.roots.json", "w"), indent=1)
open(f"{here}/env.sh", "w").write(f"""# Written by bin/setup_work_dir.py. Source this before running anything in code/ or figures/
# outside the notebooks (the notebooks set both variables themselves from .roots.json).
export ENHANCER_ROOT="{W}"
export SEQ_ROOT="{W}/sequences"
""")
print(f"""work directory ready: {W}
wrote env.sh and .roots.json; for the scripts in code/ and figures/ run

    source {here}/env.sh

next: unpack the Zenodo archives you need into {W} (external_data/ZENODO_MANIFEST.md) and, for notebooks 08/10/11, run bin/download_encode_peaks.py --dest {W}/encode4_tf_chipseq""")
