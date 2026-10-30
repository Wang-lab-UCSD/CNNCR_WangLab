#!/usr/bin/env python
"""Zenodo archives for Figures 2 and 3 (development-tree paths; not part of the repository).
  python build_fig23_zenodo.py fig2   -> fig2_core_motif_derivation.tar.gz : per-model promoter PWMs+weights of the 116 biosamples (the input of
                                        the core-motif pipeline), their expression vectors, and every intermediate of the pipeline
                                        (IC-filtered sets, self/JASPAR TOMTOM tables, representative sets, pooled set + its self-TOMTOM, the 43 motifs)
  python build_fig23_zenodo.py fig3   -> fig3_conservation_inputs.tar.gz : phyloP subset bigWig, K562 pool scans + window coordinates,
                                        per-biosample phyloP summaries of all 116 biosamples, UniProt + gnomAD RMC tables for panels f-h
Archives unpack under $ENHANCER_ROOT (multi_agent/ = the repository's code/)."""
import os, sys, glob, shutil, subprocess
# The data root: the work directory you created with bin/setup_work_dir.py and unpacked the
# GitHub and Zenodo data into. `source env.sh` exports it, or set it yourself (see PATHS.md).
ENHANCER_ROOT = os.environ.get("ENHANCER_ROOT")
if not ENHANCER_ROOT:
    raise SystemExit("set ENHANCER_ROOT to your work directory: source env.sh (see PATHS.md)")

E=ENHANCER_ROOT; MA=f"{E}/multi_agent"; G=f"{E}/github"; STG=f"{G}/_build/zenodo_staging"; OUT=f"{G}/zenodo_upload"
B=[b.strip() for b in open(f"{E}/Epimap_BSSID_nondup.txt") if b.strip()]
Z="pigz -p 8" if shutil.which("pigz") else "gzip"
def tar(name,cwd,items):
    subprocess.run(f"tar -I '{Z}' -cf {OUT}/{name} -C {cwd} {' '.join(items)}",shell=True,check=True); print("wrote",name,"%.2f GB"%(os.path.getsize(f"{OUT}/{name}")/1e9))
def cp(src,dst):
    os.makedirs(os.path.dirname(dst),exist_ok=True); shutil.copy2(src,dst)
if sys.argv[1]=="fig2":
    S=f"{STG}/fig2"; shutil.rmtree(S,ignore_errors=True)
    for b in B:
        for i in range(10):
            for f in (f"{b}_it_{i}_256MOTIF_pytorch_{i}_pro_motifs_all_256.meme",f"{b}_it_{i}_256MOTIF_pytorch_{i}_pro_motifs_weights_all_256.npy"):
                cp(f"{E}/cell_type_specific_CNN/{b}/{f}",f"{S}/cell_type_specific_CNN/{b}/{f}")
        cp(f"{E}/training_y/{b}.y.npy",f"{S}/training_y/{b}.y.npy")
        for f in glob.glob(f"{MA}/motif_pipeline/results/cell_motifs/{b}/*"): cp(f,f"{S}/multi_agent/motif_pipeline/results/cell_motifs/{b}/{os.path.basename(f)}")
    for f in ("all_feature_motifs.meme","all_feature_self_tomtom.txt","universal_regulator_motifs.meme"): cp(f"{MA}/motif_pipeline/results/{f}",f"{S}/multi_agent/motif_pipeline/results/{f}")
    cp(f"{MA}/motif_pipeline/results/plots/universal_group_coverage.png",f"{S}/multi_agent/motif_pipeline/results/plots/universal_group_coverage.png")
    open(f"{S}/README.txt","w").write("""Inputs and intermediates of the core-motif derivation (notebook 04), for the 116 EpiMap biosamples. Unpack under $ENHANCER_ROOT
(multi_agent/ is the repository's code/ folder, see PATHS.md).

cell_type_specific_CNN/{BSSID}/{BSSID}_it_{i}_256MOTIF_pytorch_{i}_pro_motifs_all_256.meme      the 2,560 promoter PWMs of model i (i = 0..9) of the biosample
cell_type_specific_CNN/{BSSID}/..._pro_motifs_weights_all_256.npy                               their contextual weights (notebook 02, part 1, for every model)
training_y/{BSSID}.y.npy                                                                        log2 FPKM of the 18,356 genes (the expression filter of step 3)
multi_agent/motif_pipeline/results/cell_motifs/{BSSID}/{BSSID}_ICfiltered.meme                  step 1: the 750 highest-weight filters of each model with IC > 10 bits (<= 7,500)
                                                        {BSSID}_self_tomtom.txt                 step 3a: TOMTOM of that set against itself (hours per biosample)
                                                        {BSSID}_jaspar_tomtom.txt               step 3b: TOMTOM of that set against JASPAR 2022
                                                        {BSSID}_cell_motifs.meme                step 3: the biosample's representative motifs (median 73)
multi_agent/motif_pipeline/results/all_feature_motifs.meme                                      step 4: the 8,441 representatives of all biosamples, pooled
multi_agent/motif_pipeline/results/all_feature_self_tomtom.txt                                  step 5: TOMTOM of the pooled set against itself (hours)
multi_agent/motif_pipeline/results/universal_regulator_motifs.meme                              step 6: the 43 core motifs (as released; the notebook regenerates them)
TOMTOM command behind every table: tomtom -no-ssc -verbosity 2 -text -dist pearson -min-overlap 7 -evalue -thresh 1 <query.meme> <target.meme>
""")
    tar("fig2_core_motif_derivation.tar.gz",S,["README.txt","cell_type_specific_CNN","training_y","multi_agent"])
elif sys.argv[1]=="fig3":
    S=f"{STG}/fig3"; shutil.rmtree(S,ignore_errors=True); CS=f"{MA}/conservation_sequence"
    cp(f"{G}/_build/fig3/phyloP100way_hg38_promoter_windows.bw",f"{S}/multi_agent/conservation_sequence/hg38.phyloP100way.promoter_windows.bw")
    for d in ("pool_scan","pool_scan_extra"): cp(f"{MA}/posthoc_analysis/results/{d}/BSS00762_fimo.txt.gz",f"{S}/multi_agent/posthoc_analysis/results/{d}/BSS00762_fimo.txt.gz")
    os.makedirs(f"{S}/data_generated",exist_ok=True)
    subprocess.run(f"""awk -F, 'BEGIN{{OFS=","}} NR==1{{print; next}} {{$12=""; print}}' {E}/data_generated/BSS00762.coord.exp.df.final.csv > {S}/data_generated/BSS00762.coord.exp.df.final.csv""",shell=True,check=True)
    for d in ("pool_conservation","pool_conservation_extra"):
        for f in glob.glob(f"{CS}/results/{d}/*_motif_summary.tsv"): cp(f,f"{S}/multi_agent/conservation_sequence/results/{d}/{os.path.basename(f)}")
    for f in ("uniprot_sprot_human.dat.gz","gnomAD_v2.1.1_transcripts_with_rmc.tsv"): cp(f"{MA}/conservation_protein/{f}",f"{S}/multi_agent/conservation_protein/{f}")
    open(f"{S}/README.txt","w").write("""Inputs of the Figure 3 notebook (06). Unpack under $ENHANCER_ROOT (multi_agent/ is the repository's code/ folder, see PATHS.md).

multi_agent/conservation_sequence/hg38.phyloP100way.promoter_windows.bw   UCSC phyloP100way (hg38) restricted to the union of the promoter windows (slots 0-9) of the
                                                                          116 biosamples; identical values to the full 9 GB file inside those windows
multi_agent/posthoc_analysis/results/pool_scan/BSS00762_fimo.txt.gz       FIMO sites of the 43 core motifs + the 289-motif control pool in the K562 promoter windows
multi_agent/posthoc_analysis/results/pool_scan_extra/BSS00762_fimo.txt.gz the 13 further pool motifs (all 116 biosamples: Zenodo fig3_pool_scans.tar.gz)
data_generated/BSS00762.coord.exp.df.final.csv                            K562 window coordinates (sequence column emptied; all biosamples: window_coordinates.tar.gz)
multi_agent/conservation_sequence/results/pool_conservation{,_extra}/{BSSID}_motif_summary.tsv
                                                                          per-biosample phyloP summaries (median/mean per motif after the best-per-gene collapse) of all
                                                                          116 biosamples = the output of 05_pool_conservation.py, from which panels c-e are computed
multi_agent/conservation_protein/uniprot_sprot_human.dat.gz               UniProt Swiss-Prot human entries (DNA-binding domain annotations, panel g)
multi_agent/conservation_protein/gnomAD_v2.1.1_transcripts_with_rmc.tsv   gnomAD v2.1.1 regional missense constraint (panel g)
""")
    tar("fig3_conservation_inputs.tar.gz",S,["README.txt","data_generated","multi_agent"])
