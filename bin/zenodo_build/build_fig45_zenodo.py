#!/usr/bin/env python
"""Zenodo archives for Figures 4 and 5 (development-tree paths). Unpack under $ENHANCER_ROOT (multi_agent/ = code/).
  fig4_eqtl   -> fig4_eqtl_inputs.tar.gz : GTEx fine-mapped eQTL benchmark (Enformer release), tissue->biosample map, window coordinates,
                 core-motif and ten-control-set FIMO scans of the 26 eQTL biosamples (notebook 07, Fig. 4a,b)
  peaks       -> fig4_fig5_motif_sites_on_peaks.tar.gz : FIMO of the 43 core motifs + control pool over the TSS-proximal ChIP-seq peaks of
                 five cell lines (notebook 08, Fig. 4d) and the 86-motif scans of K562/HEK293 peaks (notebook 11)
  fig5        -> fig5_reader_switch_inputs.tar.gz : core-motif loci per cell line and shared loci (notebook 10), feasibility table, the
                 quantile-normalised Perturb-seq effect matrices (from export_perturbseq_qn.py) and the Fig. 5d-g result caches (notebook 11)
  pack        -> tar the three staging folders"""
import os, sys, glob, shutil, subprocess
# The data root: the work directory you created with bin/setup_work_dir.py and unpacked the
# GitHub and Zenodo data into. `source env.sh` exports it, or set it yourself (see PATHS.md).
ENHANCER_ROOT = os.environ.get("ENHANCER_ROOT")
if not ENHANCER_ROOT:
    raise SystemExit("set ENHANCER_ROOT to your work directory: source env.sh (see PATHS.md)")

E=ENHANCER_ROOT; MA=f"{E}/multi_agent"; G=f"{E}/github"; STG=f"{G}/_build/zenodo_staging"; OUT=f"{G}/zenodo_upload"
Z="pigz -p 8" if shutil.which("pigz") else "gzip"
def tar(name,cwd,items): subprocess.run(f"tar -I '{Z}' -cf {OUT}/{name} -C {cwd} {' '.join(items)}",shell=True,check=True); print("wrote",name,"%.2f GB"%(os.path.getsize(f"{OUT}/{name}")/1e9))
def cp(src,dst): os.makedirs(os.path.dirname(dst),exist_ok=True); shutil.copy2(src,dst)
cmd=sys.argv[1]
if cmd=="fig4_eqtl":
    S=f"{STG}/fig4"; shutil.rmtree(S,ignore_errors=True)
    B=sorted(os.path.basename(f).split("_")[0] for f in glob.glob(f"{MA}/posthoc_analysis/results/fimo_multi/set_0/*_fimo.txt"))
    for f in glob.glob(f"{E}/enformer_comparison/eqtl-enformer/*_pos.vcf")+glob.glob(f"{E}/enformer_comparison/eqtl-enformer/*_neg.vcf"): cp(f,f"{S}/enformer_comparison/eqtl-enformer/{os.path.basename(f)}")   # not the merged files
    cp(f"{E}/geuvadis/predixscan/en_models_to_BSSID.csv",f"{S}/geuvadis/predixscan/en_models_to_BSSID.csv")
    os.makedirs(f"{S}/data_generated",exist_ok=True)
    for b in B:
        subprocess.run(f"""awk -F, 'BEGIN{{OFS=","}} NR==1{{print; next}} {{$12=""; print}}' {E}/data_generated/{b}.coord.exp.df.final.csv > {S}/data_generated/{b}.coord.exp.df.final.csv""",shell=True,check=True)
        cp(f"{MA}/posthoc_analysis/results/fimo_universal/{b}_fimo.txt",f"{S}/multi_agent/posthoc_analysis/results/fimo_universal/{b}_fimo.txt")
        for i in range(10): cp(f"{MA}/posthoc_analysis/results/fimo_multi/set_{i}/{b}_fimo.txt",f"{S}/multi_agent/posthoc_analysis/results/fimo_multi/set_{i}/{b}_fimo.txt")
    for f in ("SNP_overlap.csv","wilcoxon_stats.csv"): cp(f"{MA}/enformer_snp_overlap/results/{f}",f"{S}/multi_agent/enformer_snp_overlap/results/{f}")
    open(f"{S}/README.txt","w").write(f"""Inputs of notebook 07 (Fig. 4a,b: fine-mapped causal eQTLs at core-motif sites). Unpack under $ENHANCER_ROOT.
enformer_comparison/eqtl-enformer/{{tissue}}_{{pos,neg}}.vcf   GTEx v8 fine-mapped eQTLs as released with Enformer (Avsec et al. 2021; gs://dm-enformer/data/gtex_fine/susie_pip90):
                                                        pos = causal (SuSiE PIP > 0.9), neg = matched non-causal (PIP < 0.01); 50 tissues; only chrom and position are used
geuvadis/predixscan/en_models_to_BSSID.csv               GTEx tissue -> EpiMap biosample
data_generated/{{BSSID}}.coord.exp.df.final.csv             promoter/enhancer window coordinates of the {len(B)} eQTL biosamples (sequence column emptied)
multi_agent/posthoc_analysis/results/fimo_universal/{{BSSID}}_fimo.txt        FIMO sites (p < 1e-4) of the 43 core motifs in the promoter windows
multi_agent/posthoc_analysis/results/fimo_multi/set_{{0..9}}/{{BSSID}}_fimo.txt  the same for each of the ten random control sets of 43 JASPAR motifs
multi_agent/enformer_snp_overlap/results/SNP_overlap.csv, wilcoxon_stats.csv   the published table and test (the notebook regenerates them)
""")
    tar("fig4_eqtl_inputs.tar.gz",S,["README.txt","enformer_comparison","geuvadis","data_generated","multi_agent"])
elif cmd=="peaks":
    S=f"{STG}/peaks"; shutil.rmtree(S,ignore_errors=True); CE=f"{MA}/chipseq_enrichment"
    for c in ("K562","HepG2","A549","GM12878","MCF.7"):
        for d in ("fimo_poolscan","fimo_poolscan_extra"): cp(f"{CE}/results/{d}/{c}/fimo_peaks/fimo.tsv",f"{S}/multi_agent/chipseq_enrichment/results/{d}/{c}/fimo_peaks/fimo.tsv")
    for c in ("K562","HEK293"): cp(f"{CE}/results/fimo/{c}/fimo_peaks/fimo.tsv",f"{S}/multi_agent/chipseq_enrichment/results/fimo/{c}/fimo_peaks/fimo.tsv")
    for f in ("scan_core_plus_pool.meme","scan_oldpool_extra.meme"): cp(f"{MA}/posthoc_analysis/results/{f}",f"{S}/multi_agent/posthoc_analysis/results/{f}")
    cp(f"{CE}/combined_motifs.meme",f"{S}/multi_agent/chipseq_enrichment/combined_motifs.meme")
    for f in glob.glob(f"{CE}/results/oldpool10/fig4d_*.csv"): cp(f,f"{S}/multi_agent/chipseq_enrichment/results/oldpool10/{os.path.basename(f)}")
    open(f"{S}/README.txt","w").write("""Motif sites in ENCODE TF ChIP-seq peaks (notebooks 08 and 11). Unpack under $ENHANCER_ROOT (multi_agent/ = code/).
Peaks = summit +/- 150 bp of every IDR-thresholded narrowPeak of every TF ChIP-seq experiment of the cell line (ENCODE4), restricted to
TSS +/- 10 kb (code/chipseq_enrichment/01_prepare_peaks.py builds them from the user's ENCODE download; the sequence ids here are those peaks).
multi_agent/chipseq_enrichment/results/fimo_poolscan/{cell}/fimo_peaks/fimo.tsv        FIMO (p < 1e-4) of the 43 core motifs + the 291-motif control pool
                                                                                   (posthoc_analysis/results/scan_core_plus_pool.meme) - K562, HepG2, A549, GM12878, MCF.7
multi_agent/chipseq_enrichment/results/fimo_poolscan_extra/{cell}/fimo_peaks/fimo.tsv  the 13 further control motifs (scan_oldpool_extra.meme)
multi_agent/chipseq_enrichment/results/fimo/{K562,HEK293}/fimo_peaks/fimo.tsv          FIMO of the 43 core + 43 published control motifs (combined_motifs.meme), read by notebook 11
multi_agent/chipseq_enrichment/results/oldpool10/fig4d_*.csv                        the published Fig. 4d tables (the notebook regenerates them)
""")
    tar("fig4_fig5_motif_sites_on_peaks.tar.gz",S,["README.txt","multi_agent"])
elif cmd=="fig5":
    S=f"{STG}/fig5"; CE=f"{MA}/chipseq_enrichment"
    for f in glob.glob(f"{CE}/reader_switch/out/loci/*.parquet"): cp(f,f"{S}/multi_agent/chipseq_enrichment/reader_switch/out/loci/{os.path.basename(f)}")
    for f in ("shared_loci.parquet","reader_binding_top90.parquet","reader_binding.parquet","per_motif_switch.csv",
              # popularity / open-locus build adopted 2026-10-18 as the paper's Fig. 5b,c (notebook 10)
              "reader_binding_percell.parquet","interrogated_percell.parquet","assayed_tfs.json",
              "per_motif_setswitch_open.csv","pairs_setswitch_open.csv"): cp(f"{CE}/reader_switch/out/{f}",f"{S}/multi_agent/chipseq_enrichment/reader_switch/out/{f}")
    cp(f"{CE}/results/reader_switching/hek_k562_switch_feasibility.csv",f"{S}/multi_agent/chipseq_enrichment/results/reader_switching/hek_k562_switch_feasibility.csv")
    for f in glob.glob(f"{CE}/results/prof_plan/*.csv")+glob.glob(f"{CE}/results/prof_plan/*.pkl"): cp(f,f"{S}/multi_agent/chipseq_enrichment/results/prof_plan/{os.path.basename(f)}")   # the published tables and plot caches, not the diagnostic figures
    assert os.path.exists(f"{S}/perturbseq_qn/perturbseq_qn_K562.npz"), "run export_perturbseq_qn.py first"
    open(f"{S}/README.txt","w").write("""Inputs of notebooks 10 and 11 (Fig. 5b,c and d-g). Unpack under $ENHANCER_ROOT (multi_agent/ = code/).
multi_agent/chipseq_enrichment/reader_switch/out/loci/{cell}.parquet   genomic loci of the 43 core motifs in the promoter windows of K562, HepG2, A549, GM12878, HEK293
                                                                        (FIMO p < 1e-4 sites of the promoter-window scans, mapped to hg38 coordinates)
multi_agent/chipseq_enrichment/reader_switch/out/shared_loci.parquet    the loci of the four EpiMap lines and HEK293 that coincide in all five, used by the
                                                                        archived all-five-lines build (fig5bc_nature.py) and by notebook 11
multi_agent/chipseq_enrichment/reader_switch/out/reader_binding*.parquet, per_motif_switch.csv   the published site-level reader tables (regenerated by notebooks 10/11)
multi_agent/chipseq_enrichment/reader_switch/out/interrogated_percell.parquet, reader_binding_percell.parquet, assayed_tfs.json
                                                                        the published per-cell-line tables of the paper's Fig. 5b,c: which gene-motif loci each line
                                                                        carries in ITS OWN promoter windows, which candidate readers bind them, and which TFs have a
                                                                        ChIP-seq experiment in each line (notebook 10 regenerates all three from the loci above)
multi_agent/chipseq_enrichment/reader_switch/out/per_motif_setswitch_open.csv, pairs_setswitch_open.csv   the published Fig. 5c bars and the per-pair switch rates
multi_agent/chipseq_enrichment/results/reader_switching/hek_k562_switch_feasibility.csv         which motifs have readers perturbed in both screens
multi_agent/chipseq_enrichment/results/prof_plan/                       the published Fig. 5d-g tables and plot caches (notebook 11 regenerates them; kept for comparison)
perturbseq_qn/perturbseq_qn_{K562,HEK293T}.npz                          quantile-normalised knockout-effect vectors of the two screens (see README_perturbseq_qn.txt)
""")
    tar("fig5_reader_switch_inputs.tar.gz",S,["README.txt","multi_agent","perturbseq_qn"])
