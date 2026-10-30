# Where every input lives

The analysis scripts are archived as run. Each analysis lives in its own folder under `$ENHANCER_ROOT/multi_agent/`, writes to
`results/` inside that folder and reads the other analyses' outputs through that layout. The
repository keeps the folder names, so `$ENHANCER_ROOT/multi_agent` must be this repository's `code/` (a symlink). `bin/setup_work_dir.py
--enhancer-root DIR` creates the work directory, the symlink and every link in the first table, and writes `env.sh` (which exports
`ENHANCER_ROOT` and `SEQ_ROOT` for the scripts) and `.roots.json` (which the notebooks read). It can be rerun at any time.

## Shipped in the repository, linked into `$ENHANCER_ROOT` by `setup_work_dir.py`
| path under `$ENHANCER_ROOT` | from |
|---|---|
| `multi_agent/` | `code/` |
| `Epimap_BSSID_nondup.txt`, `EpiMapID_Name_nonDup.txt`, `EpiMapID_Name.txt`, `mnemonic_mapping.tsv` | `data/biosamples/` |
| `geuvadis/predixscan/en_models_to_BSSID.csv` | `data/biosamples/` |
| `gene_ENSG.txt`, `gene_ENSG_name.txt` | `data/genes/` |
| `genome_annotation/hg38_promoters_cage_corrected.bed`, `IGV_tracks/CAGE_TSS_hg38.bed` | `data/genes/` |
| `JASPAR2022_CORE_vertebrates_non_redundant_unique_names.meme` | `data/motifs/` |
| `motif_analysis/TF_Information.txt`, `motif_analysis/CISBP_motifs.meme` | `data/motifs/CISBP_TF_Information.txt`, `CISBP_human_motifs.meme` |
| `paper_figures/fig1`, `fig3`, `fig4`, `fonts` | `figures/` |

## Unpacked into `$ENHANCER_ROOT` from Zenodo (`tar --keep-directory-symlink -xzf <archive> -C $ENHANCER_ROOT`)
| archive | paths it creates | notebooks |
|---|---|---|
| `fig1_training_inputs_all116` | `fig1_training_inputs/{BSSID}.windows.fa.gz`, `{BSSID}.expression.tsv` | 01, 02, 05 for biosamples other than K562 (K562 ships in `data/k562_demo/`) |
| `fig1_motif_sets_all116` | `fig1_motif_sets/{BSSID}_{pro,enh}_motifs_nonredundant.meme`, `catalogue_tomtom/` | 02 |
| `fig2_core_motif_derivation` | `cell_type_specific_CNN/{BSSID}/*_pro_motifs_all_256.meme`, `*_weights_all_256.npy`; `training_y/{BSSID}.y.npy`; `multi_agent/motif_pipeline/results/` | 04 |
| `fig3_conservation_inputs` | `multi_agent/conservation_sequence/hg38.phyloP100way.promoter_windows.bw`, `results/pool_conservation*/`; `multi_agent/posthoc_analysis/results/pool_scan*/BSS00762_fimo.txt.gz`; `data_generated/BSS00762.coord.exp.df.final.csv`; `multi_agent/conservation_protein/{uniprot_sprot_human.dat.gz, gnomAD_v2.1.1_transcripts_with_rmc.tsv}` | 06 |
| `fig4_eqtl_inputs` | `enformer_comparison/eqtl-enformer/`, `geuvadis/predixscan/`, `data_generated/{26 biosamples}`, `multi_agent/posthoc_analysis/results/{fimo_universal, fimo_multi}/`, `multi_agent/enformer_snp_overlap/results/` | 07 |
| `fig4_fig5_motif_sites_on_peaks` | `multi_agent/chipseq_enrichment/results/{fimo_poolscan, fimo_poolscan_extra, fimo}/{cell}/fimo_peaks/fimo.tsv`, `results/oldpool10/` | 08, 11 |
| `fig5_reader_switch_inputs` | `multi_agent/chipseq_enrichment/reader_switch/out/`, `results/reader_switching/`, `results/prof_plan/`; `perturbseq_qn/` | 10, 11 |

## Downloaded by the user
| path under `$ENHANCER_ROOT` | how |
|---|---|
| `encode4_tf_chipseq/{cell line}/{TF}_human/{ENCFF}.bed` | `bin/download_encode_peaks.py --dest $ENHANCER_ROOT/encode4_tf_chipseq` (notebooks 08, 10, 11) |
| `genomes/hg38.fa` | the UCSC hg38 FASTA, only to re-run `code/chipseq_enrichment/reader_switch/s2_motif_loci.py`; the loci it writes ship in `fig5_reader_switch_inputs` |

`$SEQ_ROOT` (set to `$ENHANCER_ROOT/sequences` by `setup_work_dir.py`) holds two inputs no notebook reads: the per-biosample
promoter-window FASTAs that `code/posthoc_analysis/generate_10_controls.py` writes its FIMO job scripts against (the scripts are
written, not run, by notebook 04), and the raw X-Atlas/Orion download that `code/perturb_seq/orion_hek293t_process_replogle.py`
rebuilds the HEK293T Perturb-seq matrix from (notebook 11 reads the released matrix from Zenodo instead).
The external tools (`tomtom`, `bedtools`) are taken from `$PATH` (the `cnncr` environment).
