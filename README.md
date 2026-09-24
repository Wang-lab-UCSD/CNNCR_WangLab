# A compact core motif code is differentially interpreted across human cell types

Code and data to reproduce the figures of

> Shen S, Wang W. *A compact core motif code is differentially interpreted across human cell types.* (2026, in preparation)

CNN-CR (a convolutional network with contextual regression) learns transcription-factor motifs de novo while predicting gene
expression from a gene's promoter and enhancer windows. Applied to 116 EpiMap biosamples it yields a motif vocabulary per biosample;
clustering the vocabularies gives **43 core promoter motifs** that recur in at least 25 biosamples, retain most of the predictive
signal, are conserved, mark fine-mapped causal variants, generalise to a held-out cell line, and are read by different transcription
factors in different cell types.

Everything runs from **Jupyter notebooks, one per analysis step** (`notebooks/`). Each notebook runs the archived analysis scripts
(`code/`, the code used for the paper) from the right folder, explains each step and shows what it produces. **K562 (EpiMap
biosample BSS00762) is the worked example** wherever an analysis runs per biosample. Slow steps (FIMO and TOMTOM over 116 biosamples,
per-biosample clustering, genome-wide scans) are never rerun: their outputs are released on Zenodo and the notebooks start from them,
with the command that produced them documented and, where feasible, runnable behind a flag.

## Setup

```bash
git clone https://github.com/shen420/CNNCR_WangLab.git && cd CNNCR_WangLab
conda env create -f environment.yml          # ~15 min; needs conda >= 23.10 (libmamba solver) or mamba/micromamba. CPU-only: replace pytorch-cuda=11.8 by cpuonly
conda activate cnncr
python bin/setup_work_dir.py --enhancer-root /abs/path/work    # creates the work directory and points the scripts at it
jupyter lab notebooks/                        # start with 00_setup_and_data
```
Then unpack into the work directory the Zenodo archives the notebooks you want need (`external_data/ZENODO_MANIFEST.md`), and for
notebooks 08, 10 and 11 download the ENCODE peak files (`python bin/download_encode_peaks.py --dest /abs/path/work/encode4_tf_chipseq`, 1 GB).

Tested on Linux (RHEL 8) with the environment built by micromamba 2.9; the `cnncr` environment holds Python 3.9, PyTorch 2.2 (CUDA 11.8),
the MEME suite 5.5.9 (TOMTOM), bedtools, pyBigWig and the usual scientific stack. Notebooks 01, 05 and 09 train networks and want one
NVIDIA GPU (they run on a CPU, an hour or more); everything else is CPU.

## The notebooks

| notebook | figure | what it does | inputs beyond the repository | hardware, time |
|---|---|---|---|---|
| `00_setup_and_data` | | environment check, work directory, which archive each notebook needs | | |
| `01_train_cnncr` | 1c,d | window FASTA of K562 (shipped) -> one-hot tensor -> train and test the cell-type-specific CNN-CR | none (other biosamples: `fig1_training_inputs_all116`) | GPU, 85 min per model |
| `02_motif_extraction_and_analysis` | 1e,f | extract the trained model's motifs; merge the ten K562 models' motifs into the non-redundant K562 set (shipped TOMTOM tables); motif yield and catalogue concordance of all 116 biosamples | `fig1_motif_sets_all116` (part 3 only) | GPU 10 min + CPU 15 min |
| `03_linear_model` | 1h | K562 motif-count matrices (shipped) -> ridge model -> held-out predictions | none | CPU, 2 min |
| `04_core_motifs` | 2a-c | the 43 core motifs start to end from the per-biosample motifs of the 116 biosamples; JASPAR readers; control motif sets | `fig2_core_motif_derivation` | CPU, 20 min |
| `05_core43_cnncr` | 2e | the 43-motif (motif-initialised) CNN-CR trained and tested start to end on K562 | none | GPU, 3 min |
| `06_fig3_conservation` | 3c-h | phyloP at core-motif sites versus ten control sets (K562 demonstration on the phyloP subset, all 116 from shipped summaries); reader-protein constraint | `fig3_conservation_inputs` | CPU, 5 min |
| `07_eqtl_variants` | 4a,b | fine-mapped causal eQTLs at core versus control motif sites in 16 GTEx tissues | `fig4_eqtl_inputs` | CPU, 10 min |
| `08_chipseq_occupancy` | 4d | ENCODE TF ChIP-seq peaks -> fraction of each TF's TSS-proximal peaks with a core versus control site (five cell lines) | ENCODE peaks, `fig4_fig5_motif_sites_on_peaks` | CPU, 15 min |
| `09_hek293_heldout` | 4e | HEK293, a line the model never saw: linear, random forest and the 43-motif CNN-CR on the shipped HEK293 windows | none | GPU, 10 min |
| `10_reader_sites_chipseq` | 5b,c | which candidate reader TF binds each shared core-motif site in five cell lines, and how often the strongest binder switches | ENCODE peaks, `fig5_reader_switch_inputs` | CPU, 5 min |
| `11_reader_switching_perturbseq` | 5d-g | reader dominance within K562 and HEK293T and switching between them, from the quantile-normalised Perturb-seq screens | ENCODE peaks, `fig5_reader_switch_inputs`, `fig4_fig5_motif_sites_on_peaks` | CPU, 10 min |

Every notebook was executed end to end from a fresh clone before release and reproduces the corresponding numbers of the paper; where a
regenerated value can differ from the published one (GPU arithmetic in training, a newer MEME-suite version in TOMTOM, one more ENCODE
experiment per cell line than at publication) the notebook says so and prints both.

Not reproduced here: Fig. 1a,b,g (schematics and the top-50 motif composition), Fig. 3a,b (CG and information content of the MEME files),
Fig. 4c,f,g and the synovial-fibroblast half of 4e (patient data are controlled access), Fig. 5a (schematic).

## Data

**In this repository** (`data/`): the K562 training input (window FASTA and expression, 20 MB), the K562 demonstration data of the
motif de-replication and the count matrices, the released K562 model and predictions, the 43 core motifs, the four motif catalogues
(JASPAR 2022 CORE vertebrates, CIS-BP 2.0 human, Epimotifs), gene lists and TSS, the biosample tables (Supplementary Tables ST2, ST3),
the ENCODE accession table. Per-analysis result tables that the panels read ship in place under `code/<analysis>/results/`.

**On Zenodo** (DOI to be added; `external_data/ZENODO_MANIFEST.md`): seven archives, one to three per figure, 9 GB in total, unpacked
into the work directory (three of them are deposited as 0.9 GB pieces to be joined with `cat`, see the manifest). Every large input of every notebook is there, including the outputs of every step too slow for a notebook.

**ENCODE** (not redistributed): the 1,348 TF ChIP-seq IDR-thresholded peak files of K562, HepG2, A549, GM12878, MCF-7 and HEK293 listed in
`data/encode/encode_tf_chipseq_peaks.tsv`, fetched by `bin/download_encode_peaks.py`. Please cite the ENCODE consortium when using them.

No other download is needed: the phyloP subset, the UniProt and gnomAD tables, the GTEx fine-mapped eQTL benchmark (the release of
Avsec et al. 2021) and the processed Perturb-seq screens (Replogle et al. 2022; X-Atlas/Orion) are in the Zenodo archives.

## Layout

`code/<analysis>/` are the analysis folders as run for the paper; the work directory's `multi_agent/` links to `code/`, so the
scripts' relative references between folders keep working, and `bin/rewrite_paths.py` (run by `setup_work_dir.py`) rewrites the two
absolute roots of the development machine to your work directory. `PATHS.md` lists where every input lives. `figures/fig{1,3,4}/`
hold the panel scripts, `bin/zenodo_build/` the scripts that built the archives (provenance), `MANIFEST.csv` the origin of every file.

## Citation and licence

Code: MIT licence (`LICENSE`). Data: the Zenodo record's licence; the ENCODE, GTEx, JASPAR, CIS-BP, UniProt, gnomAD, EpiMap and
Perturb-seq resources under their own terms. `CITATION.cff` gives the citation of this repository.
