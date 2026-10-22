# What each notebook should print

The notebooks ship without stored outputs, so this table is the reference to compare a run against.
Values for notebooks 00-09 come from an independent rerun on a second machine (RHEL 8, conda 23.9,
RTX A6000), from a fresh clone and the Zenodo archives; values for notebooks 10 and 11 come from the
Section 8 rebuild of October 2026. Where a number can move between runs, the reason is given.

| notebook | what it checks | expected |
|---|---|---|
| **00** | imports, tools, work directory, archives | all import and data checks `ok`; 116 biosamples |
| **01** | held-out Spearman of one newly trained K562 model | **0.790-0.805** (published single models), e.g. 0.801 in 58 epochs. GPU arithmetic and a fresh seeded gene split move this by about +/-0.01; the published 0.806 of Fig. 1d is the ten-model ensemble, which `N_MODELS = 1` does not build |
| **02** (1.3) | re-extraction from the released fold-0 weights | promoter **2560/2560 PWMs identical**; enhancer 2408/2560 identical and 2477 within 0.1 (activation ties break differently per GPU); weights Spearman **1.0000** |
| **02** (2.1-2.2) | de-replication of the ten K562 models | enhancer **3076 pooled -> 621 non-redundant, identical sets**; promoter **3528 -> 729** against 3534 -> 736 released, 567/736 PWMs recovered. Expected, and explained in section 2.2: the shipped per-model TOMTOM tables were computed on a different extraction run than the shipped PWM files |
| **02** (3) | motif yield and catalogue concordance, 116 biosamples | median **705** promoter and **684** enhancer motifs; concordance promoter **62% -> 94%**, enhancer **71% -> 95%** (61.7 -> 93.9 and 71.0 -> 95.3 on rerun) |
| **02** (3.3) | TOMTOM rerun, K562 promoter vs JASPAR | **0.518** |
| **03** | ridge on K562 motif counts, six values | pro+enh / pro / enh, peak-weighted **0.763 / 0.755 / 0.420**, unweighted **0.743 / 0.735 / 0.422**. JASPAR rows 0.748 / 0.741 / 0.425 and 0.730 / 0.725 / 0.429, +/-0.003 from the random JASPAR subset |
| **04** | the core motifs, start to end | **43 of 43 released PWMs regenerated as an identical set** from 8,441 pooled representatives; **287** JASPAR reader entries; the ten control sets identical |
| **05** | 43-motif CNN-CR on K562 | **0.789** published, 0.784-0.794 on rerun (GPU arithmetic) |
| **06** (3c-h) | phyloP and protein constraint | core **0.257** vs control **0.174** median phyloP; CG-adjusted advantage **0.085**; core > every control set in **116/116** biosamples; pLI **0.706 vs 0.095**, P = 3.9e-12; DBD comparison P = 2.9e-4. The last cell draws the panels and must now run to completion |
| **07** | causal eQTLs at core vs control sites | core median **53** vs control **18** causal eQTLs; **16/16** tissues; P = **1.5e-5**; the regenerated `SNP_overlap.csv` is identical to the published one |
| **08** | fraction of peaks with a site | pooled core **7.21%** vs control **3.64%** (**1.98x**), P = 7.3e-9, 10/10 control sets P < 0.05. GM12878, MCF-7 and A549 identical; K562 and HepG2 within 0.021 percentage points, because ENCODE has since released one further experiment for each - the notebook prints the measured difference |
| **09** | HEK293 held out | linear **0.659**, random forest **0.666**, CNN-CR **0.728** (0.727 on rerun); percentiles 69th / 63rd / 65th |
| **10** (10.2) | per-cell-line motif loci | **246,571** (K562), **248,614** (HepG2), **277,309** (A549), **243,157** (GM12878), **317,970** (HEK293); **375,180** distinct (gene, motif) pairs |
| **10** (10.3) | reader binding per cell line | **90,108** (cell, gene, motif, reader) records over **1,333,621** scanned loci |
| **10** (10.4) | Fig. 5b,c | panel b motifs **35, 14, 1, 36**; 12 motifs in panel c, switch rate **3.9% to 72.2%**, **53.1% pooled over 25,608 gene comparisons** (53.1% over 25,621 pooling every motif). The regenerated `per_motif_setswitch_open.csv` is byte-identical to the published one |
| **11** (11.2) | reader target genes | **162,027** (cell, gene, motif, reader) records |
| **11** (11.3) | the chain | `dominant_reader_top90chip.csv` has **19** rows and the same motifs and leaders as the published table; **8** motifs in the pair cache |
| **11** (11.4) | Fig. 5d-g | panel f draws the four widest reciprocal motifs as all-reader curves; `switch_figready_pairs_top90chip.csv` lists motifs **29, 14, 1, 12, 35, 19, 36, 15** |

Out of scope by design (stated in the README): Fig. 1a,b,g, Fig. 3a,b, Fig. 4c,f,g, the
synovial-fibroblast half of Fig. 4e, Fig. 5a, the simulated-data control of Section 2, and the
supplementary figures.

Known harmless warnings: PyTorch 2.2 prints a one-time cuDNN `install nvrtc.so` notice on the first
GPU call, and `combine_motifs_1.py enh` prints `RuntimeWarning: invalid value encountered in divide`
for PWM columns that are constant.
