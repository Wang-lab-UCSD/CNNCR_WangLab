# Zenodo archives (DOI to be added)

Seven gzip-compressed tar archives, 9.0 GB in total; each unpacks into the work directory (`tar -xzf <archive> -C $ENHANCER_ROOT`;
`multi_agent/` inside an archive is the repository's `code/`). Every archive carries a `README.txt` describing its files; three archives are deposited as 0.9 GB pieces (see **Split archives** below). Nothing else has
to be downloaded except the ENCODE peak files (`bin/download_encode_peaks.py`).

| archive | contents | GB | notebooks |
|---|---|---|---|
| `fig1_training_inputs_all116.tar.gz` | the CNN-CR training input of every biosample: `{BSSID}.windows.fa.gz` (the 400-bp promoter and enhancer windows of the 18,356 genes, header `gene_index|ENSG|slot|type|chr:start-end|strand`) and `{BSSID}.expression.tsv` (log2 FPKM); `code/cnncr_model/cnncr_io.py` turns a pair into the one-hot tensor the models were trained on. The K562 pair ships in the repository | 2.6 | 01, 02, 05 (other biosamples) |
| `fig1_motif_sets_all116.tar.gz` | the non-redundant promoter and enhancer motif vocabulary of every biosample (notebook 02 output) with the median contextual weights, and the TOMTOM tables of every set against JASPAR 2022, CIS-BP 2.0 and the two Epimotifs catalogues | 0.1 | 02 (Fig. 1e,f) |
| `fig2_core_motif_derivation.tar.gz` | the input and every intermediate of the core-motif derivation: the ten models' promoter PWMs and contextual weights of every biosample, the expression vectors, the IC-filtered sets, the self- and JASPAR-TOMTOM tables, the representative motif sets, the pooled 8,441-motif set with its self-TOMTOM table, and the released 43 motifs | 0.6 | 04 |
| `fig3_conservation_inputs.tar.gz` | phyloP100way (hg38) restricted to the promoter windows of the 116 biosamples (176 MB; identical values to the full UCSC file inside the windows), the K562 FIMO sites of the core and control-pool motifs with the K562 window coordinates, the per-biosample phyloP summaries of all 116 biosamples, UniProt Swiss-Prot human and the gnomAD v2.1.1 regional missense constraint table | 0.34 | 06 |
| `fig4_eqtl_inputs.tar.gz` | the GTEx fine-mapped eQTL benchmark as released with Enformer (Avsec et al. 2021; causal PIP > 0.9 and matched non-causal variants, 49 tissues), the tissue-to-biosample map, the window coordinates of the 26 eQTL biosamples and their FIMO sites of the 43 core motifs and of the ten control sets | 1.4 | 07 |
| `fig4_fig5_motif_sites_on_peaks.tar.gz` | FIMO sites of the 43 core motifs and the control pool in the TSS-proximal ENCODE TF ChIP-seq peaks of K562, HepG2, A549, GM12878 and MCF-7, and of the core and published control motifs in K562 and HEK293 peaks; the published Fig. 4d tables | 3.0 | 08, 11 |
| `fig5_reader_switch_inputs.tar.gz` | the core-motif loci of the five ChIP-profiled lines and the loci shared by all five; the quantile-normalised knockout-effect vectors of the K562 (Replogle et al. 2022) and HEK293T (X-Atlas/Orion) Perturb-seq screens on their 8,236 shared genes; the reader-switch feasibility table; the published site-level reader tables and Fig. 5d-g tables | 0.92 | 10, 11 |

**Split archives.** Three archives are larger than 1 GB and are deposited as 0.9 GB pieces named `<archive>.tar.gz.part00`, `.part01`, ...
(`fig1_training_inputs_all116`: 3 pieces, `fig4_eqtl_inputs`: 2, `fig4_fig5_motif_sites_on_peaks`: 4). Download all pieces of an archive
and join them before unpacking:
```bash
cat fig4_eqtl_inputs.tar.gz.part* > fig4_eqtl_inputs.tar.gz
md5sum -c CHECKSUMS.md5 --ignore-missing      # CHECKSUMS.md5 (in the record) lists the md5 of every whole archive
tar -xzf fig4_eqtl_inputs.tar.gz -C $ENHANCER_ROOT
```

The archives were built by the scripts in `bin/zenodo_build/` (development-machine paths, kept as provenance).

Not redistributed: the ENCODE TF ChIP-seq peak files (accessions in `data/encode/encode_tf_chipseq_peaks.tsv`), and the raw Perturb-seq
screens (Replogle et al. 2022 K562 normalised bulk profiles; the X-Atlas/Orion HEK293T parquet shards on HuggingFace, processed by
`code/perturb_seq/orion_hek293t_process_replogle.py`), which are needed only to rebuild the released effect vectors.
