Motif count matrices of K562 (BSS00762) for notebook 03 (Fig. 1h demonstration): BSS00762_count_matrices.npz, int16 arrays

  {pro,enh}_cnncr_{weighted,unweighted}    genes (18,356, the order of data/genes/gene_ENSG.txt) x the biosample's non-redundant de novo motifs
                                          (736 promoter, 621 enhancer motifs; data/k562_demo/dereplication/released/)
  {pro,enh}_jaspar_{weighted,unweighted}   the same with the 841 JASPAR 2022 CORE vertebrate motifs (data/motifs/)

How they were built: every promoter (enhancer) window of every gene was scanned with FIMO (fimo --no-qvalue --norc --text,
sites with p < 1e-4) and the sites of each motif were counted per gene. 'weighted' (peak-weighted): a site is counted once for
every window that covers it, so a site under two overlapping peak windows counts twice; 'unweighted': every genomic site counts once.
The target is data/k562_demo/BSS00762.expression.tsv (log2 FPKM).
