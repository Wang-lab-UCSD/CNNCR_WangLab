Inputs of notebook 02, part 2 (de-replication of the motifs of the ten K562 models), so that TOMTOM need not be rerun.

BSS00762_it_{i}_256MOTIF_pytorch_{i}_{pro,enh}_motifs_all_256.meme     the 2,560 promoter / 2,560 enhancer PWMs extracted from model i (i = 0..9), section 2 output
BSS00762_it_{i}_256MOTIF_pytorch_{i}_{pro,enh}_motifs_weights_all_256.npy   their contextual weights
BSS00762_it_{i}_256MOTIF_pytorch_{pro,enh}_motifs_all_tomtom.txt.gz      TOMTOM of each model's PWMs against themselves (step 1 of the de-replication)
BSS00762_{pro,enh}_motifs_combined_1_tomtom.txt.gz                       TOMTOM of the pooled representatives of the ten models against themselves (step 3)
released/                                                       the pooled (combined) and the final non-redundant K562 motif sets as released, for comparison
The TOMTOM command behind every table:  tomtom -no-ssc -verbosity 2 -text -dist pearson -evalue -thresh 5 <motifs.meme> <motifs.meme>
(the enhancer PWMs of each model are stored sorted by weight, the promoter PWMs in filter order; the TOMTOM ids refer to the order in the MEME file)
