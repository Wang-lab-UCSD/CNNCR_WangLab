K562 (EpiMap BSS00762): the released CNN-CR model and its held-out predictions (notebooks 01, 02, 05).

BSS00762_it_0_256MOTIF_pytorch.pth               fold 0 of the ten released promoter+enhancer models (20 branches); notebook 02 extracts its motifs when notebook 01 was not run
BSS00762256MOTIF_pytorch_test_pred_all.npy       held-out predictions of the ten released models (10 x 1,836 genes); Fig. 1d is their median
BSS00762256MOTIF_pytorch_test_y_all.npy          the measured log2 FPKM of those genes
BSS00762256MOTIF_pytorch_test_corr_all_S.npy     held-out Spearman of each released model
BSS00762_universal_test_S.npy                    held-out Spearman of the published 43-motif CNN-CR of K562 (notebook 05 compares with it)
