CNN-CR training inputs, one pair of files per EpiMap biosample (116 biosamples; biosamples.tsv gives the names)

{BSSID}.windows.fa.gz   the regulatory windows of the 18,356 genes, 400 bp each, header
                        >{gene_index}|{ENSG}|{slot}|{promoter|enhancer}|{chr}:{start}-{end}|{strand}   (hg38)
                        slot 0      = TSS +/- 200 bp (every gene has it)
                        slot 1-9    = up to nine promoter-proximal peak windows within TSS +/- 10 kb
                        slot 10-19  = up to ten enhancer windows (EpiTensor-linked)
                        A slot the gene does not have is absent from the file and is an all-zero window for the model.
{BSSID}.expression.tsv  gene_index, ENSG, log2 FPKM (the target; same gene order in every biosample)

Reading them back with code/cnncr_model/cnncr_io.py (read_windows_fasta, read_expression) gives exactly the
one-hot tensor (genes x 20 x 400 x 4) and target vector the paper's models were trained on.
The K562 pair (BSS00762) also ships in the GitHub repository (data/k562_demo/), so the notebook demonstration needs no download.
