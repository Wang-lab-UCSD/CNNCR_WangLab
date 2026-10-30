How the Zenodo archives were built, kept as provenance and not meant to be run: these scripts read the full analysis tree,
of which the archives are the published subset. Their paths are written against $ENHANCER_ROOT like the rest of the repository.
build_fig1_zenodo.py  fig1_training_inputs_all116, fig1_motif_sets_all116 (uses code/cnncr_model/cnncr_io.py for the window FASTA format)
build_fig23_zenodo.py fig2_core_motif_derivation, fig3_conservation_inputs (subset_phylop.py builds the phyloP bigWig restricted to the promoter windows)
build_fig45_zenodo.py fig4_eqtl_inputs, fig4_fig5_motif_sites_on_peaks, fig5_reader_switch_inputs (export_perturbseq_qn.py writes the quantile-normalised Perturb-seq effect vectors)

The MEME suite records its full command line in its output, so before packing, each of the three rewrites the staged TOMTOM
tables to the $ENHANCER_ROOT form used everywhere else. The script that does it is not shipped here: it has to carry the
literal development paths to find them.
