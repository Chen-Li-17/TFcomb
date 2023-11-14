- Abstract
Cell state transitions are complicated processes that occur in various life activities. Understanding and artificially
manipulating them have been longstanding challenges. Substantial experiments reveal that the transitions could be
directed by several key transcription factors (TFs), and some computational methods have been developed to alleviate
the burden of biological experiments on identifying key TFs. However, most existing methods employ data with
resolution on the cell population, instead of individual cells, which will influence the identification quality due to cell
heterogeneity. Besides, they require collecting abundant samples for candidate cell states and are not available for
unknown cell states. As for the alternative single-cell analysis methods, they generally concentrate on differentially
expressed genes between cell states but can hardly identify key TFs responsible for directing cells state transition.
Here we present scDirect, a computational framework to identify key TFs based on single-cell multi-omics data.
scDirect models the TF identification task as a linear inverse problem, and solve it with gene regulatory networks
enhanced by a graph attention network. Through a benchmarking on a single-cell human embryonic stem cell atlas,
we systematically demonstrate the robustness and superiority of scDirect against other alternative single-cell analysis
methods on TF identification. With application on various single-cell datasets, scDirect exhibits high capability in
identifying key TFs in cell differentiation, somatic cell conversion, and cell reprogramming. Furthermore, scDirect
can quantitatively identify experimentally validated reprogramming TF combinations, which indicates the potential
of scDirect to guide and assist the experimental design in cellular engineering. We envision that scDirect could utilize
rapidly increasing single-cell datasets to identify key TFs for directing cell state transitions, and become an effective
tool to facilitate regenerative medicine and therapeutic discovery.
