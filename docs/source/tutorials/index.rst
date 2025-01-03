Tutorials
============================

Our tutorial starts from the basic data preprocessing to the main functions of TFcomb.
In our model, a cell-by-peak matrix is required to generate a base GRN. However, in most
cases the scATAC-seq data is provided with the fragment format.

We first give a tutorial of how to process the fragment files with R package ArchR. If
you already got the cell-by-peak matrix, you can directly skil to the section 2 to generate
the base GRN with python package CellOracle.

After obtaining the base GRN, in section 3 we provide the tutorial of processing scRNA-seq data
into a cell-by-gene matrix of AnnData format, which is for the TFcomb input.

In section 4, we show the example of how to identify the reprogramming TFs from fibroblasts to 
iPSCs via TFcomb.

.. toctree::
   :maxdepth: 2

   scATAC_process/index
   generate_baseGRN/index
   scRNA_process/index
   TFcomb_main/index