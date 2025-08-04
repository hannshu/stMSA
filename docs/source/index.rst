Welcome to stMSA's documentation!
===================================

.. image:: https://cdn.jsdelivr.net/gh/hannshu/stMSA/framework.png
   :alt: stMSA overview

Overview
--------

Spatial transcriptomics (ST) is a valuable methodology that integrates spatial location data with gene expression information, generating novel insights for biological research. Given the substantial number of ST datasets available, researchers are becoming more inclined to unveil potential biological features across larger datasets, thus obtaining a more comprehensive perspective. However, existing methods predominantly concentrate on cross-batch feature learning, disregarding the intricate spatial patterns within individual slices. Consequently, effectively integrating features across different slices while considering the slice-specific patterns poses a substantial challenge. To overcome this limitation and enhance the integration performance of multi-slice data, we propose a deep graph-based auto-encoder model incorporating contrastive learning techniques, named stMSA. This model is specifically tailored to generate batch-corrected representations while preserving the unique spatial patterns within each slice. It achieves this by simultaneously considering both inner-batch and cross-batch patterns during the integration process. We observe that stMSA surpasses existing state-of-the-art methods in discerning domain structures and cross-batch tissue structures across different slices, even when confronted with diverse experimental protocols and sequencing technologies. Furthermore, the representations learned by stMSA exhibit outstanding performance in matching two slices in the development dataset of a mouse embryo and aligning multi-slice mouse brain coronal sections.


Contents
--------

.. toctree::

   installation
   train_clustering
   imbalanced
   rna_protein
   cross_batch_matching
   multi_slice_alignment_representation_lerning
   multi_slice_alignment
   evaluate_model
