# PLS-Net (PyTorch)
Reimplementation of the PLS-Net architecture used for lung lobe segmentation in CT proposed by Lee et al. (2019).

The original paper can be found here: https://arxiv.org/abs/1909.07474

The implementation was used for this study by Bouget et al (2021):
https://doi.org/10.1117/1.JMI.8.2.024002

## Requirements:
* torch 1.3.1
* torchvision 0.4.2
* pytorch-lightning 0.7.3

Tested with CUDA 10.0.

## Cite:
If the source code is used in any scientific publication, please, cite the following papers:
* Lee, Hoileong, T. Matin, F. Gleeson and V. Grau. “Efficient 3D Fully Convolutional Networks for Pulmonary Lobe Segmentation in CT Images.” ArXiv abs/1909.07474 (2019): n. pag.
* David Bouget, André Pedersen, Sayied Abdol Mohieb Hosainey, Johanna Vanel, Ole Solheim, Ingerid Reinertsen, "Fast meningioma segmentation in T1-weighted magnetic resonance imaging volumes using a lightweight 3D deep learning architecture," J. Med. Imag. 8(2) 024002 (26 March 2021) https://doi.org/10.1117/1.JMI.8.2.024002
