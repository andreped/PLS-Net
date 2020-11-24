# PLS-Net (PyTorch)
Reimplementation of the PLS-Net architecture used for lung lobe segmentation in CT proposed by Lee et al. (2019).

The original paper can be found here: https://arxiv.org/abs/1909.07474

The implementation was used for this study by Bouget et al (2020):
https://arxiv.org/abs/2010.07002

### Cite:
If the source code is used in any scientific publication, please cite the following papers:
* Lee, Hoileong, T. Matin, F. Gleeson and V. Grau. “Efficient 3D Fully Convolutional Networks for Pulmonary Lobe Segmentation in CT Images.” ArXiv abs/1909.07474 (2019): n. pag.
* Bouget, D., André Pedersen, Sayied Abdol Mohieb Hosainey, Johanna Vanel, O. Solheim and I. Reinertsen. “Fast meningioma segmentation in T1-weighted MRI volumes using a lightweight 3D deep learning architecture.” ArXiv abs/2010.07002 (2020): n. pag.

### Requirements:
* torch 1.3.1 (CUDA 10.0)
* torchvision 0.4.2
* (including stuff like numpy and config_parser, but should be less dependent on those)

Also PyTorch-Lightning was tested with the architecture and worked fine. The version tested was: 0.7.3

**DISCLAIMER:** I was co-author in the study by Bouget et al.

