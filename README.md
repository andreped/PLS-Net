<div align="center">
<img src="assets/hjelde2020lobe.png" width="800">
<h1 align="center">PLS-Net (PyTorch)</h1>
<h3 align="center">Reimplementation of the PLS-Net architecture used for lung lobe segmentation in CT proposed by Lee et al. (2019).</h3>

[![test](https://github.com/andreped/PLS-Net/actions/workflows/test.yml/badge.svg)](https://github.com/andreped/PLS-Net/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://zenodo.org/badge/DOI/10.1117/1.JMI.8.2.024002.svg)](https://doi.org/10.1117/1.JMI.8.2.024002)
</div>

The implementation was made for this study by [Bouget et al. (2021)](https://doi.org/10.1117/1.JMI.8.2.024002). The original implementation can be found [here](https://arxiv.org/abs/1909.07474).

## [Usage](https://github.com/andreped/PLS-Net#usage)

The source code was tested in Python 3.6 with CUDA 10.0.

### [Clone repo and install requirements](https://github.com/andreped/PLS-Net#clone-repo-and-install-requirements)
```
git clone git+https://github.com/andreped/PLS-Net
cd "PLS-Net"
pip install -r requirements.txt
```

### [Define network](https://github.com/andreped/PLS-Net#define-network)
```
from PLS_pytorch import PLS
network = PLS()
```

(Alternatively) in PyTorch-Lightning:
```
from PLS_lightning import PLS
network = PLS()
```

Disclaimer: Note that the Lightning implementation contains some hardcoded setup and Dataloaders, and thus only serves as an example. However, PyTorch implementation should work out-of-the-box.

## [How to cite](https://github.com/andreped/PLS-Net#how-to-cite)
If the source code is used in any scientific publication, please, cite the following papers:
```
@article{bouget2021code,
  author = {David Bouget and Andr{\'e} Pedersen and Sayied Abdol Mohieb Hosainey and Johanna Vanel and Ole Solheim and Ingerid Reinertsen},
  title = {{Fast meningioma segmentation in T1-weighted magnetic resonance imaging volumes using a lightweight 3D deep learning architecture}},
  volume = {8},
  journal = {Journal of Medical Imaging},
  number = {2},
  publisher = {SPIE},
  pages = {024002},
  keywords = {three-dimensional segmentation, deep learning, meningioma, magnetic resonance imaging, clinical diagnosis, Magnetic resonance imaging, Image segmentation, Tumors, Brain, 3D image processing, 3D modeling, Image resolution, Data modeling, Neural networks, Surgery},
  year = {2021},
  doi = {10.1117/1.JMI.8.2.024002},
  url = {https://doi.org/10.1117/1.JMI.8.2.024002}
}
```
```
@misc{lee2019plsnet,
  author = {Lee, Hoileong and Matin, Tahreema and Gleeson, Fergus and Grau, Vicente},
  title = {{Efficient 3D Fully Convolutional Networks for Pulmonary Lobe Segmentation in CT Images}},
  publisher = {arXiv},
  year = {2019},
  copyright = {arXiv.org perpetual, non-exclusive license},
  keywords = {Image and Video Processing (eess.IV), Computer Vision and Pattern Recognition (cs.CV), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences},
  doi = {10.48550/ARXIV.1909.07474},
  url = {https://arxiv.org/abs/1909.07474}
}
```
