# Dual Path Morph-UNet for Road and Building Segmentation from Satellite Images

This repository contains the official Tensorflow implementation of the Dual Path Morph-UNet as described in the paper [Dual Path Morph-UNet for Road and Building Segmentation from Satellite Images](https://ieeexplore.ieee.org/abstract/document/9526886). 

## Overview
In this work we attempt to incorporate classical image processing with learning-based methods for remote sensing to increase the model's interpretability and accuracy. We propose Dual Path Morph-UNet, a novel network, for road and building segmentation to achieve better results, wherein learnable 2D morphological operations replace convolutions. The network performance is streamlined by incorporating parallel dense and residual paths for efficient information propagation, resulting in lower feature redundancy. Our network outperforms the state-of-the-art method significantly on object segmentation by utilizing multi-scale high-level morphological features with only 0.4 million parameters. Furthermore, the architecture's highly modular structure makes it suitable for other domains with minimal changes required. 

## Network Architecture

<p align="left">
  <img src="./resources/Dual_Path_Block_fig.png" width="370" />
  <img src="./resources/DPM_UNet_fig.png" width="406" /> 
</p>

Interested readers are encouraged to go through the [supplementary material](./resources/Supplementary_Material.pdf) for a deeper dive into the results and derivations.
Please cite our paper if you find the code useful. 

```
@article{dey2021dual,
  title={Dual-Path Morph-UNet for Road and Building Segmentation From Satellite Images},
  author={Dey, Moni Shankar and Chaudhuri, Ushasi and Banerjee, Biplab and Bhattacharya, Avik},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2021},
  publisher={IEEE}
}

```
