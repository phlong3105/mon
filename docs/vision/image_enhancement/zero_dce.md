---
layout      : default
title       : ZeroDCE
parent	    : Image Enhancement
grand_parent: Vision
has_children: false
has_toc     : false
permalink   : /vision/image_enhancement/hinet
---

# Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement

Chunle Guo, Chongyi Li, Jichang Guo, Chen Change Loy, Junhui Hou, Sam Kwong,
and Cong Runmin

CVPR 2020

[Website](https://li-chongyi.github.io/Proj_Zero-DCE.html){: .btn .fs-3 .mb-4 .mb-md-0 }
[Paper](data/zero_dce.pdf){: .btn .fs-3 .mb-4 .mb-md-0 }
[Supplement](data/zero_dce_sup.pdf){: .btn .fs-3 .mb-4 .mb-md-0 }
[Code](https://github.com/Li-Chongyi/Zero-DCE){: .btn .fs-3 .mb-4 .mb-md-0 }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
  1. TOC
  {:toc}
</details>

---

## Highlight

We propose the first low-light enhancement network that is **independent of
paired and unpaired training data**, thus avoiding the risk of overfitting.
As a result, our method generalizes well to various lighting conditions.

We design an image-specific curve that is able to **approximate pixel-wise
and higher-order curves by iteratively applying itself**. Such image-specific
curve can effectively perform mapping within a wide dynamic range.

We show the potential of training a deep image enhancement model in the
absence of reference images through task-specific non-reference loss functions
that indirectly evaluate enhancement quality. It is capable of processing images
in real-time **(about 500 FPS for images of size 640x480x3 on GPU)** and takes
only 30 minutes for training.

## Method

|                                                                                                                                                                                                                                ![data/zero_dce_framework.png](data/zero_dce_framework.png)                                                                                                                                                                                                                                |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| The pipeline of our method. (a) The framework of Zero-DCE. A DCE-Net is devised to estimate a set of best-fitting Light-Enhancement curves (LE-curves: LE(I(x);α)=I(x)+αI(x)(1-I(x))) to iteratively enhance a given input image. (b, c) LE-curves with different adjustment parameters α and numbers of iteration n. In (c), α1, α2, and α3 are equal to -1 while n is equal to 4. In each sub-figure, the horizontal axis represents the input pixel values while the vertical axis represents the output pixel values. |

## Ablation Studies

### 1. Contribution of Each Loss

|                                                      ![data/zero_dce_ablation_01.png](data/zero_dce_ablation_01.png)                                                       |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Ablation study of the contribution of each loss (spatial consistency loss Lspa, exposure control loss Lexp, color constancy loss Lcol, illumination smoothness loss LtvA). |

### 2. Effect of Parameter Settings

|                                                                 ![data/zero_dce_ablation_02.png](data/zero_dce_ablation_02.png)                                                                 |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Ablation study of the effect of parameter settings. l-f-n represents the proposed Zero-DCE with l convolutional layers, f feature maps of each layer (except the last layer), and n iterations. |

### 3. Impact of Training Data

|                                                                                                                                                                         ![data/zero_dce_ablation_03.png](data/zero_dce_ablation_03.png)                                                                                                                                                                          |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| To test the impact of training data, we retrain the Zero-DCE on different datasets: 1) only 900 low-light images out of 2,422 images in the original training set (Zero-DCELow), 2) 9,000 unlabeled low-light images provided in the DARK FACE dataset (Zero-DCELargeL), and 3) 4800 multi-exposure images from the data augmented combination of Part1 and Part2 subsets in the SICE dataset (Zero-DCELargeLH). |

### 4. Advantage of Three-channel Adjustment

|                 ![data/zero_dce_ablation_04.png](data/zero_dce_ablation_04.png)                 |
|:-----------------------------------------------------------------------------------------------:|
| Ablation study of the advantage of three-channel adjustment (RGB, CIE Lab, YCbCr color spaces). |

## Results

### 1. Visual Comparisons on Typical Low-light Images

| ![data/zero_dce_results_01.png](data/zero_dce_results_01.png) |
|:-------------------------------------------------------------:|


### 2. Visual Face Detection Results Before and After Enhanced by Zero-DCE

| ![data/zero_dce_results_02.png](data/zero_dce_results_02.png) |
|:-------------------------------------------------------------:|

### 3. Real Low-light Video with Variational Illumination Enanced by Zero-DCE

| ![data/zero_dce_results_03.png](data/zero_dce_results_03.png) |
|:-------------------------------------------------------------:|

### 4. Self-training (taking first 100 frames as training data) for Low-light Video Enhancement

| ![data/zero_dce_results_04.png](data/zero_dce_results_04.png) |
|:-------------------------------------------------------------:|

## Citation

```text
@Article{Zero-DCE,
    author  = {Chunle Guo and Chongyi Li and Jichang Guo and Chen Change Loy and Junhui Hou and Sam Kwong and Cong Runmin},
    title   = {Zero-reference deep curve estimation for low-light image enhancement},
    journal = {CVPR},
    pape    = {1780--1789},
    year    = {2020}
}
```
