---
layout: default
title: Image Enhancement
parent: Vision
has_children: true
has_toc: false
permalink: /vision/image_enhancement
banner: "data/image_enhancement.png"
banner_y: 1
---

# Image Enhancement

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
  1. TOC
  {:toc}
</details>

---

Image enhancement improves the original image's quality and information content before processing.

In the new era of deep learning, deep image enhancement models can perform various tasks such as:

[Low Light Enhancement](low_light_enhancement.md){: .btn .btn-primary .fs-3 .mb-4 .mb-md-0 } 
[Deraining](deraining.md){: .btn .btn-primary .fs-3 .mb-4 .mb-md-0 } 
[Dehazing](dehazing.md){: .btn .btn-primary .fs-3 .mb-4 .mb-md-0 } 

---

## Methods

| Status | Method                            | Task                                       | Learning                                | Architecture                   | Network&nbsp;Structure | Loss                                                                                                                                                       | Format | Date | Publication                     |
|:------:| --------------------------------- | ------------------------------------------ | --------------------------------------- | ------------------------------ | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ---- | ------------------------------- |
|   ✅   | [**Zero&#8209;DCE**](zero_dce.md) | [Low&nbsp;Light](low_light_enhancement.md) | [ZSL](../machine_learning/zero_shot.md) | [CNN](../deep_learning/cnn.md) | U-Net like network     | Spatial&nbsp;consistency&nbsp;loss, <br> Exposure&nbsp;control&nbsp;loss, <br> Color&nbsp;constancy&nbsp;loss, <br> Illumination&nbsp;smoothness&nbsp;loss | RGB    | 2020 | CVPR&nbsp;2020, TPAMI&nbsp;2021 |
|   ✅   | **MPNet**                         | Derain, Desnow, Dehaze, Denoise            |                                         | [CNN](../deep_learning/cnn.md) |                        |                                                                                                                                                            |        | 2021 | CVPR&nbsp;2021                  |
|   ✅   | [**HINet**](hinet.md)             | Derain, Deblur, Denoise                    |                                         | [CNN](../deep_learning/cnn.md) |                        |                                                                                                                                                            |        | 2021 | CVPR&nbsp;2021                  |

---

## Image Restoration Pipeline