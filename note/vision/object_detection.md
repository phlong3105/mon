---
layout: default
title: Object Detection
parent: Vision
has_children: true
has_toc: false
permalink: /vision/object_detection
banner: "data/object_detection.png"
banner_y: 0.492
---

# Object Detection

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
  1. TOC
  {:toc}
</details>

---

The ultimate purpose of object detection is to locate important items, draw rectangular bounding boxes around them, and determine the class of each item discovered.

Object detection is often called object recognition, object identification, image detection, and these concepts are synonymous.

|               ![](../data/object_detection_01.gif)               | 
|:----------------------------------------------------------------:|
| Here is a demo of object detection applied in autonomous vehicle |

---

## Methods

| ![](../data/milestones.png) |
|:-------------------------------------------:|
|    Major milestones of object detection     | 
 
|   ![](../data/object_detection_sota.png)   |
|:---------------------------------------:|
| SOTA results of object detection models | 

| Status | Method                                      | Architecture                   | Stage           | Anchor             | Date       | Publication    |
|:------:| ------------------------------------------- | ------------------------------ | --------------- | ------------------ | ---------- | -------------- |
|   🔄   | [**YOLOR**](yolor.md)                       | [CNN](../deep_learning/cnn.md) | One&#8209;stage | Anchor&#8209;based | 2021/05/10 | arXiv          |
|   ✅   | [**Scaled&#8209;YOLOv4**](scaled_yolov4.md) | CNN                            | One&#8209;stage | Anchor&#8209;based | 2021/06/25 | CVPR&nbsp;2021 |
|   🔄   | [**YOLOX**](yolox.md)                       | CNN                            | One&#8209;stage | Anchor&#8209;free  | 2021/08/06 | arXiv          |
|   ✅   | [**YOLOv5**](yolov5.md)                     | CNN                            | One&#8209;stage | Anchor&#8209;based |            |                |
