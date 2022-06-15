---
layout      : default
title       : Object Detection
parent		: Vision
has_children: true
has_toc     : false
permalink   : /vision/object_detection
---

![data/object_detection.png](data/object_detection.png)

# Object Detection

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
  1. TOC
  {:toc}
</details>

---

The ultimate purpose of object detection is to locate important items, draw
rectangular bounding boxes around them, and determine the class of each item
discovered.

Object detection is often called object recognition, object identification,
image detection, and these concepts are synonymous.

|  ![data/object_detection_01.gif](data/object_detection_01.gif)   |
|:----------------------------------------------------------------:|
| Here is a demo of object detection applied in autonomous vehicle | 
|                       <img height="200" />                       |

## Methods

### Road Map of Object Detection

| ![data/milestones.png](data/milestones.png) |
|:-------------------------------------------:|
|                                             | 
 
### SOTA

| ![data/object_detection_sota.png](data/object_detection_sota.png) |
|:-----------------------------------------------------------------:|
|                                                                   | 

| Status | Method                                      | Architecture | Stage           | Anchor             | Date       | Publication    |
|:------:|---------------------------------------------|--------------|-----------------|--------------------|------------|----------------|
|   🔄   | [**YOLOR**](yolor.md)                       | Deep         | One&#8209;stage | Anchor&#8209;based | 2021/05/10 | arXiv          |
|   ✅    | [**Scaled&#8209;YOLOv4**](scaled_yolov4.md) | Deep         | One&#8209;stage | Anchor&#8209;based | 2021/06/25 | CVPR&nbsp;2021 |
|   🔄   | [**YOLOX**](yolox.md)                       | Deep         | One&#8209;stage | Anchor&#8209;free  | 2021/08/06 | arXiv          |
|   ✅    | [**YOLOv5**](yolov5.md)                     | Deep         | One&#8209;stage | Anchor&#8209;based |            |                |
