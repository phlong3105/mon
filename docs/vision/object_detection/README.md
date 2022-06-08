<div align="center">
<img src="data/object_detection.png" width="1000">

Object Detection
=============================
</div>

# Overview

The ultimate purpose of object detection is to locate important items, draw
rectangular bounding boxes around them, and determine the class of each item
discovered.

Object detection is often called object recognition, object identification,
image detection, and these concepts are synonymous.

Here is a demo of object detection applied in autonomous vehicle:
<div align="center">
	<img src="data/object_detection_01.gif" height="200">
</div>

## Methods

<details open>
<summary><b style="font-size:18px">Road Map of Object Detection</b></summary>

<div align="center">
	<img src="data/milestones.png" height="500">
</div>
</details>

<details open>
<summary><b style="font-size:18px">SOTA</b></summary>

<div align="center">
	<img src="data/milestones.png" height="400">
</div>
</details>

| Status | Method                                | Architecture | Stage     | Anchor       | Date       | Publication    |
|:------:|---------------------------------------|--------------|-----------|--------------|------------|----------------|
|   📑   | [**YOLOR**](yolor.md)                 | `Deep`       | One-stage | Anchor-based | 2021/05/10 | arXiv          |
|   ✅    | [**Scaled-YOLOv4**](scaled_yolov4.md) | `Deep`       | One-stage | Anchor-based | 2021/06/25 | CVPR&nbsp;2021 |
|   📑   | [**YOLOX**](yolox.md)                 | `Deep`       | One-stage | Anchor-free  | 2021/08/06 | arXiv          |
|   ✅    | [**YOLOv5**](yolov5.md)               | `Deep`       | One-stage | Anchor-based |            |                |
