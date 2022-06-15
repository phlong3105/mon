---
layout      : default
title       : YOLOR
parent	    : Object Detection
grand_parent: Vision
has_children: false
has_toc     : false
permalink   : /vision/object_detection/yolor
---

![data/yolor.png](data/yolor.png)

# You Only Learn One Representation: Unified Network for Multiple Tasks

Chien-Yao Wang, I-Hau Yeh, and Hong-Yuan Mark Liao

arXiv

[Paper](data/yolor.pdf){: .btn .fs-3 .mb-4 .mb-md-0 }
[Code](https://github.com/WongKinYiu/yolor){: .btn .fs-3 .mb-4 .mb-md-0 }
[Reference](https://viblo.asia/p/paper-explain-yolor-su-khoi-dau-cho-mot-xu-huong-moi-Ljy5VREy5ra){: .btn .fs-3 .mb-4 .mb-md-0 }
[Reference](https://medium.com/augmented-startups/is-yolor-better-and-faster-than-yolov4-54812da66cc1){: .btn .fs-3 .mb-4 .mb-md-0 }
[Reference](https://viso.ai/deep-learning/yolor/#:~:text=YOLOR%20stands%20for%20%E2%80%9CYou%20Only,knowledge%20and%20explicit%20knowledge%20together%E2%80%9D){: .btn .fs-3 .mb-4 .mb-md-0 }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
  1. TOC
  {:toc}
</details>

---

## Highlight

YOLOR (“You Only Learn One Representation”) is a SOTA machine learning
algorithm for object detection (2021-2022).

Human beings can answer different questions given a single input. Given one
piece of data, humans can analyze the data from different angles. For example,
a photo of something may elicit different responses regarding the action
depicted,
location, etc.

YOLOR aims to give this ability to machine learning models – so that they are
able to serve many tasks given one input.

YOLOR achieved comparable object detection accuracy as the
[Scaled-YOLOv4](scaled_yolov4.md), while the inference speed was increased by
88%. This makes YOLOR one of the fastest object detection algorithms in modern
computer vision. On the MS COCO dataset, the mAP of YOLOR is 3.8% higher
compared to the [PP-YOLOv2](pp_yolov2.md), at the same inference speed.

## Method

|                                                                                                          ![data/yolor_architecture.png](data/yolor_architecture.png)                                                                                                           |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                                                                                                 YOLOR concept with implicit and explicit knowledge-based multi-task learning.                                                                                                  |
| Chúng ta có thể thấy kiến thức tường minh được tính toán từ ảnh đầu vào trong quá trình forward. Trong khi đó, kiến thức tiềm ẩn không phụ thuộc vào ảnh đầu vào trong quá trình forward mà chúng là những đặc trưng cố định đóng vai trò giống như các tham số trong mô hình. | 
|                                                                                                                              <img width="700" />                                                                                                                               |

### 1. Terminology

#### Explicit Knowledge/Information (Kiến thức tường minh)

Explicit knowledge is known as **normal learning, or things that you learn
consciously**.

Explicit knowledge is given to neural networks by providing clear metadata
or image databases that are either thoroughly annotated or well organized
(aka annotations).

Explicit knowledge for YOLOR **is obtained from the shallow layers of the
neural networks**. This knowledge directly corresponds to observations that are
supposed to be made.

#### Implicit Knowledge/Information (Kiến thức tiềm ẩn)

Implicit knowledge can effectively assist machine learning models in performing
tasks with YOLOR.

For humans, implicit knowledge refers to **the knowledge learnt subconsciously**
, sort of like riding a bike or learning how to walk. It's derived from
experience.

For neural networks, **implicit knowledge is obtained by features in the deep
layers**. The knowledge that does not correspond to observations is known as
implicit knowledge as well.

### 2. Explicit Deep Learning

Now in terms of neural networks, knowledge obtained from observation is known
as Explicit Deep Learning and corresponds to the shallow layers of a network.

Explicit Deep Learning was just briefly touched on in the paper but they
mentioned that this can be achieved using Detection Transform (DETr),
Non-Local Networks and Kernel Selection.

### 3. Implicit Deep Learning

Implicit Deep Learning corresponds to the deeper layers of a network,
usually where the features are extracted.

There’s a couple of ways in which we can implement Implicit Knowledge,
these include:

- Manifold Space Reduction,
- Kernel Alignment, and
- More Functions.

#### Manifold Space Reduction

For manifold space reduction, my understanding is that we reduce the
dimensions of the manifold space so that we are able to achieve various tasks
such as pose estimation and classification, amongst others.

#### Kernel Space Alignment

## Ablation Studies

## Results

| ![data/yolor_results_01.png](data/yolor_results_01.png) |
|:-------------------------------------------------------:|
| ![data/yolor_results_02.png](data/yolor_results_02.png) |
|      Performance of YOLOR vs. YOLO v4 and others.       |
|                   <img width="500" />                   |

## Citation

```text
@article{Wang2021,
  title   = {You Only Learn One Representation: Unified Network for Multiple Tasks},
  author  = {Chien-Yao Wang and I-Hau Yeh and Hong-Yuan Mark Liao},
  journal = {arXiv preprint arXiv:2105.04206},
  year    = {2021}
}
```
