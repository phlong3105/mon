<div align="center">
<img width="800" src="data/yolor.png">

You Only Learn One Representation: Unified Network for Multiple Tasks
=============================
Chien-Yao Wang, I-Hau Yeh, and Hong-Yuan Mark Liao

arXiv

<a href="data/yolor.pdf"><img src="../../data/badge/paper_paper.svg"></a>
<a href="https://github.com/WongKinYiu/yolor"><img src="../../data/badge/paper_code.svg"></a>
<a href="https://viblo.asia/p/paper-explain-yolor-su-khoi-dau-cho-mot-xu-huong-moi-Ljy5VREy5ra"><img src="../../data/badge/paper_reference.svg"></a>
<a href="https://medium.com/augmented-startups/is-yolor-better-and-faster-than-yolov4-54812da66cc1"><img src="../../data/badge/paper_reference.svg"></a>
<a href="https://viso.ai/deep-learning/yolor/#:~:text=YOLOR%20stands%20for%20%E2%80%9CYou%20Only,knowledge%20and%20explicit%20knowledge%20together%E2%80%9D."><img src="../../data/badge/paper_reference.svg"></a>
</div>


## Highlight
YOLOR (“You Only Learn One Representation”) is a SOTA machine learning 
algorithm for object detection (2021-2022).

Human beings can answer different questions given a single input. Given one 
piece of data, humans can analyze the data from different angles. For example, 
a photo of something may elicit different responses regarding the action depicted, 
location, etc. 

YOLOR aims to give this ability to machine learning models – so that they are 
able to serve many tasks given one input. 

YOLOR achieved comparable object detection accuracy as the 
[Scaled-YOLOv4](scaled_yolov4.md), while the inference speed was increased by 
88%. This makes YOLOR one of the fastest object detection algorithms in modern 
computer vision. On the MS COCO dataset, the mAP of YOLOR is 3.8% higher 
compared to the [PP-YOLOv2](pp_yolov2.md), at the same inference speed.


## Method
<div align="center">
	<img width="700" src="data/yolor_architecture.png">
	<p>YOLOR concept with implicit and explicit knowledge-based multi-task learning.</p>
    <p>Chúng ta có thể thấy kiến thức tường minh được tính toán từ ảnh đầu vào 
      trong quá trình forward. Trong khi đó, kiến thức tiềm ẩn không phụ thuộc 
      vào ảnh đầu vào trong quá trình forward mà chúng là những đặc trưng cố 
      định đóng vai trò giống như các tham số trong mô hình.</p>
</div>

### 1. Terminology
<details open>
<summary><b style="font-size:16px">Explicit Knowledge/Information (Kiến thức tường minh)</b></summary>

Explicit knowledge is known as **normal learning, or things that you learn 
consciously**.

Explicit knowledge is given to neural networks by providing clear metadata 
or image databases that are either thoroughly annotated or well organized 
(aka annotations).

Explicit knowledge for YOLOR **is obtained from the shallow layers of the 
neural networks**. This knowledge directly corresponds to observations that are 
supposed to be made.
</details>

<br>
<details open>
<summary><b style="font-size:16px">Implicit Knowledge/Information (Kiến thức tiềm ẩn)</b></summary>

Implicit knowledge can effectively assist machine learning models in performing 
tasks with YOLOR.

For humans, implicit knowledge refers to **the knowledge learnt subconsciously**, 
sort of like riding a bike or learning how to walk. It's derived from experience.

For neural networks, **implicit knowledge is obtained by features in the deep 
layers**. The knowledge that does not correspond to observations is known as 
implicit knowledge as well.
</details>

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

<br>
<details open>
<summary><b style="font-size:16px">Manifold Space Reduction</b></summary>

For manifold space reduction, my understanding is that we reduce the 
dimensions of the manifold space so that we are able to achieve various tasks 
such as pose estimation and classification, amongst others.
</details>

<br>
<details open>
<summary><b style="font-size:16px">Kernel Space Alignment</b></summary>

</details>


## Ablation Studies


## Results
<div align="center">
	<img width="500" src="data/yolor_results_01.png">
    <img width="500" src="data/yolor_results_02.png">
	<p>Performance of YOLOR vs. YOLO v4 and others.</p>
</div>


## Citation
```text
@article{Wang2021,
  title   = {You Only Learn One Representation: Unified Network for Multiple Tasks},
  author  = {Chien-Yao Wang and I-Hau Yeh and Hong-Yuan Mark Liao},
  journal = {arXiv preprint arXiv:2105.04206},
  year    = {2021}
}
```
