<div align="center">

CSPNet: A New Backbone That Can Enhance Learning Capability of CNN
=============================
<div>
	<p>Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao</p>
	<p>CVPR 2021</p>
</div>

<div align="center">
	<a href="../../data/pdf/cspnet.pdf">Paper</a> •
    <a href="https://github.com/WongKinYiu/ScaledYOLOv4">Code</a> •
	<a href="https://sh-tsang.medium.com/review-cspnet-a-new-backbone-that-can-enhance-learning-capability-of-cnn-da7ca51524bf">Ref01</a> •
</div>
</div>


## Highlight
- First, YOLOv4 is re-designed to form YOLOv4-CSP. 
- Then, a network scaling approach that modifies not only the depth, width, resolution, but also structure of the network, which finally forms Scaled-YOLOv4.

<div align="center">
	<img width="700" src="../../data/images/cspnet_sota.png">
	<p>CSPNet not only reduces computation cost and memory usage of the networks, but also benefit on inference speed and accuracy.</p>
</div>


## Method
In Scaled-YOLOv4, there are many prior arts used, e.g. CSPNet, OSANet, YOLOv4. It is better to know them before Scaled-YOLOv4.

### 1. Principle of Model Scaling
<details open>
<summary><b style="font-size:16px">General Principle of Model Scaling</b></summary>

<div align="center">
	<img width="400" src="../../data/images/scaled_yolov4_flop.png">
	<p>FLOPs of different computational layers with different model scaling factors.</p>
</div>

- Let the scaling factors that can be used to adjust the image size, the number of layers, and the number of channels be **_α_**, **_β_**, and **_γ_**, respectively.
- For the k-layer CNNs with _b_ base layer channels, when these scaling factors vary, the corresponding changes on FLOPs are shown as below table.

> **The scaling size, depth and width cause increase in the computation cost**. They respectively show square, linear, and square increase.

<div align="center">
	<img width="350" src="../../data/images/scaled_yolov4_flop_csp.png">
	<p>FLOPs of different computational layers with/without CSP-ization.</p>
</div>

- CSPNet is applied to ResNet, ResNeXt, and Darknet, the changes in the amount of computations are observed in the below table. 
- In brief, CSPNet splits the input into two paths. One performs convolutions. One performs no convolution. They are fused at the output.
- **CSPNet can effectively reduce the amount of computations (FLOPs)** on ResNet, ResNeXt, and Darknet by 23.5%, 46.7%, and 50.0%, respectively.

> Therefore, CSP-ized models are used as the best model for performing model scaling.

</details>


## Ablation Studies
<details open>
<summary><b style="font-size:16px">CSP-ized Model</b></summary>

<div align="center">
	<img width="400" src="../../data/images/scaled_yolov4_ablation_cspized_models.png">
	<p>Ablation study of CSP-ized models @ 608x608.</p>
</div>

- Darknet53 (D53) is used as backbone and FPN with SPP (FPNSPP) and PAN with SPP (PANSPP) are chosen as necks to design ablation studies.
- LeakyReLU (Leaky) and Mish activation function are tried.

> **CSP-ized models have greatly reduced the amount of parameters and computations by 32%, and brought improvements in both Batch 8 throughput and AP.** 
> 
> Both **CD53s-CFPNSPP-Mish, and CD53s-CPANSPP-Leaky** have the same batch 8 throughput with D53-FPNSPP-Leaky, but they respectively **have 1% and 1.6% AP improvement with lower computing resources**.
>
> Therefore, **CD53s-CPANSPP-Mish is decided to used**, as it results in the highest AP in the above table as the backbone of YOLOv4-CSP.

</details>


## Results
<details open>
<summary><b style="font-size:16px">Large-Model</b></summary>

<div align="center">
	<img width="700" src="../../data/images/scaled_yolov4_large_results.png">
	<p>Comparison of state-of-the-art object detectors</p>
</div>

- When comparing **YOLOv4-CSP** with the same accuracy of EfficientDet-D3 (47.5% vs 47.5%), **the inference speed is 1.9 times**. 
- When **YOLOv4-P5** is compared with EfficientDet-D5 with the same accuracy (51.8% vs 51.5%), **the inference speed is 2.9 times**. 
- The situation is similar to the comparisons between YOLOv4-P6 vs EfficientDet-D7 (54.5% vs 53.7%) and YOLOv4-P7 vs EfficientDet-D7x (55.5% vs 55.1%). In both cases, **YOLOv4-P6 and YOLOv4-P7 are, respectively, 3.7 times and 2.5 times faster in terms of inference speed**.

> As shown in the figure at the top of the story and also the table above, **all scaled YOLOv4 models, including YOLOv4-CSP, YOLOv4-P5, YOLOv4-P6, YOLOv4-P7, are Pareto optimal on all indicators**.

<div align="center">
	<img width="300" src="../../data/images/scaled_yolov4_large_tta_results.png">
	<p>Results of YOLOv4-large models with test-time augmentation (TTA)</p>
</div>

- With **test-time augmentation (TTA)**, YOLOv4-P5, YOLOv4-P6, and YOLOv4-P7 gets 1.1%, 0.7%, and 0.5% **higher AP**, respectively.

<div align="center">
	<img width="500" src="../../data/images/scaled_yolov4_p7_resolutions.png">
	<p>Results of YOLOv4-large models with test-time augmentation (TTA)</p>
</div>

- FPN-like architecture is a naïve once-for-all model while YOLOv4 has some stages of top-down path and detection branch. 
- YOLOv4-P7\P7 and YOLOv4-P7\P7\P6 represent the model which has removed {P7} and {P7, P6} stages from the trained YOLOv4-P7.

> As shown above, YOLOv4-P7 has the best AP at high resolution, while **YOLOv4-P7\P7 and YOLOv4-P7\P7\P6 have the best AP at middle and low resolution**, respectively. This means that we can use subnets of FPN-like models to execute the object detection task well.

</details>


## Citation
```text
@InProceedings{Wang2021,
    author    = {Chien-Yao Wang and Alexey Bochkovskiy and Hong-Yuan Mark Liao},
    title     = {{Scaled-YOLOv4}: Scaling Cross Stage Partial Network},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {13029-13038}
}
```
