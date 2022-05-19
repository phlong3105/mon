<div align="center">

HINet: Half Instance Normalization Network for Image Restoration
=============================

<div>
    <p>Liangyu Chen, Xin Lu, Jie Zhang, Xiaojie Chu, Chengpeng Chen</p>
    <p>CVPR 2021</p>
</div>

<div align="center">
    <a href="data/hinet.pdf">Paper</a> •
    <a href="https://github.com/megvii-model/HINet">Code</a>
</div>
</div>


## Highlight
- Normalization is widely used in high-level computer vision tasks:
    - Batch Normalization and IBN in classification.
    - Layer Normalization in DETR and GroupNorm in FCOS for detection.
    - Instance Normalization is used to style/domain transfer.


- This model is neck-to-neck comparable with the [MPRNet](mprnet.md)
    - Image denoising: exceed MPRNet 0.11 dB and 0.28 dB in PSNR on SIDD dataset, with only 7.5% and 30% of its multiplier-accumulator operations (MACs), 6.8× and 2.9× speedup respectively.
    - Image deblurring: get comparable performance with 22.5% of its MACs and 3.3× speedup on REDS and GoPro datasets.
    - Image deraining: exceed MPRNet by 0.3 dB in PSNR on the average result of multiple datasets following, with 1.4× speedup.


## Method
<div align="center">
    <img width="800" src="data/hinet_architecture.png"><br/>
    <p align="justify">Proposed Half Instance Normalization Network (HINet). The encoder of each subnetwork contains Half Instance Normalization Blocks (HIN Block). For simplicity, we only show 3 layers of HIN Block in the figure, and HINet has a total of 5 layers. We adopt CSFF and SAM modules from MPRNet [56].</p>
</div>


<details open>
<summary><b style="font-size:16px">1. Architecture</b></summary>

- Adopt 2 simple U-Nets architecture.
- Half Instance Normalization Block (HIN Block):
  - Carefully integrate Instance Normalization as building blocks to advance the network performance in image restoration tasks.
  - It is the first model to adopt normalization *directly* with SOTA performance in image restoration tasks.


- A multi-stage network called HINet:
    - Consists of 2 subnetworks.
    - Stacking HIN Block in each subnetwork’s encoder → the receptive field at each scale is expanded → the robustness of features is also improved.


- Adopt cross-stage feature fusion (CSFF) [55] and supervised attention module (SAM) [55] between two stages to enrich the multi-scale features and facilitate achieving performance gain respectively.

</details>

<details open>
<summary><b style="font-size:16px">2. U-Net</b></summary>

- In each stage, use one 3×3 convolutional layer to extract the initial features.
- Then those features are input into an encoder-decoder architecture with 4 down-samplings and upsamplings.
    - Use convolution with kernel size=4 for downsampling
    - Use transposed convolution with kernel size=2 for upsampling


- In the encoder: design HIN Blocks to extract features in each scale, and double the channels of features when downsampling.
- In the decoder: use ResBlocks [15] to extract high-level features, and fuse features from the encoder component to compensate for the loss of information caused by resampling.
- As for ResBlock, use leaky ReLU [29] with a negative slope equal to 0.2 and remove batch normalization.
- Finally, get the residual output of the reconstructed image by using one 3×3 convolution.  

</details>

<details open>
<summary><b style="font-size:16px">3. Linking Sub-networks</b></summary>

- Use **cross-stage feature fusion (CSFF)** module and **supervised attention module (SAM)** to connect 2 subnetworks.
- For **CSFF**: use $3 × 3$ convolution to transform the features from one stage to the next stage for aggregation

> Enrich the multi-scale features of the next stage.

- For **SAM**: replace the 1×1 convolutions in the original module with 3×3 convolutions and add bias in each convolution. 

> By introducing SAM, the useful features at the current stage can propagate to the next stage and the less informative ones will be suppressed by the attention masks

</details>

<details open>
<summary><b style="font-size:16px">4. Half Instance Normalization Block</b></summary>

> Because of variance of small image patches differ a lot among mini-batches and the different formulations of training and testing, BN is not commonly used in low-level tasks.

- Instead, Instance Normalization (IN) keeps the same normalization procedure consistent in both training and inference.
- Further, IN re-calibrates the mean and variance of features without the influence of batch dimension, which can keep more scale information than BN.

<div align="center">
    <img width="400" src="data/hinet_hin_block.png"><br/>
</div>

- HIN block firstly takes the input features $F_{in} ∈ \mathbb{R}^{C_{in}×H×W}$ and generates intermediate features $F_{mid} ∈ \mathbb{R}^{C_{out}×H×W}$ with $3 × 3$ convolution, where $C_{in}$/$C_{out}$ is the number of input/output channels for HIN block.
- Then, the features $F_{mid}$ are divided into two parts ($F_{mid1}$ /$F_{mid2} ∈ \mathbb{R}^{C_{out}/2×H×W}$).
- The first part $F_{mid1}$ is normalized by IN with learnable affine parameters and then concatenates with $F_{mid2}$ in channel dimension.
- HIN blocks use IN on the half of the channels and keep context information by the other half of the channels.
- Later experiments will also show that this design is more friendly to features in shallow layers of the network. After the concat operation, the residual features $R_{out} ∈ \mathbb{R}^{C_{out}×H×W}$ are obtained by passing features to one $3 × 3$ convolution layer and two leaky ReLU layers, which is shown in Figure 3 a.
- Finally, HIN blocks output Fout by add Rout with shortcut features (obtained after 1×1 convolution).

</details>


## Ablation Studies


## Results


## Citation
```text
@inproceeding{Chen2021,
    author    = {Liangyu Chen and Xin Lu and Jie Zhang and Xiaojie Chu and Chengpeng Chen},
    title     = {HINet: Half Instance Normalization Network for Image Restoration},
    booktitle = {CVPRW},
    month     = {June},
    year      = {2021},
    pages     = {182-192}
}
```
