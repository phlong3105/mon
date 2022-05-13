<div align="center">

Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement
=============================

<div align="center">
    <a href="https://li-chongyi.github.io/Proj_Zero-DCE.html">Website</a> •
    <a href="https://github.com/phlong3105/one/blob/master/data/pdf/zero_dce.pdf">Paper</a> •
    <a href="https://github.com/phlong3105/one/blob/master/data/pdf/zero_dce_sup.pdf">Supplement</a> •
    <a href="https://github.com/Li-Chongyi/Zero-DCE">Code</a>
</div>
</div>


## Highlight
1. We propose the first low-light enhancement network that is **independent of paired and unpaired training data**, thus avoiding the risk of overfitting. As a result, our method generalizes well to various lighting conditions. 
2. We design an image-specific curve that is able to **approximate pixel-wise and higher-order curves by iteratively applying itself**. Such image-specific curve can effectively perform mapping within a wide dynamic range. 
3. We show the potential of training a deep image enhancement model in the absence of reference images through task-specific non-reference loss functions that indirectly evaluate enhancement quality. It is capable of processing images in real-time **(about 500 FPS for images of size 640x480x3 on GPU)** and takes only 30 minutes for training.


## Method
<div align="center">
    <img width="800" src="../../data/images/zero_dce_framework.png"><br/>
    <p align="justify">The pipeline of our method. (a) The framework of Zero-DCE. A DCE-Net is devised to estimate a set of best-fitting Light-Enhancement curves (LE-curves: LE(I(x);α)=I(x)+αI(x)(1-I(x))) to iteratively enhance a given input image. (b, c) LE-curves with different adjustment parameters α and numbers of iteration n. In (c), α1, α2, and α3 are equal to -1 while n is equal to 4. In each subfigure, the horizontal axis represents the input pixel values while the vertical axis represents the output pixel values.</p>
</div>


## Ablation Studies
<details open>
<summary><b>1. Contribution of Each Loss</b></summary>

<div align="center">
    <img width="600" src="../../data/images/zero_dce_ablation_01.png"><br/>
    <p align="justify">Ablation study of the contribution of each loss (spatial consistency loss Lspa, exposure control loss Lexp, color constancy loss Lcol, illumination smoothness loss LtvA).</p>
</div>
</details>

<details open>
<summary><b>2. Effect of Parameter Settings</b></summary>

<div align="center">
    <img width="600" src="../../data/images/zero_dce_ablation_02.png"><br/>
    <p align="justify">Ablation study of the effect of parameter settings. l-f-n represents the proposed Zero-DCE with l convolutional layers, f feature maps of each layer (except the last layer), and n iterations.</p>
</div>
</details>

<details open>
<summary><b>3. Impact of Training Data</b></summary>

<div align="center">
    <img width="600" src="../../data/images/zero_dce_ablation_03.png"><br/>
    <p align="justify">To test the impact of training data, we retrain the Zero-DCE on different datasets: 1) only 900 low-light images out of 2,422 images in the original training set (Zero-DCELow), 2) 9,000 unlabeled low-light images provided in the DARK FACE dataset (Zero-DCELargeL), and 3) 4800 multi-exposure images from the data augmented combination of Part1 and Part2 subsets in the SICE dataset (Zero-DCELargeLH).</p>
</div>
</details>

<details open>
<summary><b>4. Advantage of Three-channel Adjustment</b></summary>

<div align="center">
    <img width="600" src="../../data/images/zero_dce_ablation_04.png"><br/>
    <p align="justify">Ablation study of the advantage of three-channel adjustment (RGB, CIE Lab, YCbCr color spaces).</p>
</div>
</details>


## Results
<details open>
<summary><b>1. Visual Comparisons on Typical Low-light Images</b></summary>

<div align="center">
    <img width="600" src="../../data/images/zero_dce_results_01.png"><br/>
</div>
</details>

<details open>
<summary><b>2. Visual Face Detection Results Before and After Enhanced by Zero-DCE</b></summary>

<div align="center">
    <img width="400" src="../../data/images/zero_dce_results_02.png"><br/>
</div>
</details>

<details open>
<summary><b>3. Real Low-light Video with Variational Illumination Enanced by Zero-DCE</b></summary>

<div align="center">
    <img width="600" src="../../data/images/zero_dce_results_03.gif"><br/>
</div>
</details>

<details open>
<summary><b>4. Self-training (taking first 100 frames as training data) for Low-light Video Enhancement</b></summary>

<div align="center">
    <img width="600" src="../../data/images/zero_dce_results_04.gif"><br/>
</div>
</details>
