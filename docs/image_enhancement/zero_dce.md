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
3. We show the potential of training a deep image enhancement model in the absence of reference images through task-specific non-reference loss functions that indirectly evaluate enhancement quality. It is capable of processing images in real-time **(about 500 FPS for images of size 640*480*3 on GPU)** and takes only 30 minutes for training.


## Method
<div align="center">
    <img width="800" src="../../data/images/zero_dce_framework.png"><br/>
    <p align="justify">
        The pipeline of our method. (a) The framework of Zero-DCE. A DCE-Net is devised to estimate a set of best-fitting Light-Enhancement curves (LE-curves: LE(I(x);α)=I(x)+αI(x)(1-I(x))) to iteratively enhance a given input image. (b, c) LE-curves with different adjustment parameters α and numbers of iteration n. In (c), α1, α2, and α3 are equal to -1 while n is equal to 4. In each subfigure, the horizontal axis represents the input pixel values while the vertical axis represents the output pixel values.
    </p>
</div>


## Data


## Results
