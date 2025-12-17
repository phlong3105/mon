<div align="center">
<h1>Depth Anything At Any Condition</h1>
  
[**Boyuan Sun**](https://bbbbchan.github.io)<sup>1*</sup> · [**Modi Jin**](https://ghost233lism.github.io/)<sup>1*</sup> · [**Bowen Yin**](https://yinbow.github.io/)<sup>1</sup> · [**Qibin Hou**](https://houqb.github.io/)<sup>1&dagger;</sup>

<sup>1</sup>VCIP, Nankai University

*Equal contribution &emsp;&dagger;Corresponding author



**English | [简体中文](README_zh.md)**

<a href="https://arxiv.org/abs/2507.01634"><img src='https://img.shields.io/badge/Paper-2507.01634-red' alt='Paper PDF'></a>
<a href="https://ghost233lism.github.io/depthanything-AC-page/ "><img src='https://img.shields.io/badge/Project-Page-green' alt='Project Page'></a>
<a href='https://huggingface.co/ghost233lism/DepthAnything-AC'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
<a href='https://huggingface.co/spaces/ghost233lism/DepthAnything-AC'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-orange' alt='Demo'></a>
<a href='https://huggingface.co/papers/2507.01634'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow'></a>
</div>

**DepthAnything-AC** is a robust monocular depth estimation (MDE) model fine-tuned from [DepthAnything-V2](https://github.com/DepthAnything/Depth-Anything-V2), designed for **zero-shot depth estimation under diverse and challenging environmental conditions**, including low light, adverse weather, and sensor distortions.

To address the lack of high-quality annotations in corrupted scenes, we introduce a lightweight **unsupervised consistency regularization** framework that enables training on unlabeled data. Additionally, our proposed **Spatial Distance Constraint** helps the model learn patch-level geometric relationships, enhancing semantic boundaries and fine details.

![teaser](assets/teaser.png)

<div align="center">
<img src="assets/depthanything-AC-video.gif" alt="video" width="100%">
</div>

## News

- **2025-07-03:** 🚀  DepthAnything-AC was ranked **#3 Paper of the Day** on [HuggingFace Daily Papers](https://huggingface.co/papers/date/2025-07-03).
- **2025-07-03:** 🔥  The paper of [DepthAnything-AC](https://arxiv.org/abs/2507.01634) is released. 
- **2025-07-02:** 🔥  The code of [DepthAnything-AC](HVision-NKU/DepthAnythingAC) is released.
- **2025-07-02:** 🔥  The DepthAnything-AC model and evaluation benchmarks are released

## TODO List
- [ ] Instructions for training dataset download and process.
- [ ] Jittor implementation of DepthAnything-AC.
- [ ] Longer and more comprehensive video demo.
## Model Architecture

![architecture](assets/architecture.png)

## Installation

### Requirements

- Python>=3.9
- torch==2.3.0
- torchvision==0.18.0
- torchaudio==2.3.0
- cuda==12.1

### Setup

```bash
git clone https://github.com/your-repo/Depth-Anything-AC
cd Depth-Anything-AC
conda create -n depth_anything_ac python=3.9
conda activate depth_anything_ac
pip install -r requirements.txt
```



## Usage
### Get Depth-Anything-AC Model
Download the pre-trained checkpoints from huggingface:
```bash
mkdir checkpoints
cd checkpoints

# (Optional) Using huggingface mirrors
export HF_ENDPOINT=https://hf-mirror.com

# download DepthAnything-AC model from huggingface
huggingface-cli download --resume-download ghost233lism/DepthAnything-AC --local-dir ghost233lism/DepthAnything-AC
```

We also provide the DepthAnything-AC model on Google Drive: [Download](https://drive.google.com/drive/folders/1yjM7_V9XQlL-taoRTbMq7aoCh1-Xr-ya?usp=sharing)


### Quick Inference

We provide the quick inference scripts for single/batch image input in `tools/`.  Please refer to [infer](./tools/README.md) for detailed information.

### Training
We provide the full training process of DepthAnything-AC, including consistency regularization, spatial distance extraction/constraint and wide-used Affine-Invariant Loss Function.

Prepare your configuration in `configs/` file and run:

```bash
bash tools/train.sh <num_gpu> <port>
```

### Evaluation
We provide the direct evaluation for DA-2K, enhanced DA-2K, KITTI, NYU-D, Sintel, ETH3D, DIODE, NuScenes-Night, RobotCar-night, DS-rain/cloud/fog, KITTI-C benchmarks. You may refer to `configs/` for more details.

```bash
bash tools/val.sh <num_gpu> <port> <dataset>
```

## Results

### Quantitative Results

#### DA-2K Multi-Condition Robustness Results

Quantitative results on the enhanced multi-condition DA-2K benchmark, including complex light and climate conditions. The evaluation metric is **Accuracy** ↑.

| Method | Encoder | **DA-2K** | **DA-2K dark** | **DA-2K fog** | **DA-2K snow** | **DA-2K blur** |
|:-------|:-------:|:---------:|:---------------:|:--------------:|:---------------:|:---------------:|
| DynaDepth | ResNet | 0.655 | 0.652 | 0.613 | 0.605 | 0.633 |
| EC-Depth | ViT-S | 0.753 | 0.732 | 0.724 | 0.713 | 0.701 |
| STEPS | ResNet | 0.577 | 0.587 | 0.581 | 0.561 | 0.577 |
| RobustDepth | ViT-S | 0.724 | 0.716 | 0.686 | 0.668 | 0.680 |
| Weather-Depth | ViT-S | 0.745 | 0.724 | 0.716 | 0.697 | 0.666 |
| DepthPro | ViT-S | 0.947 | 0.872 | 0.902 | 0.793 | 0.772 |
| DepthAnything V1 | ViT-S | 0.884 | 0.859 | 0.836 | 0.880 | 0.821 |
| DepthAnything V2 | ViT-S | 0.952 | 0.910 | 0.922 | 0.880 | 0.862 |
| **Depth Anything AC** | ViT-S | **0.953** | **0.923** | **0.929** | **0.892** | **0.880** |

#### Zero-shot Relative Depth Estimation on Real Complex Benchmarks

Zero-shot evaluation results on challenging real-world scenarios including night scenes, adverse weather conditions, and complex environmental factors. All results use ViT-S encoder.

| Method | Encoder | **NuScenes-night** | | **RobotCar-night** | | **DS-rain** | | **DS-cloud** | | **DS-fog** | |
|:-------|:-------:|:----------------:|:---:|:----------------:|:---:|:---------:|:---:|:----------:|:---:|:--------:|:---:|
| | | AbsRel ↓ | δ₁ ↑ | AbsRel ↓ | δ₁ ↑ | AbsRel ↓ | δ₁ ↑ | AbsRel ↓ | δ₁ ↑ | AbsRel ↓ | δ₁ ↑ |
| DynaDepth | ResNet | 0.381 | 0.394 | 0.512 | 0.294 | 0.239 | 0.606 | 0.172 | 0.608 | 0.144 | 0.901 |
| EC-Depth | ViT-S | 0.243 | 0.623 | 0.228 | 0.552 | 0.155 | 0.766 | 0.158 | 0.767 | 0.109 | 0.861 |
| STEPS | ResNet | 0.252 | 0.588 | 0.350 | 0.367 | 0.301 | 0.480 | 0.252 | 0.588 | 0.216 | 0.641 |
| RobustDepth | ViT-S | 0.260 | 0.597 | 0.311 | 0.521 | 0.167 | 0.755 | 0.168 | 0.775 | 0.105 | 0.882 |
| Weather-Depth | ViT-S | - | - | - | - | 0.158 | 0.764 | 0.160 | 0.767 | 0.105 | 0.879 |
| Syn2Real | ViT-S | - | - | - | - | 0.171 | 0.729 | - | - | 0.128 | 0.845 |
| DepthPro | ViT-S | 0.218 | 0.669 | 0.237 | 0.534 | **0.124** | **0.841** | 0.158 | 0.779 | **0.102** | **0.892** |
| DepthAnything V1 | ViT-S | 0.232 | 0.679 | 0.239 | 0.518 | 0.133 | 0.819 | 0.150 | **0.801** | 0.098 | 0.891 |
| DepthAnything V2 | ViT-S | 0.200 | 0.725 | 0.239 | 0.518 | 0.125 | 0.840 | 0.151 | 0.798 | 0.103 | 0.890 |
| **Depth Anything AC** | ViT-S | **0.198** | **0.727** | **0.227** | **0.555** | 0.125 | 0.840 | **0.149** | **0.801** | 0.103 | 0.889 |

*Bold: Best performance, Underlined: Second best performance. NuScenes-night and RobotCar-night represent nighttime driving scenarios. DS-rain, DS-cloud, and DS-fog are DrivingStereo weather variation datasets.*

#### Zero-shot Relative Depth Estimation on Synthetic KITTI-C Benchmarks

Zero-shot evaluation results on synthetic KITTI-C corruption benchmarks, testing robustness against various image degradations and corruptions.

| Method | Encoder | **Dark** | | **Snow** | | **Motion** | | **Gaussian** | |
|:-------|:-------:|:--------:|:---:|:--------:|:---:|:----------:|:---:|:------------:|:---:|
| | | AbsRel ↓ | δ₁ ↑ | AbsRel ↓ | δ₁ ↑ | AbsRel ↓ | δ₁ ↑ | AbsRel ↓ | δ₁ ↑ |
| DynaDepth | ResNet | 0.163 | 0.752 | 0.338 | 0.393 | 0.234 | 0.609 | 0.274 | 0.501 |
| STEPS | ResNet | 0.230 | 0.631 | 0.242 | 0.622 | 0.291 | 0.508 | 0.204 | 0.692 |
| DepthPro | ViT-S | 0.145 | 0.793 | 0.197 | 0.685 | 0.170 | 0.746 | 0.170 | 0.745 |
| DepthAnything V2 | ViT-S | **0.130** | 0.832 | 0.115 | 0.872 | 0.127 | 0.840 | 0.157 | 0.785 |
| **Depth Anything AC** | ViT-S | **0.130** | **0.834** | **0.114** | **0.873** | **0.126** | **0.841** | **0.153** | **0.793** |

*KITTI-C includes synthetic corruptions: Dark (low-light conditions), Snow (weather simulation), Motion (motion blur), and Gaussian (noise corruption).*


## Citation

If you find this work useful, please consider citing:

```bibtex
@article{sun2025depth,
  title={Depth Anything at Any Condition},
  author={Sun, Boyuan and Modi Jin and Bowen Yin and Hou, Qibin},
  journal={arXiv preprint arXiv:2507.01634},
  year={2025}
}
```


## License

This code is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only.
Please note that any commercial use of this code requires formal permission prior to use.

## Contact

For technical questions, please contact 
sbysbysby123[AT]gmail.com or jin_modi[AT]mail.nankai.edu.cn

For commercial licensing, please contact andrewhoux[AT]gmail.com.

## Acknowledgements

We thank the authors of [DepthAnything](https://github.com/LiheYoung/Depth-Anything) and [DepthAnything V2](https://github.com/DepthAnything/Depth-Anything-V2) for their foundational work. We also acknowledge [DINOv2](https://github.com/facebookresearch/dinov2) for the robust visual encoder, [CorrMatch](https://github.com/BBBBchan/CorrMatch) for their codebase, and [RoboDepth](https://github.com/ldkong1205/RoboDepth) for their contributions.

<p align="center"> 
<img src="https://api.star-history.com/svg?repos=HVision-NKU/DepthAnythingAC&type=Date" style="width:70%"/>
</p>




