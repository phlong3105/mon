#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Deep Learning Models.

This package contains deep learning models for various tasks.

Taxonomy:
::

    Computer Vision
    ├── Low-Level
    │   ├── Restoration
    │   │   ├── Deblurring
    │   │   ├── Denoising
    │   │   ├── Dehazing
    │   │   ├── Deraining
    │   │   ├── Desnowing
    │   │   ├── Demoireing
    │   │   ├── Demosaicing
    │   │   ├── Image Inpainting
    │   │   └── Super-Resolution
    │   ├── Enhancement
    │   │   ├── Low-Light Enhancement
    │   │   ├── Multi-Exposure Fusion
    │   │   ├── Color Correction
    │   │   ├── Colorization
    │   │   ├── Retouching
    │   │   ├── Sharpening
    │   │   ├── Style Transfer
    │   │   └── Tone Mapping
    │   └── Filtering
    ├── Mid-Level
    │   ├── Segmentation
    │   ├── Depth & Geometry
    │   │   ├── 3D Reconstruction
    │   │   ├── Monocular Depth Estimation
    │   │   └── Stereo Matching
    │   ├── Motion
    │   │   ├── Background Subtraction
    │   │   ├── Optical Flow
    │   │   └── Object Tracking
    │   └── Keypoint Estimation
    │       ├── Facial Landmarks
    │       └── Human Pose Estimation
    └── High-Level
        ├── Recognition
        │   └── Classification
        ├── Localization
        │   ├── Object Detection
        │   └── Anomaly Detection
        ├── Reasoning
        │   ├── Visual Question Answering
        │   └── Scene Graph Generation
        └── Generation
"""

from __future__ import annotations

from .bgsubtract import *
from .classify import *
from .detect import *
from .enhance import *
from .framework import *
from .monodepth import *
from .restore import *
from .segment import *
from .track import *
