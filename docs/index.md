---
layout   : default
title    : Home
nav_order: 1
permalink: /
---

![One](data/one.png)

---

# One Research Framework

`One` is a comprehensive research framework and knowledge base of our works 
related to computer vision, machine learning, and deep learning.

[Getting Started](#getting-started){: .btn .btn-primary .fs-3 .mb-4 .mb-md-0 } 
[Knowledge Base](#knowledge-base){: .btn .fs-3 .mb-4 .mb-md-0 } 
[Cite](#cite){: .btn .fs-3 .mb-4 .mb-md-0 } 
[Contact](#contact){: .btn .fs-3 .mb-4 .mb-md-0 } 

---

## Getting Started

### Prerequisite

|            | Requirements                                                                                                                                                                                                                                         |
|:-----------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **OS**     | [**Ubuntu 20.04 / 22.04**](https://ubuntu.com/download/desktop) (fully supports), `Windows 10` and `MacOS` (partially supports)                                                                                                                      |
| **Env**    | [**Python>=3.9.0**](https://www.python.org/), [**PyTorch>=1.11.0**](https://pytorch.org/get-started/locally/), [**cudatoolkit=11.3**](https://pytorch.org/get-started/locally/), with [**anaconda**](https://www.anaconda.com/products/distribution) |	
| **Editor** | [**PyCharm**](https://www.jetbrains.com/pycharm/download)                                                                                                                                                                                            |

### Directory

```text
one               # root directory
 |__ data         # contains data
 |__ docs
 |__ install      # helpful installation scripts       
 |__ pretrained   # pretrained models weights
 |__ scripts      # main scripts
 |__ src
 |      |__ one
 |      |__ project1
 |      |__ project2
 |      |__ ..
 |__ tests
 |__ third_party
```

### Installation using `conda`

```shell
cd <to-where-you-want-to-save-one-dir>
mkdir -p one
mkdir -p one/data
cd one

# Install `aic22_track4` package
git clone git@github.com:phlong3105/one
cd one/install
chmod +x install.sh
conda init bash

# Install package. When prompt to input the dataset directory path, you should 
# enter: <some-path>/one/datasets
bash -i install.sh
cd ..
pip install --upgrade -e .
```

## Knowledge Base

### [Machine Learning](machine_learning/README.md)

|                                                                                            <img width="150"/>                                                                                            |                                                                                   <img width="150"/>                                                                                   |                                                                             <img width="150"/>                                                                             |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [![Data Processing](machine_learning/data_processing/data/data_processing_small.gif)](machine_learning/data_processing/README.md) <br> [**Data Processing**](machine_learning/data_processing/README.md) | [![Training](machine_learning/model_learning/data/training_small.gif)](machine_learning/model_learning/README.md) <br> [**Model Learning**](machine_learning/model_learning/README.md) | [![Serving](machine_learning/model_serving/data/serving.gif)](machine_learning/model_serving/README.md) <br> [**Model Serving**](machine_learning/model_serving/README.md) |
|                           [![Classification](data/photo.png)](machine_learning/classification/README.md) <br> [**Classification**](machine_learning/classification/README.md)                            |                          [![Clustering](data/photo.png)](machine_learning/clustering/README.md) <br> [**Clustering**](machine_learning/clustering/README.md)                           |              [![Deep Learning](data/photo.png)](machine_learning/deep_learning/README.md) <br> [**Deep Learning**](machine_learning/deep_learning/README.md)               |
|     [![Dimensionality Reduction](data/photo.png)](machine_learning/dimensionality_reduction/README.md) <br> [**Dimensionality <br> Reduction**](machine_learning/dimensionality_reduction/README.md)     |             [![Neural Network](data/photo.png)](machine_learning/neural_network/README.md) <br> [**Neural Network<br>&nbsp;**](machine_learning/neural_network/README.md)              |               [![Regression](data/photo.png)](machine_learning/regression/README.md) <br> [**Regression<br>&nbsp;**](machine_learning/regression/README.md)                |

### [Vision](vision/README.md)

|                                                                                                 <img width="150"/>                                                                                                  |                                                                                                 <img width="150"/>                                                                                                  |                                                                               <img width="150"/>                                                                               |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|         [![Data Processing](vision/action_recognition/data/action_recognition_small.gif)](vision/action_recognition/README.md) <br> [**Action Recognition<br>&nbsp;**](vision/action_recognition/README.md)         |             [![Action Detection](vision/action_detection/data/action_detection_small.gif)](vision/action_detection/README.md) <br> [**Action Detection<br>&nbsp;**](vision/action_detection/README.md)              |           [![Image Classification](data/photo.png)](vision/image_classification/README.md) <br> [**Image<br>Classification**](vision/image_classification/README.md)           |
|              [![Image Enhancement](vision/image_enhancement/data/image_enhancement_small.gif)](vision/image_enhancement/README.md) <br> [**Image<br>Enhancement**](vision/image_enhancement/README.md)              | [![Instance Segmentation](vision/instance_segmentation/data/instance_segmentation_small.gif)](vision/instance_segmentation/README.md) <br> [**Instance <br> Segmentation**](vision/instance_segmentation/README.md) | [![Lane Detection](vision/lane_detection/data/lane_detection_small.gif)](vision/lane_detection/README.md) <br> [**Lane Detection<br>&nbsp;**](vision/lane_detection/README.md) |
|                  [![Object Detection](vision/object_detection/data/object_detection_small.gif)](vision/object_detection/README.md) <br> [**Object Detection**](vision/object_detection/README.md)                   |                                         [![Object Tracking](data/photo.png)](vision/object_tracking/README.md) <br> [**Object Tracking**](vision/object_tracking/README.md)                                         |                    [![Reidentification](data/photo.png)](vision/reidentification/README.md) <br>  [**Reidentification**](vision/reidentification/README.md)                    |
| [![Semantic Segmentation](vision/semantic_segmentation/data/semantic_segmentation_small.gif)](vision/semantic_segmentation/README.md) <br> [**Semantic <br> Segmentation**](vision/semantic_segmentation/README.md) |                                                                                                                                                                                                                     |                                                                                                                                                                                |

### [Image Processing](image_processing/README.md)

|                                                                         <img width="150"/>                                                                          |                                                                    <img width="150"/>                                                                     |                                                                     <img width="150"/>                                                                      |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [![Camera Calibration](data/photo.png)](image_processing/camera_calibration/README.md) <br> [**Camera Calibration**](image_processing/camera_calibration/README.md) | [![Feature Extraction](data/photo.png)](image_processing/feature_extraction) <br> [**Feature Extraction**](image_processing/feature_extraction/README.md) |               [![Filtering](data/photo.png)](image_processing/filtering/README.md) <br> [**Filtering**](image_processing/filtering/README.md)               |
|                   [![Histogram](data/photo.png)](image_processing/histogram/README.md) <br> [**Histogram**](image_processing/histogram/README.md)                   |                       [![Spatial](data/photo.png)](image_processing/spatial) <br> [**Spatial**](image_processing/spatial/README.md)                       | [![Spatial Temporal](data/photo.png)](image_processing/spatial_temporal/README.md) <br> [**Spatial Temporal**](image_processing/spatial_temporal/README.md) |

### [Tools](tools/README.md)

|                                           <img width="150"/>                                           |                                      <img width="150"/>                                      |                                      <img width="150"/>                                      |
|:------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|
| [![Anaconda](tools/data/anaconda_small.gif)](tools/anaconda.md) <br> [**Anaconda**](tools/anaconda.md) | [![Docker](tools/data/docker_small.gif)](tools/docker.md) <br> [**Docker**](tools/docker.md) | [![Python](tools/data/python_small.gif)](tools/python.md) <br> [**Python**](tools/python.md) |
|        [![Swift](tools/data/apple_small.gif)](tools/swift.md) <br> [**Swift**](tools/swift.md)         |                                                                                              |                                                                                              |

## Projects

### [Challenges](challenges/README.md)

|                                                        <img width=150/>                                                        |                                                 <img width=150/>                                                 |                                                               <img width=150/>                                                               |
|:------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------:|
| [![AI City](challenges/ai_city/data/ai_city_small.gif)](challenges/aic/README.md) <br> [**AI City**](challenges/aic/README.md) |   [![AutoNue](data/photo.png)](challenges/autonue/README.md) <br> [**AutoNue**](challenges/autonue/README.md)    | [![ChaLearn](challenges/chalearn/data/chalearn_small.gif)](challenges/chalearn/README.md) <br> [**ChaLearn**](challenges/chalearn/README.md) |
|            [![KATECH](data/photo.png)](challenges/katech/README.md) <br> [**KATECH**](challenges/katech/README.md)             |       [![KODAS](data/photo.png)](challenges/kodas/README.md) <br> [**KODAS**](challenges/kodas/README.md)        |                       [![NICO](data/photo.png)](challenges/nico/README.md) <br> [**NICO**](challenges/nico/README.md)                        |
 |              [![NTIRE](data/photo.png)](challenges/ntire/README.md) <br> [**NTIRE**](challenges/ntire/README.md)               | [![UG2+](challenges/ug2/data/ug2_small.gif)](challenges/ug2/README.md) <br> [**UG2+**](challenges/ug2/README.md) |               [![VisDrone](data/photo.png)](challenges/visdrone/README.md) <br> [**VisDrone**](challenges/visdrone/README.md)                |
 |        [![VIPriors](data/photo.png)](challenges/vipriors/README.md) <br> [**VIPriors**](challenges/vipriors/README.md)         |       [![Waymo](data/photo.png)](challenges/waymo/README.md) <br> [**Waymo+**](challenges/waymo/README.md)       |                                                                                                                                              |

[//]: # ()
[//]: # (### [Autonomous Vehicle]&#40;autonomous_vehicle&#41;)
[//]: # ()
[//]: # (|                                                                                                   <img width=150/>                                                                                                   |                                                                                                       <img width=150/>                                                                                                       | <img width=150/> |)
[//]: # (|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------:|)
[//]: # (| [![Autonomous Sensor]&#40;data/photo.png&#41;]&#40;autonomous_vehicle/autonomous_sensor&#41; <br> [**Autonomous<br>Sensor**]&#40;autonomous_vehicle/autonomous_sensor&#41; | [![Scene Understanding]&#40;data/photo.png&#41;]&#40;autonomous_vehicle/scene_understanding&#41; <br> [**Scene<br>Understanding**]&#40;autonomous_vehicle/scene_understanding&#41; |                  |)
[//]: # ()
[//]: # (### [Surveillance System]&#40;surveillance_system&#41;)
[//]: # ()
[//]: # (|                                                                               <img width=150/>                                                                               | <img width=150/> | <img width=150/> |)
[//]: # (|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------:|:----------------:|)
[//]: # (| [![Edge TSS]&#40;data/photo.png&#41;]&#40;https://phlong3105.github.io/surveillance_system/edge_tss&#41; <br>  [**Edge TSS**]&#40;surveillance_system/edge_tss&#41; |                  |                  |)

## Publications

A list of publications mentioning/using `One` package:

* [Runner-Up (3rd) submission to CVPR 2022 5th UG2+ Challenge](http://www.ug2challenge.org/)
* [DeepACO: A Robust Deep Learning-based Automatic Checkout System](https://openaccess.thecvf.com/content/CVPR2022W/AICity/html/Pham_DeepACO_A_Robust_Deep_Learning-Based_Automatic_Checkout_System_CVPRW_2022_paper.html)
* [A Region-and-Trajectory Movement Matching for Multiple Turn-counts at Road Intersection on Edge Device](https://ieeexplore.ieee.org/abstract/document/9523072)

## Cite

If you find our work useful, please cite the following:

```text
@misc{Pham2022,  
    author       = {Long Hoang Pham},  
    title        = {One: One Research Framework},  
    publisher    = {GitHub},
    journal      = {GitHub repository},
    howpublished = {\url{https://github.com/phlong3105/one}},
    year         = {2022},
}
```

## Contact

If you have any questions, feel free to contact `Long Pham` 
([longpham3105@gmail](longpham3105@gmail) or [phlong@skku.edu](phlong@skku.edu))


<script type="text/javascript" id="clustrmaps" src="//clustrmaps.com/map_v2.js?d=JUeNLvGJNmhIBDXVZ8UaNFwKXabm78dcdcwW8trsAXQ&cl=ffffff&w=a"></script>
