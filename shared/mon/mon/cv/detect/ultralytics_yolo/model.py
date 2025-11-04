#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Ultralytics YOLOs model for object detection, classification,
segmentation, orientation bounding box detection, and pose estimation.

References:
    - Code: https://github.com/ultralytics/ultralytics
"""

__all__ = [
    "YOLOv11l",
    "YOLOv11l_CLS",
    "YOLOv11l_OBB",
    "YOLOv11l_POSE",
    "YOLOv11l_SEG",
    "YOLOv11m",
    "YOLOv11m_CLS",
    "YOLOv11m_OBB",
    "YOLOv11m_POSE",
    "YOLOv11m_SEG",
    "YOLOv11n",
    "YOLOv11n_CLS",
    "YOLOv11n_OBB",
    "YOLOv11n_POSE",
    "YOLOv11n_SEG",
    "YOLOv11s",
    "YOLOv11s_CLS",
    "YOLOv11s_OBB",
    "YOLOv11s_POSE",
    "YOLOv11s_SEG",
    "YOLOv11x",
    "YOLOv11x_CLS",
    "YOLOv11x_OBB",
    "YOLOv11x_POSE",
    "YOLOv11x_POSE",
    "YOLOv11x_SEG",
    "YOLOv12",
    "YOLOv12l",
    "YOLOv12m",
    "YOLOv12n",
    "YOLOv12s",
    "YOLOv12x",
]

from typing import Any

import box
from ultralytics import YOLO

import mon.nn as nn
import mon.nn as nn
from mon.constants import MODELS, ROOT_DIR
from mon.core import MLType, Path, Task

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- YOLOv11 -----
class YOLOv11(YOLO, nn.ModelMixin):
    """Ultralytics YOLOs model for object detection.
    
    References:
        - Code: https://github.com/ultralytics/ultralytics
    """
    
    arch     : str          = "yolov11"
    name     : str          = "yolov11"
    tasks    : list[Task]   = [Task.DETECT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, weights: Any = "coco80", *args, **kwargs):
        weights, path, num_classes = self.parse_weights(weights)
        super().__init__(model=path, *args, **kwargs)
        

@MODELS.register(name="yolov11n", arch="yolov11")
class YOLOv11n(YOLOv11):
    
    name: str  = "yolov11n"
    zoo : dict = box.Box({
        "coco80"   : {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11n/coco80/yolov11n_coco80.pt",
            "num_classes": 80,
        },
        "widerface": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11n/widerface/yolov11n_widerface.pt",
            "num_classes": 1,
        },
    })
    

@MODELS.register(name="yolov11s", arch="yolov11")
class YOLOv11s(YOLOv11):
    
    name: str  = "yolov11s"
    zoo : dict = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11s/coco80/yolov11s_coco80.pt",
            "num_classes": 80,
        },
        "widerface": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11s/widerface/yolov11s_widerface.pt",
            "num_classes": 1,
        },
    })


@MODELS.register(name="yolov11m", arch="yolov11")
class YOLOv11m(YOLOv11):
    
    name: str  = "yolov11m"
    zoo : dict = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11m/coco80/yolov11m_coco80.pt",
            "num_classes": 80,
        },
    })


@MODELS.register(name="yolov11l", arch="yolov11")
class YOLOv11l(YOLOv11):
    
    name: str  = "yolov11l"
    zoo : dict = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11l/coco80/yolov11l_coco80.pt",
            "num_classes": 80,
        },
    })


@MODELS.register(name="yolov11x", arch="yolov11")
class YOLOv11x(YOLOv11):
    
    name: str  = "yolov11x"
    zoo : dict = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11x/coco80/yolov11x_coco80.pt",
            "num_classes": 80,
        },
    })


# ----- YOLOv11-OBB -----
class YOLOv11_OBB(YOLO, nn.ModelMixin):
    """Ultralytics YOLOs model for orientation bounding box detection.
    
    References:
        - Code: https://github.com/ultralytics/ultralytics
    """
    
    arch     : str          = "yolov11_obb"
    name     : str          = "yolov11_obb"
    tasks    : list[Task]   = [Task.OBB]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, weights: Any = "coco80", *args, **kwargs):
        weights, path, num_classes = self.parse_weights(weights)
        super().__init__(model=path, *args, **kwargs)
        
        
@MODELS.register(name="yolov11n_obb", arch="yolov11_obb")
class YOLOv11n_OBB(YOLOv11_OBB):
    
    name: str  = "yolov11n_obb"
    zoo : dict = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11n_obb/dotav1/yolov11n_obb_dotav1.pt",
            "num_classes": 15,
        },
    })


@MODELS.register(name="yolov11s_obb", arch="yolov11_obb")
class YOLOv11s_OBB(YOLOv11_OBB):
    
    name: str  = "yolov11s_obb"
    zoo : dict = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11s_obb/dotav1/yolov11s_obb_dotav1.pt",
            "num_classes": 15,
        },
    })


@MODELS.register(name="yolov11m_obb", arch="yolov11_obb")
class YOLOv11m_OBB(YOLOv11_OBB):
    
    name: str  = "yolov11m_obb"
    zoo : dict = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11m_obb/dotav1/yolov11m_obb_dotav1.pt",
            "num_classes": 15,
        },
    })


@MODELS.register(name="yolov11l_obb", arch="yolov11_obb")
class YOLOv11l_OBB(YOLOv11_OBB):
    
    name: str  = "yolov11l_obb"
    zoo : dict = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11l_obb/dotav1/yolov11l_obb_dotav1.pt",
            "num_classes": 15,
        },
    })


@MODELS.register(name="yolov11x_obb", arch="yolov11_obb")
class YOLOv11x_OBB(YOLOv11_OBB):
    
    name: str  = "yolov11x_obb"
    zoo : dict = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11x_obb/dotav1/yolov11x_obb_dotav1.pt",
            "num_classes": 15,
        },
    })


# ----- YOLOv11-SEG -----
class YOLOv11_SEG(YOLO, nn.ModelMixin):
    """Ultralytics YOLOs model for segmentation.
    
    References:
        - Code: https://github.com/ultralytics/ultralytics
    """
    
    arch     : str          = "yolov11_seg"
    name     : str          = "yolov11_seg"
    tasks    : list[Task]   = [Task.SEGMENT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, weights: Any = "coco80", *args, **kwargs):
        weights, path, num_classes = self.parse_weights(weights)
        super().__init__(model=path, *args, **kwargs)
        
        
@MODELS.register(name="yolov11n_seg", arch="yolov11_seg")
class YOLOv11n_SEG(YOLOv11_SEG):
    
    name: str  = "yolov11n_seg"
    zoo : dict = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11n_seg/coco80/yolov11n_seg_coco80.pt",
            "num_classes": 80,
        },
    })


@MODELS.register(name="yolov11s_seg", arch="yolov11_seg")
class YOLOv11s_SEG(YOLOv11_SEG):
    
    name: str  = "yolov11s_seg"
    zoo : dict = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11s_seg/coco80/yolov11s_seg_coco80.pt",
            "num_classes": 80,
        },
    })
    
    
@MODELS.register(name="yolov11m_seg", arch="yolov11_seg")
class YOLOv11m_SEG(YOLOv11_SEG):
    
    name: str  = "yolov11m_seg"
    zoo : dict = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11m_seg/coco80/yolov11m_seg_coco80.pt",
            "num_classes": 80,
        },
    })


@MODELS.register(name="yolov11l_seg", arch="yolov11_seg")
class YOLOv11l_SEG(YOLOv11_SEG):
    
    name: str  = "yolov11l_seg"
    zoo : dict = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11l_seg/coco80/yolov11l_seg_coco80.pt",
            "num_classes": 80,
        },
    })


@MODELS.register(name="yolov11x_seg", arch="yolov11_seg")
class YOLOv11x_SEG(YOLOv11_SEG):
    
    name: str  = "yolov11l_seg"
    zoo : dict = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11x_seg/coco80/yolov11x_seg_coco80.pt",
            "num_classes": 80,
        },
    })


# ----- YOLOv11-CLS -----
class YOLOv11_CLS(YOLO, nn.ModelMixin):
    """Ultralytics YOLOs model for classification.
    
    References:
        - Code: https://github.com/ultralytics/ultralytics
    """
    
    arch     : str          = "yolov11_cls"
    name     : str          = "yolov11n_cls"
    tasks    : list[Task]   = [Task.CLASSIFY]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, weights: Any = "imagenet", *args, **kwargs):
        weights, path, num_classes = self.parse_weights(weights)
        super().__init__(model=path, *args, **kwargs)
        
        
@MODELS.register(name="yolov11n_cls", arch="yolov11_cls")
class YOLOv11n_CLS(YOLOv11_CLS):
    
    name: str  = "yolov11n_cls"
    zoo : dict = box.Box({
        "imagenet": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11n_cls/imagenet/yolov11n_cls_imagenet.pt",
            "num_classes": 1000,
        },
    })


@MODELS.register(name="yolov11s_cls", arch="yolov11_cls")
class YOLOv11s_CLS(YOLOv11_CLS):
    
    name: str  = "yolov11s_cls"
    zoo : dict = box.Box({
        "imagenet": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11s_cls/imagenet/yolov11s_cls_imagenet.pt",
            "num_classes": 1000,
        },
    })


@MODELS.register(name="yolov11m_cls", arch="yolov11_cls")
class YOLOv11m_CLS(YOLOv11_CLS):
    
    name: str  = "yolov11m_cls"
    zoo : dict = box.Box({
        "imagenet": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11m_cls/imagenet/yolov11m_cls_imagenet.pt",
            "num_classes": 1000,
        },
    })


@MODELS.register(name="yolov11l_cls", arch="yolov11_cls")
class YOLOv11l_CLS(YOLOv11_CLS):
    
    name: str  = "yolov11l_cls"
    zoo : dict = box.Box({
        "imagenet": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11l_cls/imagenet/yolov11l_cls_imagenet.pt",
            "num_classes": 1000,
        },
    })


@MODELS.register(name="yolov11x_cls", arch="yolov11_cls")
class YOLOv11x_CLS(YOLOv11_CLS):
    
    name: str  = "yolov11x_cls"
    zoo : dict = box.Box({
        "imagenet": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11x_cls/imagenet/yolov11x_cls_imagenet.pt",
            "num_classes": 1000,
        },
    })


# ----- YOLOv11-POSE -----
class YOLOv11_POSE(YOLO, nn.ModelMixin):
    """Ultralytics YOLOs model for pose estimation.
    
    References:
        - Code: https://github.com/ultralytics/ultralytics
    """
    
    arch     : str          = "yolov11_pose"
    name     : str          = "yolov11n_pose"
    tasks    : list[Task]   = [Task.POSE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, weights: Any = "coco1", *args, **kwargs):
        weights, path, num_classes = self.parse_weights(weights)
        super().__init__(model=path, *args, **kwargs)
        
        
@MODELS.register(name="yolov11n_pose", arch="yolov11_pose")
class YOLOv11n_POSE(YOLOv11_POSE):
    
    name: str  = "yolov11n_pose"
    zoo : dict = box.Box({
        "coco1": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11n_pose/coco1/yolov11n_pose_coco1.pt",
            "num_classes": 1,
        },
    })


@MODELS.register(name="yolov11s_pose", arch="yolov11_pose")
class YOLOv11s_POSE(YOLOv11_POSE):
    
    name: str  = "yolov11s_pose"
    zoo : dict = box.Box({
        "coco1": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11s_pose/coco1/yolov11s_pose_coco1.pt",
            "num_classes": 1,
        },
    })


@MODELS.register(name="yolov11m_pose", arch="yolov11_pose")
class YOLOv11m_POSE(YOLOv11_POSE):
    
    name: str  = "yolov11m_pose"
    zoo : dict = box.Box({
        "coco1": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11m_pose/coco1/yolov11m_pose_coco1.pt",
            "num_classes": 1,
        },
    })


@MODELS.register(name="yolov11l_pose", arch="yolov11_pose")
class YOLOv11l_POSE(YOLOv11_POSE):
    
    name: str  = "yolov11l_pose"
    zoo : dict = box.Box({
        "coco1": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11l_pose/coco1/yolov11l_pose_coco1.pt",
            "num_classes": 1,
        },
    })
    

@MODELS.register(name="yolov11x_pose", arch="yolov11_pose")
class YOLOv11x_POSE(YOLOv11_POSE):
    
    name: str  = "yolov11x_pose"
    zoo : dict = box.Box({
        "coco1": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov11/yolov11x_pose/coco1/yolov11x_pose_coco1.pt",
            "num_classes": 1,
        },
    })


# ----- YOLOv12 -----
class YOLOv12(YOLO, nn.ModelMixin):
    """Ultralytics YOLOs model for object detection.
    
    References:
        - Code: https://github.com/ultralytics/ultralytics
    """
    
    arch     : str          = "yolov12"
    name     : str          = "yolov12"
    tasks    : list[Task]   = [Task.DETECT]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, weights: Any = "coco80", *args, **kwargs):
        weights, path, num_classes = self.parse_weights(weights)
        super().__init__(model=path, *args, **kwargs)
        

@MODELS.register(name="yolov12n", arch="yolov12")
class YOLOv12n(YOLOv12):
    
    name: str  = "yolov12n"
    zoo : dict = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov12/yolov12n/coco80/yolov12n_coco80.pt",
            "num_classes": 80,
        },
    })
    

@MODELS.register(name="yolov12s", arch="yolov12")
class YOLOv12s(YOLOv12):
    
    name: str  = "yolov12s"
    zoo : dict = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov12/yolov12s/coco80/yolov12s_coco80.pt",
            "num_classes": 80,
        },
    })


@MODELS.register(name="yolov12m", arch="yolov12")
class YOLOv12m(YOLOv12):
    
    name: str  = "yolov12m"
    zoo : dict = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov12/yolov12m/coco80/yolov12m_coco80.pt",
            "num_classes": 80,
        },
    })


@MODELS.register(name="yolov12l", arch="yolov12")
class YOLOv12l(YOLOv12):
    
    name: str  = "yolov12l"
    zoo : dict = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov12/yolov12l/coco80/yolov12l_coco80.pt",
            "num_classes": 80,
        },
    })


@MODELS.register(name="yolov12x", arch="yolov12")
class YOLOv12x(YOLOv12):
    
    name: str  = "yolov12x"
    zoo : dict = box.Box({
        "coco80": {
            "url"        : "",
            "path"       : ROOT_DIR / "zoo/cv/ultralytics/yolov12/yolov12x/coco80/yolov12x_coco80.pt",
            "num_classes": 80,
        },
    })
