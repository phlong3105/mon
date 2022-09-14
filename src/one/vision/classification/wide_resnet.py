#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from one.nn import *
from one.vision.classification.resnet import ResNet

CURRENT_DIR = Path(__file__).resolve().parent
CFG_DIR     = CURRENT_DIR / "cfg"


# H1: - Model ------------------------------------------------------------------

cfgs = {
    "wide_resnet50": {
        "channels": 3,
        "backbone": [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [64, 7, 2, 3, 1, 1, False]],                                          # 0
            [-1,     1,      BatchNorm2d,       []],                                                                  # 1
            [-1,     1,      ReLU,              [True]],                                                              # 2
            [-1,     1,      MaxPool2d,         [3, 2, 1]],                                                           # 3
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 3, 64,   64,  1, 1, 1, 128, False, BatchNorm2d]],  # 4
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 4, 256,  128, 2, 1, 1, 128, False, BatchNorm2d]],  # 5
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 6, 512,  256, 2, 1, 1, 128, False, BatchNorm2d]],  # 6
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 3, 1024, 512, 2, 1, 1, 128, False, BatchNorm2d]],  # 7
            [-1,     1,      AdaptiveAvgPool2d, [1]],                                                                 # 8
        ],
        "head": [
            [-1,     1,      ResNetClassifier,  [2048]],                                                              # 9
        ]
    },
    "wide_resnet101": {
        "channels": 3,
        "backbone": [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [64, 7, 2, 3, 1, 1, False]],                                           # 0
            [-1,     1,      BatchNorm2d,       []],                                                                   # 1
            [-1,     1,      ReLU,              [True]],                                                               # 2
            [-1,     1,      MaxPool2d,         [3, 2, 1]],                                                            # 3
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 3,  64,   64,  1, 1, 1, 128, False, BatchNorm2d]],  # 4
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 4,  256,  128, 2, 1, 1, 128, False, BatchNorm2d]],  # 5
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 23, 512,  256, 2, 1, 1, 128, False, BatchNorm2d]],  # 6
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 3,  1024, 512, 2, 1, 1, 128, False, BatchNorm2d]],  # 7
            [-1,     1,      AdaptiveAvgPool2d, [1]],                                                                  # 8
        ],
        "head": [
            [-1,     1,      ResNetClassifier,  [2048]],                                                               # 9
        ]
    },
}


@MODELS.register(name="wide_resnet50")
class Wide_ResNet50(ResNet):
    """
    
    Args:
        cfg (dict | Path_ | None): Model's layers configuration. It can be an
            external .yaml path or a dictionary. Defaults to None means you
            should define each layer manually in `self.parse_model()` method.
        root (Path_): The root directory of the model. Defaults to RUNS_DIR.
        name (str | None): Model's name. In case None is given, it will be
            `self.__class__.__name__`. Defaults to None.
        fullname (str | None): Model's fullname in the following format:
            {name}-{data_name}-{postfix}. In case None is given, it will be
            `self.name`. Defaults to None.
        channels (int): Input channel. Defaults to 3.
        num_classes (int | None): Number of classes for classification or
            detection tasks. Defaults to None.
        classlabels (ClassLabels | None): ClassLabels object that contains all
            labels in the dataset. Defaults to None.
        pretrained (Pretrained): Initialize weights from pretrained.
            - If True, use the original pretrained described by the author
              (usually, ImageNet or COCO). By default, it is the first element
              in the `model_zoo` dictionary.
            - If str and is a file/path, then load weights from saved file.
            - In each inherited model, `pretrained` can be a dictionary's
              key to get the corresponding local file or url of the weight.
        phase (ModelPhase_): Model's running phase. Defaults to training.
        loss (Losses_ | None): Loss function for training model.
            Defaults to None.
        metrics (Metrics_ | None): Metric(s) for validating and testing model.
            Defaults to None.
        optimizers (Optimizers_ | None): Optimizer(s) for training model.
            Defaults to None.
        debug (dict | Munch | None): Debug configs. Defaults to None.
        verbose (bool): Verbosity.
    """
    
    model_zoo = {
        "imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth",
            filename    = "wide_resnet50-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "wide_resnet50.yaml",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "wide_resnet",
        fullname   : str  | None         = "wide_resnet50",
        channels   : int                 = 3,
        num_classes: int  | None 		 = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = None,
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg = cfg or "wide_resnet50"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg
        
        super().__init__(
            cfg         = cfg,
            root        = root,
            name        = name,
            fullname    = fullname,
            channels    = channels,
            num_classes = num_classes,
            pretrained  = Wide_ResNet50.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def load_pretrained(self):
        """
        Load pretrained weights. It only loads the intersection layers of
        matching keys and shapes between current model and pretrained.
        """
        if is_dict(self.pretrained) and self.pretrained["name"] == "imagenet":
            state_dict = load_state_dict_from_path(
                model_dir=self.pretrained_dir, **self.pretrained
            )
            """
            for k in sorted(self.model.state_dict().keys()):
                print(f"model_state_dict[\"{k}\"] = ")
            for k in sorted(state_dict.keys()):
                print(f"state_dict[\"{k}\"]")
            """
            model_state_dict = self.model.state_dict()
            model_state_dict["0.weight"]                                   = state_dict["conv1.weight"]
            model_state_dict["1.weight"]                                   = state_dict["bn1.weight"]
            model_state_dict["1.bias"]                                     = state_dict["bn1.bias"]
            model_state_dict["1.running_mean"]                             = state_dict["bn1.running_mean"]
            model_state_dict["1.running_var"]                              = state_dict["bn1.running_var"]
            model_state_dict["4.convs.0.bn1.bias"]                         = state_dict["layer1.0.bn1.bias"]
            model_state_dict["4.convs.0.bn1.num_batches_tracked"]          = state_dict["layer1.0.bn1.num_batches_tracked"]
            model_state_dict["4.convs.0.bn1.running_mean"]                 = state_dict["layer1.0.bn1.running_mean"]
            model_state_dict["4.convs.0.bn1.running_var"]                  = state_dict["layer1.0.bn1.running_var"]
            model_state_dict["4.convs.0.bn1.weight"]                       = state_dict["layer1.0.bn1.weight"]
            model_state_dict["4.convs.0.bn2.bias"]                         = state_dict["layer1.0.bn2.bias"]
            model_state_dict["4.convs.0.bn2.num_batches_tracked"]          = state_dict["layer1.0.bn2.num_batches_tracked"]
            model_state_dict["4.convs.0.bn2.running_mean"]                 = state_dict["layer1.0.bn2.running_mean"]
            model_state_dict["4.convs.0.bn2.running_var"]                  = state_dict["layer1.0.bn2.running_var"]
            model_state_dict["4.convs.0.bn2.weight"]                       = state_dict["layer1.0.bn2.weight"]
            model_state_dict["4.convs.0.bn3.bias"]                         = state_dict["layer1.0.bn3.bias"]
            model_state_dict["4.convs.0.bn3.num_batches_tracked"]          = state_dict["layer1.0.bn3.num_batches_tracked"]
            model_state_dict["4.convs.0.bn3.running_mean"]                 = state_dict["layer1.0.bn3.running_mean"]
            model_state_dict["4.convs.0.bn3.running_var"]                  = state_dict["layer1.0.bn3.running_var"]
            model_state_dict["4.convs.0.bn3.weight"]                       = state_dict["layer1.0.bn3.weight"]
            model_state_dict["4.convs.0.conv1.weight"]                     = state_dict["layer1.0.conv1.weight"]
            model_state_dict["4.convs.0.conv2.weight"]                     = state_dict["layer1.0.conv2.weight"]
            model_state_dict["4.convs.0.conv3.weight"]                     = state_dict["layer1.0.conv3.weight"]
            model_state_dict["4.convs.0.downsample.0.weight"]              = state_dict["layer1.0.downsample.0.weight"]
            model_state_dict["4.convs.0.downsample.1.bias"]                = state_dict["layer1.0.downsample.1.bias"]
            model_state_dict["4.convs.0.downsample.1.num_batches_tracked"] = state_dict["layer1.0.downsample.1.num_batches_tracked"]
            model_state_dict["4.convs.0.downsample.1.running_mean"]        = state_dict["layer1.0.downsample.1.running_mean"]
            model_state_dict["4.convs.0.downsample.1.running_var"]         = state_dict["layer1.0.downsample.1.running_var"]
            model_state_dict["4.convs.0.downsample.1.weight"]              = state_dict["layer1.0.downsample.1.weight"]
            model_state_dict["4.convs.1.bn1.bias"]                         = state_dict["layer1.1.bn1.bias"]
            model_state_dict["4.convs.1.bn1.num_batches_tracked"]          = state_dict["layer1.1.bn1.num_batches_tracked"]
            model_state_dict["4.convs.1.bn1.running_mean"]                 = state_dict["layer1.1.bn1.running_mean"]
            model_state_dict["4.convs.1.bn1.running_var"]                  = state_dict["layer1.1.bn1.running_var"]
            model_state_dict["4.convs.1.bn1.weight"]                       = state_dict["layer1.1.bn1.weight"]
            model_state_dict["4.convs.1.bn2.bias"]                         = state_dict["layer1.1.bn2.bias"]
            model_state_dict["4.convs.1.bn2.num_batches_tracked"]          = state_dict["layer1.1.bn2.num_batches_tracked"]
            model_state_dict["4.convs.1.bn2.running_mean"]                 = state_dict["layer1.1.bn2.running_mean"]
            model_state_dict["4.convs.1.bn2.running_var"]                  = state_dict["layer1.1.bn2.running_var"]
            model_state_dict["4.convs.1.bn2.weight"]                       = state_dict["layer1.1.bn2.weight"]
            model_state_dict["4.convs.1.bn3.bias"]                         = state_dict["layer1.1.bn3.bias"]
            model_state_dict["4.convs.1.bn3.num_batches_tracked"]          = state_dict["layer1.1.bn3.num_batches_tracked"]
            model_state_dict["4.convs.1.bn3.running_mean"]                 = state_dict["layer1.1.bn3.running_mean"]
            model_state_dict["4.convs.1.bn3.running_var"]                  = state_dict["layer1.1.bn3.running_var"]
            model_state_dict["4.convs.1.bn3.weight"]                       = state_dict["layer1.1.bn3.weight"]
            model_state_dict["4.convs.1.conv1.weight"]                     = state_dict["layer1.1.conv1.weight"]
            model_state_dict["4.convs.1.conv2.weight"]                     = state_dict["layer1.1.conv2.weight"]
            model_state_dict["4.convs.1.conv3.weight"]                     = state_dict["layer1.1.conv3.weight"]
            model_state_dict["4.convs.2.bn1.bias"]                         = state_dict["layer1.2.bn1.bias"]
            model_state_dict["4.convs.2.bn1.num_batches_tracked"]          = state_dict["layer1.2.bn1.num_batches_tracked"]
            model_state_dict["4.convs.2.bn1.running_mean"]                 = state_dict["layer1.2.bn1.running_mean"]
            model_state_dict["4.convs.2.bn1.running_var"]                  = state_dict["layer1.2.bn1.running_var"]
            model_state_dict["4.convs.2.bn1.weight"]                       = state_dict["layer1.2.bn1.weight"]
            model_state_dict["4.convs.2.bn2.bias"]                         = state_dict["layer1.2.bn2.bias"]
            model_state_dict["4.convs.2.bn2.num_batches_tracked"]          = state_dict["layer1.2.bn2.num_batches_tracked"]
            model_state_dict["4.convs.2.bn2.running_mean"]                 = state_dict["layer1.2.bn2.running_mean"]
            model_state_dict["4.convs.2.bn2.running_var"]                  = state_dict["layer1.2.bn2.running_var"]
            model_state_dict["4.convs.2.bn2.weight"]                       = state_dict["layer1.2.bn2.weight"]
            model_state_dict["4.convs.2.bn3.bias"]                         = state_dict["layer1.2.bn3.bias"]
            model_state_dict["4.convs.2.bn3.num_batches_tracked"]          = state_dict["layer1.2.bn3.num_batches_tracked"]
            model_state_dict["4.convs.2.bn3.running_mean"]                 = state_dict["layer1.2.bn3.running_mean"]
            model_state_dict["4.convs.2.bn3.running_var"]                  = state_dict["layer1.2.bn3.running_var"]
            model_state_dict["4.convs.2.bn3.weight"]                       = state_dict["layer1.2.bn3.weight"]
            model_state_dict["4.convs.2.conv1.weight"]                     = state_dict["layer1.2.conv1.weight"]
            model_state_dict["4.convs.2.conv2.weight"]                     = state_dict["layer1.2.conv2.weight"]
            model_state_dict["4.convs.2.conv3.weight"]                     = state_dict["layer1.2.conv3.weight"]
            model_state_dict["5.convs.0.bn1.bias"]                         = state_dict["layer2.0.bn1.bias"]
            model_state_dict["5.convs.0.bn1.num_batches_tracked"]          = state_dict["layer2.0.bn1.num_batches_tracked"]
            model_state_dict["5.convs.0.bn1.running_mean"]                 = state_dict["layer2.0.bn1.running_mean"]
            model_state_dict["5.convs.0.bn1.running_var"]                  = state_dict["layer2.0.bn1.running_var"]
            model_state_dict["5.convs.0.bn1.weight"]                       = state_dict["layer2.0.bn1.weight"]
            model_state_dict["5.convs.0.bn2.bias"]                         = state_dict["layer2.0.bn2.bias"]
            model_state_dict["5.convs.0.bn2.num_batches_tracked"]          = state_dict["layer2.0.bn2.num_batches_tracked"]
            model_state_dict["5.convs.0.bn2.running_mean"]                 = state_dict["layer2.0.bn2.running_mean"]
            model_state_dict["5.convs.0.bn2.running_var"]                  = state_dict["layer2.0.bn2.running_var"]
            model_state_dict["5.convs.0.bn2.weight"]                       = state_dict["layer2.0.bn2.weight"]
            model_state_dict["5.convs.0.bn3.bias"]                         = state_dict["layer2.0.bn3.bias"]
            model_state_dict["5.convs.0.bn3.num_batches_tracked"]          = state_dict["layer2.0.bn3.num_batches_tracked"]
            model_state_dict["5.convs.0.bn3.running_mean"]                 = state_dict["layer2.0.bn3.running_mean"]
            model_state_dict["5.convs.0.bn3.running_var"]                  = state_dict["layer2.0.bn3.running_var"]
            model_state_dict["5.convs.0.bn3.weight"]                       = state_dict["layer2.0.bn3.weight"]
            model_state_dict["5.convs.0.conv1.weight"]                     = state_dict["layer2.0.conv1.weight"]
            model_state_dict["5.convs.0.conv2.weight"]                     = state_dict["layer2.0.conv2.weight"]
            model_state_dict["5.convs.0.conv3.weight"]                     = state_dict["layer2.0.conv3.weight"]
            model_state_dict["5.convs.0.downsample.0.weight"]              = state_dict["layer2.0.downsample.0.weight"]
            model_state_dict["5.convs.0.downsample.1.bias"]                = state_dict["layer2.0.downsample.1.bias"]
            model_state_dict["5.convs.0.downsample.1.num_batches_tracked"] = state_dict["layer2.0.downsample.1.num_batches_tracked"]
            model_state_dict["5.convs.0.downsample.1.running_mean"]        = state_dict["layer2.0.downsample.1.running_mean"]
            model_state_dict["5.convs.0.downsample.1.running_var"]         = state_dict["layer2.0.downsample.1.running_var"]
            model_state_dict["5.convs.0.downsample.1.weight"]              = state_dict["layer2.0.downsample.1.weight"]
            model_state_dict["5.convs.1.bn1.bias"]                         = state_dict["layer2.1.bn1.bias"]
            model_state_dict["5.convs.1.bn1.num_batches_tracked"]          = state_dict["layer2.1.bn1.num_batches_tracked"]
            model_state_dict["5.convs.1.bn1.running_mean"]                 = state_dict["layer2.1.bn1.running_mean"]
            model_state_dict["5.convs.1.bn1.running_var"]                  = state_dict["layer2.1.bn1.running_var"]
            model_state_dict["5.convs.1.bn1.weight"]                       = state_dict["layer2.1.bn1.weight"]
            model_state_dict["5.convs.1.bn2.bias"]                         = state_dict["layer2.1.bn2.bias"]
            model_state_dict["5.convs.1.bn2.num_batches_tracked"]          = state_dict["layer2.1.bn2.num_batches_tracked"]
            model_state_dict["5.convs.1.bn2.running_mean"]                 = state_dict["layer2.1.bn2.running_mean"]
            model_state_dict["5.convs.1.bn2.running_var"]                  = state_dict["layer2.1.bn2.running_var"]
            model_state_dict["5.convs.1.bn2.weight"]                       = state_dict["layer2.1.bn2.weight"]
            model_state_dict["5.convs.1.bn3.bias"]                         = state_dict["layer2.1.bn3.bias"]
            model_state_dict["5.convs.1.bn3.num_batches_tracked"]          = state_dict["layer2.1.bn3.num_batches_tracked"]
            model_state_dict["5.convs.1.bn3.running_mean"]                 = state_dict["layer2.1.bn3.running_mean"]
            model_state_dict["5.convs.1.bn3.running_var"]                  = state_dict["layer2.1.bn3.running_var"]
            model_state_dict["5.convs.1.bn3.weight"]                       = state_dict["layer2.1.bn3.weight"]
            model_state_dict["5.convs.1.conv1.weight"]                     = state_dict["layer2.1.conv1.weight"]
            model_state_dict["5.convs.1.conv2.weight"]                     = state_dict["layer2.1.conv2.weight"]
            model_state_dict["5.convs.1.conv3.weight"]                     = state_dict["layer2.1.conv3.weight"]
            model_state_dict["5.convs.2.bn1.bias"]                         = state_dict["layer2.2.bn1.bias"]
            model_state_dict["5.convs.2.bn1.num_batches_tracked"]          = state_dict["layer2.2.bn1.num_batches_tracked"]
            model_state_dict["5.convs.2.bn1.running_mean"]                 = state_dict["layer2.2.bn1.running_mean"]
            model_state_dict["5.convs.2.bn1.running_var"]                  = state_dict["layer2.2.bn1.running_var"]
            model_state_dict["5.convs.2.bn1.weight"]                       = state_dict["layer2.2.bn1.weight"]
            model_state_dict["5.convs.2.bn2.bias"]                         = state_dict["layer2.2.bn2.bias"]
            model_state_dict["5.convs.2.bn2.num_batches_tracked"]          = state_dict["layer2.2.bn2.num_batches_tracked"]
            model_state_dict["5.convs.2.bn2.running_mean"]                 = state_dict["layer2.2.bn2.running_mean"]
            model_state_dict["5.convs.2.bn2.running_var"]                  = state_dict["layer2.2.bn2.running_var"]
            model_state_dict["5.convs.2.bn2.weight"]                       = state_dict["layer2.2.bn2.weight"]
            model_state_dict["5.convs.2.bn3.bias"]                         = state_dict["layer2.2.bn3.bias"]
            model_state_dict["5.convs.2.bn3.num_batches_tracked"]          = state_dict["layer2.2.bn3.num_batches_tracked"]
            model_state_dict["5.convs.2.bn3.running_mean"]                 = state_dict["layer2.2.bn3.running_mean"]
            model_state_dict["5.convs.2.bn3.running_var"]                  = state_dict["layer2.2.bn3.running_var"]
            model_state_dict["5.convs.2.bn3.weight"]                       = state_dict["layer2.2.bn3.weight"]
            model_state_dict["5.convs.2.conv1.weight"]                     = state_dict["layer2.2.conv1.weight"]
            model_state_dict["5.convs.2.conv2.weight"]                     = state_dict["layer2.2.conv2.weight"]
            model_state_dict["5.convs.2.conv3.weight"]                     = state_dict["layer2.2.conv3.weight"]
            model_state_dict["5.convs.3.bn1.bias"]                         = state_dict["layer2.3.bn1.bias"]
            model_state_dict["5.convs.3.bn1.num_batches_tracked"]          = state_dict["layer2.3.bn1.num_batches_tracked"]
            model_state_dict["5.convs.3.bn1.running_mean"]                 = state_dict["layer2.3.bn1.running_mean"]
            model_state_dict["5.convs.3.bn1.running_var"]                  = state_dict["layer2.3.bn1.running_var"]
            model_state_dict["5.convs.3.bn1.weight"]                       = state_dict["layer2.3.bn1.weight"]
            model_state_dict["5.convs.3.bn2.bias"]                         = state_dict["layer2.3.bn2.bias"]
            model_state_dict["5.convs.3.bn2.num_batches_tracked"]          = state_dict["layer2.3.bn2.num_batches_tracked"]
            model_state_dict["5.convs.3.bn2.running_mean"]                 = state_dict["layer2.3.bn2.running_mean"]
            model_state_dict["5.convs.3.bn2.running_var"]                  = state_dict["layer2.3.bn2.running_var"]
            model_state_dict["5.convs.3.bn2.weight"]                       = state_dict["layer2.3.bn2.weight"]
            model_state_dict["5.convs.3.bn3.bias"]                         = state_dict["layer2.3.bn3.bias"]
            model_state_dict["5.convs.3.bn3.num_batches_tracked"]          = state_dict["layer2.3.bn3.num_batches_tracked"]
            model_state_dict["5.convs.3.bn3.running_mean"]                 = state_dict["layer2.3.bn3.running_mean"]
            model_state_dict["5.convs.3.bn3.running_var"]                  = state_dict["layer2.3.bn3.running_var"]
            model_state_dict["5.convs.3.bn3.weight"]                       = state_dict["layer2.3.bn3.weight"]
            model_state_dict["5.convs.3.conv1.weight"]                     = state_dict["layer2.3.conv1.weight"]
            model_state_dict["5.convs.3.conv2.weight"]                     = state_dict["layer2.3.conv2.weight"]
            model_state_dict["5.convs.3.conv3.weight"]                     = state_dict["layer2.3.conv3.weight"]
            model_state_dict["6.convs.0.bn1.bias"]                         = state_dict["layer3.0.bn1.bias"]
            model_state_dict["6.convs.0.bn1.num_batches_tracked"]          = state_dict["layer3.0.bn1.num_batches_tracked"]
            model_state_dict["6.convs.0.bn1.running_mean"]                 = state_dict["layer3.0.bn1.running_mean"]
            model_state_dict["6.convs.0.bn1.running_var"]                  = state_dict["layer3.0.bn1.running_var"]
            model_state_dict["6.convs.0.bn1.weight"]                       = state_dict["layer3.0.bn1.weight"]
            model_state_dict["6.convs.0.bn2.bias"]                         = state_dict["layer3.0.bn2.bias"]
            model_state_dict["6.convs.0.bn2.num_batches_tracked"]          = state_dict["layer3.0.bn2.num_batches_tracked"]
            model_state_dict["6.convs.0.bn2.running_mean"]                 = state_dict["layer3.0.bn2.running_mean"]
            model_state_dict["6.convs.0.bn2.running_var"]                  = state_dict["layer3.0.bn2.running_var"]
            model_state_dict["6.convs.0.bn2.weight"]                       = state_dict["layer3.0.bn2.weight"]
            model_state_dict["6.convs.0.bn3.bias"]                         = state_dict["layer3.0.bn3.bias"]
            model_state_dict["6.convs.0.bn3.num_batches_tracked"]          = state_dict["layer3.0.bn3.num_batches_tracked"]
            model_state_dict["6.convs.0.bn3.running_mean"]                 = state_dict["layer3.0.bn3.running_mean"]
            model_state_dict["6.convs.0.bn3.running_var"]                  = state_dict["layer3.0.bn3.running_var"]
            model_state_dict["6.convs.0.bn3.weight"]                       = state_dict["layer3.0.bn3.weight"]
            model_state_dict["6.convs.0.conv1.weight"]                     = state_dict["layer3.0.conv1.weight"]
            model_state_dict["6.convs.0.conv2.weight"]                     = state_dict["layer3.0.conv2.weight"]
            model_state_dict["6.convs.0.conv3.weight"]                     = state_dict["layer3.0.conv3.weight"]
            model_state_dict["6.convs.0.downsample.0.weight"]              = state_dict["layer3.0.downsample.0.weight"]
            model_state_dict["6.convs.0.downsample.1.bias"]                = state_dict["layer3.0.downsample.1.bias"]
            model_state_dict["6.convs.0.downsample.1.num_batches_tracked"] = state_dict["layer3.0.downsample.1.num_batches_tracked"]
            model_state_dict["6.convs.0.downsample.1.running_mean"]        = state_dict["layer3.0.downsample.1.running_mean"]
            model_state_dict["6.convs.0.downsample.1.running_var"]         = state_dict["layer3.0.downsample.1.running_var"]
            model_state_dict["6.convs.0.downsample.1.weight"]              = state_dict["layer3.0.downsample.1.weight"]
            model_state_dict["6.convs.1.bn1.bias"]                         = state_dict["layer3.1.bn1.bias"]
            model_state_dict["6.convs.1.bn1.num_batches_tracked"]          = state_dict["layer3.1.bn1.num_batches_tracked"]
            model_state_dict["6.convs.1.bn1.running_mean"]                 = state_dict["layer3.1.bn1.running_mean"]
            model_state_dict["6.convs.1.bn1.running_var"]                  = state_dict["layer3.1.bn1.running_var"]
            model_state_dict["6.convs.1.bn1.weight"]                       = state_dict["layer3.1.bn1.weight"]
            model_state_dict["6.convs.1.bn2.bias"]                         = state_dict["layer3.1.bn2.bias"]
            model_state_dict["6.convs.1.bn2.num_batches_tracked"]          = state_dict["layer3.1.bn2.num_batches_tracked"]
            model_state_dict["6.convs.1.bn2.running_mean"]                 = state_dict["layer3.1.bn2.running_mean"]
            model_state_dict["6.convs.1.bn2.running_var"]                  = state_dict["layer3.1.bn2.running_var"]
            model_state_dict["6.convs.1.bn2.weight"]                       = state_dict["layer3.1.bn2.weight"]
            model_state_dict["6.convs.1.bn3.bias"]                         = state_dict["layer3.1.bn3.bias"]
            model_state_dict["6.convs.1.bn3.num_batches_tracked"]          = state_dict["layer3.1.bn3.num_batches_tracked"]
            model_state_dict["6.convs.1.bn3.running_mean"]                 = state_dict["layer3.1.bn3.running_mean"]
            model_state_dict["6.convs.1.bn3.running_var"]                  = state_dict["layer3.1.bn3.running_var"]
            model_state_dict["6.convs.1.bn3.weight"]                       = state_dict["layer3.1.bn3.weight"]
            model_state_dict["6.convs.1.conv1.weight"]                     = state_dict["layer3.1.conv1.weight"]
            model_state_dict["6.convs.1.conv2.weight"]                     = state_dict["layer3.1.conv2.weight"]
            model_state_dict["6.convs.1.conv3.weight"]                     = state_dict["layer3.1.conv3.weight"]
            model_state_dict["6.convs.2.bn1.bias"]                         = state_dict["layer3.2.bn1.bias"]
            model_state_dict["6.convs.2.bn1.num_batches_tracked"]          = state_dict["layer3.2.bn1.num_batches_tracked"]
            model_state_dict["6.convs.2.bn1.running_mean"]                 = state_dict["layer3.2.bn1.running_mean"]
            model_state_dict["6.convs.2.bn1.running_var"]                  = state_dict["layer3.2.bn1.running_var"]
            model_state_dict["6.convs.2.bn1.weight"]                       = state_dict["layer3.2.bn1.weight"]
            model_state_dict["6.convs.2.bn2.bias"]                         = state_dict["layer3.2.bn2.bias"]
            model_state_dict["6.convs.2.bn2.num_batches_tracked"]          = state_dict["layer3.2.bn2.num_batches_tracked"]
            model_state_dict["6.convs.2.bn2.running_mean"]                 = state_dict["layer3.2.bn2.running_mean"]
            model_state_dict["6.convs.2.bn2.running_var"]                  = state_dict["layer3.2.bn2.running_var"]
            model_state_dict["6.convs.2.bn2.weight"]                       = state_dict["layer3.2.bn2.weight"]
            model_state_dict["6.convs.2.bn3.bias"]                         = state_dict["layer3.2.bn3.bias"]
            model_state_dict["6.convs.2.bn3.num_batches_tracked"]          = state_dict["layer3.2.bn3.num_batches_tracked"]
            model_state_dict["6.convs.2.bn3.running_mean"]                 = state_dict["layer3.2.bn3.running_mean"]
            model_state_dict["6.convs.2.bn3.running_var"]                  = state_dict["layer3.2.bn3.running_var"]
            model_state_dict["6.convs.2.bn3.weight"]                       = state_dict["layer3.2.bn3.weight"]
            model_state_dict["6.convs.2.conv1.weight"]                     = state_dict["layer3.2.conv1.weight"]
            model_state_dict["6.convs.2.conv2.weight"]                     = state_dict["layer3.2.conv2.weight"]
            model_state_dict["6.convs.2.conv3.weight"]                     = state_dict["layer3.2.conv3.weight"]
            model_state_dict["6.convs.3.bn1.bias"]                         = state_dict["layer3.3.bn1.bias"]
            model_state_dict["6.convs.3.bn1.num_batches_tracked"]          = state_dict["layer3.3.bn1.num_batches_tracked"]
            model_state_dict["6.convs.3.bn1.running_mean"]                 = state_dict["layer3.3.bn1.running_mean"]
            model_state_dict["6.convs.3.bn1.running_var"]                  = state_dict["layer3.3.bn1.running_var"]
            model_state_dict["6.convs.3.bn1.weight"]                       = state_dict["layer3.3.bn1.weight"]
            model_state_dict["6.convs.3.bn2.bias"]                         = state_dict["layer3.3.bn2.bias"]
            model_state_dict["6.convs.3.bn2.num_batches_tracked"]          = state_dict["layer3.3.bn2.num_batches_tracked"]
            model_state_dict["6.convs.3.bn2.running_mean"]                 = state_dict["layer3.3.bn2.running_mean"]
            model_state_dict["6.convs.3.bn2.running_var"]                  = state_dict["layer3.3.bn2.running_var"]
            model_state_dict["6.convs.3.bn2.weight"]                       = state_dict["layer3.3.bn2.weight"]
            model_state_dict["6.convs.3.bn3.bias"]                         = state_dict["layer3.3.bn3.bias"]
            model_state_dict["6.convs.3.bn3.num_batches_tracked"]          = state_dict["layer3.3.bn3.num_batches_tracked"]
            model_state_dict["6.convs.3.bn3.running_mean"]                 = state_dict["layer3.3.bn3.running_mean"]
            model_state_dict["6.convs.3.bn3.running_var"]                  = state_dict["layer3.3.bn3.running_var"]
            model_state_dict["6.convs.3.bn3.weight"]                       = state_dict["layer3.3.bn3.weight"]
            model_state_dict["6.convs.3.conv1.weight"]                     = state_dict["layer3.3.conv1.weight"]
            model_state_dict["6.convs.3.conv2.weight"]                     = state_dict["layer3.3.conv2.weight"]
            model_state_dict["6.convs.3.conv3.weight"]                     = state_dict["layer3.3.conv3.weight"]
            model_state_dict["6.convs.4.bn1.bias"]                         = state_dict["layer3.4.bn1.bias"]
            model_state_dict["6.convs.4.bn1.num_batches_tracked"]          = state_dict["layer3.4.bn1.num_batches_tracked"]
            model_state_dict["6.convs.4.bn1.running_mean"]                 = state_dict["layer3.4.bn1.running_mean"]
            model_state_dict["6.convs.4.bn1.running_var"]                  = state_dict["layer3.4.bn1.running_var"]
            model_state_dict["6.convs.4.bn1.weight"]                       = state_dict["layer3.4.bn1.weight"]
            model_state_dict["6.convs.4.bn2.bias"]                         = state_dict["layer3.4.bn2.bias"]
            model_state_dict["6.convs.4.bn2.num_batches_tracked"]          = state_dict["layer3.4.bn2.num_batches_tracked"]
            model_state_dict["6.convs.4.bn2.running_mean"]                 = state_dict["layer3.4.bn2.running_mean"]
            model_state_dict["6.convs.4.bn2.running_var"]                  = state_dict["layer3.4.bn2.running_var"]
            model_state_dict["6.convs.4.bn2.weight"]                       = state_dict["layer3.4.bn2.weight"]
            model_state_dict["6.convs.4.bn3.bias"]                         = state_dict["layer3.4.bn3.bias"]
            model_state_dict["6.convs.4.bn3.num_batches_tracked"]          = state_dict["layer3.4.bn3.num_batches_tracked"]
            model_state_dict["6.convs.4.bn3.running_mean"]                 = state_dict["layer3.4.bn3.running_mean"]
            model_state_dict["6.convs.4.bn3.running_var"]                  = state_dict["layer3.4.bn3.running_var"]
            model_state_dict["6.convs.4.bn3.weight"]                       = state_dict["layer3.4.bn3.weight"]
            model_state_dict["6.convs.4.conv1.weight"]                     = state_dict["layer3.4.conv1.weight"]
            model_state_dict["6.convs.4.conv2.weight"]                     = state_dict["layer3.4.conv2.weight"]
            model_state_dict["6.convs.4.conv3.weight"]                     = state_dict["layer3.4.conv3.weight"]
            model_state_dict["6.convs.5.bn1.bias"]                         = state_dict["layer3.5.bn1.bias"]
            model_state_dict["6.convs.5.bn1.num_batches_tracked"]          = state_dict["layer3.5.bn1.num_batches_tracked"]
            model_state_dict["6.convs.5.bn1.running_mean"]                 = state_dict["layer3.5.bn1.running_mean"]
            model_state_dict["6.convs.5.bn1.running_var"]                  = state_dict["layer3.5.bn1.running_var"]
            model_state_dict["6.convs.5.bn1.weight"]                       = state_dict["layer3.5.bn1.weight"]
            model_state_dict["6.convs.5.bn2.bias"]                         = state_dict["layer3.5.bn2.bias"]
            model_state_dict["6.convs.5.bn2.num_batches_tracked"]          = state_dict["layer3.5.bn2.num_batches_tracked"]
            model_state_dict["6.convs.5.bn2.running_mean"]                 = state_dict["layer3.5.bn2.running_mean"]
            model_state_dict["6.convs.5.bn2.running_var"]                  = state_dict["layer3.5.bn2.running_var"]
            model_state_dict["6.convs.5.bn2.weight"]                       = state_dict["layer3.5.bn2.weight"]
            model_state_dict["6.convs.5.bn3.bias"]                         = state_dict["layer3.5.bn3.bias"]
            model_state_dict["6.convs.5.bn3.num_batches_tracked"]          = state_dict["layer3.5.bn3.num_batches_tracked"]
            model_state_dict["6.convs.5.bn3.running_mean"]                 = state_dict["layer3.5.bn3.running_mean"]
            model_state_dict["6.convs.5.bn3.running_var"]                  = state_dict["layer3.5.bn3.running_var"]
            model_state_dict["6.convs.5.bn3.weight"]                       = state_dict["layer3.5.bn3.weight"]
            model_state_dict["6.convs.5.conv1.weight"]                     = state_dict["layer3.5.conv1.weight"]
            model_state_dict["6.convs.5.conv2.weight"]                     = state_dict["layer3.5.conv2.weight"]
            model_state_dict["6.convs.5.conv3.weight"]                     = state_dict["layer3.5.conv3.weight"]
            model_state_dict["7.convs.0.bn1.bias"]                         = state_dict["layer4.0.bn1.bias"]
            model_state_dict["7.convs.0.bn1.num_batches_tracked"]          = state_dict["layer4.0.bn1.num_batches_tracked"]
            model_state_dict["7.convs.0.bn1.running_mean"]                 = state_dict["layer4.0.bn1.running_mean"]
            model_state_dict["7.convs.0.bn1.running_var"]                  = state_dict["layer4.0.bn1.running_var"]
            model_state_dict["7.convs.0.bn1.weight"]                       = state_dict["layer4.0.bn1.weight"]
            model_state_dict["7.convs.0.bn2.bias"]                         = state_dict["layer4.0.bn2.bias"]
            model_state_dict["7.convs.0.bn2.num_batches_tracked"]          = state_dict["layer4.0.bn2.num_batches_tracked"]
            model_state_dict["7.convs.0.bn2.running_mean"]                 = state_dict["layer4.0.bn2.running_mean"]
            model_state_dict["7.convs.0.bn2.running_var"]                  = state_dict["layer4.0.bn2.running_var"]
            model_state_dict["7.convs.0.bn2.weight"]                       = state_dict["layer4.0.bn2.weight"]
            model_state_dict["7.convs.0.bn3.bias"]                         = state_dict["layer4.0.bn3.bias"]
            model_state_dict["7.convs.0.bn3.num_batches_tracked"]          = state_dict["layer4.0.bn3.num_batches_tracked"]
            model_state_dict["7.convs.0.bn3.running_mean"]                 = state_dict["layer4.0.bn3.running_mean"]
            model_state_dict["7.convs.0.bn3.running_var"]                  = state_dict["layer4.0.bn3.running_var"]
            model_state_dict["7.convs.0.bn3.weight"]                       = state_dict["layer4.0.bn3.weight"]
            model_state_dict["7.convs.0.conv1.weight"]                     = state_dict["layer4.0.conv1.weight"]
            model_state_dict["7.convs.0.conv2.weight"]                     = state_dict["layer4.0.conv2.weight"]
            model_state_dict["7.convs.0.conv3.weight"]                     = state_dict["layer4.0.conv3.weight"]
            model_state_dict["7.convs.0.downsample.0.weight"]              = state_dict["layer4.0.downsample.0.weight"]
            model_state_dict["7.convs.0.downsample.1.bias"]                = state_dict["layer4.0.downsample.1.bias"]
            model_state_dict["7.convs.0.downsample.1.num_batches_tracked"] = state_dict["layer4.0.downsample.1.num_batches_tracked"]
            model_state_dict["7.convs.0.downsample.1.running_mean"]        = state_dict["layer4.0.downsample.1.running_mean"]
            model_state_dict["7.convs.0.downsample.1.running_var"]         = state_dict["layer4.0.downsample.1.running_var"]
            model_state_dict["7.convs.0.downsample.1.weight"]              = state_dict["layer4.0.downsample.1.weight"]
            model_state_dict["7.convs.1.bn1.bias"]                         = state_dict["layer4.1.bn1.bias"]
            model_state_dict["7.convs.1.bn1.num_batches_tracked"]          = state_dict["layer4.1.bn1.num_batches_tracked"]
            model_state_dict["7.convs.1.bn1.running_mean"]                 = state_dict["layer4.1.bn1.running_mean"]
            model_state_dict["7.convs.1.bn1.running_var"]                  = state_dict["layer4.1.bn1.running_var"]
            model_state_dict["7.convs.1.bn1.weight"]                       = state_dict["layer4.1.bn1.weight"]
            model_state_dict["7.convs.1.bn2.bias"]                         = state_dict["layer4.1.bn2.bias"]
            model_state_dict["7.convs.1.bn2.num_batches_tracked"]          = state_dict["layer4.1.bn2.num_batches_tracked"]
            model_state_dict["7.convs.1.bn2.running_mean"]                 = state_dict["layer4.1.bn2.running_mean"]
            model_state_dict["7.convs.1.bn2.running_var"]                  = state_dict["layer4.1.bn2.running_var"]
            model_state_dict["7.convs.1.bn2.weight"]                       = state_dict["layer4.1.bn2.weight"]
            model_state_dict["7.convs.1.bn3.bias"]                         = state_dict["layer4.1.bn3.bias"]
            model_state_dict["7.convs.1.bn3.num_batches_tracked"]          = state_dict["layer4.1.bn3.num_batches_tracked"]
            model_state_dict["7.convs.1.bn3.running_mean"]                 = state_dict["layer4.1.bn3.running_mean"]
            model_state_dict["7.convs.1.bn3.running_var"]                  = state_dict["layer4.1.bn3.running_var"]
            model_state_dict["7.convs.1.bn3.weight"]                       = state_dict["layer4.1.bn3.weight"]
            model_state_dict["7.convs.1.conv1.weight"]                     = state_dict["layer4.1.conv1.weight"]
            model_state_dict["7.convs.1.conv2.weight"]                     = state_dict["layer4.1.conv2.weight"]
            model_state_dict["7.convs.1.conv3.weight"]                     = state_dict["layer4.1.conv3.weight"]
            model_state_dict["7.convs.2.bn1.bias"]                         = state_dict["layer4.2.bn1.bias"]
            model_state_dict["7.convs.2.bn1.num_batches_tracked"]          = state_dict["layer4.2.bn1.num_batches_tracked"]
            model_state_dict["7.convs.2.bn1.running_mean"]                 = state_dict["layer4.2.bn1.running_mean"]
            model_state_dict["7.convs.2.bn1.running_var"]                  = state_dict["layer4.2.bn1.running_var"]
            model_state_dict["7.convs.2.bn1.weight"]                       = state_dict["layer4.2.bn1.weight"]
            model_state_dict["7.convs.2.bn2.bias"]                         = state_dict["layer4.2.bn2.bias"]
            model_state_dict["7.convs.2.bn2.num_batches_tracked"]          = state_dict["layer4.2.bn2.num_batches_tracked"]
            model_state_dict["7.convs.2.bn2.running_mean"]                 = state_dict["layer4.2.bn2.running_mean"]
            model_state_dict["7.convs.2.bn2.running_var"]                  = state_dict["layer4.2.bn2.running_var"]
            model_state_dict["7.convs.2.bn2.weight"]                       = state_dict["layer4.2.bn2.weight"]
            model_state_dict["7.convs.2.bn3.bias"]                         = state_dict["layer4.2.bn3.bias"]
            model_state_dict["7.convs.2.bn3.num_batches_tracked"]          = state_dict["layer4.2.bn3.num_batches_tracked"]
            model_state_dict["7.convs.2.bn3.running_mean"]                 = state_dict["layer4.2.bn3.running_mean"]
            model_state_dict["7.convs.2.bn3.running_var"]                  = state_dict["layer4.2.bn3.running_var"]
            model_state_dict["7.convs.2.bn3.weight"]                       = state_dict["layer4.2.bn3.weight"]
            model_state_dict["7.convs.2.conv1.weight"]                     = state_dict["layer4.2.conv1.weight"]
            model_state_dict["7.convs.2.conv2.weight"]                     = state_dict["layer4.2.conv2.weight"]
            model_state_dict["7.convs.2.conv3.weight"]                     = state_dict["layer4.2.conv3.weight"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["9.linear.weight"] = state_dict["fc.weight"]
                model_state_dict["9.linear.bias"]   = state_dict["fc.bias"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()


@MODELS.register(name="wide_resnet101")
class Wide_ResNet101(ResNet):
    """
    
    Args:
        cfg (dict | Path_ | None): Model's layers configuration. It can be an
            external .yaml path or a dictionary. Defaults to None means you
            should define each layer manually in `self.parse_model()` method.
        root (Path_): The root directory of the model. Defaults to RUNS_DIR.
        name (str | None): Model's name. In case None is given, it will be
            `self.__class__.__name__`. Defaults to None.
        fullname (str | None): Model's fullname in the following format:
            {name}-{data_name}-{postfix}. In case None is given, it will be
            `self.name`. Defaults to None.
        channels (int): Input channel. Defaults to 3.
        num_classes (int | None): Number of classes for classification or
            detection tasks. Defaults to None.
        classlabels (ClassLabels | None): ClassLabels object that contains all
            labels in the dataset. Defaults to None.
        pretrained (Pretrained): Initialize weights from pretrained.
            - If True, use the original pretrained described by the author
              (usually, ImageNet or COCO). By default, it is the first element
              in the `model_zoo` dictionary.
            - If str and is a file/path, then load weights from saved file.
            - In each inherited model, `pretrained` can be a dictionary's
              key to get the corresponding local file or url of the weight.
        phase (ModelPhase_): Model's running phase. Defaults to training.
        loss (Losses_ | None): Loss function for training model.
            Defaults to None.
        metrics (Metrics_ | None): Metric(s) for validating and testing model.
            Defaults to None.
        optimizers (Optimizers_ | None): Optimizer(s) for training model.
            Defaults to None.
        debug (dict | Munch | None): Debug configs. Defaults to None.
        verbose (bool): Verbosity.
    """
    
    model_zoo = {
        "imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth",
            filename    = "wide_resnet101-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "wide_resnet101.yaml",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "wide_resnet",
        fullname   : str  | None         = "wide_resnet101",
        channels   : int                 = 3,
        num_classes: int  | None 		 = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = None,
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg = cfg or "wide_resnet101"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg
        
        super().__init__(
            cfg         = cfg,
            root        = root,
            name        = name,
            fullname    = fullname,
            channels    = channels,
            num_classes = num_classes,
            pretrained  = Wide_ResNet101.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def load_pretrained(self):
        """
        Load pretrained weights. It only loads the intersection layers of
        matching keys and shapes between current model and pretrained.
        """
        if is_dict(self.pretrained) and self.pretrained["name"] == "imagenet":
            state_dict = load_state_dict_from_path(
                model_dir=self.pretrained_dir, **self.pretrained
            )
            """
            for k in sorted(self.model.state_dict().keys()):
                print(f"model_state_dict[\"{k}\"] = ")
            for k in sorted(state_dict.keys()):
                print(f"state_dict[\"{k}\"]")
            """
            model_state_dict = self.model.state_dict()
            model_state_dict["0.weight"]                                   = state_dict["conv1.weight"]
            model_state_dict["1.weight"]                                   = state_dict["bn1.weight"]
            model_state_dict["1.bias"]                                     = state_dict["bn1.bias"]
            model_state_dict["1.running_mean"]                             = state_dict["bn1.running_mean"]
            model_state_dict["1.running_var"]                              = state_dict["bn1.running_var"]
            model_state_dict["4.convs.0.bn1.bias"]                         = state_dict["layer1.0.bn1.bias"]
            model_state_dict["4.convs.0.bn1.num_batches_tracked"]          = state_dict["layer1.0.bn1.num_batches_tracked"]
            model_state_dict["4.convs.0.bn1.running_mean"]                 = state_dict["layer1.0.bn1.running_mean"]
            model_state_dict["4.convs.0.bn1.running_var"]                  = state_dict["layer1.0.bn1.running_var"]
            model_state_dict["4.convs.0.bn1.weight"]                       = state_dict["layer1.0.bn1.weight"]
            model_state_dict["4.convs.0.bn2.bias"]                         = state_dict["layer1.0.bn2.bias"]
            model_state_dict["4.convs.0.bn2.num_batches_tracked"]          = state_dict["layer1.0.bn2.num_batches_tracked"]
            model_state_dict["4.convs.0.bn2.running_mean"]                 = state_dict["layer1.0.bn2.running_mean"]
            model_state_dict["4.convs.0.bn2.running_var"]                  = state_dict["layer1.0.bn2.running_var"]
            model_state_dict["4.convs.0.bn2.weight"]                       = state_dict["layer1.0.bn2.weight"]
            model_state_dict["4.convs.0.bn3.bias"]                         = state_dict["layer1.0.bn3.bias"]
            model_state_dict["4.convs.0.bn3.num_batches_tracked"]          = state_dict["layer1.0.bn3.num_batches_tracked"]
            model_state_dict["4.convs.0.bn3.running_mean"]                 = state_dict["layer1.0.bn3.running_mean"]
            model_state_dict["4.convs.0.bn3.running_var"]                  = state_dict["layer1.0.bn3.running_var"]
            model_state_dict["4.convs.0.bn3.weight"]                       = state_dict["layer1.0.bn3.weight"]
            model_state_dict["4.convs.0.conv1.weight"]                     = state_dict["layer1.0.conv1.weight"]
            model_state_dict["4.convs.0.conv2.weight"]                     = state_dict["layer1.0.conv2.weight"]
            model_state_dict["4.convs.0.conv3.weight"]                     = state_dict["layer1.0.conv3.weight"]
            model_state_dict["4.convs.0.downsample.0.weight"]              = state_dict["layer1.0.downsample.0.weight"]
            model_state_dict["4.convs.0.downsample.1.bias"]                = state_dict["layer1.0.downsample.1.bias"]
            model_state_dict["4.convs.0.downsample.1.num_batches_tracked"] = state_dict["layer1.0.downsample.1.num_batches_tracked"]
            model_state_dict["4.convs.0.downsample.1.running_mean"]        = state_dict["layer1.0.downsample.1.running_mean"]
            model_state_dict["4.convs.0.downsample.1.running_var"]         = state_dict["layer1.0.downsample.1.running_var"]
            model_state_dict["4.convs.0.downsample.1.weight"]              = state_dict["layer1.0.downsample.1.weight"]
            model_state_dict["4.convs.1.bn1.bias"]                         = state_dict["layer1.1.bn1.bias"]
            model_state_dict["4.convs.1.bn1.num_batches_tracked"]          = state_dict["layer1.1.bn1.num_batches_tracked"]
            model_state_dict["4.convs.1.bn1.running_mean"]                 = state_dict["layer1.1.bn1.running_mean"]
            model_state_dict["4.convs.1.bn1.running_var"]                  = state_dict["layer1.1.bn1.running_var"]
            model_state_dict["4.convs.1.bn1.weight"]                       = state_dict["layer1.1.bn1.weight"]
            model_state_dict["4.convs.1.bn2.bias"]                         = state_dict["layer1.1.bn2.bias"]
            model_state_dict["4.convs.1.bn2.num_batches_tracked"]          = state_dict["layer1.1.bn2.num_batches_tracked"]
            model_state_dict["4.convs.1.bn2.running_mean"]                 = state_dict["layer1.1.bn2.running_mean"]
            model_state_dict["4.convs.1.bn2.running_var"]                  = state_dict["layer1.1.bn2.running_var"]
            model_state_dict["4.convs.1.bn2.weight"]                       = state_dict["layer1.1.bn2.weight"]
            model_state_dict["4.convs.1.bn3.bias"]                         = state_dict["layer1.1.bn3.bias"]
            model_state_dict["4.convs.1.bn3.num_batches_tracked"]          = state_dict["layer1.1.bn3.num_batches_tracked"]
            model_state_dict["4.convs.1.bn3.running_mean"]                 = state_dict["layer1.1.bn3.running_mean"]
            model_state_dict["4.convs.1.bn3.running_var"]                  = state_dict["layer1.1.bn3.running_var"]
            model_state_dict["4.convs.1.bn3.weight"]                       = state_dict["layer1.1.bn3.weight"]
            model_state_dict["4.convs.1.conv1.weight"]                     = state_dict["layer1.1.conv1.weight"]
            model_state_dict["4.convs.1.conv2.weight"]                     = state_dict["layer1.1.conv2.weight"]
            model_state_dict["4.convs.1.conv3.weight"]                     = state_dict["layer1.1.conv3.weight"]
            model_state_dict["4.convs.2.bn1.bias"]                         = state_dict["layer1.2.bn1.bias"]
            model_state_dict["4.convs.2.bn1.num_batches_tracked"]          = state_dict["layer1.2.bn1.num_batches_tracked"]
            model_state_dict["4.convs.2.bn1.running_mean"]                 = state_dict["layer1.2.bn1.running_mean"]
            model_state_dict["4.convs.2.bn1.running_var"]                  = state_dict["layer1.2.bn1.running_var"]
            model_state_dict["4.convs.2.bn1.weight"]                       = state_dict["layer1.2.bn1.weight"]
            model_state_dict["4.convs.2.bn2.bias"]                         = state_dict["layer1.2.bn2.bias"]
            model_state_dict["4.convs.2.bn2.num_batches_tracked"]          = state_dict["layer1.2.bn2.num_batches_tracked"]
            model_state_dict["4.convs.2.bn2.running_mean"]                 = state_dict["layer1.2.bn2.running_mean"]
            model_state_dict["4.convs.2.bn2.running_var"]                  = state_dict["layer1.2.bn2.running_var"]
            model_state_dict["4.convs.2.bn2.weight"]                       = state_dict["layer1.2.bn2.weight"]
            model_state_dict["4.convs.2.bn3.bias"]                         = state_dict["layer1.2.bn3.bias"]
            model_state_dict["4.convs.2.bn3.num_batches_tracked"]          = state_dict["layer1.2.bn3.num_batches_tracked"]
            model_state_dict["4.convs.2.bn3.running_mean"]                 = state_dict["layer1.2.bn3.running_mean"]
            model_state_dict["4.convs.2.bn3.running_var"]                  = state_dict["layer1.2.bn3.running_var"]
            model_state_dict["4.convs.2.bn3.weight"]                       = state_dict["layer1.2.bn3.weight"]
            model_state_dict["4.convs.2.conv1.weight"]                     = state_dict["layer1.2.conv1.weight"]
            model_state_dict["4.convs.2.conv2.weight"]                     = state_dict["layer1.2.conv2.weight"]
            model_state_dict["4.convs.2.conv3.weight"]                     = state_dict["layer1.2.conv3.weight"]
            model_state_dict["5.convs.0.bn1.bias"]                         = state_dict["layer2.0.bn1.bias"]
            model_state_dict["5.convs.0.bn1.num_batches_tracked"]          = state_dict["layer2.0.bn1.num_batches_tracked"]
            model_state_dict["5.convs.0.bn1.running_mean"]                 = state_dict["layer2.0.bn1.running_mean"]
            model_state_dict["5.convs.0.bn1.running_var"]                  = state_dict["layer2.0.bn1.running_var"]
            model_state_dict["5.convs.0.bn1.weight"]                       = state_dict["layer2.0.bn1.weight"]
            model_state_dict["5.convs.0.bn2.bias"]                         = state_dict["layer2.0.bn2.bias"]
            model_state_dict["5.convs.0.bn2.num_batches_tracked"]          = state_dict["layer2.0.bn2.num_batches_tracked"]
            model_state_dict["5.convs.0.bn2.running_mean"]                 = state_dict["layer2.0.bn2.running_mean"]
            model_state_dict["5.convs.0.bn2.running_var"]                  = state_dict["layer2.0.bn2.running_var"]
            model_state_dict["5.convs.0.bn2.weight"]                       = state_dict["layer2.0.bn2.weight"]
            model_state_dict["5.convs.0.bn3.bias"]                         = state_dict["layer2.0.bn3.bias"]
            model_state_dict["5.convs.0.bn3.num_batches_tracked"]          = state_dict["layer2.0.bn3.num_batches_tracked"]
            model_state_dict["5.convs.0.bn3.running_mean"]                 = state_dict["layer2.0.bn3.running_mean"]
            model_state_dict["5.convs.0.bn3.running_var"]                  = state_dict["layer2.0.bn3.running_var"]
            model_state_dict["5.convs.0.bn3.weight"]                       = state_dict["layer2.0.bn3.weight"]
            model_state_dict["5.convs.0.conv1.weight"]                     = state_dict["layer2.0.conv1.weight"]
            model_state_dict["5.convs.0.conv2.weight"]                     = state_dict["layer2.0.conv2.weight"]
            model_state_dict["5.convs.0.conv3.weight"]                     = state_dict["layer2.0.conv3.weight"]
            model_state_dict["5.convs.0.downsample.0.weight"]              = state_dict["layer2.0.downsample.0.weight"]
            model_state_dict["5.convs.0.downsample.1.bias"]                = state_dict["layer2.0.downsample.1.bias"]
            model_state_dict["5.convs.0.downsample.1.num_batches_tracked"] = state_dict["layer2.0.downsample.1.num_batches_tracked"]
            model_state_dict["5.convs.0.downsample.1.running_mean"]        = state_dict["layer2.0.downsample.1.running_mean"]
            model_state_dict["5.convs.0.downsample.1.running_var"]         = state_dict["layer2.0.downsample.1.running_var"]
            model_state_dict["5.convs.0.downsample.1.weight"]              = state_dict["layer2.0.downsample.1.weight"]
            model_state_dict["5.convs.1.bn1.bias"]                         = state_dict["layer2.1.bn1.bias"]
            model_state_dict["5.convs.1.bn1.num_batches_tracked"]          = state_dict["layer2.1.bn1.num_batches_tracked"]
            model_state_dict["5.convs.1.bn1.running_mean"]                 = state_dict["layer2.1.bn1.running_mean"]
            model_state_dict["5.convs.1.bn1.running_var"]                  = state_dict["layer2.1.bn1.running_var"]
            model_state_dict["5.convs.1.bn1.weight"]                       = state_dict["layer2.1.bn1.weight"]
            model_state_dict["5.convs.1.bn2.bias"]                         = state_dict["layer2.1.bn2.bias"]
            model_state_dict["5.convs.1.bn2.num_batches_tracked"]          = state_dict["layer2.1.bn2.num_batches_tracked"]
            model_state_dict["5.convs.1.bn2.running_mean"]                 = state_dict["layer2.1.bn2.running_mean"]
            model_state_dict["5.convs.1.bn2.running_var"]                  = state_dict["layer2.1.bn2.running_var"]
            model_state_dict["5.convs.1.bn2.weight"]                       = state_dict["layer2.1.bn2.weight"]
            model_state_dict["5.convs.1.bn3.bias"]                         = state_dict["layer2.1.bn3.bias"]
            model_state_dict["5.convs.1.bn3.num_batches_tracked"]          = state_dict["layer2.1.bn3.num_batches_tracked"]
            model_state_dict["5.convs.1.bn3.running_mean"]                 = state_dict["layer2.1.bn3.running_mean"]
            model_state_dict["5.convs.1.bn3.running_var"]                  = state_dict["layer2.1.bn3.running_var"]
            model_state_dict["5.convs.1.bn3.weight"]                       = state_dict["layer2.1.bn3.weight"]
            model_state_dict["5.convs.1.conv1.weight"]                     = state_dict["layer2.1.conv1.weight"]
            model_state_dict["5.convs.1.conv2.weight"]                     = state_dict["layer2.1.conv2.weight"]
            model_state_dict["5.convs.1.conv3.weight"]                     = state_dict["layer2.1.conv3.weight"]
            model_state_dict["5.convs.2.bn1.bias"]                         = state_dict["layer2.2.bn1.bias"]
            model_state_dict["5.convs.2.bn1.num_batches_tracked"]          = state_dict["layer2.2.bn1.num_batches_tracked"]
            model_state_dict["5.convs.2.bn1.running_mean"]                 = state_dict["layer2.2.bn1.running_mean"]
            model_state_dict["5.convs.2.bn1.running_var"]                  = state_dict["layer2.2.bn1.running_var"]
            model_state_dict["5.convs.2.bn1.weight"]                       = state_dict["layer2.2.bn1.weight"]
            model_state_dict["5.convs.2.bn2.bias"]                         = state_dict["layer2.2.bn2.bias"]
            model_state_dict["5.convs.2.bn2.num_batches_tracked"]          = state_dict["layer2.2.bn2.num_batches_tracked"]
            model_state_dict["5.convs.2.bn2.running_mean"]                 = state_dict["layer2.2.bn2.running_mean"]
            model_state_dict["5.convs.2.bn2.running_var"]                  = state_dict["layer2.2.bn2.running_var"]
            model_state_dict["5.convs.2.bn2.weight"]                       = state_dict["layer2.2.bn2.weight"]
            model_state_dict["5.convs.2.bn3.bias"]                         = state_dict["layer2.2.bn3.bias"]
            model_state_dict["5.convs.2.bn3.num_batches_tracked"]          = state_dict["layer2.2.bn3.num_batches_tracked"]
            model_state_dict["5.convs.2.bn3.running_mean"]                 = state_dict["layer2.2.bn3.running_mean"]
            model_state_dict["5.convs.2.bn3.running_var"]                  = state_dict["layer2.2.bn3.running_var"]
            model_state_dict["5.convs.2.bn3.weight"]                       = state_dict["layer2.2.bn3.weight"]
            model_state_dict["5.convs.2.conv1.weight"]                     = state_dict["layer2.2.conv1.weight"]
            model_state_dict["5.convs.2.conv2.weight"]                     = state_dict["layer2.2.conv2.weight"]
            model_state_dict["5.convs.2.conv3.weight"]                     = state_dict["layer2.2.conv3.weight"]
            model_state_dict["5.convs.3.bn1.bias"]                         = state_dict["layer2.3.bn1.bias"]
            model_state_dict["5.convs.3.bn1.num_batches_tracked"]          = state_dict["layer2.3.bn1.num_batches_tracked"]
            model_state_dict["5.convs.3.bn1.running_mean"]                 = state_dict["layer2.3.bn1.running_mean"]
            model_state_dict["5.convs.3.bn1.running_var"]                  = state_dict["layer2.3.bn1.running_var"]
            model_state_dict["5.convs.3.bn1.weight"]                       = state_dict["layer2.3.bn1.weight"]
            model_state_dict["5.convs.3.bn2.bias"]                         = state_dict["layer2.3.bn2.bias"]
            model_state_dict["5.convs.3.bn2.num_batches_tracked"]          = state_dict["layer2.3.bn2.num_batches_tracked"]
            model_state_dict["5.convs.3.bn2.running_mean"]                 = state_dict["layer2.3.bn2.running_mean"]
            model_state_dict["5.convs.3.bn2.running_var"]                  = state_dict["layer2.3.bn2.running_var"]
            model_state_dict["5.convs.3.bn2.weight"]                       = state_dict["layer2.3.bn2.weight"]
            model_state_dict["5.convs.3.bn3.bias"]                         = state_dict["layer2.3.bn3.bias"]
            model_state_dict["5.convs.3.bn3.num_batches_tracked"]          = state_dict["layer2.3.bn3.num_batches_tracked"]
            model_state_dict["5.convs.3.bn3.running_mean"]                 = state_dict["layer2.3.bn3.running_mean"]
            model_state_dict["5.convs.3.bn3.running_var"]                  = state_dict["layer2.3.bn3.running_var"]
            model_state_dict["5.convs.3.bn3.weight"]                       = state_dict["layer2.3.bn3.weight"]
            model_state_dict["5.convs.3.conv1.weight"]                     = state_dict["layer2.3.conv1.weight"]
            model_state_dict["5.convs.3.conv2.weight"]                     = state_dict["layer2.3.conv2.weight"]
            model_state_dict["5.convs.3.conv3.weight"]                     = state_dict["layer2.3.conv3.weight"]
            model_state_dict["6.convs.0.bn1.bias"]                         = state_dict["layer3.0.bn1.bias"]
            model_state_dict["6.convs.0.bn1.num_batches_tracked"]          = state_dict["layer3.0.bn1.num_batches_tracked"]
            model_state_dict["6.convs.0.bn1.running_mean"]                 = state_dict["layer3.0.bn1.running_mean"]
            model_state_dict["6.convs.0.bn1.running_var"]                  = state_dict["layer3.0.bn1.running_var"]
            model_state_dict["6.convs.0.bn1.weight"]                       = state_dict["layer3.0.bn1.weight"]
            model_state_dict["6.convs.0.bn2.bias"]                         = state_dict["layer3.0.bn2.bias"]
            model_state_dict["6.convs.0.bn2.num_batches_tracked"]          = state_dict["layer3.0.bn2.num_batches_tracked"]
            model_state_dict["6.convs.0.bn2.running_mean"]                 = state_dict["layer3.0.bn2.running_mean"]
            model_state_dict["6.convs.0.bn2.running_var"]                  = state_dict["layer3.0.bn2.running_var"]
            model_state_dict["6.convs.0.bn2.weight"]                       = state_dict["layer3.0.bn2.weight"]
            model_state_dict["6.convs.0.bn3.bias"]                         = state_dict["layer3.0.bn3.bias"]
            model_state_dict["6.convs.0.bn3.num_batches_tracked"]          = state_dict["layer3.0.bn3.num_batches_tracked"]
            model_state_dict["6.convs.0.bn3.running_mean"]                 = state_dict["layer3.0.bn3.running_mean"]
            model_state_dict["6.convs.0.bn3.running_var"]                  = state_dict["layer3.0.bn3.running_var"]
            model_state_dict["6.convs.0.bn3.weight"]                       = state_dict["layer3.0.bn3.weight"]
            model_state_dict["6.convs.0.conv1.weight"]                     = state_dict["layer3.0.conv1.weight"]
            model_state_dict["6.convs.0.conv2.weight"]                     = state_dict["layer3.0.conv2.weight"]
            model_state_dict["6.convs.0.conv3.weight"]                     = state_dict["layer3.0.conv3.weight"]
            model_state_dict["6.convs.0.downsample.0.weight"]              = state_dict["layer3.0.downsample.0.weight"]
            model_state_dict["6.convs.0.downsample.1.bias"]                = state_dict["layer3.0.downsample.1.bias"]
            model_state_dict["6.convs.0.downsample.1.num_batches_tracked"] = state_dict["layer3.0.downsample.1.num_batches_tracked"]
            model_state_dict["6.convs.0.downsample.1.running_mean"]        = state_dict["layer3.0.downsample.1.running_mean"]
            model_state_dict["6.convs.0.downsample.1.running_var"]         = state_dict["layer3.0.downsample.1.running_var"]
            model_state_dict["6.convs.0.downsample.1.weight"]              = state_dict["layer3.0.downsample.1.weight"]
            model_state_dict["6.convs.1.bn1.bias"]                         = state_dict["layer3.1.bn1.bias"]
            model_state_dict["6.convs.1.bn1.num_batches_tracked"]          = state_dict["layer3.1.bn1.num_batches_tracked"]
            model_state_dict["6.convs.1.bn1.running_mean"]                 = state_dict["layer3.1.bn1.running_mean"]
            model_state_dict["6.convs.1.bn1.running_var"]                  = state_dict["layer3.1.bn1.running_var"]
            model_state_dict["6.convs.1.bn1.weight"]                       = state_dict["layer3.1.bn1.weight"]
            model_state_dict["6.convs.1.bn2.bias"]                         = state_dict["layer3.1.bn2.bias"]
            model_state_dict["6.convs.1.bn2.num_batches_tracked"]          = state_dict["layer3.1.bn2.num_batches_tracked"]
            model_state_dict["6.convs.1.bn2.running_mean"]                 = state_dict["layer3.1.bn2.running_mean"]
            model_state_dict["6.convs.1.bn2.running_var"]                  = state_dict["layer3.1.bn2.running_var"]
            model_state_dict["6.convs.1.bn2.weight"]                       = state_dict["layer3.1.bn2.weight"]
            model_state_dict["6.convs.1.bn3.bias"]                         = state_dict["layer3.1.bn3.bias"]
            model_state_dict["6.convs.1.bn3.num_batches_tracked"]          = state_dict["layer3.1.bn3.num_batches_tracked"]
            model_state_dict["6.convs.1.bn3.running_mean"]                 = state_dict["layer3.1.bn3.running_mean"]
            model_state_dict["6.convs.1.bn3.running_var"]                  = state_dict["layer3.1.bn3.running_var"]
            model_state_dict["6.convs.1.bn3.weight"]                       = state_dict["layer3.1.bn3.weight"]
            model_state_dict["6.convs.1.conv1.weight"]                     = state_dict["layer3.1.conv1.weight"]
            model_state_dict["6.convs.1.conv2.weight"]                     = state_dict["layer3.1.conv2.weight"]
            model_state_dict["6.convs.1.conv3.weight"]                     = state_dict["layer3.1.conv3.weight"]
            model_state_dict["6.convs.2.bn1.bias"]                         = state_dict["layer3.2.bn1.bias"]
            model_state_dict["6.convs.2.bn1.num_batches_tracked"]          = state_dict["layer3.2.bn1.num_batches_tracked"]
            model_state_dict["6.convs.2.bn1.running_mean"]                 = state_dict["layer3.2.bn1.running_mean"]
            model_state_dict["6.convs.2.bn1.running_var"]                  = state_dict["layer3.2.bn1.running_var"]
            model_state_dict["6.convs.2.bn1.weight"]                       = state_dict["layer3.2.bn1.weight"]
            model_state_dict["6.convs.2.bn2.bias"]                         = state_dict["layer3.2.bn2.bias"]
            model_state_dict["6.convs.2.bn2.num_batches_tracked"]          = state_dict["layer3.2.bn2.num_batches_tracked"]
            model_state_dict["6.convs.2.bn2.running_mean"]                 = state_dict["layer3.2.bn2.running_mean"]
            model_state_dict["6.convs.2.bn2.running_var"]                  = state_dict["layer3.2.bn2.running_var"]
            model_state_dict["6.convs.2.bn2.weight"]                       = state_dict["layer3.2.bn2.weight"]
            model_state_dict["6.convs.2.bn3.bias"]                         = state_dict["layer3.2.bn3.bias"]
            model_state_dict["6.convs.2.bn3.num_batches_tracked"]          = state_dict["layer3.2.bn3.num_batches_tracked"]
            model_state_dict["6.convs.2.bn3.running_mean"]                 = state_dict["layer3.2.bn3.running_mean"]
            model_state_dict["6.convs.2.bn3.running_var"]                  = state_dict["layer3.2.bn3.running_var"]
            model_state_dict["6.convs.2.bn3.weight"]                       = state_dict["layer3.2.bn3.weight"]
            model_state_dict["6.convs.2.conv1.weight"]                     = state_dict["layer3.2.conv1.weight"]
            model_state_dict["6.convs.2.conv2.weight"]                     = state_dict["layer3.2.conv2.weight"]
            model_state_dict["6.convs.2.conv3.weight"]                     = state_dict["layer3.2.conv3.weight"]
            model_state_dict["6.convs.3.bn1.bias"]                         = state_dict["layer3.3.bn1.bias"]
            model_state_dict["6.convs.3.bn1.num_batches_tracked"]          = state_dict["layer3.3.bn1.num_batches_tracked"]
            model_state_dict["6.convs.3.bn1.running_mean"]                 = state_dict["layer3.3.bn1.running_mean"]
            model_state_dict["6.convs.3.bn1.running_var"]                  = state_dict["layer3.3.bn1.running_var"]
            model_state_dict["6.convs.3.bn1.weight"]                       = state_dict["layer3.3.bn1.weight"]
            model_state_dict["6.convs.3.bn2.bias"]                         = state_dict["layer3.3.bn2.bias"]
            model_state_dict["6.convs.3.bn2.num_batches_tracked"]          = state_dict["layer3.3.bn2.num_batches_tracked"]
            model_state_dict["6.convs.3.bn2.running_mean"]                 = state_dict["layer3.3.bn2.running_mean"]
            model_state_dict["6.convs.3.bn2.running_var"]                  = state_dict["layer3.3.bn2.running_var"]
            model_state_dict["6.convs.3.bn2.weight"]                       = state_dict["layer3.3.bn2.weight"]
            model_state_dict["6.convs.3.bn3.bias"]                         = state_dict["layer3.3.bn3.bias"]
            model_state_dict["6.convs.3.bn3.num_batches_tracked"]          = state_dict["layer3.3.bn3.num_batches_tracked"]
            model_state_dict["6.convs.3.bn3.running_mean"]                 = state_dict["layer3.3.bn3.running_mean"]
            model_state_dict["6.convs.3.bn3.running_var"]                  = state_dict["layer3.3.bn3.running_var"]
            model_state_dict["6.convs.3.bn3.weight"]                       = state_dict["layer3.3.bn3.weight"]
            model_state_dict["6.convs.3.conv1.weight"]                     = state_dict["layer3.3.conv1.weight"]
            model_state_dict["6.convs.3.conv2.weight"]                     = state_dict["layer3.3.conv2.weight"]
            model_state_dict["6.convs.3.conv3.weight"]                     = state_dict["layer3.3.conv3.weight"]
            model_state_dict["6.convs.4.bn1.bias"]                         = state_dict["layer3.4.bn1.bias"]
            model_state_dict["6.convs.4.bn1.num_batches_tracked"]          = state_dict["layer3.4.bn1.num_batches_tracked"]
            model_state_dict["6.convs.4.bn1.running_mean"]                 = state_dict["layer3.4.bn1.running_mean"]
            model_state_dict["6.convs.4.bn1.running_var"]                  = state_dict["layer3.4.bn1.running_var"]
            model_state_dict["6.convs.4.bn1.weight"]                       = state_dict["layer3.4.bn1.weight"]
            model_state_dict["6.convs.4.bn2.bias"]                         = state_dict["layer3.4.bn2.bias"]
            model_state_dict["6.convs.4.bn2.num_batches_tracked"]          = state_dict["layer3.4.bn2.num_batches_tracked"]
            model_state_dict["6.convs.4.bn2.running_mean"]                 = state_dict["layer3.4.bn2.running_mean"]
            model_state_dict["6.convs.4.bn2.running_var"]                  = state_dict["layer3.4.bn2.running_var"]
            model_state_dict["6.convs.4.bn2.weight"]                       = state_dict["layer3.4.bn2.weight"]
            model_state_dict["6.convs.4.bn3.bias"]                         = state_dict["layer3.4.bn3.bias"]
            model_state_dict["6.convs.4.bn3.num_batches_tracked"]          = state_dict["layer3.4.bn3.num_batches_tracked"]
            model_state_dict["6.convs.4.bn3.running_mean"]                 = state_dict["layer3.4.bn3.running_mean"]
            model_state_dict["6.convs.4.bn3.running_var"]                  = state_dict["layer3.4.bn3.running_var"]
            model_state_dict["6.convs.4.bn3.weight"]                       = state_dict["layer3.4.bn3.weight"]
            model_state_dict["6.convs.4.conv1.weight"]                     = state_dict["layer3.4.conv1.weight"]
            model_state_dict["6.convs.4.conv2.weight"]                     = state_dict["layer3.4.conv2.weight"]
            model_state_dict["6.convs.4.conv3.weight"]                     = state_dict["layer3.4.conv3.weight"]
            model_state_dict["6.convs.5.bn1.bias"]                         = state_dict["layer3.5.bn1.bias"]
            model_state_dict["6.convs.5.bn1.num_batches_tracked"]          = state_dict["layer3.5.bn1.num_batches_tracked"]
            model_state_dict["6.convs.5.bn1.running_mean"]                 = state_dict["layer3.5.bn1.running_mean"]
            model_state_dict["6.convs.5.bn1.running_var"]                  = state_dict["layer3.5.bn1.running_var"]
            model_state_dict["6.convs.5.bn1.weight"]                       = state_dict["layer3.5.bn1.weight"]
            model_state_dict["6.convs.5.bn2.bias"]                         = state_dict["layer3.5.bn2.bias"]
            model_state_dict["6.convs.5.bn2.num_batches_tracked"]          = state_dict["layer3.5.bn2.num_batches_tracked"]
            model_state_dict["6.convs.5.bn2.running_mean"]                 = state_dict["layer3.5.bn2.running_mean"]
            model_state_dict["6.convs.5.bn2.running_var"]                  = state_dict["layer3.5.bn2.running_var"]
            model_state_dict["6.convs.5.bn2.weight"]                       = state_dict["layer3.5.bn2.weight"]
            model_state_dict["6.convs.5.bn3.bias"]                         = state_dict["layer3.5.bn3.bias"]
            model_state_dict["6.convs.5.bn3.num_batches_tracked"]          = state_dict["layer3.5.bn3.num_batches_tracked"]
            model_state_dict["6.convs.5.bn3.running_mean"]                 = state_dict["layer3.5.bn3.running_mean"]
            model_state_dict["6.convs.5.bn3.running_var"]                  = state_dict["layer3.5.bn3.running_var"]
            model_state_dict["6.convs.5.bn3.weight"]                       = state_dict["layer3.5.bn3.weight"]
            model_state_dict["6.convs.5.conv1.weight"]                     = state_dict["layer3.5.conv1.weight"]
            model_state_dict["6.convs.5.conv2.weight"]                     = state_dict["layer3.5.conv2.weight"]
            model_state_dict["6.convs.5.conv3.weight"]                     = state_dict["layer3.5.conv3.weight"]
            model_state_dict["7.convs.0.bn1.bias"]                         = state_dict["layer4.0.bn1.bias"]
            model_state_dict["7.convs.0.bn1.num_batches_tracked"]          = state_dict["layer4.0.bn1.num_batches_tracked"]
            model_state_dict["7.convs.0.bn1.running_mean"]                 = state_dict["layer4.0.bn1.running_mean"]
            model_state_dict["7.convs.0.bn1.running_var"]                  = state_dict["layer4.0.bn1.running_var"]
            model_state_dict["7.convs.0.bn1.weight"]                       = state_dict["layer4.0.bn1.weight"]
            model_state_dict["7.convs.0.bn2.bias"]                         = state_dict["layer4.0.bn2.bias"]
            model_state_dict["7.convs.0.bn2.num_batches_tracked"]          = state_dict["layer4.0.bn2.num_batches_tracked"]
            model_state_dict["7.convs.0.bn2.running_mean"]                 = state_dict["layer4.0.bn2.running_mean"]
            model_state_dict["7.convs.0.bn2.running_var"]                  = state_dict["layer4.0.bn2.running_var"]
            model_state_dict["7.convs.0.bn2.weight"]                       = state_dict["layer4.0.bn2.weight"]
            model_state_dict["7.convs.0.bn3.bias"]                         = state_dict["layer4.0.bn3.bias"]
            model_state_dict["7.convs.0.bn3.num_batches_tracked"]          = state_dict["layer4.0.bn3.num_batches_tracked"]
            model_state_dict["7.convs.0.bn3.running_mean"]                 = state_dict["layer4.0.bn3.running_mean"]
            model_state_dict["7.convs.0.bn3.running_var"]                  = state_dict["layer4.0.bn3.running_var"]
            model_state_dict["7.convs.0.bn3.weight"]                       = state_dict["layer4.0.bn3.weight"]
            model_state_dict["7.convs.0.conv1.weight"]                     = state_dict["layer4.0.conv1.weight"]
            model_state_dict["7.convs.0.conv2.weight"]                     = state_dict["layer4.0.conv2.weight"]
            model_state_dict["7.convs.0.conv3.weight"]                     = state_dict["layer4.0.conv3.weight"]
            model_state_dict["7.convs.0.downsample.0.weight"]              = state_dict["layer4.0.downsample.0.weight"]
            model_state_dict["7.convs.0.downsample.1.bias"]                = state_dict["layer4.0.downsample.1.bias"]
            model_state_dict["7.convs.0.downsample.1.num_batches_tracked"] = state_dict["layer4.0.downsample.1.num_batches_tracked"]
            model_state_dict["7.convs.0.downsample.1.running_mean"]        = state_dict["layer4.0.downsample.1.running_mean"]
            model_state_dict["7.convs.0.downsample.1.running_var"]         = state_dict["layer4.0.downsample.1.running_var"]
            model_state_dict["7.convs.0.downsample.1.weight"]              = state_dict["layer4.0.downsample.1.weight"]
            model_state_dict["7.convs.1.bn1.bias"]                         = state_dict["layer4.1.bn1.bias"]
            model_state_dict["7.convs.1.bn1.num_batches_tracked"]          = state_dict["layer4.1.bn1.num_batches_tracked"]
            model_state_dict["7.convs.1.bn1.running_mean"]                 = state_dict["layer4.1.bn1.running_mean"]
            model_state_dict["7.convs.1.bn1.running_var"]                  = state_dict["layer4.1.bn1.running_var"]
            model_state_dict["7.convs.1.bn1.weight"]                       = state_dict["layer4.1.bn1.weight"]
            model_state_dict["7.convs.1.bn2.bias"]                         = state_dict["layer4.1.bn2.bias"]
            model_state_dict["7.convs.1.bn2.num_batches_tracked"]          = state_dict["layer4.1.bn2.num_batches_tracked"]
            model_state_dict["7.convs.1.bn2.running_mean"]                 = state_dict["layer4.1.bn2.running_mean"]
            model_state_dict["7.convs.1.bn2.running_var"]                  = state_dict["layer4.1.bn2.running_var"]
            model_state_dict["7.convs.1.bn2.weight"]                       = state_dict["layer4.1.bn2.weight"]
            model_state_dict["7.convs.1.bn3.bias"]                         = state_dict["layer4.1.bn3.bias"]
            model_state_dict["7.convs.1.bn3.num_batches_tracked"]          = state_dict["layer4.1.bn3.num_batches_tracked"]
            model_state_dict["7.convs.1.bn3.running_mean"]                 = state_dict["layer4.1.bn3.running_mean"]
            model_state_dict["7.convs.1.bn3.running_var"]                  = state_dict["layer4.1.bn3.running_var"]
            model_state_dict["7.convs.1.bn3.weight"]                       = state_dict["layer4.1.bn3.weight"]
            model_state_dict["7.convs.1.conv1.weight"]                     = state_dict["layer4.1.conv1.weight"]
            model_state_dict["7.convs.1.conv2.weight"]                     = state_dict["layer4.1.conv2.weight"]
            model_state_dict["7.convs.1.conv3.weight"]                     = state_dict["layer4.1.conv3.weight"]
            model_state_dict["7.convs.2.bn1.bias"]                         = state_dict["layer4.2.bn1.bias"]
            model_state_dict["7.convs.2.bn1.num_batches_tracked"]          = state_dict["layer4.2.bn1.num_batches_tracked"]
            model_state_dict["7.convs.2.bn1.running_mean"]                 = state_dict["layer4.2.bn1.running_mean"]
            model_state_dict["7.convs.2.bn1.running_var"]                  = state_dict["layer4.2.bn1.running_var"]
            model_state_dict["7.convs.2.bn1.weight"]                       = state_dict["layer4.2.bn1.weight"]
            model_state_dict["7.convs.2.bn2.bias"]                         = state_dict["layer4.2.bn2.bias"]
            model_state_dict["7.convs.2.bn2.num_batches_tracked"]          = state_dict["layer4.2.bn2.num_batches_tracked"]
            model_state_dict["7.convs.2.bn2.running_mean"]                 = state_dict["layer4.2.bn2.running_mean"]
            model_state_dict["7.convs.2.bn2.running_var"]                  = state_dict["layer4.2.bn2.running_var"]
            model_state_dict["7.convs.2.bn2.weight"]                       = state_dict["layer4.2.bn2.weight"]
            model_state_dict["7.convs.2.bn3.bias"]                         = state_dict["layer4.2.bn3.bias"]
            model_state_dict["7.convs.2.bn3.num_batches_tracked"]          = state_dict["layer4.2.bn3.num_batches_tracked"]
            model_state_dict["7.convs.2.bn3.running_mean"]                 = state_dict["layer4.2.bn3.running_mean"]
            model_state_dict["7.convs.2.bn3.running_var"]                  = state_dict["layer4.2.bn3.running_var"]
            model_state_dict["7.convs.2.bn3.weight"]                       = state_dict["layer4.2.bn3.weight"]
            model_state_dict["7.convs.2.conv1.weight"]                     = state_dict["layer4.2.conv1.weight"]
            model_state_dict["7.convs.2.conv2.weight"]                     = state_dict["layer4.2.conv2.weight"]
            model_state_dict["7.convs.2.conv3.weight"]                     = state_dict["layer4.2.conv3.weight"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["9.linear.weight"] = state_dict["fc.weight"]
                model_state_dict["9.linear.bias"]   = state_dict["fc.bias"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()
