#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from one.nn import *

CURRENT_DIR = Path(__file__).resolve().parent
CFG_DIR     = CURRENT_DIR / "cfg"


# H1: - Model ------------------------------------------------------------------

cfgs = {
    "resnet18": {
        "zero_init_residual": False,
        "channels": 3,
        "backbone": [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [64, 7, 2, 3, 1, 1, False]],                                        # 0
            [-1,     1,      BatchNorm2d,       []],                                                                # 1
            [-1,     1,      ReLU,              [True]],                                                            # 2
            [-1,     1,      MaxPool2d,         [3, 2, 1]],                                                         # 3
            [-1,     1,      ResNetBlock,       [ResNetBasicBlock, 2, 64,  64,  1, 1, 1, 64, False, BatchNorm2d]],  # 4
            [-1,     1,      ResNetBlock,       [ResNetBasicBlock, 2, 64,  128, 2, 1, 1, 64, False, BatchNorm2d]],  # 5
            [-1,     1,      ResNetBlock,       [ResNetBasicBlock, 2, 128, 256, 2, 1, 1, 64, False, BatchNorm2d]],  # 6
            [-1,     1,      ResNetBlock,       [ResNetBasicBlock, 2, 256, 512, 2, 1, 1, 64, False, BatchNorm2d]],  # 7
            [-1,     1,      AdaptiveAvgPool2d, [1]],                                                               # 8
        ],
        "head": [
            [-1,     1,      LinearClassifier,  [512]],                                                             # 9
        ]
    },
    "resnet34": {
        "zero_init_residual": False,
        "channels": 3,
        "backbone": [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [64, 7, 2, 3, 1, 1, False]],                                        # 0
            [-1,     1,      BatchNorm2d,       []],                                                                # 1
            [-1,     1,      ReLU,              [True]],                                                            # 2
            [-1,     1,      MaxPool2d,         [3, 2, 1]],                                                         # 3
            [-1,     1,      ResNetBlock,       [ResNetBasicBlock, 3, 64,  64,  1, 1, 1, 64, False, BatchNorm2d]],  # 4
            [-1,     1,      ResNetBlock,       [ResNetBasicBlock, 4, 64,  128, 2, 1, 1, 64, False, BatchNorm2d]],  # 5
            [-1,     1,      ResNetBlock,       [ResNetBasicBlock, 6, 128, 256, 2, 1, 1, 64, False, BatchNorm2d]],  # 6
            [-1,     1,      ResNetBlock,       [ResNetBasicBlock, 3, 256, 512, 2, 1, 1, 64, False, BatchNorm2d]],  # 7
            [-1,     1,      AdaptiveAvgPool2d, [1]],                                                               # 8
        ],
        "head": [
            [-1,     1,      LinearClassifier,  [512]],                                                             # 9
        ]
    },
    "resnet50": {
        "zero_init_residual": False,
        "channels": 3,
        "backbone": [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [64, 7, 2, 3, 1, 1, False]],                                         # 0
            [-1,     1,      BatchNorm2d,       []],                                                                 # 1
            [-1,     1,      ReLU,              [True]],                                                             # 2
            [-1,     1,      MaxPool2d,         [3, 2, 1]],                                                          # 3
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 3, 64,   64,  1, 1, 1, 64, False, BatchNorm2d]],  # 4
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 4, 256,  128, 2, 1, 1, 64, False, BatchNorm2d]],  # 5
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 6, 512,  256, 2, 1, 1, 64, False, BatchNorm2d]],  # 6
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 3, 1024, 512, 2, 1, 1, 64, False, BatchNorm2d]],  # 7
            [-1,     1,      AdaptiveAvgPool2d, [1]],                                                                # 8
        ],
        "head": [
            [-1,     1,      LinearClassifier,  [2048]],                                                             # 9
        ]
    },
    "resnet101": {
        "zero_init_residual": False,
        "channels": 3,
        "backbone": [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [64, 7, 2, 3, 1, 1, False]],                                          # 0
            [-1,     1,      BatchNorm2d,       []],                                                                  # 1
            [-1,     1,      ReLU,              [True]],                                                              # 2
            [-1,     1,      MaxPool2d,         [3, 2, 1]],                                                           # 3
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 3,  64,   64,  1, 1, 1, 64, False, BatchNorm2d]],  # 4
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 4,  256,  128, 2, 1, 1, 64, False, BatchNorm2d]],  # 5
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 23, 512,  256, 2, 1, 1, 64, False, BatchNorm2d]],  # 6
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 3,  1024, 512, 2, 1, 1, 64, False, BatchNorm2d]],  # 7
            [-1,     1,      AdaptiveAvgPool2d, [1]],                                                                 # 8
        ],
        "head": [
            [-1,     1,      LinearClassifier,  [2048]],                                                              # 9
        ]
    },
    "resnet152": {
        "zero_init_residual": False,
        "channels": 3,
        "backbone": [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [64, 7, 2, 3, 1, 1, False]],                                          # 0
            [-1,     1,      BatchNorm2d,       []],                                                                  # 1
            [-1,     1,      ReLU,              [True]],                                                              # 2
            [-1,     1,      MaxPool2d,         [3, 2, 1]],                                                           # 3
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 3,  64,   64,  1, 1, 1, 64, False, BatchNorm2d]],  # 4
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 8,  256,  128, 2, 1, 1, 64, False, BatchNorm2d]],  # 5
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 36, 512,  256, 2, 1, 1, 64, False, BatchNorm2d]],  # 6
            [-1,     1,      ResNetBlock,       [ResNetBottleneck, 3,  1024, 512, 2, 1, 1, 64, False, BatchNorm2d]],  # 7
            [-1,     1,      AdaptiveAvgPool2d, [1]],                                                                 # 8
        ],
        "head": [
            [-1,     1,      LinearClassifier,  [2048]],                                                              # 9
        ]
    },
}


@MODELS.register(name="resnet")
class ResNet(ImageClassificationModel):
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
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "resnet18",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "resnet",
        fullname   : str  | None         = "resnet",
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
        cfg = cfg or "resnet18"
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
            pretrained  = pretrained,
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )

    def init_weights(self, m: Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                nn.init.kaiming_normal_(m.conv.weight, mode="fan_out", nonlinearity="relu")
            else:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif classname.find("BatchNorm") != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif classname.find("GroupNorm") != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        zero_init_residual = self.cfg["zero_init_residual"]
        if zero_init_residual:
            if isinstance(m, ResNetBottleneck) and m.bn3.weight is not None:
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, ResNetBottleneck) and m.bn2.weight is not None:
                nn.init.constant_(m.bn2.weight, 0)
        

@MODELS.register(name="resnet18")
class ResNet18(ResNet):
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
            path        = "https://download.pytorch.org/models/resnet18-f37072fd.pth",
            filename    = "resnet18-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "resnet18.yaml",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "resnet",
        fullname   : str  | None         = "resnet18",
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
        cfg = cfg or "resnet18"
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
            pretrained  = ResNet18.init_pretrained(pretrained),
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
            model_state_dict["0.weight"]                            = state_dict["conv1.weight"]
            model_state_dict["1.weight"]                            = state_dict["bn1.weight"]
            model_state_dict["1.bias"]                              = state_dict["bn1.bias"]
            model_state_dict["1.running_mean"]                      = state_dict["bn1.running_mean"]
            model_state_dict["1.running_var"]                       = state_dict["bn1.running_var"]
            model_state_dict["4.convs.0.conv1.weight"]              = state_dict["layer1.0.conv1.weight"]
            model_state_dict["4.convs.0.bn1.weight"]                = state_dict["layer1.0.bn1.weight"]
            model_state_dict["4.convs.0.bn1.bias"]                  = state_dict["layer1.0.bn1.bias"]
            model_state_dict["4.convs.0.bn1.running_mean"]          = state_dict["layer1.0.bn1.running_mean"]
            model_state_dict["4.convs.0.bn1.running_var"]           = state_dict["layer1.0.bn1.running_var"]
            model_state_dict["4.convs.0.conv2.weight"]              = state_dict["layer1.0.conv2.weight"]
            model_state_dict["4.convs.0.bn2.weight"]                = state_dict["layer1.0.bn2.weight"]
            model_state_dict["4.convs.0.bn2.bias"]                  = state_dict["layer1.0.bn2.bias"]
            model_state_dict["4.convs.0.bn2.running_mean"]          = state_dict["layer1.0.bn2.running_mean"]
            model_state_dict["4.convs.0.bn2.running_var"]           = state_dict["layer1.0.bn2.running_var"]
            model_state_dict["4.convs.1.conv1.weight"]              = state_dict["layer1.1.conv1.weight"]
            model_state_dict["4.convs.1.bn1.weight"]                = state_dict["layer1.1.bn1.weight"]
            model_state_dict["4.convs.1.bn1.bias"]                  = state_dict["layer1.1.bn1.bias"]
            model_state_dict["4.convs.1.bn1.running_mean"]          = state_dict["layer1.1.bn1.running_mean"]
            model_state_dict["4.convs.1.bn1.running_var"]           = state_dict["layer1.1.bn1.running_var"]
            model_state_dict["4.convs.1.conv2.weight"]              = state_dict["layer1.1.conv2.weight"]
            model_state_dict["4.convs.1.bn2.weight"]                = state_dict["layer1.1.bn2.weight"]
            model_state_dict["4.convs.1.bn2.bias"]                  = state_dict["layer1.1.bn2.bias"]
            model_state_dict["4.convs.1.bn2.running_mean"]          = state_dict["layer1.1.bn2.running_mean"]
            model_state_dict["4.convs.1.bn2.running_var"]           = state_dict["layer1.1.bn2.running_var"]
            model_state_dict["5.convs.0.conv1.weight"]              = state_dict["layer2.0.conv1.weight"]
            model_state_dict["5.convs.0.bn1.weight"]                = state_dict["layer2.0.bn1.weight"]
            model_state_dict["5.convs.0.bn1.bias"]                  = state_dict["layer2.0.bn1.bias"]
            model_state_dict["5.convs.0.bn1.running_mean"]          = state_dict["layer2.0.bn1.running_mean"]
            model_state_dict["5.convs.0.bn1.running_var"]           = state_dict["layer2.0.bn1.running_var"]
            model_state_dict["5.convs.0.conv2.weight"]              = state_dict["layer2.0.conv2.weight"]
            model_state_dict["5.convs.0.bn2.weight"]                = state_dict["layer2.0.bn2.weight"]
            model_state_dict["5.convs.0.bn2.bias"]                  = state_dict["layer2.0.bn2.bias"]
            model_state_dict["5.convs.0.bn2.running_mean"]          = state_dict["layer2.0.bn2.running_mean"]
            model_state_dict["5.convs.0.bn2.running_var"]           = state_dict["layer2.0.bn2.running_var"]
            model_state_dict["5.convs.0.downsample.0.weight"]       = state_dict["layer2.0.downsample.0.weight"]
            model_state_dict["5.convs.0.downsample.1.weight"]       = state_dict["layer2.0.downsample.1.weight"]
            model_state_dict["5.convs.0.downsample.1.bias"]         = state_dict["layer2.0.downsample.1.bias"]
            model_state_dict["5.convs.0.downsample.1.running_mean"] = state_dict["layer2.0.downsample.1.running_mean"]
            model_state_dict["5.convs.0.downsample.1.running_var"]  = state_dict["layer2.0.downsample.1.running_var"]
            model_state_dict["5.convs.1.conv1.weight"]              = state_dict["layer2.1.conv1.weight"]
            model_state_dict["5.convs.1.bn1.weight"]                = state_dict["layer2.1.bn1.weight"]
            model_state_dict["5.convs.1.bn1.bias"]                  = state_dict["layer2.1.bn1.bias"]
            model_state_dict["5.convs.1.bn1.running_mean"]          = state_dict["layer2.1.bn1.running_mean"]
            model_state_dict["5.convs.1.bn1.running_var"]           = state_dict["layer2.1.bn1.running_var"]
            model_state_dict["5.convs.1.conv2.weight"]              = state_dict["layer2.1.conv2.weight"]
            model_state_dict["5.convs.1.bn2.weight"]                = state_dict["layer2.1.bn2.weight"]
            model_state_dict["5.convs.1.bn2.bias"]                  = state_dict["layer2.1.bn2.bias"]
            model_state_dict["5.convs.1.bn2.running_mean"]          = state_dict["layer2.1.bn2.running_mean"]
            model_state_dict["5.convs.1.bn2.running_var"]           = state_dict["layer2.1.bn2.running_var"]
            model_state_dict["6.convs.0.conv1.weight"]              = state_dict["layer3.0.conv1.weight"]
            model_state_dict["6.convs.0.bn1.weight"]                = state_dict["layer3.0.bn1.weight"]
            model_state_dict["6.convs.0.bn1.bias"]                  = state_dict["layer3.0.bn1.bias"]
            model_state_dict["6.convs.0.bn1.running_mean"]          = state_dict["layer3.0.bn1.running_mean"]
            model_state_dict["6.convs.0.bn1.running_var"]           = state_dict["layer3.0.bn1.running_var"]
            model_state_dict["6.convs.0.conv2.weight"]              = state_dict["layer3.0.conv2.weight"]
            model_state_dict["6.convs.0.bn2.weight"]                = state_dict["layer3.0.bn2.weight"]
            model_state_dict["6.convs.0.bn2.bias"]                  = state_dict["layer3.0.bn2.bias"]
            model_state_dict["6.convs.0.bn2.running_mean"]          = state_dict["layer3.0.bn2.running_mean"]
            model_state_dict["6.convs.0.bn2.running_var"]           = state_dict["layer3.0.bn2.running_var"]
            model_state_dict["6.convs.0.downsample.0.weight"]       = state_dict["layer3.0.downsample.0.weight"]
            model_state_dict["6.convs.0.downsample.1.weight"]       = state_dict["layer3.0.downsample.1.weight"]
            model_state_dict["6.convs.0.downsample.1.bias"]         = state_dict["layer3.0.downsample.1.bias"]
            model_state_dict["6.convs.0.downsample.1.running_mean"] = state_dict["layer3.0.downsample.1.running_mean"]
            model_state_dict["6.convs.0.downsample.1.running_var"]  = state_dict["layer3.0.downsample.1.running_var"]
            model_state_dict["6.convs.1.conv1.weight"]              = state_dict["layer3.1.conv1.weight"]
            model_state_dict["6.convs.1.bn1.weight"]                = state_dict["layer3.1.bn1.weight"]
            model_state_dict["6.convs.1.bn1.bias"]                  = state_dict["layer3.1.bn1.bias"]
            model_state_dict["6.convs.1.bn1.running_mean"]          = state_dict["layer3.1.bn1.running_mean"]
            model_state_dict["6.convs.1.bn1.running_var"]           = state_dict["layer3.1.bn1.running_var"]
            model_state_dict["6.convs.1.conv2.weight"]              = state_dict["layer3.1.conv2.weight"]
            model_state_dict["6.convs.1.bn2.weight"]                = state_dict["layer3.1.bn2.weight"]
            model_state_dict["6.convs.1.bn2.bias"]                  = state_dict["layer3.1.bn2.bias"]
            model_state_dict["6.convs.1.bn2.running_mean"]          = state_dict["layer3.1.bn2.running_mean"]
            model_state_dict["6.convs.1.bn2.running_var"]           = state_dict["layer3.1.bn2.running_var"]
            model_state_dict["7.convs.0.conv1.weight"]              = state_dict["layer4.0.conv1.weight"]
            model_state_dict["7.convs.0.bn1.weight"]                = state_dict["layer4.0.bn1.weight"]
            model_state_dict["7.convs.0.bn1.bias"]                  = state_dict["layer4.0.bn1.bias"]
            model_state_dict["7.convs.0.bn1.running_mean"]          = state_dict["layer4.0.bn1.running_mean"]
            model_state_dict["7.convs.0.bn1.running_var"]           = state_dict["layer4.0.bn1.running_var"]
            model_state_dict["7.convs.0.conv2.weight"]              = state_dict["layer4.0.conv2.weight"]
            model_state_dict["7.convs.0.bn2.weight"]                = state_dict["layer4.0.bn2.weight"]
            model_state_dict["7.convs.0.bn2.bias"]                  = state_dict["layer4.0.bn2.bias"]
            model_state_dict["7.convs.0.bn2.running_mean"]          = state_dict["layer4.0.bn2.running_mean"]
            model_state_dict["7.convs.0.bn2.running_var"]           = state_dict["layer4.0.bn2.running_var"]
            model_state_dict["7.convs.0.downsample.0.weight"]       = state_dict["layer4.0.downsample.0.weight"]
            model_state_dict["7.convs.0.downsample.1.weight"]       = state_dict["layer4.0.downsample.1.weight"]
            model_state_dict["7.convs.0.downsample.1.bias"]         = state_dict["layer4.0.downsample.1.bias"]
            model_state_dict["7.convs.0.downsample.1.running_mean"] = state_dict["layer4.0.downsample.1.running_mean"]
            model_state_dict["7.convs.0.downsample.1.running_var"]  = state_dict["layer4.0.downsample.1.running_var"]
            model_state_dict["7.convs.1.conv1.weight"]              = state_dict["layer4.1.conv1.weight"]
            model_state_dict["7.convs.1.bn1.weight"]                = state_dict["layer4.1.bn1.weight"]
            model_state_dict["7.convs.1.bn1.bias"]                  = state_dict["layer4.1.bn1.bias"]
            model_state_dict["7.convs.1.bn1.running_mean"]          = state_dict["layer4.1.bn1.running_mean"]
            model_state_dict["7.convs.1.bn1.running_var"]           = state_dict["layer4.1.bn1.running_var"]
            model_state_dict["7.convs.1.conv2.weight"]              = state_dict["layer4.1.conv2.weight"]
            model_state_dict["7.convs.1.bn2.weight"]                = state_dict["layer4.1.bn2.weight"]
            model_state_dict["7.convs.1.bn2.bias"]                  = state_dict["layer4.1.bn2.bias"]
            model_state_dict["7.convs.1.bn2.running_mean"]          = state_dict["layer4.1.bn2.running_mean"]
            model_state_dict["7.convs.1.bn2.running_var"]           = state_dict["layer4.1.bn2.running_var"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["9.linear.weight"] = state_dict["fc.weight"]
                model_state_dict["9.linear.bias"]   = state_dict["fc.bias"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()


@MODELS.register(name="resnet34")
class ResNet34(ResNet):
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
            path        = "https://download.pytorch.org/models/resnet34-b627a593.pth",
            filename    = "resnet34-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "resnet34.yaml",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "resnet",
        fullname   : str  | None         = "resnet34",
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
        cfg = cfg or "resnet34"
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
            pretrained  = ResNet34.init_pretrained(pretrained),
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
            model_state_dict["0.weight"]                            = state_dict["conv1.weight"]
            model_state_dict["1.weight"]                            = state_dict["bn1.weight"]
            model_state_dict["1.bias"]                              = state_dict["bn1.bias"]
            model_state_dict["1.running_mean"]                      = state_dict["bn1.running_mean"]
            model_state_dict["1.running_var"]                       = state_dict["bn1.running_var"]
            model_state_dict["4.convs.0.conv1.weight"]              = state_dict["layer1.0.conv1.weight"]
            model_state_dict["4.convs.0.bn1.weight"]                = state_dict["layer1.0.bn1.weight"]
            model_state_dict["4.convs.0.bn1.bias"]                  = state_dict["layer1.0.bn1.bias"]
            model_state_dict["4.convs.0.bn1.running_mean"]          = state_dict["layer1.0.bn1.running_mean"]
            model_state_dict["4.convs.0.bn1.running_var"]           = state_dict["layer1.0.bn1.running_var"]
            model_state_dict["4.convs.0.conv2.weight"]              = state_dict["layer1.0.conv2.weight"]
            model_state_dict["4.convs.0.bn2.weight"]                = state_dict["layer1.0.bn2.weight"]
            model_state_dict["4.convs.0.bn2.bias"]                  = state_dict["layer1.0.bn2.bias"]
            model_state_dict["4.convs.0.bn2.running_mean"]          = state_dict["layer1.0.bn2.running_mean"]
            model_state_dict["4.convs.0.bn2.running_var"]           = state_dict["layer1.0.bn2.running_var"]
            model_state_dict["4.convs.1.conv1.weight"]              = state_dict["layer1.1.conv1.weight"]
            model_state_dict["4.convs.1.bn1.weight"]                = state_dict["layer1.1.bn1.weight"]
            model_state_dict["4.convs.1.bn1.bias"]                  = state_dict["layer1.1.bn1.bias"]
            model_state_dict["4.convs.1.bn1.running_mean"]          = state_dict["layer1.1.bn1.running_mean"]
            model_state_dict["4.convs.1.bn1.running_var"]           = state_dict["layer1.1.bn1.running_var"]
            model_state_dict["4.convs.1.conv2.weight"]              = state_dict["layer1.1.conv2.weight"]
            model_state_dict["4.convs.1.bn2.weight"]                = state_dict["layer1.1.bn2.weight"]
            model_state_dict["4.convs.1.bn2.bias"]                  = state_dict["layer1.1.bn2.bias"]
            model_state_dict["4.convs.1.bn2.running_mean"]          = state_dict["layer1.1.bn2.running_mean"]
            model_state_dict["4.convs.1.bn2.running_var"]           = state_dict["layer1.1.bn2.running_var"]
            model_state_dict["4.convs.2.conv1.weight"]              = state_dict["layer1.2.conv1.weight"]
            model_state_dict["4.convs.2.bn1.weight"]                = state_dict["layer1.2.bn1.weight"]
            model_state_dict["4.convs.2.bn1.bias"]                  = state_dict["layer1.2.bn1.bias"]
            model_state_dict["4.convs.2.bn1.running_mean"]          = state_dict["layer1.2.bn1.running_mean"]
            model_state_dict["4.convs.2.bn1.running_var"]           = state_dict["layer1.2.bn1.running_var"]
            model_state_dict["4.convs.2.conv2.weight"]              = state_dict["layer1.2.conv2.weight"]
            model_state_dict["4.convs.2.bn2.weight"]                = state_dict["layer1.2.bn2.weight"]
            model_state_dict["4.convs.2.bn2.bias"]                  = state_dict["layer1.2.bn2.bias"]
            model_state_dict["4.convs.2.bn2.running_mean"]          = state_dict["layer1.2.bn2.running_mean"]
            model_state_dict["4.convs.2.bn2.running_var"]           = state_dict["layer1.2.bn2.running_var"]
            model_state_dict["5.convs.0.conv1.weight"]              = state_dict["layer2.0.conv1.weight"]
            model_state_dict["5.convs.0.bn1.weight"]                = state_dict["layer2.0.bn1.weight"]
            model_state_dict["5.convs.0.bn1.bias"]                  = state_dict["layer2.0.bn1.bias"]
            model_state_dict["5.convs.0.bn1.running_mean"]          = state_dict["layer2.0.bn1.running_mean"]
            model_state_dict["5.convs.0.bn1.running_var"]           = state_dict["layer2.0.bn1.running_var"]
            model_state_dict["5.convs.0.conv2.weight"]              = state_dict["layer2.0.conv2.weight"]
            model_state_dict["5.convs.0.bn2.weight"]                = state_dict["layer2.0.bn2.weight"]
            model_state_dict["5.convs.0.bn2.bias"]                  = state_dict["layer2.0.bn2.bias"]
            model_state_dict["5.convs.0.bn2.running_mean"]          = state_dict["layer2.0.bn2.running_mean"]
            model_state_dict["5.convs.0.bn2.running_var"]           = state_dict["layer2.0.bn2.running_var"]
            model_state_dict["5.convs.0.downsample.0.weight"]       = state_dict["layer2.0.downsample.0.weight"]
            model_state_dict["5.convs.0.downsample.1.weight"]       = state_dict["layer2.0.downsample.1.weight"]
            model_state_dict["5.convs.0.downsample.1.bias"]         = state_dict["layer2.0.downsample.1.bias"]
            model_state_dict["5.convs.0.downsample.1.running_mean"] = state_dict["layer2.0.downsample.1.running_mean"]
            model_state_dict["5.convs.0.downsample.1.running_var"]  = state_dict["layer2.0.downsample.1.running_var"]
            model_state_dict["5.convs.1.conv1.weight"]              = state_dict["layer2.1.conv1.weight"]
            model_state_dict["5.convs.1.bn1.weight"]                = state_dict["layer2.1.bn1.weight"]
            model_state_dict["5.convs.1.bn1.bias"]                  = state_dict["layer2.1.bn1.bias"]
            model_state_dict["5.convs.1.bn1.running_mean"]          = state_dict["layer2.1.bn1.running_mean"]
            model_state_dict["5.convs.1.bn1.running_var"]           = state_dict["layer2.1.bn1.running_var"]
            model_state_dict["5.convs.1.conv2.weight"]              = state_dict["layer2.1.conv2.weight"]
            model_state_dict["5.convs.1.bn2.weight"]                = state_dict["layer2.1.bn2.weight"]
            model_state_dict["5.convs.1.bn2.bias"]                  = state_dict["layer2.1.bn2.bias"]
            model_state_dict["5.convs.1.bn2.running_mean"]          = state_dict["layer2.1.bn2.running_mean"]
            model_state_dict["5.convs.1.bn2.running_var"]           = state_dict["layer2.1.bn2.running_var"]
            model_state_dict["5.convs.2.conv1.weight"]              = state_dict["layer2.2.conv1.weight"]
            model_state_dict["5.convs.2.bn1.weight"]                = state_dict["layer2.2.bn1.weight"]
            model_state_dict["5.convs.2.bn1.bias"]                  = state_dict["layer2.2.bn1.bias"]
            model_state_dict["5.convs.2.bn1.running_mean"]          = state_dict["layer2.2.bn1.running_mean"]
            model_state_dict["5.convs.2.bn1.running_var"]           = state_dict["layer2.2.bn1.running_var"]
            model_state_dict["5.convs.2.conv2.weight"]              = state_dict["layer2.2.conv2.weight"]
            model_state_dict["5.convs.2.bn2.weight"]                = state_dict["layer2.2.bn2.weight"]
            model_state_dict["5.convs.2.bn2.bias"]                  = state_dict["layer2.2.bn2.bias"]
            model_state_dict["5.convs.2.bn2.running_mean"]          = state_dict["layer2.2.bn2.running_mean"]
            model_state_dict["5.convs.2.bn2.running_var"]           = state_dict["layer2.2.bn2.running_var"]
            model_state_dict["5.convs.3.conv1.weight"]              = state_dict["layer2.3.conv1.weight"]
            model_state_dict["5.convs.3.bn1.weight"]                = state_dict["layer2.3.bn1.weight"]
            model_state_dict["5.convs.3.bn1.bias"]                  = state_dict["layer2.3.bn1.bias"]
            model_state_dict["5.convs.3.bn1.running_mean"]          = state_dict["layer2.3.bn1.running_mean"]
            model_state_dict["5.convs.3.bn1.running_var"]           = state_dict["layer2.3.bn1.running_var"]
            model_state_dict["5.convs.3.conv2.weight"]              = state_dict["layer2.3.conv2.weight"]
            model_state_dict["5.convs.3.bn2.weight"]                = state_dict["layer2.3.bn2.weight"]
            model_state_dict["5.convs.3.bn2.bias"]                  = state_dict["layer2.3.bn2.bias"]
            model_state_dict["5.convs.3.bn2.running_mean"]          = state_dict["layer2.3.bn2.running_mean"]
            model_state_dict["5.convs.3.bn2.running_var"]           = state_dict["layer2.3.bn2.running_var"]
            model_state_dict["6.convs.0.conv1.weight"]              = state_dict["layer3.0.conv1.weight"]
            model_state_dict["6.convs.0.bn1.weight"]                = state_dict["layer3.0.bn1.weight"]
            model_state_dict["6.convs.0.bn1.bias"]                  = state_dict["layer3.0.bn1.bias"]
            model_state_dict["6.convs.0.bn1.running_mean"]          = state_dict["layer3.0.bn1.running_mean"]
            model_state_dict["6.convs.0.bn1.running_var"]           = state_dict["layer3.0.bn1.running_var"]
            model_state_dict["6.convs.0.conv2.weight"]              = state_dict["layer3.0.conv2.weight"]
            model_state_dict["6.convs.0.bn2.weight"]                = state_dict["layer3.0.bn2.weight"]
            model_state_dict["6.convs.0.bn2.bias"]                  = state_dict["layer3.0.bn2.bias"]
            model_state_dict["6.convs.0.bn2.running_mean"]          = state_dict["layer3.0.bn2.running_mean"]
            model_state_dict["6.convs.0.bn2.running_var"]           = state_dict["layer3.0.bn2.running_var"]
            model_state_dict["6.convs.0.downsample.0.weight"]       = state_dict["layer3.0.downsample.0.weight"]
            model_state_dict["6.convs.0.downsample.1.weight"]       = state_dict["layer3.0.downsample.1.weight"]
            model_state_dict["6.convs.0.downsample.1.bias"]         = state_dict["layer3.0.downsample.1.bias"]
            model_state_dict["6.convs.0.downsample.1.running_mean"] = state_dict["layer3.0.downsample.1.running_mean"]
            model_state_dict["6.convs.0.downsample.1.running_var"]  = state_dict["layer3.0.downsample.1.running_var"]
            model_state_dict["6.convs.1.conv1.weight"]              = state_dict["layer3.1.conv1.weight"]
            model_state_dict["6.convs.1.bn1.weight"]                = state_dict["layer3.1.bn1.weight"]
            model_state_dict["6.convs.1.bn1.bias"]                  = state_dict["layer3.1.bn1.bias"]
            model_state_dict["6.convs.1.bn1.running_mean"]          = state_dict["layer3.1.bn1.running_mean"]
            model_state_dict["6.convs.1.bn1.running_var"]           = state_dict["layer3.1.bn1.running_var"]
            model_state_dict["6.convs.1.conv2.weight"]              = state_dict["layer3.1.conv2.weight"]
            model_state_dict["6.convs.1.bn2.weight"]                = state_dict["layer3.1.bn2.weight"]
            model_state_dict["6.convs.1.bn2.bias"]                  = state_dict["layer3.1.bn2.bias"]
            model_state_dict["6.convs.1.bn2.running_mean"]          = state_dict["layer3.1.bn2.running_mean"]
            model_state_dict["6.convs.1.bn2.running_var"]           = state_dict["layer3.1.bn2.running_var"]
            model_state_dict["6.convs.2.conv1.weight"]              = state_dict["layer3.2.conv1.weight"]
            model_state_dict["6.convs.2.bn1.weight"]                = state_dict["layer3.2.bn1.weight"]
            model_state_dict["6.convs.2.bn1.bias"]                  = state_dict["layer3.2.bn1.bias"]
            model_state_dict["6.convs.2.bn1.running_mean"]          = state_dict["layer3.2.bn1.running_mean"]
            model_state_dict["6.convs.2.bn1.running_var"]           = state_dict["layer3.2.bn1.running_var"]
            model_state_dict["6.convs.2.conv2.weight"]              = state_dict["layer3.2.conv2.weight"]
            model_state_dict["6.convs.2.bn2.weight"]                = state_dict["layer3.2.bn2.weight"]
            model_state_dict["6.convs.2.bn2.bias"]                  = state_dict["layer3.2.bn2.bias"]
            model_state_dict["6.convs.2.bn2.running_mean"]          = state_dict["layer3.2.bn2.running_mean"]
            model_state_dict["6.convs.2.bn2.running_var"]           = state_dict["layer3.2.bn2.running_var"]
            model_state_dict["6.convs.3.conv1.weight"]              = state_dict["layer3.3.conv1.weight"]
            model_state_dict["6.convs.3.bn1.weight"]                = state_dict["layer3.3.bn1.weight"]
            model_state_dict["6.convs.3.bn1.bias"]                  = state_dict["layer3.3.bn1.bias"]
            model_state_dict["6.convs.3.bn1.running_mean"]          = state_dict["layer3.3.bn1.running_mean"]
            model_state_dict["6.convs.3.bn1.running_var"]           = state_dict["layer3.3.bn1.running_var"]
            model_state_dict["6.convs.3.conv2.weight"]              = state_dict["layer3.3.conv2.weight"]
            model_state_dict["6.convs.3.bn2.weight"]                = state_dict["layer3.3.bn2.weight"]
            model_state_dict["6.convs.3.bn2.bias"]                  = state_dict["layer3.3.bn2.bias"]
            model_state_dict["6.convs.3.bn2.running_mean"]          = state_dict["layer3.3.bn2.running_mean"]
            model_state_dict["6.convs.3.bn2.running_var"]           = state_dict["layer3.3.bn2.running_var"]
            model_state_dict["6.convs.4.conv1.weight"]              = state_dict["layer3.4.conv1.weight"]
            model_state_dict["6.convs.4.bn1.weight"]                = state_dict["layer3.4.bn1.weight"]
            model_state_dict["6.convs.4.bn1.bias"]                  = state_dict["layer3.4.bn1.bias"]
            model_state_dict["6.convs.4.bn1.running_mean"]          = state_dict["layer3.4.bn1.running_mean"]
            model_state_dict["6.convs.4.bn1.running_var"]           = state_dict["layer3.4.bn1.running_var"]
            model_state_dict["6.convs.4.conv2.weight"]              = state_dict["layer3.4.conv2.weight"]
            model_state_dict["6.convs.4.bn2.weight"]                = state_dict["layer3.4.bn2.weight"]
            model_state_dict["6.convs.4.bn2.bias"]                  = state_dict["layer3.4.bn2.bias"]
            model_state_dict["6.convs.4.bn2.running_mean"]          = state_dict["layer3.4.bn2.running_mean"]
            model_state_dict["6.convs.4.bn2.running_var"]           = state_dict["layer3.4.bn2.running_var"]
            model_state_dict["6.convs.5.conv1.weight"]              = state_dict["layer3.5.conv1.weight"]
            model_state_dict["6.convs.5.bn1.weight"]                = state_dict["layer3.5.bn1.weight"]
            model_state_dict["6.convs.5.bn1.bias"]                  = state_dict["layer3.5.bn1.bias"]
            model_state_dict["6.convs.5.bn1.running_mean"]          = state_dict["layer3.5.bn1.running_mean"]
            model_state_dict["6.convs.5.bn1.running_var"]           = state_dict["layer3.5.bn1.running_var"]
            model_state_dict["6.convs.5.conv2.weight"]              = state_dict["layer3.5.conv2.weight"]
            model_state_dict["6.convs.5.bn2.weight"]                = state_dict["layer3.5.bn2.weight"]
            model_state_dict["6.convs.5.bn2.bias"]                  = state_dict["layer3.5.bn2.bias"]
            model_state_dict["6.convs.5.bn2.running_mean"]          = state_dict["layer3.5.bn2.running_mean"]
            model_state_dict["6.convs.5.bn2.running_var"]           = state_dict["layer3.5.bn2.running_var"]
            model_state_dict["7.convs.0.conv1.weight"]              = state_dict["layer4.0.conv1.weight"]
            model_state_dict["7.convs.0.bn1.weight"]                = state_dict["layer4.0.bn1.weight"]
            model_state_dict["7.convs.0.bn1.bias"]                  = state_dict["layer4.0.bn1.bias"]
            model_state_dict["7.convs.0.bn1.running_mean"]          = state_dict["layer4.0.bn1.running_mean"]
            model_state_dict["7.convs.0.bn1.running_var"]           = state_dict["layer4.0.bn1.running_var"]
            model_state_dict["7.convs.0.conv2.weight"]              = state_dict["layer4.0.conv2.weight"]
            model_state_dict["7.convs.0.bn2.weight"]                = state_dict["layer4.0.bn2.weight"]
            model_state_dict["7.convs.0.bn2.bias"]                  = state_dict["layer4.0.bn2.bias"]
            model_state_dict["7.convs.0.bn2.running_mean"]          = state_dict["layer4.0.bn2.running_mean"]
            model_state_dict["7.convs.0.bn2.running_var"]           = state_dict["layer4.0.bn2.running_var"]
            model_state_dict["7.convs.0.downsample.0.weight"]       = state_dict["layer4.0.downsample.0.weight"]
            model_state_dict["7.convs.0.downsample.1.weight"]       = state_dict["layer4.0.downsample.1.weight"]
            model_state_dict["7.convs.0.downsample.1.bias"]         = state_dict["layer4.0.downsample.1.bias"]
            model_state_dict["7.convs.0.downsample.1.running_mean"] = state_dict["layer4.0.downsample.1.running_mean"]
            model_state_dict["7.convs.0.downsample.1.running_var"]  = state_dict["layer4.0.downsample.1.running_var"]
            model_state_dict["7.convs.1.conv1.weight"]              = state_dict["layer4.1.conv1.weight"]
            model_state_dict["7.convs.1.bn1.weight"]                = state_dict["layer4.1.bn1.weight"]
            model_state_dict["7.convs.1.bn1.bias"]                  = state_dict["layer4.1.bn1.bias"]
            model_state_dict["7.convs.1.bn1.running_mean"]          = state_dict["layer4.1.bn1.running_mean"]
            model_state_dict["7.convs.1.bn1.running_var"]           = state_dict["layer4.1.bn1.running_var"]
            model_state_dict["7.convs.1.conv2.weight"]              = state_dict["layer4.1.conv2.weight"]
            model_state_dict["7.convs.1.bn2.weight"]                = state_dict["layer4.1.bn2.weight"]
            model_state_dict["7.convs.1.bn2.bias"]                  = state_dict["layer4.1.bn2.bias"]
            model_state_dict["7.convs.1.bn2.running_mean"]          = state_dict["layer4.1.bn2.running_mean"]
            model_state_dict["7.convs.1.bn2.running_var"]           = state_dict["layer4.1.bn2.running_var"]
            model_state_dict["7.convs.2.conv1.weight"]              = state_dict["layer4.2.conv1.weight"]
            model_state_dict["7.convs.2.bn1.weight"]                = state_dict["layer4.2.bn1.weight"]
            model_state_dict["7.convs.2.bn1.bias"]                  = state_dict["layer4.2.bn1.bias"]
            model_state_dict["7.convs.2.bn1.running_mean"]          = state_dict["layer4.2.bn1.running_mean"]
            model_state_dict["7.convs.2.bn1.running_var"]           = state_dict["layer4.2.bn1.running_var"]
            model_state_dict["7.convs.2.conv2.weight"]              = state_dict["layer4.2.conv2.weight"]
            model_state_dict["7.convs.2.bn2.weight"]                = state_dict["layer4.2.bn2.weight"]
            model_state_dict["7.convs.2.bn2.bias"]                  = state_dict["layer4.2.bn2.bias"]
            model_state_dict["7.convs.2.bn2.running_mean"]          = state_dict["layer4.2.bn2.running_mean"]
            model_state_dict["7.convs.2.bn2.running_var"]           = state_dict["layer4.2.bn2.running_var"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["9.linear.weight"] = state_dict["fc.weight"]
                model_state_dict["9.linear.bias"]   = state_dict["fc.bias"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()


@MODELS.register(name="resnet50")
class ResNet50(ResNet):
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
            path        = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
            filename    = "resnet50-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "resnet50.yaml",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "resnet",
        fullname   : str  | None         = "resnet50",
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
        cfg = cfg or "resnet50"
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
            pretrained  = ResNet50.init_pretrained(pretrained),
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
            model_state_dict["1.bias"]                                     = state_dict["bn1.bias"]
            model_state_dict["1.num_batches_tracked"]                      = state_dict["bn1.num_batches_tracked"]
            model_state_dict["1.running_mean"]                             = state_dict["bn1.running_mean"]
            model_state_dict["1.running_var"]                              = state_dict["bn1.running_var"]
            model_state_dict["1.weight"]                                   = state_dict["bn1.weight"]
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


@MODELS.register(name="resnet101")
class ResNet101(ResNet):
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
            path        = "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
            filename    = "resnet101-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "resnet101.yaml",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "resnet",
        fullname   : str  | None         = "resnet101",
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
        cfg = cfg or "resnet101"
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
            pretrained  = ResNet101.init_pretrained(pretrained),
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
            model_state_dict["1.bias"]                                     = state_dict["bn1.bias"]
            model_state_dict["1.num_batches_tracked"]                      = state_dict["bn1.num_batches_tracked"]
            model_state_dict["1.running_mean"]                             = state_dict["bn1.running_mean"]
            model_state_dict["1.running_var"]                              = state_dict["bn1.running_var"]
            model_state_dict["1.weight"]                                   = state_dict["bn1.weight"]
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
            model_state_dict["6.convs.10.bn1.bias"]                        = state_dict["layer3.10.bn1.bias"]
            model_state_dict["6.convs.10.bn1.num_batches_tracked"]         = state_dict["layer3.10.bn1.num_batches_tracked"]
            model_state_dict["6.convs.10.bn1.running_mean"]                = state_dict["layer3.10.bn1.running_mean"]
            model_state_dict["6.convs.10.bn1.running_var"]                 = state_dict["layer3.10.bn1.running_var"]
            model_state_dict["6.convs.10.bn1.weight"]                      = state_dict["layer3.10.bn1.weight"]
            model_state_dict["6.convs.10.bn2.bias"]                        = state_dict["layer3.10.bn2.bias"]
            model_state_dict["6.convs.10.bn2.num_batches_tracked"]         = state_dict["layer3.10.bn2.num_batches_tracked"]
            model_state_dict["6.convs.10.bn2.running_mean"]                = state_dict["layer3.10.bn2.running_mean"]
            model_state_dict["6.convs.10.bn2.running_var"]                 = state_dict["layer3.10.bn2.running_var"]
            model_state_dict["6.convs.10.bn2.weight"]                      = state_dict["layer3.10.bn2.weight"]
            model_state_dict["6.convs.10.bn3.bias"]                        = state_dict["layer3.10.bn3.bias"]
            model_state_dict["6.convs.10.bn3.num_batches_tracked"]         = state_dict["layer3.10.bn3.num_batches_tracked"]
            model_state_dict["6.convs.10.bn3.running_mean"]                = state_dict["layer3.10.bn3.running_mean"]
            model_state_dict["6.convs.10.bn3.running_var"]                 = state_dict["layer3.10.bn3.running_var"]
            model_state_dict["6.convs.10.bn3.weight"]                      = state_dict["layer3.10.bn3.weight"]
            model_state_dict["6.convs.10.conv1.weight"]                    = state_dict["layer3.10.conv1.weight"]
            model_state_dict["6.convs.10.conv2.weight"]                    = state_dict["layer3.10.conv2.weight"]
            model_state_dict["6.convs.10.conv3.weight"]                    = state_dict["layer3.10.conv3.weight"]
            model_state_dict["6.convs.11.bn1.bias"]                        = state_dict["layer3.11.bn1.bias"]
            model_state_dict["6.convs.11.bn1.num_batches_tracked"]         = state_dict["layer3.11.bn1.num_batches_tracked"]
            model_state_dict["6.convs.11.bn1.running_mean"]                = state_dict["layer3.11.bn1.running_mean"]
            model_state_dict["6.convs.11.bn1.running_var"]                 = state_dict["layer3.11.bn1.running_var"]
            model_state_dict["6.convs.11.bn1.weight"]                      = state_dict["layer3.11.bn1.weight"]
            model_state_dict["6.convs.11.bn2.bias"]                        = state_dict["layer3.11.bn2.bias"]
            model_state_dict["6.convs.11.bn2.num_batches_tracked"]         = state_dict["layer3.11.bn2.num_batches_tracked"]
            model_state_dict["6.convs.11.bn2.running_mean"]                = state_dict["layer3.11.bn2.running_mean"]
            model_state_dict["6.convs.11.bn2.running_var"]                 = state_dict["layer3.11.bn2.running_var"]
            model_state_dict["6.convs.11.bn2.weight"]                      = state_dict["layer3.11.bn2.weight"]
            model_state_dict["6.convs.11.bn3.bias"]                        = state_dict["layer3.11.bn3.bias"]
            model_state_dict["6.convs.11.bn3.num_batches_tracked"]         = state_dict["layer3.11.bn3.num_batches_tracked"]
            model_state_dict["6.convs.11.bn3.running_mean"]                = state_dict["layer3.11.bn3.running_mean"]
            model_state_dict["6.convs.11.bn3.running_var"]                 = state_dict["layer3.11.bn3.running_var"]
            model_state_dict["6.convs.11.bn3.weight"]                      = state_dict["layer3.11.bn3.weight"]
            model_state_dict["6.convs.11.conv1.weight"]                    = state_dict["layer3.11.conv1.weight"]
            model_state_dict["6.convs.11.conv2.weight"]                    = state_dict["layer3.11.conv2.weight"]
            model_state_dict["6.convs.11.conv3.weight"]                    = state_dict["layer3.11.conv3.weight"]
            model_state_dict["6.convs.12.bn1.bias"]                        = state_dict["layer3.12.bn1.bias"]
            model_state_dict["6.convs.12.bn1.num_batches_tracked"]         = state_dict["layer3.12.bn1.num_batches_tracked"]
            model_state_dict["6.convs.12.bn1.running_mean"]                = state_dict["layer3.12.bn1.running_mean"]
            model_state_dict["6.convs.12.bn1.running_var"]                 = state_dict["layer3.12.bn1.running_var"]
            model_state_dict["6.convs.12.bn1.weight"]                      = state_dict["layer3.12.bn1.weight"]
            model_state_dict["6.convs.12.bn2.bias"]                        = state_dict["layer3.12.bn2.bias"]
            model_state_dict["6.convs.12.bn2.num_batches_tracked"]         = state_dict["layer3.12.bn2.num_batches_tracked"]
            model_state_dict["6.convs.12.bn2.running_mean"]                = state_dict["layer3.12.bn2.running_mean"]
            model_state_dict["6.convs.12.bn2.running_var"]                 = state_dict["layer3.12.bn2.running_var"]
            model_state_dict["6.convs.12.bn2.weight"]                      = state_dict["layer3.12.bn2.weight"]
            model_state_dict["6.convs.12.bn3.bias"]                        = state_dict["layer3.12.bn3.bias"]
            model_state_dict["6.convs.12.bn3.num_batches_tracked"]         = state_dict["layer3.12.bn3.num_batches_tracked"]
            model_state_dict["6.convs.12.bn3.running_mean"]                = state_dict["layer3.12.bn3.running_mean"]
            model_state_dict["6.convs.12.bn3.running_var"]                 = state_dict["layer3.12.bn3.running_var"]
            model_state_dict["6.convs.12.bn3.weight"]                      = state_dict["layer3.12.bn3.weight"]
            model_state_dict["6.convs.12.conv1.weight"]                    = state_dict["layer3.12.conv1.weight"]
            model_state_dict["6.convs.12.conv2.weight"]                    = state_dict["layer3.12.conv2.weight"]
            model_state_dict["6.convs.12.conv3.weight"]                    = state_dict["layer3.12.conv3.weight"]
            model_state_dict["6.convs.13.bn1.bias"]                        = state_dict["layer3.13.bn1.bias"]
            model_state_dict["6.convs.13.bn1.num_batches_tracked"]         = state_dict["layer3.13.bn1.num_batches_tracked"]
            model_state_dict["6.convs.13.bn1.running_mean"]                = state_dict["layer3.13.bn1.running_mean"]
            model_state_dict["6.convs.13.bn1.running_var"]                 = state_dict["layer3.13.bn1.running_var"]
            model_state_dict["6.convs.13.bn1.weight"]                      = state_dict["layer3.13.bn1.weight"]
            model_state_dict["6.convs.13.bn2.bias"]                        = state_dict["layer3.13.bn2.bias"]
            model_state_dict["6.convs.13.bn2.num_batches_tracked"]         = state_dict["layer3.13.bn2.num_batches_tracked"]
            model_state_dict["6.convs.13.bn2.running_mean"]                = state_dict["layer3.13.bn2.running_mean"]
            model_state_dict["6.convs.13.bn2.running_var"]                 = state_dict["layer3.13.bn2.running_var"]
            model_state_dict["6.convs.13.bn2.weight"]                      = state_dict["layer3.13.bn2.weight"]
            model_state_dict["6.convs.13.bn3.bias"]                        = state_dict["layer3.13.bn3.bias"]
            model_state_dict["6.convs.13.bn3.num_batches_tracked"]         = state_dict["layer3.13.bn3.num_batches_tracked"]
            model_state_dict["6.convs.13.bn3.running_mean"]                = state_dict["layer3.13.bn3.running_mean"]
            model_state_dict["6.convs.13.bn3.running_var"]                 = state_dict["layer3.13.bn3.running_var"]
            model_state_dict["6.convs.13.bn3.weight"]                      = state_dict["layer3.13.bn3.weight"]
            model_state_dict["6.convs.13.conv1.weight"]                    = state_dict["layer3.13.conv1.weight"]
            model_state_dict["6.convs.13.conv2.weight"]                    = state_dict["layer3.13.conv2.weight"]
            model_state_dict["6.convs.13.conv3.weight"]                    = state_dict["layer3.13.conv3.weight"]
            model_state_dict["6.convs.14.bn1.bias"]                        = state_dict["layer3.14.bn1.bias"]
            model_state_dict["6.convs.14.bn1.num_batches_tracked"]         = state_dict["layer3.14.bn1.num_batches_tracked"]
            model_state_dict["6.convs.14.bn1.running_mean"]                = state_dict["layer3.14.bn1.running_mean"]
            model_state_dict["6.convs.14.bn1.running_var"]                 = state_dict["layer3.14.bn1.running_var"]
            model_state_dict["6.convs.14.bn1.weight"]                      = state_dict["layer3.14.bn1.weight"]
            model_state_dict["6.convs.14.bn2.bias"]                        = state_dict["layer3.14.bn2.bias"]
            model_state_dict["6.convs.14.bn2.num_batches_tracked"]         = state_dict["layer3.14.bn2.num_batches_tracked"]
            model_state_dict["6.convs.14.bn2.running_mean"]                = state_dict["layer3.14.bn2.running_mean"]
            model_state_dict["6.convs.14.bn2.running_var"]                 = state_dict["layer3.14.bn2.running_var"]
            model_state_dict["6.convs.14.bn2.weight"]                      = state_dict["layer3.14.bn2.weight"]
            model_state_dict["6.convs.14.bn3.bias"]                        = state_dict["layer3.14.bn3.bias"]
            model_state_dict["6.convs.14.bn3.num_batches_tracked"]         = state_dict["layer3.14.bn3.num_batches_tracked"]
            model_state_dict["6.convs.14.bn3.running_mean"]                = state_dict["layer3.14.bn3.running_mean"]
            model_state_dict["6.convs.14.bn3.running_var"]                 = state_dict["layer3.14.bn3.running_var"]
            model_state_dict["6.convs.14.bn3.weight"]                      = state_dict["layer3.14.bn3.weight"]
            model_state_dict["6.convs.14.conv1.weight"]                    = state_dict["layer3.14.conv1.weight"]
            model_state_dict["6.convs.14.conv2.weight"]                    = state_dict["layer3.14.conv2.weight"]
            model_state_dict["6.convs.14.conv3.weight"]                    = state_dict["layer3.14.conv3.weight"]
            model_state_dict["6.convs.15.bn1.bias"]                        = state_dict["layer3.15.bn1.bias"]
            model_state_dict["6.convs.15.bn1.num_batches_tracked"]         = state_dict["layer3.15.bn1.num_batches_tracked"]
            model_state_dict["6.convs.15.bn1.running_mean"]                = state_dict["layer3.15.bn1.running_mean"]
            model_state_dict["6.convs.15.bn1.running_var"]                 = state_dict["layer3.15.bn1.running_var"]
            model_state_dict["6.convs.15.bn1.weight"]                      = state_dict["layer3.15.bn1.weight"]
            model_state_dict["6.convs.15.bn2.bias"]                        = state_dict["layer3.15.bn2.bias"]
            model_state_dict["6.convs.15.bn2.num_batches_tracked"]         = state_dict["layer3.15.bn2.num_batches_tracked"]
            model_state_dict["6.convs.15.bn2.running_mean"]                = state_dict["layer3.15.bn2.running_mean"]
            model_state_dict["6.convs.15.bn2.running_var"]                 = state_dict["layer3.15.bn2.running_var"]
            model_state_dict["6.convs.15.bn2.weight"]                      = state_dict["layer3.15.bn2.weight"]
            model_state_dict["6.convs.15.bn3.bias"]                        = state_dict["layer3.15.bn3.bias"]
            model_state_dict["6.convs.15.bn3.num_batches_tracked"]         = state_dict["layer3.15.bn3.num_batches_tracked"]
            model_state_dict["6.convs.15.bn3.running_mean"]                = state_dict["layer3.15.bn3.running_mean"]
            model_state_dict["6.convs.15.bn3.running_var"]                 = state_dict["layer3.15.bn3.running_var"]
            model_state_dict["6.convs.15.bn3.weight"]                      = state_dict["layer3.15.bn3.weight"]
            model_state_dict["6.convs.15.conv1.weight"]                    = state_dict["layer3.15.conv1.weight"]
            model_state_dict["6.convs.15.conv2.weight"]                    = state_dict["layer3.15.conv2.weight"]
            model_state_dict["6.convs.15.conv3.weight"]                    = state_dict["layer3.15.conv3.weight"]
            model_state_dict["6.convs.16.bn1.bias"]                        = state_dict["layer3.16.bn1.bias"]
            model_state_dict["6.convs.16.bn1.num_batches_tracked"]         = state_dict["layer3.16.bn1.num_batches_tracked"]
            model_state_dict["6.convs.16.bn1.running_mean"]                = state_dict["layer3.16.bn1.running_mean"]
            model_state_dict["6.convs.16.bn1.running_var"]                 = state_dict["layer3.16.bn1.running_var"]
            model_state_dict["6.convs.16.bn1.weight"]                      = state_dict["layer3.16.bn1.weight"]
            model_state_dict["6.convs.16.bn2.bias"]                        = state_dict["layer3.16.bn2.bias"]
            model_state_dict["6.convs.16.bn2.num_batches_tracked"]         = state_dict["layer3.16.bn2.num_batches_tracked"]
            model_state_dict["6.convs.16.bn2.running_mean"]                = state_dict["layer3.16.bn2.running_mean"]
            model_state_dict["6.convs.16.bn2.running_var"]                 = state_dict["layer3.16.bn2.running_var"]
            model_state_dict["6.convs.16.bn2.weight"]                      = state_dict["layer3.16.bn2.weight"]
            model_state_dict["6.convs.16.bn3.bias"]                        = state_dict["layer3.16.bn3.bias"]
            model_state_dict["6.convs.16.bn3.num_batches_tracked"]         = state_dict["layer3.16.bn3.num_batches_tracked"]
            model_state_dict["6.convs.16.bn3.running_mean"]                = state_dict["layer3.16.bn3.running_mean"]
            model_state_dict["6.convs.16.bn3.running_var"]                 = state_dict["layer3.16.bn3.running_var"]
            model_state_dict["6.convs.16.bn3.weight"]                      = state_dict["layer3.16.bn3.weight"]
            model_state_dict["6.convs.16.conv1.weight"]                    = state_dict["layer3.16.conv1.weight"]
            model_state_dict["6.convs.16.conv2.weight"]                    = state_dict["layer3.16.conv2.weight"]
            model_state_dict["6.convs.16.conv3.weight"]                    = state_dict["layer3.16.conv3.weight"]
            model_state_dict["6.convs.17.bn1.bias"]                        = state_dict["layer3.17.bn1.bias"]
            model_state_dict["6.convs.17.bn1.num_batches_tracked"]         = state_dict["layer3.17.bn1.num_batches_tracked"]
            model_state_dict["6.convs.17.bn1.running_mean"]                = state_dict["layer3.17.bn1.running_mean"]
            model_state_dict["6.convs.17.bn1.running_var"]                 = state_dict["layer3.17.bn1.running_var"]
            model_state_dict["6.convs.17.bn1.weight"]                      = state_dict["layer3.17.bn1.weight"]
            model_state_dict["6.convs.17.bn2.bias"]                        = state_dict["layer3.17.bn2.bias"]
            model_state_dict["6.convs.17.bn2.num_batches_tracked"]         = state_dict["layer3.17.bn2.num_batches_tracked"]
            model_state_dict["6.convs.17.bn2.running_mean"]                = state_dict["layer3.17.bn2.running_mean"]
            model_state_dict["6.convs.17.bn2.running_var"]                 = state_dict["layer3.17.bn2.running_var"]
            model_state_dict["6.convs.17.bn2.weight"]                      = state_dict["layer3.17.bn2.weight"]
            model_state_dict["6.convs.17.bn3.bias"]                        = state_dict["layer3.17.bn3.bias"]
            model_state_dict["6.convs.17.bn3.num_batches_tracked"]         = state_dict["layer3.17.bn3.num_batches_tracked"]
            model_state_dict["6.convs.17.bn3.running_mean"]                = state_dict["layer3.17.bn3.running_mean"]
            model_state_dict["6.convs.17.bn3.running_var"]                 = state_dict["layer3.17.bn3.running_var"]
            model_state_dict["6.convs.17.bn3.weight"]                      = state_dict["layer3.17.bn3.weight"]
            model_state_dict["6.convs.17.conv1.weight"]                    = state_dict["layer3.17.conv1.weight"]
            model_state_dict["6.convs.17.conv2.weight"]                    = state_dict["layer3.17.conv2.weight"]
            model_state_dict["6.convs.17.conv3.weight"]                    = state_dict["layer3.17.conv3.weight"]
            model_state_dict["6.convs.18.bn1.bias"]                        = state_dict["layer3.18.bn1.bias"]
            model_state_dict["6.convs.18.bn1.num_batches_tracked"]         = state_dict["layer3.18.bn1.num_batches_tracked"]
            model_state_dict["6.convs.18.bn1.running_mean"]                = state_dict["layer3.18.bn1.running_mean"]
            model_state_dict["6.convs.18.bn1.running_var"]                 = state_dict["layer3.18.bn1.running_var"]
            model_state_dict["6.convs.18.bn1.weight"]                      = state_dict["layer3.18.bn1.weight"]
            model_state_dict["6.convs.18.bn2.bias"]                        = state_dict["layer3.18.bn2.bias"]
            model_state_dict["6.convs.18.bn2.num_batches_tracked"]         = state_dict["layer3.18.bn2.num_batches_tracked"]
            model_state_dict["6.convs.18.bn2.running_mean"]                = state_dict["layer3.18.bn2.running_mean"]
            model_state_dict["6.convs.18.bn2.running_var"]                 = state_dict["layer3.18.bn2.running_var"]
            model_state_dict["6.convs.18.bn2.weight"]                      = state_dict["layer3.18.bn2.weight"]
            model_state_dict["6.convs.18.bn3.bias"]                        = state_dict["layer3.18.bn3.bias"]
            model_state_dict["6.convs.18.bn3.num_batches_tracked"]         = state_dict["layer3.18.bn3.num_batches_tracked"]
            model_state_dict["6.convs.18.bn3.running_mean"]                = state_dict["layer3.18.bn3.running_mean"]
            model_state_dict["6.convs.18.bn3.running_var"]                 = state_dict["layer3.18.bn3.running_var"]
            model_state_dict["6.convs.18.bn3.weight"]                      = state_dict["layer3.18.bn3.weight"]
            model_state_dict["6.convs.18.conv1.weight"]                    = state_dict["layer3.18.conv1.weight"]
            model_state_dict["6.convs.18.conv2.weight"]                    = state_dict["layer3.18.conv2.weight"]
            model_state_dict["6.convs.18.conv3.weight"]                    = state_dict["layer3.18.conv3.weight"]
            model_state_dict["6.convs.19.bn1.bias"]                        = state_dict["layer3.19.bn1.bias"]
            model_state_dict["6.convs.19.bn1.num_batches_tracked"]         = state_dict["layer3.19.bn1.num_batches_tracked"]
            model_state_dict["6.convs.19.bn1.running_mean"]                = state_dict["layer3.19.bn1.running_mean"]
            model_state_dict["6.convs.19.bn1.running_var"]                 = state_dict["layer3.19.bn1.running_var"]
            model_state_dict["6.convs.19.bn1.weight"]                      = state_dict["layer3.19.bn1.weight"]
            model_state_dict["6.convs.19.bn2.bias"]                        = state_dict["layer3.19.bn2.bias"]
            model_state_dict["6.convs.19.bn2.num_batches_tracked"]         = state_dict["layer3.19.bn2.num_batches_tracked"]
            model_state_dict["6.convs.19.bn2.running_mean"]                = state_dict["layer3.19.bn2.running_mean"]
            model_state_dict["6.convs.19.bn2.running_var"]                 = state_dict["layer3.19.bn2.running_var"]
            model_state_dict["6.convs.19.bn2.weight"]                      = state_dict["layer3.19.bn2.weight"]
            model_state_dict["6.convs.19.bn3.bias"]                        = state_dict["layer3.19.bn3.bias"]
            model_state_dict["6.convs.19.bn3.num_batches_tracked"]         = state_dict["layer3.19.bn3.num_batches_tracked"]
            model_state_dict["6.convs.19.bn3.running_mean"]                = state_dict["layer3.19.bn3.running_mean"]
            model_state_dict["6.convs.19.bn3.running_var"]                 = state_dict["layer3.19.bn3.running_var"]
            model_state_dict["6.convs.19.bn3.weight"]                      = state_dict["layer3.19.bn3.weight"]
            model_state_dict["6.convs.19.conv1.weight"]                    = state_dict["layer3.19.conv1.weight"]
            model_state_dict["6.convs.19.conv2.weight"]                    = state_dict["layer3.19.conv2.weight"]
            model_state_dict["6.convs.19.conv3.weight"]                    = state_dict["layer3.19.conv3.weight"]
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
            model_state_dict["6.convs.20.bn1.bias"]                        = state_dict["layer3.20.bn1.bias"]
            model_state_dict["6.convs.20.bn1.num_batches_tracked"]         = state_dict["layer3.20.bn1.num_batches_tracked"]
            model_state_dict["6.convs.20.bn1.running_mean"]                = state_dict["layer3.20.bn1.running_mean"]
            model_state_dict["6.convs.20.bn1.running_var"]                 = state_dict["layer3.20.bn1.running_var"]
            model_state_dict["6.convs.20.bn1.weight"]                      = state_dict["layer3.20.bn1.weight"]
            model_state_dict["6.convs.20.bn2.bias"]                        = state_dict["layer3.20.bn2.bias"]
            model_state_dict["6.convs.20.bn2.num_batches_tracked"]         = state_dict["layer3.20.bn2.num_batches_tracked"]
            model_state_dict["6.convs.20.bn2.running_mean"]                = state_dict["layer3.20.bn2.running_mean"]
            model_state_dict["6.convs.20.bn2.running_var"]                 = state_dict["layer3.20.bn2.running_var"]
            model_state_dict["6.convs.20.bn2.weight"]                      = state_dict["layer3.20.bn2.weight"]
            model_state_dict["6.convs.20.bn3.bias"]                        = state_dict["layer3.20.bn3.bias"]
            model_state_dict["6.convs.20.bn3.num_batches_tracked"]         = state_dict["layer3.20.bn3.num_batches_tracked"]
            model_state_dict["6.convs.20.bn3.running_mean"]                = state_dict["layer3.20.bn3.running_mean"]
            model_state_dict["6.convs.20.bn3.running_var"]                 = state_dict["layer3.20.bn3.running_var"]
            model_state_dict["6.convs.20.bn3.weight"]                      = state_dict["layer3.20.bn3.weight"]
            model_state_dict["6.convs.20.conv1.weight"]                    = state_dict["layer3.20.conv1.weight"]
            model_state_dict["6.convs.20.conv2.weight"]                    = state_dict["layer3.20.conv2.weight"]
            model_state_dict["6.convs.20.conv3.weight"]                    = state_dict["layer3.20.conv3.weight"]
            model_state_dict["6.convs.21.bn1.bias"]                        = state_dict["layer3.21.bn1.bias"]
            model_state_dict["6.convs.21.bn1.num_batches_tracked"]         = state_dict["layer3.21.bn1.num_batches_tracked"]
            model_state_dict["6.convs.21.bn1.running_mean"]                = state_dict["layer3.21.bn1.running_mean"]
            model_state_dict["6.convs.21.bn1.running_var"]                 = state_dict["layer3.21.bn1.running_var"]
            model_state_dict["6.convs.21.bn1.weight"]                      = state_dict["layer3.21.bn1.weight"]
            model_state_dict["6.convs.21.bn2.bias"]                        = state_dict["layer3.21.bn2.bias"]
            model_state_dict["6.convs.21.bn2.num_batches_tracked"]         = state_dict["layer3.21.bn2.num_batches_tracked"]
            model_state_dict["6.convs.21.bn2.running_mean"]                = state_dict["layer3.21.bn2.running_mean"]
            model_state_dict["6.convs.21.bn2.running_var"]                 = state_dict["layer3.21.bn2.running_var"]
            model_state_dict["6.convs.21.bn2.weight"]                      = state_dict["layer3.21.bn2.weight"]
            model_state_dict["6.convs.21.bn3.bias"]                        = state_dict["layer3.21.bn3.bias"]
            model_state_dict["6.convs.21.bn3.num_batches_tracked"]         = state_dict["layer3.21.bn3.num_batches_tracked"]
            model_state_dict["6.convs.21.bn3.running_mean"]                = state_dict["layer3.21.bn3.running_mean"]
            model_state_dict["6.convs.21.bn3.running_var"]                 = state_dict["layer3.21.bn3.running_var"]
            model_state_dict["6.convs.21.bn3.weight"]                      = state_dict["layer3.21.bn3.weight"]
            model_state_dict["6.convs.21.conv1.weight"]                    = state_dict["layer3.21.conv1.weight"]
            model_state_dict["6.convs.21.conv2.weight"]                    = state_dict["layer3.21.conv2.weight"]
            model_state_dict["6.convs.21.conv3.weight"]                    = state_dict["layer3.21.conv3.weight"]
            model_state_dict["6.convs.22.bn1.bias"]                        = state_dict["layer3.22.bn1.bias"]
            model_state_dict["6.convs.22.bn1.num_batches_tracked"]         = state_dict["layer3.22.bn1.num_batches_tracked"]
            model_state_dict["6.convs.22.bn1.running_mean"]                = state_dict["layer3.22.bn1.running_mean"]
            model_state_dict["6.convs.22.bn1.running_var"]                 = state_dict["layer3.22.bn1.running_var"]
            model_state_dict["6.convs.22.bn1.weight"]                      = state_dict["layer3.22.bn1.weight"]
            model_state_dict["6.convs.22.bn2.bias"]                        = state_dict["layer3.22.bn2.bias"]
            model_state_dict["6.convs.22.bn2.num_batches_tracked"]         = state_dict["layer3.22.bn2.num_batches_tracked"]
            model_state_dict["6.convs.22.bn2.running_mean"]                = state_dict["layer3.22.bn2.running_mean"]
            model_state_dict["6.convs.22.bn2.running_var"]                 = state_dict["layer3.22.bn2.running_var"]
            model_state_dict["6.convs.22.bn2.weight"]                      = state_dict["layer3.22.bn2.weight"]
            model_state_dict["6.convs.22.bn3.bias"]                        = state_dict["layer3.22.bn3.bias"]
            model_state_dict["6.convs.22.bn3.num_batches_tracked"]         = state_dict["layer3.22.bn3.num_batches_tracked"]
            model_state_dict["6.convs.22.bn3.running_mean"]                = state_dict["layer3.22.bn3.running_mean"]
            model_state_dict["6.convs.22.bn3.running_var"]                 = state_dict["layer3.22.bn3.running_var"]
            model_state_dict["6.convs.22.bn3.weight"]                      = state_dict["layer3.22.bn3.weight"]
            model_state_dict["6.convs.22.conv1.weight"]                    = state_dict["layer3.22.conv1.weight"]
            model_state_dict["6.convs.22.conv2.weight"]                    = state_dict["layer3.22.conv2.weight"]
            model_state_dict["6.convs.22.conv3.weight"]                    = state_dict["layer3.22.conv3.weight"]
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
            model_state_dict["6.convs.6.bn1.bias"]                         = state_dict["layer3.6.bn1.bias"]
            model_state_dict["6.convs.6.bn1.num_batches_tracked"]          = state_dict["layer3.6.bn1.num_batches_tracked"]
            model_state_dict["6.convs.6.bn1.running_mean"]                 = state_dict["layer3.6.bn1.running_mean"]
            model_state_dict["6.convs.6.bn1.running_var"]                  = state_dict["layer3.6.bn1.running_var"]
            model_state_dict["6.convs.6.bn1.weight"]                       = state_dict["layer3.6.bn1.weight"]
            model_state_dict["6.convs.6.bn2.bias"]                         = state_dict["layer3.6.bn2.bias"]
            model_state_dict["6.convs.6.bn2.num_batches_tracked"]          = state_dict["layer3.6.bn2.num_batches_tracked"]
            model_state_dict["6.convs.6.bn2.running_mean"]                 = state_dict["layer3.6.bn2.running_mean"]
            model_state_dict["6.convs.6.bn2.running_var"]                  = state_dict["layer3.6.bn2.running_var"]
            model_state_dict["6.convs.6.bn2.weight"]                       = state_dict["layer3.6.bn2.weight"]
            model_state_dict["6.convs.6.bn3.bias"]                         = state_dict["layer3.6.bn3.bias"]
            model_state_dict["6.convs.6.bn3.num_batches_tracked"]          = state_dict["layer3.6.bn3.num_batches_tracked"]
            model_state_dict["6.convs.6.bn3.running_mean"]                 = state_dict["layer3.6.bn3.running_mean"]
            model_state_dict["6.convs.6.bn3.running_var"]                  = state_dict["layer3.6.bn3.running_var"]
            model_state_dict["6.convs.6.bn3.weight"]                       = state_dict["layer3.6.bn3.weight"]
            model_state_dict["6.convs.6.conv1.weight"]                     = state_dict["layer3.6.conv1.weight"]
            model_state_dict["6.convs.6.conv2.weight"]                     = state_dict["layer3.6.conv2.weight"]
            model_state_dict["6.convs.6.conv3.weight"]                     = state_dict["layer3.6.conv3.weight"]
            model_state_dict["6.convs.7.bn1.bias"]                         = state_dict["layer3.7.bn1.bias"]
            model_state_dict["6.convs.7.bn1.num_batches_tracked"]          = state_dict["layer3.7.bn1.num_batches_tracked"]
            model_state_dict["6.convs.7.bn1.running_mean"]                 = state_dict["layer3.7.bn1.running_mean"]
            model_state_dict["6.convs.7.bn1.running_var"]                  = state_dict["layer3.7.bn1.running_var"]
            model_state_dict["6.convs.7.bn1.weight"]                       = state_dict["layer3.7.bn1.weight"]
            model_state_dict["6.convs.7.bn2.bias"]                         = state_dict["layer3.7.bn2.bias"]
            model_state_dict["6.convs.7.bn2.num_batches_tracked"]          = state_dict["layer3.7.bn2.num_batches_tracked"]
            model_state_dict["6.convs.7.bn2.running_mean"]                 = state_dict["layer3.7.bn2.running_mean"]
            model_state_dict["6.convs.7.bn2.running_var"]                  = state_dict["layer3.7.bn2.running_var"]
            model_state_dict["6.convs.7.bn2.weight"]                       = state_dict["layer3.7.bn2.weight"]
            model_state_dict["6.convs.7.bn3.bias"]                         = state_dict["layer3.7.bn3.bias"]
            model_state_dict["6.convs.7.bn3.num_batches_tracked"]          = state_dict["layer3.7.bn3.num_batches_tracked"]
            model_state_dict["6.convs.7.bn3.running_mean"]                 = state_dict["layer3.7.bn3.running_mean"]
            model_state_dict["6.convs.7.bn3.running_var"]                  = state_dict["layer3.7.bn3.running_var"]
            model_state_dict["6.convs.7.bn3.weight"]                       = state_dict["layer3.7.bn3.weight"]
            model_state_dict["6.convs.7.conv1.weight"]                     = state_dict["layer3.7.conv1.weight"]
            model_state_dict["6.convs.7.conv2.weight"]                     = state_dict["layer3.7.conv2.weight"]
            model_state_dict["6.convs.7.conv3.weight"]                     = state_dict["layer3.7.conv3.weight"]
            model_state_dict["6.convs.8.bn1.bias"]                         = state_dict["layer3.8.bn1.bias"]
            model_state_dict["6.convs.8.bn1.num_batches_tracked"]          = state_dict["layer3.8.bn1.num_batches_tracked"]
            model_state_dict["6.convs.8.bn1.running_mean"]                 = state_dict["layer3.8.bn1.running_mean"]
            model_state_dict["6.convs.8.bn1.running_var"]                  = state_dict["layer3.8.bn1.running_var"]
            model_state_dict["6.convs.8.bn1.weight"]                       = state_dict["layer3.8.bn1.weight"]
            model_state_dict["6.convs.8.bn2.bias"]                         = state_dict["layer3.8.bn2.bias"]
            model_state_dict["6.convs.8.bn2.num_batches_tracked"]          = state_dict["layer3.8.bn2.num_batches_tracked"]
            model_state_dict["6.convs.8.bn2.running_mean"]                 = state_dict["layer3.8.bn2.running_mean"]
            model_state_dict["6.convs.8.bn2.running_var"]                  = state_dict["layer3.8.bn2.running_var"]
            model_state_dict["6.convs.8.bn2.weight"]                       = state_dict["layer3.8.bn2.weight"]
            model_state_dict["6.convs.8.bn3.bias"]                         = state_dict["layer3.8.bn3.bias"]
            model_state_dict["6.convs.8.bn3.num_batches_tracked"]          = state_dict["layer3.8.bn3.num_batches_tracked"]
            model_state_dict["6.convs.8.bn3.running_mean"]                 = state_dict["layer3.8.bn3.running_mean"]
            model_state_dict["6.convs.8.bn3.running_var"]                  = state_dict["layer3.8.bn3.running_var"]
            model_state_dict["6.convs.8.bn3.weight"]                       = state_dict["layer3.8.bn3.weight"]
            model_state_dict["6.convs.8.conv1.weight"]                     = state_dict["layer3.8.conv1.weight"]
            model_state_dict["6.convs.8.conv2.weight"]                     = state_dict["layer3.8.conv2.weight"]
            model_state_dict["6.convs.8.conv3.weight"]                     = state_dict["layer3.8.conv3.weight"]
            model_state_dict["6.convs.9.bn1.bias"]                         = state_dict["layer3.9.bn1.bias"]
            model_state_dict["6.convs.9.bn1.num_batches_tracked"]          = state_dict["layer3.9.bn1.num_batches_tracked"]
            model_state_dict["6.convs.9.bn1.running_mean"]                 = state_dict["layer3.9.bn1.running_mean"]
            model_state_dict["6.convs.9.bn1.running_var"]                  = state_dict["layer3.9.bn1.running_var"]
            model_state_dict["6.convs.9.bn1.weight"]                       = state_dict["layer3.9.bn1.weight"]
            model_state_dict["6.convs.9.bn2.bias"]                         = state_dict["layer3.9.bn2.bias"]
            model_state_dict["6.convs.9.bn2.num_batches_tracked"]          = state_dict["layer3.9.bn2.num_batches_tracked"]
            model_state_dict["6.convs.9.bn2.running_mean"]                 = state_dict["layer3.9.bn2.running_mean"]
            model_state_dict["6.convs.9.bn2.running_var"]                  = state_dict["layer3.9.bn2.running_var"]
            model_state_dict["6.convs.9.bn2.weight"]                       = state_dict["layer3.9.bn2.weight"]
            model_state_dict["6.convs.9.bn3.bias"]                         = state_dict["layer3.9.bn3.bias"]
            model_state_dict["6.convs.9.bn3.num_batches_tracked"]          = state_dict["layer3.9.bn3.num_batches_tracked"]
            model_state_dict["6.convs.9.bn3.running_mean"]                 = state_dict["layer3.9.bn3.running_mean"]
            model_state_dict["6.convs.9.bn3.running_var"]                  = state_dict["layer3.9.bn3.running_var"]
            model_state_dict["6.convs.9.bn3.weight"]                       = state_dict["layer3.9.bn3.weight"]
            model_state_dict["6.convs.9.conv1.weight"]                     = state_dict["layer3.9.conv1.weight"]
            model_state_dict["6.convs.9.conv2.weight"]                     = state_dict["layer3.9.conv2.weight"]
            model_state_dict["6.convs.9.conv3.weight"]                     = state_dict["layer3.9.conv3.weight"]
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


@MODELS.register(name="resnet152")
class ResNet152(ResNet):
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
            path        = "https://download.pytorch.org/models/resnet152-f82ba261.pth",
            filename    = "resnet152-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "resnet152.yaml",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "resnet",
        fullname   : str  | None         = "resnet152",
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
        cfg = cfg or "resnet152"
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
            pretrained  = ResNet152.init_pretrained(pretrained),
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
            model_state_dict["1.bias"]                                     = state_dict["bn1.bias"]
            model_state_dict["1.num_batches_tracked"]                      = state_dict["bn1.num_batches_tracked"]
            model_state_dict["1.running_mean"]                             = state_dict["bn1.running_mean"]
            model_state_dict["1.running_var"]                              = state_dict["bn1.running_var"]
            model_state_dict["1.weight"]                                   = state_dict["bn1.weight"]
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
            model_state_dict["5.convs.4.bn1.bias"]                         = state_dict["layer2.4.bn1.bias"]
            model_state_dict["5.convs.4.bn1.num_batches_tracked"]          = state_dict["layer2.4.bn1.num_batches_tracked"]
            model_state_dict["5.convs.4.bn1.running_mean"]                 = state_dict["layer2.4.bn1.running_mean"]
            model_state_dict["5.convs.4.bn1.running_var"]                  = state_dict["layer2.4.bn1.running_var"]
            model_state_dict["5.convs.4.bn1.weight"]                       = state_dict["layer2.4.bn1.weight"]
            model_state_dict["5.convs.4.bn2.bias"]                         = state_dict["layer2.4.bn2.bias"]
            model_state_dict["5.convs.4.bn2.num_batches_tracked"]          = state_dict["layer2.4.bn2.num_batches_tracked"]
            model_state_dict["5.convs.4.bn2.running_mean"]                 = state_dict["layer2.4.bn2.running_mean"]
            model_state_dict["5.convs.4.bn2.running_var"]                  = state_dict["layer2.4.bn2.running_var"]
            model_state_dict["5.convs.4.bn2.weight"]                       = state_dict["layer2.4.bn2.weight"]
            model_state_dict["5.convs.4.bn3.bias"]                         = state_dict["layer2.4.bn3.bias"]
            model_state_dict["5.convs.4.bn3.num_batches_tracked"]          = state_dict["layer2.4.bn3.num_batches_tracked"]
            model_state_dict["5.convs.4.bn3.running_mean"]                 = state_dict["layer2.4.bn3.running_mean"]
            model_state_dict["5.convs.4.bn3.running_var"]                  = state_dict["layer2.4.bn3.running_var"]
            model_state_dict["5.convs.4.bn3.weight"]                       = state_dict["layer2.4.bn3.weight"]
            model_state_dict["5.convs.4.conv1.weight"]                     = state_dict["layer2.4.conv1.weight"]
            model_state_dict["5.convs.4.conv2.weight"]                     = state_dict["layer2.4.conv2.weight"]
            model_state_dict["5.convs.4.conv3.weight"]                     = state_dict["layer2.4.conv3.weight"]
            model_state_dict["5.convs.5.bn1.bias"]                         = state_dict["layer2.5.bn1.bias"]
            model_state_dict["5.convs.5.bn1.num_batches_tracked"]          = state_dict["layer2.5.bn1.num_batches_tracked"]
            model_state_dict["5.convs.5.bn1.running_mean"]                 = state_dict["layer2.5.bn1.running_mean"]
            model_state_dict["5.convs.5.bn1.running_var"]                  = state_dict["layer2.5.bn1.running_var"]
            model_state_dict["5.convs.5.bn1.weight"]                       = state_dict["layer2.5.bn1.weight"]
            model_state_dict["5.convs.5.bn2.bias"]                         = state_dict["layer2.5.bn2.bias"]
            model_state_dict["5.convs.5.bn2.num_batches_tracked"]          = state_dict["layer2.5.bn2.num_batches_tracked"]
            model_state_dict["5.convs.5.bn2.running_mean"]                 = state_dict["layer2.5.bn2.running_mean"]
            model_state_dict["5.convs.5.bn2.running_var"]                  = state_dict["layer2.5.bn2.running_var"]
            model_state_dict["5.convs.5.bn2.weight"]                       = state_dict["layer2.5.bn2.weight"]
            model_state_dict["5.convs.5.bn3.bias"]                         = state_dict["layer2.5.bn3.bias"]
            model_state_dict["5.convs.5.bn3.num_batches_tracked"]          = state_dict["layer2.5.bn3.num_batches_tracked"]
            model_state_dict["5.convs.5.bn3.running_mean"]                 = state_dict["layer2.5.bn3.running_mean"]
            model_state_dict["5.convs.5.bn3.running_var"]                  = state_dict["layer2.5.bn3.running_var"]
            model_state_dict["5.convs.5.bn3.weight"]                       = state_dict["layer2.5.bn3.weight"]
            model_state_dict["5.convs.5.conv1.weight"]                     = state_dict["layer2.5.conv1.weight"]
            model_state_dict["5.convs.5.conv2.weight"]                     = state_dict["layer2.5.conv2.weight"]
            model_state_dict["5.convs.5.conv3.weight"]                     = state_dict["layer2.5.conv3.weight"]
            model_state_dict["5.convs.6.bn1.bias"]                         = state_dict["layer2.6.bn1.bias"]
            model_state_dict["5.convs.6.bn1.num_batches_tracked"]          = state_dict["layer2.6.bn1.num_batches_tracked"]
            model_state_dict["5.convs.6.bn1.running_mean"]                 = state_dict["layer2.6.bn1.running_mean"]
            model_state_dict["5.convs.6.bn1.running_var"]                  = state_dict["layer2.6.bn1.running_var"]
            model_state_dict["5.convs.6.bn1.weight"]                       = state_dict["layer2.6.bn1.weight"]
            model_state_dict["5.convs.6.bn2.bias"]                         = state_dict["layer2.6.bn2.bias"]
            model_state_dict["5.convs.6.bn2.num_batches_tracked"]          = state_dict["layer2.6.bn2.num_batches_tracked"]
            model_state_dict["5.convs.6.bn2.running_mean"]                 = state_dict["layer2.6.bn2.running_mean"]
            model_state_dict["5.convs.6.bn2.running_var"]                  = state_dict["layer2.6.bn2.running_var"]
            model_state_dict["5.convs.6.bn2.weight"]                       = state_dict["layer2.6.bn2.weight"]
            model_state_dict["5.convs.6.bn3.bias"]                         = state_dict["layer2.6.bn3.bias"]
            model_state_dict["5.convs.6.bn3.num_batches_tracked"]          = state_dict["layer2.6.bn3.num_batches_tracked"]
            model_state_dict["5.convs.6.bn3.running_mean"]                 = state_dict["layer2.6.bn3.running_mean"]
            model_state_dict["5.convs.6.bn3.running_var"]                  = state_dict["layer2.6.bn3.running_var"]
            model_state_dict["5.convs.6.bn3.weight"]                       = state_dict["layer2.6.bn3.weight"]
            model_state_dict["5.convs.6.conv1.weight"]                     = state_dict["layer2.6.conv1.weight"]
            model_state_dict["5.convs.6.conv2.weight"]                     = state_dict["layer2.6.conv2.weight"]
            model_state_dict["5.convs.6.conv3.weight"]                     = state_dict["layer2.6.conv3.weight"]
            model_state_dict["5.convs.7.bn1.bias"]                         = state_dict["layer2.7.bn1.bias"]
            model_state_dict["5.convs.7.bn1.num_batches_tracked"]          = state_dict["layer2.7.bn1.num_batches_tracked"]
            model_state_dict["5.convs.7.bn1.running_mean"]                 = state_dict["layer2.7.bn1.running_mean"]
            model_state_dict["5.convs.7.bn1.running_var"]                  = state_dict["layer2.7.bn1.running_var"]
            model_state_dict["5.convs.7.bn1.weight"]                       = state_dict["layer2.7.bn1.weight"]
            model_state_dict["5.convs.7.bn2.bias"]                         = state_dict["layer2.7.bn2.bias"]
            model_state_dict["5.convs.7.bn2.num_batches_tracked"]          = state_dict["layer2.7.bn2.num_batches_tracked"]
            model_state_dict["5.convs.7.bn2.running_mean"]                 = state_dict["layer2.7.bn2.running_mean"]
            model_state_dict["5.convs.7.bn2.running_var"]                  = state_dict["layer2.7.bn2.running_var"]
            model_state_dict["5.convs.7.bn2.weight"]                       = state_dict["layer2.7.bn2.weight"]
            model_state_dict["5.convs.7.bn3.bias"]                         = state_dict["layer2.7.bn3.bias"]
            model_state_dict["5.convs.7.bn3.num_batches_tracked"]          = state_dict["layer2.7.bn3.num_batches_tracked"]
            model_state_dict["5.convs.7.bn3.running_mean"]                 = state_dict["layer2.7.bn3.running_mean"]
            model_state_dict["5.convs.7.bn3.running_var"]                  = state_dict["layer2.7.bn3.running_var"]
            model_state_dict["5.convs.7.bn3.weight"]                       = state_dict["layer2.7.bn3.weight"]
            model_state_dict["5.convs.7.conv1.weight"]                     = state_dict["layer2.7.conv1.weight"]
            model_state_dict["5.convs.7.conv2.weight"]                     = state_dict["layer2.7.conv2.weight"]
            model_state_dict["5.convs.7.conv3.weight"]                     = state_dict["layer2.7.conv3.weight"]
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
            model_state_dict["6.convs.10.bn1.bias"]                        = state_dict["layer3.10.bn1.bias"]
            model_state_dict["6.convs.10.bn1.num_batches_tracked"]         = state_dict["layer3.10.bn1.num_batches_tracked"]
            model_state_dict["6.convs.10.bn1.running_mean"]                = state_dict["layer3.10.bn1.running_mean"]
            model_state_dict["6.convs.10.bn1.running_var"]                 = state_dict["layer3.10.bn1.running_var"]
            model_state_dict["6.convs.10.bn1.weight"]                      = state_dict["layer3.10.bn1.weight"]
            model_state_dict["6.convs.10.bn2.bias"]                        = state_dict["layer3.10.bn2.bias"]
            model_state_dict["6.convs.10.bn2.num_batches_tracked"]         = state_dict["layer3.10.bn2.num_batches_tracked"]
            model_state_dict["6.convs.10.bn2.running_mean"]                = state_dict["layer3.10.bn2.running_mean"]
            model_state_dict["6.convs.10.bn2.running_var"]                 = state_dict["layer3.10.bn2.running_var"]
            model_state_dict["6.convs.10.bn2.weight"]                      = state_dict["layer3.10.bn2.weight"]
            model_state_dict["6.convs.10.bn3.bias"]                        = state_dict["layer3.10.bn3.bias"]
            model_state_dict["6.convs.10.bn3.num_batches_tracked"]         = state_dict["layer3.10.bn3.num_batches_tracked"]
            model_state_dict["6.convs.10.bn3.running_mean"]                = state_dict["layer3.10.bn3.running_mean"]
            model_state_dict["6.convs.10.bn3.running_var"]                 = state_dict["layer3.10.bn3.running_var"]
            model_state_dict["6.convs.10.bn3.weight"]                      = state_dict["layer3.10.bn3.weight"]
            model_state_dict["6.convs.10.conv1.weight"]                    = state_dict["layer3.10.conv1.weight"]
            model_state_dict["6.convs.10.conv2.weight"]                    = state_dict["layer3.10.conv2.weight"]
            model_state_dict["6.convs.10.conv3.weight"]                    = state_dict["layer3.10.conv3.weight"]
            model_state_dict["6.convs.11.bn1.bias"]                        = state_dict["layer3.11.bn1.bias"]
            model_state_dict["6.convs.11.bn1.num_batches_tracked"]         = state_dict["layer3.11.bn1.num_batches_tracked"]
            model_state_dict["6.convs.11.bn1.running_mean"]                = state_dict["layer3.11.bn1.running_mean"]
            model_state_dict["6.convs.11.bn1.running_var"]                 = state_dict["layer3.11.bn1.running_var"]
            model_state_dict["6.convs.11.bn1.weight"]                      = state_dict["layer3.11.bn1.weight"]
            model_state_dict["6.convs.11.bn2.bias"]                        = state_dict["layer3.11.bn2.bias"]
            model_state_dict["6.convs.11.bn2.num_batches_tracked"]         = state_dict["layer3.11.bn2.num_batches_tracked"]
            model_state_dict["6.convs.11.bn2.running_mean"]                = state_dict["layer3.11.bn2.running_mean"]
            model_state_dict["6.convs.11.bn2.running_var"]                 = state_dict["layer3.11.bn2.running_var"]
            model_state_dict["6.convs.11.bn2.weight"]                      = state_dict["layer3.11.bn2.weight"]
            model_state_dict["6.convs.11.bn3.bias"]                        = state_dict["layer3.11.bn3.bias"]
            model_state_dict["6.convs.11.bn3.num_batches_tracked"]         = state_dict["layer3.11.bn3.num_batches_tracked"]
            model_state_dict["6.convs.11.bn3.running_mean"]                = state_dict["layer3.11.bn3.running_mean"]
            model_state_dict["6.convs.11.bn3.running_var"]                 = state_dict["layer3.11.bn3.running_var"]
            model_state_dict["6.convs.11.bn3.weight"]                      = state_dict["layer3.11.bn3.weight"]
            model_state_dict["6.convs.11.conv1.weight"]                    = state_dict["layer3.11.conv1.weight"]
            model_state_dict["6.convs.11.conv2.weight"]                    = state_dict["layer3.11.conv2.weight"]
            model_state_dict["6.convs.11.conv3.weight"]                    = state_dict["layer3.11.conv3.weight"]
            model_state_dict["6.convs.12.bn1.bias"]                        = state_dict["layer3.12.bn1.bias"]
            model_state_dict["6.convs.12.bn1.num_batches_tracked"]         = state_dict["layer3.12.bn1.num_batches_tracked"]
            model_state_dict["6.convs.12.bn1.running_mean"]                = state_dict["layer3.12.bn1.running_mean"]
            model_state_dict["6.convs.12.bn1.running_var"]                 = state_dict["layer3.12.bn1.running_var"]
            model_state_dict["6.convs.12.bn1.weight"]                      = state_dict["layer3.12.bn1.weight"]
            model_state_dict["6.convs.12.bn2.bias"]                        = state_dict["layer3.12.bn2.bias"]
            model_state_dict["6.convs.12.bn2.num_batches_tracked"]         = state_dict["layer3.12.bn2.num_batches_tracked"]
            model_state_dict["6.convs.12.bn2.running_mean"]                = state_dict["layer3.12.bn2.running_mean"]
            model_state_dict["6.convs.12.bn2.running_var"]                 = state_dict["layer3.12.bn2.running_var"]
            model_state_dict["6.convs.12.bn2.weight"]                      = state_dict["layer3.12.bn2.weight"]
            model_state_dict["6.convs.12.bn3.bias"]                        = state_dict["layer3.12.bn3.bias"]
            model_state_dict["6.convs.12.bn3.num_batches_tracked"]         = state_dict["layer3.12.bn3.num_batches_tracked"]
            model_state_dict["6.convs.12.bn3.running_mean"]                = state_dict["layer3.12.bn3.running_mean"]
            model_state_dict["6.convs.12.bn3.running_var"]                 = state_dict["layer3.12.bn3.running_var"]
            model_state_dict["6.convs.12.bn3.weight"]                      = state_dict["layer3.12.bn3.weight"]
            model_state_dict["6.convs.12.conv1.weight"]                    = state_dict["layer3.12.conv1.weight"]
            model_state_dict["6.convs.12.conv2.weight"]                    = state_dict["layer3.12.conv2.weight"]
            model_state_dict["6.convs.12.conv3.weight"]                    = state_dict["layer3.12.conv3.weight"]
            model_state_dict["6.convs.13.bn1.bias"]                        = state_dict["layer3.13.bn1.bias"]
            model_state_dict["6.convs.13.bn1.num_batches_tracked"]         = state_dict["layer3.13.bn1.num_batches_tracked"]
            model_state_dict["6.convs.13.bn1.running_mean"]                = state_dict["layer3.13.bn1.running_mean"]
            model_state_dict["6.convs.13.bn1.running_var"]                 = state_dict["layer3.13.bn1.running_var"]
            model_state_dict["6.convs.13.bn1.weight"]                      = state_dict["layer3.13.bn1.weight"]
            model_state_dict["6.convs.13.bn2.bias"]                        = state_dict["layer3.13.bn2.bias"]
            model_state_dict["6.convs.13.bn2.num_batches_tracked"]         = state_dict["layer3.13.bn2.num_batches_tracked"]
            model_state_dict["6.convs.13.bn2.running_mean"]                = state_dict["layer3.13.bn2.running_mean"]
            model_state_dict["6.convs.13.bn2.running_var"]                 = state_dict["layer3.13.bn2.running_var"]
            model_state_dict["6.convs.13.bn2.weight"]                      = state_dict["layer3.13.bn2.weight"]
            model_state_dict["6.convs.13.bn3.bias"]                        = state_dict["layer3.13.bn3.bias"]
            model_state_dict["6.convs.13.bn3.num_batches_tracked"]         = state_dict["layer3.13.bn3.num_batches_tracked"]
            model_state_dict["6.convs.13.bn3.running_mean"]                = state_dict["layer3.13.bn3.running_mean"]
            model_state_dict["6.convs.13.bn3.running_var"]                 = state_dict["layer3.13.bn3.running_var"]
            model_state_dict["6.convs.13.bn3.weight"]                      = state_dict["layer3.13.bn3.weight"]
            model_state_dict["6.convs.13.conv1.weight"]                    = state_dict["layer3.13.conv1.weight"]
            model_state_dict["6.convs.13.conv2.weight"]                    = state_dict["layer3.13.conv2.weight"]
            model_state_dict["6.convs.13.conv3.weight"]                    = state_dict["layer3.13.conv3.weight"]
            model_state_dict["6.convs.14.bn1.bias"]                        = state_dict["layer3.14.bn1.bias"]
            model_state_dict["6.convs.14.bn1.num_batches_tracked"]         = state_dict["layer3.14.bn1.num_batches_tracked"]
            model_state_dict["6.convs.14.bn1.running_mean"]                = state_dict["layer3.14.bn1.running_mean"]
            model_state_dict["6.convs.14.bn1.running_var"]                 = state_dict["layer3.14.bn1.running_var"]
            model_state_dict["6.convs.14.bn1.weight"]                      = state_dict["layer3.14.bn1.weight"]
            model_state_dict["6.convs.14.bn2.bias"]                        = state_dict["layer3.14.bn2.bias"]
            model_state_dict["6.convs.14.bn2.num_batches_tracked"]         = state_dict["layer3.14.bn2.num_batches_tracked"]
            model_state_dict["6.convs.14.bn2.running_mean"]                = state_dict["layer3.14.bn2.running_mean"]
            model_state_dict["6.convs.14.bn2.running_var"]                 = state_dict["layer3.14.bn2.running_var"]
            model_state_dict["6.convs.14.bn2.weight"]                      = state_dict["layer3.14.bn2.weight"]
            model_state_dict["6.convs.14.bn3.bias"]                        = state_dict["layer3.14.bn3.bias"]
            model_state_dict["6.convs.14.bn3.num_batches_tracked"]         = state_dict["layer3.14.bn3.num_batches_tracked"]
            model_state_dict["6.convs.14.bn3.running_mean"]                = state_dict["layer3.14.bn3.running_mean"]
            model_state_dict["6.convs.14.bn3.running_var"]                 = state_dict["layer3.14.bn3.running_var"]
            model_state_dict["6.convs.14.bn3.weight"]                      = state_dict["layer3.14.bn3.weight"]
            model_state_dict["6.convs.14.conv1.weight"]                    = state_dict["layer3.14.conv1.weight"]
            model_state_dict["6.convs.14.conv2.weight"]                    = state_dict["layer3.14.conv2.weight"]
            model_state_dict["6.convs.14.conv3.weight"]                    = state_dict["layer3.14.conv3.weight"]
            model_state_dict["6.convs.15.bn1.bias"]                        = state_dict["layer3.15.bn1.bias"]
            model_state_dict["6.convs.15.bn1.num_batches_tracked"]         = state_dict["layer3.15.bn1.num_batches_tracked"]
            model_state_dict["6.convs.15.bn1.running_mean"]                = state_dict["layer3.15.bn1.running_mean"]
            model_state_dict["6.convs.15.bn1.running_var"]                 = state_dict["layer3.15.bn1.running_var"]
            model_state_dict["6.convs.15.bn1.weight"]                      = state_dict["layer3.15.bn1.weight"]
            model_state_dict["6.convs.15.bn2.bias"]                        = state_dict["layer3.15.bn2.bias"]
            model_state_dict["6.convs.15.bn2.num_batches_tracked"]         = state_dict["layer3.15.bn2.num_batches_tracked"]
            model_state_dict["6.convs.15.bn2.running_mean"]                = state_dict["layer3.15.bn2.running_mean"]
            model_state_dict["6.convs.15.bn2.running_var"]                 = state_dict["layer3.15.bn2.running_var"]
            model_state_dict["6.convs.15.bn2.weight"]                      = state_dict["layer3.15.bn2.weight"]
            model_state_dict["6.convs.15.bn3.bias"]                        = state_dict["layer3.15.bn3.bias"]
            model_state_dict["6.convs.15.bn3.num_batches_tracked"]         = state_dict["layer3.15.bn3.num_batches_tracked"]
            model_state_dict["6.convs.15.bn3.running_mean"]                = state_dict["layer3.15.bn3.running_mean"]
            model_state_dict["6.convs.15.bn3.running_var"]                 = state_dict["layer3.15.bn3.running_var"]
            model_state_dict["6.convs.15.bn3.weight"]                      = state_dict["layer3.15.bn3.weight"]
            model_state_dict["6.convs.15.conv1.weight"]                    = state_dict["layer3.15.conv1.weight"]
            model_state_dict["6.convs.15.conv2.weight"]                    = state_dict["layer3.15.conv2.weight"]
            model_state_dict["6.convs.15.conv3.weight"]                    = state_dict["layer3.15.conv3.weight"]
            model_state_dict["6.convs.16.bn1.bias"]                        = state_dict["layer3.16.bn1.bias"]
            model_state_dict["6.convs.16.bn1.num_batches_tracked"]         = state_dict["layer3.16.bn1.num_batches_tracked"]
            model_state_dict["6.convs.16.bn1.running_mean"]                = state_dict["layer3.16.bn1.running_mean"]
            model_state_dict["6.convs.16.bn1.running_var"]                 = state_dict["layer3.16.bn1.running_var"]
            model_state_dict["6.convs.16.bn1.weight"]                      = state_dict["layer3.16.bn1.weight"]
            model_state_dict["6.convs.16.bn2.bias"]                        = state_dict["layer3.16.bn2.bias"]
            model_state_dict["6.convs.16.bn2.num_batches_tracked"]         = state_dict["layer3.16.bn2.num_batches_tracked"]
            model_state_dict["6.convs.16.bn2.running_mean"]                = state_dict["layer3.16.bn2.running_mean"]
            model_state_dict["6.convs.16.bn2.running_var"]                 = state_dict["layer3.16.bn2.running_var"]
            model_state_dict["6.convs.16.bn2.weight"]                      = state_dict["layer3.16.bn2.weight"]
            model_state_dict["6.convs.16.bn3.bias"]                        = state_dict["layer3.16.bn3.bias"]
            model_state_dict["6.convs.16.bn3.num_batches_tracked"]         = state_dict["layer3.16.bn3.num_batches_tracked"]
            model_state_dict["6.convs.16.bn3.running_mean"]                = state_dict["layer3.16.bn3.running_mean"]
            model_state_dict["6.convs.16.bn3.running_var"]                 = state_dict["layer3.16.bn3.running_var"]
            model_state_dict["6.convs.16.bn3.weight"]                      = state_dict["layer3.16.bn3.weight"]
            model_state_dict["6.convs.16.conv1.weight"]                    = state_dict["layer3.16.conv1.weight"]
            model_state_dict["6.convs.16.conv2.weight"]                    = state_dict["layer3.16.conv2.weight"]
            model_state_dict["6.convs.16.conv3.weight"]                    = state_dict["layer3.16.conv3.weight"]
            model_state_dict["6.convs.17.bn1.bias"]                        = state_dict["layer3.17.bn1.bias"]
            model_state_dict["6.convs.17.bn1.num_batches_tracked"]         = state_dict["layer3.17.bn1.num_batches_tracked"]
            model_state_dict["6.convs.17.bn1.running_mean"]                = state_dict["layer3.17.bn1.running_mean"]
            model_state_dict["6.convs.17.bn1.running_var"]                 = state_dict["layer3.17.bn1.running_var"]
            model_state_dict["6.convs.17.bn1.weight"]                      = state_dict["layer3.17.bn1.weight"]
            model_state_dict["6.convs.17.bn2.bias"]                        = state_dict["layer3.17.bn2.bias"]
            model_state_dict["6.convs.17.bn2.num_batches_tracked"]         = state_dict["layer3.17.bn2.num_batches_tracked"]
            model_state_dict["6.convs.17.bn2.running_mean"]                = state_dict["layer3.17.bn2.running_mean"]
            model_state_dict["6.convs.17.bn2.running_var"]                 = state_dict["layer3.17.bn2.running_var"]
            model_state_dict["6.convs.17.bn2.weight"]                      = state_dict["layer3.17.bn2.weight"]
            model_state_dict["6.convs.17.bn3.bias"]                        = state_dict["layer3.17.bn3.bias"]
            model_state_dict["6.convs.17.bn3.num_batches_tracked"]         = state_dict["layer3.17.bn3.num_batches_tracked"]
            model_state_dict["6.convs.17.bn3.running_mean"]                = state_dict["layer3.17.bn3.running_mean"]
            model_state_dict["6.convs.17.bn3.running_var"]                 = state_dict["layer3.17.bn3.running_var"]
            model_state_dict["6.convs.17.bn3.weight"]                      = state_dict["layer3.17.bn3.weight"]
            model_state_dict["6.convs.17.conv1.weight"]                    = state_dict["layer3.17.conv1.weight"]
            model_state_dict["6.convs.17.conv2.weight"]                    = state_dict["layer3.17.conv2.weight"]
            model_state_dict["6.convs.17.conv3.weight"]                    = state_dict["layer3.17.conv3.weight"]
            model_state_dict["6.convs.18.bn1.bias"]                        = state_dict["layer3.18.bn1.bias"]
            model_state_dict["6.convs.18.bn1.num_batches_tracked"]         = state_dict["layer3.18.bn1.num_batches_tracked"]
            model_state_dict["6.convs.18.bn1.running_mean"]                = state_dict["layer3.18.bn1.running_mean"]
            model_state_dict["6.convs.18.bn1.running_var"]                 = state_dict["layer3.18.bn1.running_var"]
            model_state_dict["6.convs.18.bn1.weight"]                      = state_dict["layer3.18.bn1.weight"]
            model_state_dict["6.convs.18.bn2.bias"]                        = state_dict["layer3.18.bn2.bias"]
            model_state_dict["6.convs.18.bn2.num_batches_tracked"]         = state_dict["layer3.18.bn2.num_batches_tracked"]
            model_state_dict["6.convs.18.bn2.running_mean"]                = state_dict["layer3.18.bn2.running_mean"]
            model_state_dict["6.convs.18.bn2.running_var"]                 = state_dict["layer3.18.bn2.running_var"]
            model_state_dict["6.convs.18.bn2.weight"]                      = state_dict["layer3.18.bn2.weight"]
            model_state_dict["6.convs.18.bn3.bias"]                        = state_dict["layer3.18.bn3.bias"]
            model_state_dict["6.convs.18.bn3.num_batches_tracked"]         = state_dict["layer3.18.bn3.num_batches_tracked"]
            model_state_dict["6.convs.18.bn3.running_mean"]                = state_dict["layer3.18.bn3.running_mean"]
            model_state_dict["6.convs.18.bn3.running_var"]                 = state_dict["layer3.18.bn3.running_var"]
            model_state_dict["6.convs.18.bn3.weight"]                      = state_dict["layer3.18.bn3.weight"]
            model_state_dict["6.convs.18.conv1.weight"]                    = state_dict["layer3.18.conv1.weight"]
            model_state_dict["6.convs.18.conv2.weight"]                    = state_dict["layer3.18.conv2.weight"]
            model_state_dict["6.convs.18.conv3.weight"]                    = state_dict["layer3.18.conv3.weight"]
            model_state_dict["6.convs.19.bn1.bias"]                        = state_dict["layer3.19.bn1.bias"]
            model_state_dict["6.convs.19.bn1.num_batches_tracked"]         = state_dict["layer3.19.bn1.num_batches_tracked"]
            model_state_dict["6.convs.19.bn1.running_mean"]                = state_dict["layer3.19.bn1.running_mean"]
            model_state_dict["6.convs.19.bn1.running_var"]                 = state_dict["layer3.19.bn1.running_var"]
            model_state_dict["6.convs.19.bn1.weight"]                      = state_dict["layer3.19.bn1.weight"]
            model_state_dict["6.convs.19.bn2.bias"]                        = state_dict["layer3.19.bn2.bias"]
            model_state_dict["6.convs.19.bn2.num_batches_tracked"]         = state_dict["layer3.19.bn2.num_batches_tracked"]
            model_state_dict["6.convs.19.bn2.running_mean"]                = state_dict["layer3.19.bn2.running_mean"]
            model_state_dict["6.convs.19.bn2.running_var"]                 = state_dict["layer3.19.bn2.running_var"]
            model_state_dict["6.convs.19.bn2.weight"]                      = state_dict["layer3.19.bn2.weight"]
            model_state_dict["6.convs.19.bn3.bias"]                        = state_dict["layer3.19.bn3.bias"]
            model_state_dict["6.convs.19.bn3.num_batches_tracked"]         = state_dict["layer3.19.bn3.num_batches_tracked"]
            model_state_dict["6.convs.19.bn3.running_mean"]                = state_dict["layer3.19.bn3.running_mean"]
            model_state_dict["6.convs.19.bn3.running_var"]                 = state_dict["layer3.19.bn3.running_var"]
            model_state_dict["6.convs.19.bn3.weight"]                      = state_dict["layer3.19.bn3.weight"]
            model_state_dict["6.convs.19.conv1.weight"]                    = state_dict["layer3.19.conv1.weight"]
            model_state_dict["6.convs.19.conv2.weight"]                    = state_dict["layer3.19.conv2.weight"]
            model_state_dict["6.convs.19.conv3.weight"]                    = state_dict["layer3.19.conv3.weight"]
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
            model_state_dict["6.convs.20.bn1.bias"]                        = state_dict["layer3.20.bn1.bias"]
            model_state_dict["6.convs.20.bn1.num_batches_tracked"]         = state_dict["layer3.20.bn1.num_batches_tracked"]
            model_state_dict["6.convs.20.bn1.running_mean"]                = state_dict["layer3.20.bn1.running_mean"]
            model_state_dict["6.convs.20.bn1.running_var"]                 = state_dict["layer3.20.bn1.running_var"]
            model_state_dict["6.convs.20.bn1.weight"]                      = state_dict["layer3.20.bn1.weight"]
            model_state_dict["6.convs.20.bn2.bias"]                        = state_dict["layer3.20.bn2.bias"]
            model_state_dict["6.convs.20.bn2.num_batches_tracked"]         = state_dict["layer3.20.bn2.num_batches_tracked"]
            model_state_dict["6.convs.20.bn2.running_mean"]                = state_dict["layer3.20.bn2.running_mean"]
            model_state_dict["6.convs.20.bn2.running_var"]                 = state_dict["layer3.20.bn2.running_var"]
            model_state_dict["6.convs.20.bn2.weight"]                      = state_dict["layer3.20.bn2.weight"]
            model_state_dict["6.convs.20.bn3.bias"]                        = state_dict["layer3.20.bn3.bias"]
            model_state_dict["6.convs.20.bn3.num_batches_tracked"]         = state_dict["layer3.20.bn3.num_batches_tracked"]
            model_state_dict["6.convs.20.bn3.running_mean"]                = state_dict["layer3.20.bn3.running_mean"]
            model_state_dict["6.convs.20.bn3.running_var"]                 = state_dict["layer3.20.bn3.running_var"]
            model_state_dict["6.convs.20.bn3.weight"]                      = state_dict["layer3.20.bn3.weight"]
            model_state_dict["6.convs.20.conv1.weight"]                    = state_dict["layer3.20.conv1.weight"]
            model_state_dict["6.convs.20.conv2.weight"]                    = state_dict["layer3.20.conv2.weight"]
            model_state_dict["6.convs.20.conv3.weight"]                    = state_dict["layer3.20.conv3.weight"]
            model_state_dict["6.convs.21.bn1.bias"]                        = state_dict["layer3.21.bn1.bias"]
            model_state_dict["6.convs.21.bn1.num_batches_tracked"]         = state_dict["layer3.21.bn1.num_batches_tracked"]
            model_state_dict["6.convs.21.bn1.running_mean"]                = state_dict["layer3.21.bn1.running_mean"]
            model_state_dict["6.convs.21.bn1.running_var"]                 = state_dict["layer3.21.bn1.running_var"]
            model_state_dict["6.convs.21.bn1.weight"]                      = state_dict["layer3.21.bn1.weight"]
            model_state_dict["6.convs.21.bn2.bias"]                        = state_dict["layer3.21.bn2.bias"]
            model_state_dict["6.convs.21.bn2.num_batches_tracked"]         = state_dict["layer3.21.bn2.num_batches_tracked"]
            model_state_dict["6.convs.21.bn2.running_mean"]                = state_dict["layer3.21.bn2.running_mean"]
            model_state_dict["6.convs.21.bn2.running_var"]                 = state_dict["layer3.21.bn2.running_var"]
            model_state_dict["6.convs.21.bn2.weight"]                      = state_dict["layer3.21.bn2.weight"]
            model_state_dict["6.convs.21.bn3.bias"]                        = state_dict["layer3.21.bn3.bias"]
            model_state_dict["6.convs.21.bn3.num_batches_tracked"]         = state_dict["layer3.21.bn3.num_batches_tracked"]
            model_state_dict["6.convs.21.bn3.running_mean"]                = state_dict["layer3.21.bn3.running_mean"]
            model_state_dict["6.convs.21.bn3.running_var"]                 = state_dict["layer3.21.bn3.running_var"]
            model_state_dict["6.convs.21.bn3.weight"]                      = state_dict["layer3.21.bn3.weight"]
            model_state_dict["6.convs.21.conv1.weight"]                    = state_dict["layer3.21.conv1.weight"]
            model_state_dict["6.convs.21.conv2.weight"]                    = state_dict["layer3.21.conv2.weight"]
            model_state_dict["6.convs.21.conv3.weight"]                    = state_dict["layer3.21.conv3.weight"]
            model_state_dict["6.convs.22.bn1.bias"]                        = state_dict["layer3.22.bn1.bias"]
            model_state_dict["6.convs.22.bn1.num_batches_tracked"]         = state_dict["layer3.22.bn1.num_batches_tracked"]
            model_state_dict["6.convs.22.bn1.running_mean"]                = state_dict["layer3.22.bn1.running_mean"]
            model_state_dict["6.convs.22.bn1.running_var"]                 = state_dict["layer3.22.bn1.running_var"]
            model_state_dict["6.convs.22.bn1.weight"]                      = state_dict["layer3.22.bn1.weight"]
            model_state_dict["6.convs.22.bn2.bias"]                        = state_dict["layer3.22.bn2.bias"]
            model_state_dict["6.convs.22.bn2.num_batches_tracked"]         = state_dict["layer3.22.bn2.num_batches_tracked"]
            model_state_dict["6.convs.22.bn2.running_mean"]                = state_dict["layer3.22.bn2.running_mean"]
            model_state_dict["6.convs.22.bn2.running_var"]                 = state_dict["layer3.22.bn2.running_var"]
            model_state_dict["6.convs.22.bn2.weight"]                      = state_dict["layer3.22.bn2.weight"]
            model_state_dict["6.convs.22.bn3.bias"]                        = state_dict["layer3.22.bn3.bias"]
            model_state_dict["6.convs.22.bn3.num_batches_tracked"]         = state_dict["layer3.22.bn3.num_batches_tracked"]
            model_state_dict["6.convs.22.bn3.running_mean"]                = state_dict["layer3.22.bn3.running_mean"]
            model_state_dict["6.convs.22.bn3.running_var"]                 = state_dict["layer3.22.bn3.running_var"]
            model_state_dict["6.convs.22.bn3.weight"]                      = state_dict["layer3.22.bn3.weight"]
            model_state_dict["6.convs.22.conv1.weight"]                    = state_dict["layer3.22.conv1.weight"]
            model_state_dict["6.convs.22.conv2.weight"]                    = state_dict["layer3.22.conv2.weight"]
            model_state_dict["6.convs.22.conv3.weight"]                    = state_dict["layer3.22.conv3.weight"]
            model_state_dict["6.convs.23.bn1.bias"]                        = state_dict["layer3.23.bn1.bias"]
            model_state_dict["6.convs.23.bn1.num_batches_tracked"]         = state_dict["layer3.23.bn1.num_batches_tracked"]
            model_state_dict["6.convs.23.bn1.running_mean"]                = state_dict["layer3.23.bn1.running_mean"]
            model_state_dict["6.convs.23.bn1.running_var"]                 = state_dict["layer3.23.bn1.running_var"]
            model_state_dict["6.convs.23.bn1.weight"]                      = state_dict["layer3.23.bn1.weight"]
            model_state_dict["6.convs.23.bn2.bias"]                        = state_dict["layer3.23.bn2.bias"]
            model_state_dict["6.convs.23.bn2.num_batches_tracked"]         = state_dict["layer3.23.bn2.num_batches_tracked"]
            model_state_dict["6.convs.23.bn2.running_mean"]                = state_dict["layer3.23.bn2.running_mean"]
            model_state_dict["6.convs.23.bn2.running_var"]                 = state_dict["layer3.23.bn2.running_var"]
            model_state_dict["6.convs.23.bn2.weight"]                      = state_dict["layer3.23.bn2.weight"]
            model_state_dict["6.convs.23.bn3.bias"]                        = state_dict["layer3.23.bn3.bias"]
            model_state_dict["6.convs.23.bn3.num_batches_tracked"]         = state_dict["layer3.23.bn3.num_batches_tracked"]
            model_state_dict["6.convs.23.bn3.running_mean"]                = state_dict["layer3.23.bn3.running_mean"]
            model_state_dict["6.convs.23.bn3.running_var"]                 = state_dict["layer3.23.bn3.running_var"]
            model_state_dict["6.convs.23.bn3.weight"]                      = state_dict["layer3.23.bn3.weight"]
            model_state_dict["6.convs.23.conv1.weight"]                    = state_dict["layer3.23.conv1.weight"]
            model_state_dict["6.convs.23.conv2.weight"]                    = state_dict["layer3.23.conv2.weight"]
            model_state_dict["6.convs.23.conv3.weight"]                    = state_dict["layer3.23.conv3.weight"]
            model_state_dict["6.convs.24.bn1.bias"]                        = state_dict["layer3.24.bn1.bias"]
            model_state_dict["6.convs.24.bn1.num_batches_tracked"]         = state_dict["layer3.24.bn1.num_batches_tracked"]
            model_state_dict["6.convs.24.bn1.running_mean"]                = state_dict["layer3.24.bn1.running_mean"]
            model_state_dict["6.convs.24.bn1.running_var"]                 = state_dict["layer3.24.bn1.running_var"]
            model_state_dict["6.convs.24.bn1.weight"]                      = state_dict["layer3.24.bn1.weight"]
            model_state_dict["6.convs.24.bn2.bias"]                        = state_dict["layer3.24.bn2.bias"]
            model_state_dict["6.convs.24.bn2.num_batches_tracked"]         = state_dict["layer3.24.bn2.num_batches_tracked"]
            model_state_dict["6.convs.24.bn2.running_mean"]                = state_dict["layer3.24.bn2.running_mean"]
            model_state_dict["6.convs.24.bn2.running_var"]                 = state_dict["layer3.24.bn2.running_var"]
            model_state_dict["6.convs.24.bn2.weight"]                      = state_dict["layer3.24.bn2.weight"]
            model_state_dict["6.convs.24.bn3.bias"]                        = state_dict["layer3.24.bn3.bias"]
            model_state_dict["6.convs.24.bn3.num_batches_tracked"]         = state_dict["layer3.24.bn3.num_batches_tracked"]
            model_state_dict["6.convs.24.bn3.running_mean"]                = state_dict["layer3.24.bn3.running_mean"]
            model_state_dict["6.convs.24.bn3.running_var"]                 = state_dict["layer3.24.bn3.running_var"]
            model_state_dict["6.convs.24.bn3.weight"]                      = state_dict["layer3.24.bn3.weight"]
            model_state_dict["6.convs.24.conv1.weight"]                    = state_dict["layer3.24.conv1.weight"]
            model_state_dict["6.convs.24.conv2.weight"]                    = state_dict["layer3.24.conv2.weight"]
            model_state_dict["6.convs.24.conv3.weight"]                    = state_dict["layer3.24.conv3.weight"]
            model_state_dict["6.convs.25.bn1.bias"]                        = state_dict["layer3.25.bn1.bias"]
            model_state_dict["6.convs.25.bn1.num_batches_tracked"]         = state_dict["layer3.25.bn1.num_batches_tracked"]
            model_state_dict["6.convs.25.bn1.running_mean"]                = state_dict["layer3.25.bn1.running_mean"]
            model_state_dict["6.convs.25.bn1.running_var"]                 = state_dict["layer3.25.bn1.running_var"]
            model_state_dict["6.convs.25.bn1.weight"]                      = state_dict["layer3.25.bn1.weight"]
            model_state_dict["6.convs.25.bn2.bias"]                        = state_dict["layer3.25.bn2.bias"]
            model_state_dict["6.convs.25.bn2.num_batches_tracked"]         = state_dict["layer3.25.bn2.num_batches_tracked"]
            model_state_dict["6.convs.25.bn2.running_mean"]                = state_dict["layer3.25.bn2.running_mean"]
            model_state_dict["6.convs.25.bn2.running_var"]                 = state_dict["layer3.25.bn2.running_var"]
            model_state_dict["6.convs.25.bn2.weight"]                      = state_dict["layer3.25.bn2.weight"]
            model_state_dict["6.convs.25.bn3.bias"]                        = state_dict["layer3.25.bn3.bias"]
            model_state_dict["6.convs.25.bn3.num_batches_tracked"]         = state_dict["layer3.25.bn3.num_batches_tracked"]
            model_state_dict["6.convs.25.bn3.running_mean"]                = state_dict["layer3.25.bn3.running_mean"]
            model_state_dict["6.convs.25.bn3.running_var"]                 = state_dict["layer3.25.bn3.running_var"]
            model_state_dict["6.convs.25.bn3.weight"]                      = state_dict["layer3.25.bn3.weight"]
            model_state_dict["6.convs.25.conv1.weight"]                    = state_dict["layer3.25.conv1.weight"]
            model_state_dict["6.convs.25.conv2.weight"]                    = state_dict["layer3.25.conv2.weight"]
            model_state_dict["6.convs.25.conv3.weight"]                    = state_dict["layer3.25.conv3.weight"]
            model_state_dict["6.convs.26.bn1.bias"]                        = state_dict["layer3.26.bn1.bias"]
            model_state_dict["6.convs.26.bn1.num_batches_tracked"]         = state_dict["layer3.26.bn1.num_batches_tracked"]
            model_state_dict["6.convs.26.bn1.running_mean"]                = state_dict["layer3.26.bn1.running_mean"]
            model_state_dict["6.convs.26.bn1.running_var"]                 = state_dict["layer3.26.bn1.running_var"]
            model_state_dict["6.convs.26.bn1.weight"]                      = state_dict["layer3.26.bn1.weight"]
            model_state_dict["6.convs.26.bn2.bias"]                        = state_dict["layer3.26.bn2.bias"]
            model_state_dict["6.convs.26.bn2.num_batches_tracked"]         = state_dict["layer3.26.bn2.num_batches_tracked"]
            model_state_dict["6.convs.26.bn2.running_mean"]                = state_dict["layer3.26.bn2.running_mean"]
            model_state_dict["6.convs.26.bn2.running_var"]                 = state_dict["layer3.26.bn2.running_var"]
            model_state_dict["6.convs.26.bn2.weight"]                      = state_dict["layer3.26.bn2.weight"]
            model_state_dict["6.convs.26.bn3.bias"]                        = state_dict["layer3.26.bn3.bias"]
            model_state_dict["6.convs.26.bn3.num_batches_tracked"]         = state_dict["layer3.26.bn3.num_batches_tracked"]
            model_state_dict["6.convs.26.bn3.running_mean"]                = state_dict["layer3.26.bn3.running_mean"]
            model_state_dict["6.convs.26.bn3.running_var"]                 = state_dict["layer3.26.bn3.running_var"]
            model_state_dict["6.convs.26.bn3.weight"]                      = state_dict["layer3.26.bn3.weight"]
            model_state_dict["6.convs.26.conv1.weight"]                    = state_dict["layer3.26.conv1.weight"]
            model_state_dict["6.convs.26.conv2.weight"]                    = state_dict["layer3.26.conv2.weight"]
            model_state_dict["6.convs.26.conv3.weight"]                    = state_dict["layer3.26.conv3.weight"]
            model_state_dict["6.convs.27.bn1.bias"]                        = state_dict["layer3.27.bn1.bias"]
            model_state_dict["6.convs.27.bn1.num_batches_tracked"]         = state_dict["layer3.27.bn1.num_batches_tracked"]
            model_state_dict["6.convs.27.bn1.running_mean"]                = state_dict["layer3.27.bn1.running_mean"]
            model_state_dict["6.convs.27.bn1.running_var"]                 = state_dict["layer3.27.bn1.running_var"]
            model_state_dict["6.convs.27.bn1.weight"]                      = state_dict["layer3.27.bn1.weight"]
            model_state_dict["6.convs.27.bn2.bias"]                        = state_dict["layer3.27.bn2.bias"]
            model_state_dict["6.convs.27.bn2.num_batches_tracked"]         = state_dict["layer3.27.bn2.num_batches_tracked"]
            model_state_dict["6.convs.27.bn2.running_mean"]                = state_dict["layer3.27.bn2.running_mean"]
            model_state_dict["6.convs.27.bn2.running_var"]                 = state_dict["layer3.27.bn2.running_var"]
            model_state_dict["6.convs.27.bn2.weight"]                      = state_dict["layer3.27.bn2.weight"]
            model_state_dict["6.convs.27.bn3.bias"]                        = state_dict["layer3.27.bn3.bias"]
            model_state_dict["6.convs.27.bn3.num_batches_tracked"]         = state_dict["layer3.27.bn3.num_batches_tracked"]
            model_state_dict["6.convs.27.bn3.running_mean"]                = state_dict["layer3.27.bn3.running_mean"]
            model_state_dict["6.convs.27.bn3.running_var"]                 = state_dict["layer3.27.bn3.running_var"]
            model_state_dict["6.convs.27.bn3.weight"]                      = state_dict["layer3.27.bn3.weight"]
            model_state_dict["6.convs.27.conv1.weight"]                    = state_dict["layer3.27.conv1.weight"]
            model_state_dict["6.convs.27.conv2.weight"]                    = state_dict["layer3.27.conv2.weight"]
            model_state_dict["6.convs.27.conv3.weight"]                    = state_dict["layer3.27.conv3.weight"]
            model_state_dict["6.convs.28.bn1.bias"]                        = state_dict["layer3.28.bn1.bias"]
            model_state_dict["6.convs.28.bn1.num_batches_tracked"]         = state_dict["layer3.28.bn1.num_batches_tracked"]
            model_state_dict["6.convs.28.bn1.running_mean"]                = state_dict["layer3.28.bn1.running_mean"]
            model_state_dict["6.convs.28.bn1.running_var"]                 = state_dict["layer3.28.bn1.running_var"]
            model_state_dict["6.convs.28.bn1.weight"]                      = state_dict["layer3.28.bn1.weight"]
            model_state_dict["6.convs.28.bn2.bias"]                        = state_dict["layer3.28.bn2.bias"]
            model_state_dict["6.convs.28.bn2.num_batches_tracked"]         = state_dict["layer3.28.bn2.num_batches_tracked"]
            model_state_dict["6.convs.28.bn2.running_mean"]                = state_dict["layer3.28.bn2.running_mean"]
            model_state_dict["6.convs.28.bn2.running_var"]                 = state_dict["layer3.28.bn2.running_var"]
            model_state_dict["6.convs.28.bn2.weight"]                      = state_dict["layer3.28.bn2.weight"]
            model_state_dict["6.convs.28.bn3.bias"]                        = state_dict["layer3.28.bn3.bias"]
            model_state_dict["6.convs.28.bn3.num_batches_tracked"]         = state_dict["layer3.28.bn3.num_batches_tracked"]
            model_state_dict["6.convs.28.bn3.running_mean"]                = state_dict["layer3.28.bn3.running_mean"]
            model_state_dict["6.convs.28.bn3.running_var"]                 = state_dict["layer3.28.bn3.running_var"]
            model_state_dict["6.convs.28.bn3.weight"]                      = state_dict["layer3.28.bn3.weight"]
            model_state_dict["6.convs.28.conv1.weight"]                    = state_dict["layer3.28.conv1.weight"]
            model_state_dict["6.convs.28.conv2.weight"]                    = state_dict["layer3.28.conv2.weight"]
            model_state_dict["6.convs.28.conv3.weight"]                    = state_dict["layer3.28.conv3.weight"]
            model_state_dict["6.convs.29.bn1.bias"]                        = state_dict["layer3.29.bn1.bias"]
            model_state_dict["6.convs.29.bn1.num_batches_tracked"]         = state_dict["layer3.29.bn1.num_batches_tracked"]
            model_state_dict["6.convs.29.bn1.running_mean"]                = state_dict["layer3.29.bn1.running_mean"]
            model_state_dict["6.convs.29.bn1.running_var"]                 = state_dict["layer3.29.bn1.running_var"]
            model_state_dict["6.convs.29.bn1.weight"]                      = state_dict["layer3.29.bn1.weight"]
            model_state_dict["6.convs.29.bn2.bias"]                        = state_dict["layer3.29.bn2.bias"]
            model_state_dict["6.convs.29.bn2.num_batches_tracked"]         = state_dict["layer3.29.bn2.num_batches_tracked"]
            model_state_dict["6.convs.29.bn2.running_mean"]                = state_dict["layer3.29.bn2.running_mean"]
            model_state_dict["6.convs.29.bn2.running_var"]                 = state_dict["layer3.29.bn2.running_var"]
            model_state_dict["6.convs.29.bn2.weight"]                      = state_dict["layer3.29.bn2.weight"]
            model_state_dict["6.convs.29.bn3.bias"]                        = state_dict["layer3.29.bn3.bias"]
            model_state_dict["6.convs.29.bn3.num_batches_tracked"]         = state_dict["layer3.29.bn3.num_batches_tracked"]
            model_state_dict["6.convs.29.bn3.running_mean"]                = state_dict["layer3.29.bn3.running_mean"]
            model_state_dict["6.convs.29.bn3.running_var"]                 = state_dict["layer3.29.bn3.running_var"]
            model_state_dict["6.convs.29.bn3.weight"]                      = state_dict["layer3.29.bn3.weight"]
            model_state_dict["6.convs.29.conv1.weight"]                    = state_dict["layer3.29.conv1.weight"]
            model_state_dict["6.convs.29.conv2.weight"]                    = state_dict["layer3.29.conv2.weight"]
            model_state_dict["6.convs.29.conv3.weight"]                    = state_dict["layer3.29.conv3.weight"]
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
            model_state_dict["6.convs.30.bn1.bias"]                        = state_dict["layer3.30.bn1.bias"]
            model_state_dict["6.convs.30.bn1.num_batches_tracked"]         = state_dict["layer3.30.bn1.num_batches_tracked"]
            model_state_dict["6.convs.30.bn1.running_mean"]                = state_dict["layer3.30.bn1.running_mean"]
            model_state_dict["6.convs.30.bn1.running_var"]                 = state_dict["layer3.30.bn1.running_var"]
            model_state_dict["6.convs.30.bn1.weight"]                      = state_dict["layer3.30.bn1.weight"]
            model_state_dict["6.convs.30.bn2.bias"]                        = state_dict["layer3.30.bn2.bias"]
            model_state_dict["6.convs.30.bn2.num_batches_tracked"]         = state_dict["layer3.30.bn2.num_batches_tracked"]
            model_state_dict["6.convs.30.bn2.running_mean"]                = state_dict["layer3.30.bn2.running_mean"]
            model_state_dict["6.convs.30.bn2.running_var"]                 = state_dict["layer3.30.bn2.running_var"]
            model_state_dict["6.convs.30.bn2.weight"]                      = state_dict["layer3.30.bn2.weight"]
            model_state_dict["6.convs.30.bn3.bias"]                        = state_dict["layer3.30.bn3.bias"]
            model_state_dict["6.convs.30.bn3.num_batches_tracked"]         = state_dict["layer3.30.bn3.num_batches_tracked"]
            model_state_dict["6.convs.30.bn3.running_mean"]                = state_dict["layer3.30.bn3.running_mean"]
            model_state_dict["6.convs.30.bn3.running_var"]                 = state_dict["layer3.30.bn3.running_var"]
            model_state_dict["6.convs.30.bn3.weight"]                      = state_dict["layer3.30.bn3.weight"]
            model_state_dict["6.convs.30.conv1.weight"]                    = state_dict["layer3.30.conv1.weight"]
            model_state_dict["6.convs.30.conv2.weight"]                    = state_dict["layer3.30.conv2.weight"]
            model_state_dict["6.convs.30.conv3.weight"]                    = state_dict["layer3.30.conv3.weight"]
            model_state_dict["6.convs.31.bn1.bias"]                        = state_dict["layer3.31.bn1.bias"]
            model_state_dict["6.convs.31.bn1.num_batches_tracked"]         = state_dict["layer3.31.bn1.num_batches_tracked"]
            model_state_dict["6.convs.31.bn1.running_mean"]                = state_dict["layer3.31.bn1.running_mean"]
            model_state_dict["6.convs.31.bn1.running_var"]                 = state_dict["layer3.31.bn1.running_var"]
            model_state_dict["6.convs.31.bn1.weight"]                      = state_dict["layer3.31.bn1.weight"]
            model_state_dict["6.convs.31.bn2.bias"]                        = state_dict["layer3.31.bn2.bias"]
            model_state_dict["6.convs.31.bn2.num_batches_tracked"]         = state_dict["layer3.31.bn2.num_batches_tracked"]
            model_state_dict["6.convs.31.bn2.running_mean"]                = state_dict["layer3.31.bn2.running_mean"]
            model_state_dict["6.convs.31.bn2.running_var"]                 = state_dict["layer3.31.bn2.running_var"]
            model_state_dict["6.convs.31.bn2.weight"]                      = state_dict["layer3.31.bn2.weight"]
            model_state_dict["6.convs.31.bn3.bias"]                        = state_dict["layer3.31.bn3.bias"]
            model_state_dict["6.convs.31.bn3.num_batches_tracked"]         = state_dict["layer3.31.bn3.num_batches_tracked"]
            model_state_dict["6.convs.31.bn3.running_mean"]                = state_dict["layer3.31.bn3.running_mean"]
            model_state_dict["6.convs.31.bn3.running_var"]                 = state_dict["layer3.31.bn3.running_var"]
            model_state_dict["6.convs.31.bn3.weight"]                      = state_dict["layer3.31.bn3.weight"]
            model_state_dict["6.convs.31.conv1.weight"]                    = state_dict["layer3.31.conv1.weight"]
            model_state_dict["6.convs.31.conv2.weight"]                    = state_dict["layer3.31.conv2.weight"]
            model_state_dict["6.convs.31.conv3.weight"]                    = state_dict["layer3.31.conv3.weight"]
            model_state_dict["6.convs.32.bn1.bias"]                        = state_dict["layer3.32.bn1.bias"]
            model_state_dict["6.convs.32.bn1.num_batches_tracked"]         = state_dict["layer3.32.bn1.num_batches_tracked"]
            model_state_dict["6.convs.32.bn1.running_mean"]                = state_dict["layer3.32.bn1.running_mean"]
            model_state_dict["6.convs.32.bn1.running_var"]                 = state_dict["layer3.32.bn1.running_var"]
            model_state_dict["6.convs.32.bn1.weight"]                      = state_dict["layer3.32.bn1.weight"]
            model_state_dict["6.convs.32.bn2.bias"]                        = state_dict["layer3.32.bn2.bias"]
            model_state_dict["6.convs.32.bn2.num_batches_tracked"]         = state_dict["layer3.32.bn2.num_batches_tracked"]
            model_state_dict["6.convs.32.bn2.running_mean"]                = state_dict["layer3.32.bn2.running_mean"]
            model_state_dict["6.convs.32.bn2.running_var"]                 = state_dict["layer3.32.bn2.running_var"]
            model_state_dict["6.convs.32.bn2.weight"]                      = state_dict["layer3.32.bn2.weight"]
            model_state_dict["6.convs.32.bn3.bias"]                        = state_dict["layer3.32.bn3.bias"]
            model_state_dict["6.convs.32.bn3.num_batches_tracked"]         = state_dict["layer3.32.bn3.num_batches_tracked"]
            model_state_dict["6.convs.32.bn3.running_mean"]                = state_dict["layer3.32.bn3.running_mean"]
            model_state_dict["6.convs.32.bn3.running_var"]                 = state_dict["layer3.32.bn3.running_var"]
            model_state_dict["6.convs.32.bn3.weight"]                      = state_dict["layer3.32.bn3.weight"]
            model_state_dict["6.convs.32.conv1.weight"]                    = state_dict["layer3.32.conv1.weight"]
            model_state_dict["6.convs.32.conv2.weight"]                    = state_dict["layer3.32.conv2.weight"]
            model_state_dict["6.convs.32.conv3.weight"]                    = state_dict["layer3.32.conv3.weight"]
            model_state_dict["6.convs.33.bn1.bias"]                        = state_dict["layer3.33.bn1.bias"]
            model_state_dict["6.convs.33.bn1.num_batches_tracked"]         = state_dict["layer3.33.bn1.num_batches_tracked"]
            model_state_dict["6.convs.33.bn1.running_mean"]                = state_dict["layer3.33.bn1.running_mean"]
            model_state_dict["6.convs.33.bn1.running_var"]                 = state_dict["layer3.33.bn1.running_var"]
            model_state_dict["6.convs.33.bn1.weight"]                      = state_dict["layer3.33.bn1.weight"]
            model_state_dict["6.convs.33.bn2.bias"]                        = state_dict["layer3.33.bn2.bias"]
            model_state_dict["6.convs.33.bn2.num_batches_tracked"]         = state_dict["layer3.33.bn2.num_batches_tracked"]
            model_state_dict["6.convs.33.bn2.running_mean"]                = state_dict["layer3.33.bn2.running_mean"]
            model_state_dict["6.convs.33.bn2.running_var"]                 = state_dict["layer3.33.bn2.running_var"]
            model_state_dict["6.convs.33.bn2.weight"]                      = state_dict["layer3.33.bn2.weight"]
            model_state_dict["6.convs.33.bn3.bias"]                        = state_dict["layer3.33.bn3.bias"]
            model_state_dict["6.convs.33.bn3.num_batches_tracked"]         = state_dict["layer3.33.bn3.num_batches_tracked"]
            model_state_dict["6.convs.33.bn3.running_mean"]                = state_dict["layer3.33.bn3.running_mean"]
            model_state_dict["6.convs.33.bn3.running_var"]                 = state_dict["layer3.33.bn3.running_var"]
            model_state_dict["6.convs.33.bn3.weight"]                      = state_dict["layer3.33.bn3.weight"]
            model_state_dict["6.convs.33.conv1.weight"]                    = state_dict["layer3.33.conv1.weight"]
            model_state_dict["6.convs.33.conv2.weight"]                    = state_dict["layer3.33.conv2.weight"]
            model_state_dict["6.convs.33.conv3.weight"]                    = state_dict["layer3.33.conv3.weight"]
            model_state_dict["6.convs.34.bn1.bias"]                        = state_dict["layer3.34.bn1.bias"]
            model_state_dict["6.convs.34.bn1.num_batches_tracked"]         = state_dict["layer3.34.bn1.num_batches_tracked"]
            model_state_dict["6.convs.34.bn1.running_mean"]                = state_dict["layer3.34.bn1.running_mean"]
            model_state_dict["6.convs.34.bn1.running_var"]                 = state_dict["layer3.34.bn1.running_var"]
            model_state_dict["6.convs.34.bn1.weight"]                      = state_dict["layer3.34.bn1.weight"]
            model_state_dict["6.convs.34.bn2.bias"]                        = state_dict["layer3.34.bn2.bias"]
            model_state_dict["6.convs.34.bn2.num_batches_tracked"]         = state_dict["layer3.34.bn2.num_batches_tracked"]
            model_state_dict["6.convs.34.bn2.running_mean"]                = state_dict["layer3.34.bn2.running_mean"]
            model_state_dict["6.convs.34.bn2.running_var"]                 = state_dict["layer3.34.bn2.running_var"]
            model_state_dict["6.convs.34.bn2.weight"]                      = state_dict["layer3.34.bn2.weight"]
            model_state_dict["6.convs.34.bn3.bias"]                        = state_dict["layer3.34.bn3.bias"]
            model_state_dict["6.convs.34.bn3.num_batches_tracked"]         = state_dict["layer3.34.bn3.num_batches_tracked"]
            model_state_dict["6.convs.34.bn3.running_mean"]                = state_dict["layer3.34.bn3.running_mean"]
            model_state_dict["6.convs.34.bn3.running_var"]                 = state_dict["layer3.34.bn3.running_var"]
            model_state_dict["6.convs.34.bn3.weight"]                      = state_dict["layer3.34.bn3.weight"]
            model_state_dict["6.convs.34.conv1.weight"]                    = state_dict["layer3.34.conv1.weight"]
            model_state_dict["6.convs.34.conv2.weight"]                    = state_dict["layer3.34.conv2.weight"]
            model_state_dict["6.convs.34.conv3.weight"]                    = state_dict["layer3.34.conv3.weight"]
            model_state_dict["6.convs.35.bn1.bias"]                        = state_dict["layer3.35.bn1.bias"]
            model_state_dict["6.convs.35.bn1.num_batches_tracked"]         = state_dict["layer3.35.bn1.num_batches_tracked"]
            model_state_dict["6.convs.35.bn1.running_mean"]                = state_dict["layer3.35.bn1.running_mean"]
            model_state_dict["6.convs.35.bn1.running_var"]                 = state_dict["layer3.35.bn1.running_var"]
            model_state_dict["6.convs.35.bn1.weight"]                      = state_dict["layer3.35.bn1.weight"]
            model_state_dict["6.convs.35.bn2.bias"]                        = state_dict["layer3.35.bn2.bias"]
            model_state_dict["6.convs.35.bn2.num_batches_tracked"]         = state_dict["layer3.35.bn2.num_batches_tracked"]
            model_state_dict["6.convs.35.bn2.running_mean"]                = state_dict["layer3.35.bn2.running_mean"]
            model_state_dict["6.convs.35.bn2.running_var"]                 = state_dict["layer3.35.bn2.running_var"]
            model_state_dict["6.convs.35.bn2.weight"]                      = state_dict["layer3.35.bn2.weight"]
            model_state_dict["6.convs.35.bn3.bias"]                        = state_dict["layer3.35.bn3.bias"]
            model_state_dict["6.convs.35.bn3.num_batches_tracked"]         = state_dict["layer3.35.bn3.num_batches_tracked"]
            model_state_dict["6.convs.35.bn3.running_mean"]                = state_dict["layer3.35.bn3.running_mean"]
            model_state_dict["6.convs.35.bn3.running_var"]                 = state_dict["layer3.35.bn3.running_var"]
            model_state_dict["6.convs.35.bn3.weight"]                      = state_dict["layer3.35.bn3.weight"]
            model_state_dict["6.convs.35.conv1.weight"]                    = state_dict["layer3.35.conv1.weight"]
            model_state_dict["6.convs.35.conv2.weight"]                    = state_dict["layer3.35.conv2.weight"]
            model_state_dict["6.convs.35.conv3.weight"]                    = state_dict["layer3.35.conv3.weight"]
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
            model_state_dict["6.convs.6.bn1.bias"]                         = state_dict["layer3.6.bn1.bias"]
            model_state_dict["6.convs.6.bn1.num_batches_tracked"]          = state_dict["layer3.6.bn1.num_batches_tracked"]
            model_state_dict["6.convs.6.bn1.running_mean"]                 = state_dict["layer3.6.bn1.running_mean"]
            model_state_dict["6.convs.6.bn1.running_var"]                  = state_dict["layer3.6.bn1.running_var"]
            model_state_dict["6.convs.6.bn1.weight"]                       = state_dict["layer3.6.bn1.weight"]
            model_state_dict["6.convs.6.bn2.bias"]                         = state_dict["layer3.6.bn2.bias"]
            model_state_dict["6.convs.6.bn2.num_batches_tracked"]          = state_dict["layer3.6.bn2.num_batches_tracked"]
            model_state_dict["6.convs.6.bn2.running_mean"]                 = state_dict["layer3.6.bn2.running_mean"]
            model_state_dict["6.convs.6.bn2.running_var"]                  = state_dict["layer3.6.bn2.running_var"]
            model_state_dict["6.convs.6.bn2.weight"]                       = state_dict["layer3.6.bn2.weight"]
            model_state_dict["6.convs.6.bn3.bias"]                         = state_dict["layer3.6.bn3.bias"]
            model_state_dict["6.convs.6.bn3.num_batches_tracked"]          = state_dict["layer3.6.bn3.num_batches_tracked"]
            model_state_dict["6.convs.6.bn3.running_mean"]                 = state_dict["layer3.6.bn3.running_mean"]
            model_state_dict["6.convs.6.bn3.running_var"]                  = state_dict["layer3.6.bn3.running_var"]
            model_state_dict["6.convs.6.bn3.weight"]                       = state_dict["layer3.6.bn3.weight"]
            model_state_dict["6.convs.6.conv1.weight"]                     = state_dict["layer3.6.conv1.weight"]
            model_state_dict["6.convs.6.conv2.weight"]                     = state_dict["layer3.6.conv2.weight"]
            model_state_dict["6.convs.6.conv3.weight"]                     = state_dict["layer3.6.conv3.weight"]
            model_state_dict["6.convs.7.bn1.bias"]                         = state_dict["layer3.7.bn1.bias"]
            model_state_dict["6.convs.7.bn1.num_batches_tracked"]          = state_dict["layer3.7.bn1.num_batches_tracked"]
            model_state_dict["6.convs.7.bn1.running_mean"]                 = state_dict["layer3.7.bn1.running_mean"]
            model_state_dict["6.convs.7.bn1.running_var"]                  = state_dict["layer3.7.bn1.running_var"]
            model_state_dict["6.convs.7.bn1.weight"]                       = state_dict["layer3.7.bn1.weight"]
            model_state_dict["6.convs.7.bn2.bias"]                         = state_dict["layer3.7.bn2.bias"]
            model_state_dict["6.convs.7.bn2.num_batches_tracked"]          = state_dict["layer3.7.bn2.num_batches_tracked"]
            model_state_dict["6.convs.7.bn2.running_mean"]                 = state_dict["layer3.7.bn2.running_mean"]
            model_state_dict["6.convs.7.bn2.running_var"]                  = state_dict["layer3.7.bn2.running_var"]
            model_state_dict["6.convs.7.bn2.weight"]                       = state_dict["layer3.7.bn2.weight"]
            model_state_dict["6.convs.7.bn3.bias"]                         = state_dict["layer3.7.bn3.bias"]
            model_state_dict["6.convs.7.bn3.num_batches_tracked"]          = state_dict["layer3.7.bn3.num_batches_tracked"]
            model_state_dict["6.convs.7.bn3.running_mean"]                 = state_dict["layer3.7.bn3.running_mean"]
            model_state_dict["6.convs.7.bn3.running_var"]                  = state_dict["layer3.7.bn3.running_var"]
            model_state_dict["6.convs.7.bn3.weight"]                       = state_dict["layer3.7.bn3.weight"]
            model_state_dict["6.convs.7.conv1.weight"]                     = state_dict["layer3.7.conv1.weight"]
            model_state_dict["6.convs.7.conv2.weight"]                     = state_dict["layer3.7.conv2.weight"]
            model_state_dict["6.convs.7.conv3.weight"]                     = state_dict["layer3.7.conv3.weight"]
            model_state_dict["6.convs.8.bn1.bias"]                         = state_dict["layer3.8.bn1.bias"]
            model_state_dict["6.convs.8.bn1.num_batches_tracked"]          = state_dict["layer3.8.bn1.num_batches_tracked"]
            model_state_dict["6.convs.8.bn1.running_mean"]                 = state_dict["layer3.8.bn1.running_mean"]
            model_state_dict["6.convs.8.bn1.running_var"]                  = state_dict["layer3.8.bn1.running_var"]
            model_state_dict["6.convs.8.bn1.weight"]                       = state_dict["layer3.8.bn1.weight"]
            model_state_dict["6.convs.8.bn2.bias"]                         = state_dict["layer3.8.bn2.bias"]
            model_state_dict["6.convs.8.bn2.num_batches_tracked"]          = state_dict["layer3.8.bn2.num_batches_tracked"]
            model_state_dict["6.convs.8.bn2.running_mean"]                 = state_dict["layer3.8.bn2.running_mean"]
            model_state_dict["6.convs.8.bn2.running_var"]                  = state_dict["layer3.8.bn2.running_var"]
            model_state_dict["6.convs.8.bn2.weight"]                       = state_dict["layer3.8.bn2.weight"]
            model_state_dict["6.convs.8.bn3.bias"]                         = state_dict["layer3.8.bn3.bias"]
            model_state_dict["6.convs.8.bn3.num_batches_tracked"]          = state_dict["layer3.8.bn3.num_batches_tracked"]
            model_state_dict["6.convs.8.bn3.running_mean"]                 = state_dict["layer3.8.bn3.running_mean"]
            model_state_dict["6.convs.8.bn3.running_var"]                  = state_dict["layer3.8.bn3.running_var"]
            model_state_dict["6.convs.8.bn3.weight"]                       = state_dict["layer3.8.bn3.weight"]
            model_state_dict["6.convs.8.conv1.weight"]                     = state_dict["layer3.8.conv1.weight"]
            model_state_dict["6.convs.8.conv2.weight"]                     = state_dict["layer3.8.conv2.weight"]
            model_state_dict["6.convs.8.conv3.weight"]                     = state_dict["layer3.8.conv3.weight"]
            model_state_dict["6.convs.9.bn1.bias"]                         = state_dict["layer3.9.bn1.bias"]
            model_state_dict["6.convs.9.bn1.num_batches_tracked"]          = state_dict["layer3.9.bn1.num_batches_tracked"]
            model_state_dict["6.convs.9.bn1.running_mean"]                 = state_dict["layer3.9.bn1.running_mean"]
            model_state_dict["6.convs.9.bn1.running_var"]                  = state_dict["layer3.9.bn1.running_var"]
            model_state_dict["6.convs.9.bn1.weight"]                       = state_dict["layer3.9.bn1.weight"]
            model_state_dict["6.convs.9.bn2.bias"]                         = state_dict["layer3.9.bn2.bias"]
            model_state_dict["6.convs.9.bn2.num_batches_tracked"]          = state_dict["layer3.9.bn2.num_batches_tracked"]
            model_state_dict["6.convs.9.bn2.running_mean"]                 = state_dict["layer3.9.bn2.running_mean"]
            model_state_dict["6.convs.9.bn2.running_var"]                  = state_dict["layer3.9.bn2.running_var"]
            model_state_dict["6.convs.9.bn2.weight"]                       = state_dict["layer3.9.bn2.weight"]
            model_state_dict["6.convs.9.bn3.bias"]                         = state_dict["layer3.9.bn3.bias"]
            model_state_dict["6.convs.9.bn3.num_batches_tracked"]          = state_dict["layer3.9.bn3.num_batches_tracked"]
            model_state_dict["6.convs.9.bn3.running_mean"]                 = state_dict["layer3.9.bn3.running_mean"]
            model_state_dict["6.convs.9.bn3.running_var"]                  = state_dict["layer3.9.bn3.running_var"]
            model_state_dict["6.convs.9.bn3.weight"]                       = state_dict["layer3.9.bn3.weight"]
            model_state_dict["6.convs.9.conv1.weight"]                     = state_dict["layer3.9.conv1.weight"]
            model_state_dict["6.convs.9.conv2.weight"]                     = state_dict["layer3.9.conv2.weight"]
            model_state_dict["6.convs.9.conv3.weight"]                     = state_dict["layer3.9.conv3.weight"]
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
