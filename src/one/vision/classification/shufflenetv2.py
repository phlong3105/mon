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
    "shufflenet_v2-x0.5": {
        "channels": 3,
        "backbone": [
            # [from, number, module,           args(out_channels, ...)]
            # Stage 1
            [-1,     1,      Conv2d,           [24, 3, 2, 1, 1, 1, False]],    # 0
            [-1,     1,      BatchNorm2d,      [24]],                          # 1
            [-1,     1,      ReLU,             [True]],                        # 2
            [-1,     1,      MaxPool2d,        [3, 2, 1]],                     # 3
            # Stage 2                                                          
            [-1,     1,      InvertedResidual, [48,  2]],                      # 4
            [-1,     1,      InvertedResidual, [48,  1]],                      # 5
            [-1,     1,      InvertedResidual, [48,  1]],                      # 6
            [-1,     1,      InvertedResidual, [48,  1]],                      # 7
            # Stage 3
            [-1,     1,      InvertedResidual, [96,  2]],                      # 8
            [-1,     1,      InvertedResidual, [96,  1]],                      # 9
            [-1,     1,      InvertedResidual, [96,  1]],                      # 10
            [-1,     1,      InvertedResidual, [96,  1]],                      # 11
            [-1,     1,      InvertedResidual, [96,  1]],                      # 12
            [-1,     1,      InvertedResidual, [96,  1]],                      # 13
            [-1,     1,      InvertedResidual, [96,  1]],                      # 14
            [-1,     1,      InvertedResidual, [96,  1]],                      # 15
            # Stage 4                                                          
            [-1,     1,      InvertedResidual, [192, 2]],                      # 16
            [-1,     1,      InvertedResidual, [192, 1]],                      # 17
            [-1,     1,      InvertedResidual, [192, 1]],                      # 18
            [-1,     1,      InvertedResidual, [192, 1]],                      # 19
            # Stage 5
            [-1,     1,      Conv2d,           [1024, 1, 1, 0, 1, 1, False]],  # 20
            [-1,     1,      BatchNorm2d,      [1024]],                        # 21
            [-1,     1,      ReLU,             [True]],                        # 22
        ],
        "head": [
            [-1,     1,      ShuffleNetV2Classifier, [1024]],                  # 23
        ]
    },
    "shufflenet_v2-x1.0": {
        "channels": 3,
        "backbone": [
            # [from, number, module,           args(out_channels, ...)]
            # Stage 1
            [-1,     1,      Conv2d,           [24, 3, 2, 1, 1, 1, False]],    # 0
            [-1,     1,      BatchNorm2d,      [24]],                          # 1
            [-1,     1,      ReLU,             [True]],                        # 2
            [-1,     1,      MaxPool2d,        [3, 2, 1]],                     # 3
            # Stage 2
            [-1,     1,      InvertedResidual, [116, 2]],                      # 4
            [-1,     1,      InvertedResidual, [116, 1]],                      # 5
            [-1,     1,      InvertedResidual, [116, 1]],                      # 6
            [-1,     1,      InvertedResidual, [116, 1]],                      # 7
            # Stage 3
            [-1,     1,      InvertedResidual, [232, 2]],                      # 8
            [-1,     1,      InvertedResidual, [232, 1]],                      # 9
            [-1,     1,      InvertedResidual, [232, 1]],                      # 10
            [-1,     1,      InvertedResidual, [232, 1]],                      # 11
            [-1,     1,      InvertedResidual, [232, 1]],                      # 12
            [-1,     1,      InvertedResidual, [232, 1]],                      # 13
            [-1,     1,      InvertedResidual, [232, 1]],                      # 14
            [-1,     1,      InvertedResidual, [232, 1]],                      # 15
            # Stage 4
            [-1,     1,      InvertedResidual, [464, 2]],                      # 16
            [-1,     1,      InvertedResidual, [464, 1]],                      # 17
            [-1,     1,      InvertedResidual, [464, 1]],                      # 18
            [-1,     1,      InvertedResidual, [464, 1]],                      # 19
            # Stage 5
            [-1,     1,      Conv2d,           [1024, 1, 1, 0, 1, 1, False]],  # 20
            [-1,     1,      BatchNorm2d,      [1024]],                        # 21
            [-1,     1,      ReLU,             [True]],                        # 22
        ],
        "head": [
            [-1,     1,      ShuffleNetV2Classifier, [1024]],                  # 23
        ]
    },
    "shufflenet_v2-x1.5": {
        "channels": 3,
        "backbone": [
            # [from, number, module,           args(out_channels, ...)]
            # Stage 1
            [-1,     1,      Conv2d,           [24, 3, 2, 1, 1, 1, False]],    # 0
            [-1,     1,      BatchNorm2d,      [24]],                          # 1
            [-1,     1,      ReLU,             [True]],                        # 2
            [-1,     1,      MaxPool2d,        [3, 2, 1]],                     # 3
            # Stage 2
            [-1,     1,      InvertedResidual, [176, 2]],                      # 4
            [-1,     1,      InvertedResidual, [176, 1]],                      # 5
            [-1,     1,      InvertedResidual, [176, 1]],                      # 6
            [-1,     1,      InvertedResidual, [176, 1]],                      # 7
            # Stage 3
            [-1,     1,      InvertedResidual, [352, 2]],                      # 8
            [-1,     1,      InvertedResidual, [352, 1]],                      # 9
            [-1,     1,      InvertedResidual, [352, 1]],                      # 10
            [-1,     1,      InvertedResidual, [352, 1]],                      # 11
            [-1,     1,      InvertedResidual, [352, 1]],                      # 12
            [-1,     1,      InvertedResidual, [352, 1]],                      # 13
            [-1,     1,      InvertedResidual, [352, 1]],                      # 14
            [-1,     1,      InvertedResidual, [352, 1]],                      # 15
            # Stage 4
            [-1,     1,      InvertedResidual, [704, 2]],                      # 16
            [-1,     1,      InvertedResidual, [704, 1]],                      # 17
            [-1,     1,      InvertedResidual, [704, 1]],                      # 18
            [-1,     1,      InvertedResidual, [704, 1]],                      # 19
            # Stage 5
            [-1,     1,      Conv2d,           [1024, 1, 1, 0, 1, 1, False]],  # 20
            [-1,     1,      BatchNorm2d,      [1024]],                        # 21
            [-1,     1,      ReLU,             [True]],                        # 22
        ],
        "head": [
            [-1,     1,      ShuffleNetV2Classifier, [1024]],                  # 23
        ]
    },
    "shufflenet_v2-x2.0": {
        "channels": 3,
        "backbone": [
            # [from, number, module,           args(out_channels, ...)]
            # Stage 1
            [-1,     1,      Conv2d,           [24, 3, 2, 1, 1, 1, False]],    # 0
            [-1,     1,      BatchNorm2d,      [24]],                          # 1
            [-1,     1,      ReLU,             [True]],                        # 2
            [-1,     1,      MaxPool2d,        [3, 2, 1]],                     # 3
            # Stage 2
            [-1,     1,      InvertedResidual, [244, 2]],                      # 4
            [-1,     1,      InvertedResidual, [244, 1]],                      # 5
            [-1,     1,      InvertedResidual, [244, 1]],                      # 6
            [-1,     1,      InvertedResidual, [244, 1]],                      # 7
            # Stage 3
            [-1,     1,      InvertedResidual, [488, 2]],                      # 8
            [-1,     1,      InvertedResidual, [488, 1]],                      # 9
            [-1,     1,      InvertedResidual, [488, 1]],                      # 10
            [-1,     1,      InvertedResidual, [488, 1]],                      # 11
            [-1,     1,      InvertedResidual, [488, 1]],                      # 12
            [-1,     1,      InvertedResidual, [488, 1]],                      # 13
            [-1,     1,      InvertedResidual, [488, 1]],                      # 14
            [-1,     1,      InvertedResidual, [488, 1]],                      # 15
            # Stage 4
            [-1,     1,      InvertedResidual, [976, 2]],                      # 16
            [-1,     1,      InvertedResidual, [976, 1]],                      # 17
            [-1,     1,      InvertedResidual, [976, 1]],                      # 18
            [-1,     1,      InvertedResidual, [976, 1]],                      # 19
            # Stage 5
            [-1,     1,      Conv2d,           [2048, 1, 1, 0, 1, 1, False]],  # 20
            [-1,     1,      BatchNorm2d,      [2048]],                        # 21
            [-1,     1,      ReLU,             [True]],                        # 22
        ],
        "head": [
            [-1,     1,      ShuffleNetV2Classifier, [2048]],                  # 23
        ]
    },
}


@MODELS.register(name="shufflenet_v2")
class ShuffleNetV2(ImageClassificationModel):
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
        cfg        : dict | Path_ | None = "shufflenet_v2_x0.5",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "shufflenet_v2",
        fullname   : str  | None         = "shufflenet_v2_x0.5",
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
        cfg = cfg or "shufflenet_v2_x0.5"
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
        pass
   

@MODELS.register(name="shufflenet_v2_x0.5")
class ShuffleNetV2_x0_5(ShuffleNetV2):
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
            path        = "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
            filename    = "shufflenet_v2-x0.5-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "shufflenet_v2_x0.5.yaml",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "shufflenet_v2",
        fullname   : str  | None         = "shufflenet_v2-x0.5",
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
        cfg = cfg or "shufflenet_v2-x0.5"
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
            pretrained  = ShuffleNetV2_x0_5.init_pretrained(pretrained),
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
            model_state_dict["0.weight"]                  = state_dict["conv1.0.weight"]
            model_state_dict["1.bias"]                    = state_dict["conv1.1.bias"]
            model_state_dict["1.running_mean"]            = state_dict["conv1.1.running_mean"]
            model_state_dict["1.running_var"]             = state_dict["conv1.1.running_var"]
            model_state_dict["1.weight"]                  = state_dict["conv1.1.weight"]
            model_state_dict["4.branch1.0.weight"]        = state_dict["stage2.0.branch1.0.weight"]
            model_state_dict["4.branch1.1.bias"]          = state_dict["stage2.0.branch1.1.bias"]
            model_state_dict["4.branch1.1.running_mean"]  = state_dict["stage2.0.branch1.1.running_mean"]
            model_state_dict["4.branch1.1.running_var"]   = state_dict["stage2.0.branch1.1.running_var"]
            model_state_dict["4.branch1.1.weight"]        = state_dict["stage2.0.branch1.1.weight"]
            model_state_dict["4.branch1.2.weight"]        = state_dict["stage2.0.branch1.2.weight"]
            model_state_dict["4.branch1.3.bias"]          = state_dict["stage2.0.branch1.3.bias"]
            model_state_dict["4.branch1.3.running_mean"]  = state_dict["stage2.0.branch1.3.running_mean"]
            model_state_dict["4.branch1.3.running_var"]   = state_dict["stage2.0.branch1.3.running_var"]
            model_state_dict["4.branch1.3.weight"]        = state_dict["stage2.0.branch1.3.weight"]
            model_state_dict["4.branch2.0.weight"]        = state_dict["stage2.0.branch2.0.weight"]
            model_state_dict["4.branch2.1.bias"]          = state_dict["stage2.0.branch2.1.bias"]
            model_state_dict["4.branch2.1.running_mean"]  = state_dict["stage2.0.branch2.1.running_mean"]
            model_state_dict["4.branch2.1.running_var"]   = state_dict["stage2.0.branch2.1.running_var"]
            model_state_dict["4.branch2.1.weight"]        = state_dict["stage2.0.branch2.1.weight"]
            model_state_dict["4.branch2.3.weight"]        = state_dict["stage2.0.branch2.3.weight"]
            model_state_dict["4.branch2.4.bias"]          = state_dict["stage2.0.branch2.4.bias"]
            model_state_dict["4.branch2.4.running_mean"]  = state_dict["stage2.0.branch2.4.running_mean"]
            model_state_dict["4.branch2.4.running_var"]   = state_dict["stage2.0.branch2.4.running_var"]
            model_state_dict["4.branch2.4.weight"]        = state_dict["stage2.0.branch2.4.weight"]
            model_state_dict["4.branch2.5.weight"]        = state_dict["stage2.0.branch2.5.weight"]
            model_state_dict["4.branch2.6.bias"]          = state_dict["stage2.0.branch2.6.bias"]
            model_state_dict["4.branch2.6.running_mean"]  = state_dict["stage2.0.branch2.6.running_mean"]
            model_state_dict["4.branch2.6.running_var"]   = state_dict["stage2.0.branch2.6.running_var"]
            model_state_dict["4.branch2.6.weight"]        = state_dict["stage2.0.branch2.6.weight"]
            model_state_dict["5.branch2.0.weight"]        = state_dict["stage2.1.branch2.0.weight"]
            model_state_dict["5.branch2.1.bias"]          = state_dict["stage2.1.branch2.1.bias"]
            model_state_dict["5.branch2.1.running_mean"]  = state_dict["stage2.1.branch2.1.running_mean"]
            model_state_dict["5.branch2.1.running_var"]   = state_dict["stage2.1.branch2.1.running_var"]
            model_state_dict["5.branch2.1.weight"]        = state_dict["stage2.1.branch2.1.weight"]
            model_state_dict["5.branch2.3.weight"]        = state_dict["stage2.1.branch2.3.weight"]
            model_state_dict["5.branch2.4.bias"]          = state_dict["stage2.1.branch2.4.bias"]
            model_state_dict["5.branch2.4.running_mean"]  = state_dict["stage2.1.branch2.4.running_mean"]
            model_state_dict["5.branch2.4.running_var"]   = state_dict["stage2.1.branch2.4.running_var"]
            model_state_dict["5.branch2.4.weight"]        = state_dict["stage2.1.branch2.4.weight"]
            model_state_dict["5.branch2.5.weight"]        = state_dict["stage2.1.branch2.5.weight"]
            model_state_dict["5.branch2.6.bias"]          = state_dict["stage2.1.branch2.6.bias"]
            model_state_dict["5.branch2.6.running_mean"]  = state_dict["stage2.1.branch2.6.running_mean"]
            model_state_dict["5.branch2.6.running_var"]   = state_dict["stage2.1.branch2.6.running_var"]
            model_state_dict["5.branch2.6.weight"]        = state_dict["stage2.1.branch2.6.weight"]
            model_state_dict["6.branch2.0.weight"]        = state_dict["stage2.2.branch2.0.weight"]
            model_state_dict["6.branch2.1.bias"]          = state_dict["stage2.2.branch2.1.bias"]
            model_state_dict["6.branch2.1.running_mean"]  = state_dict["stage2.2.branch2.1.running_mean"]
            model_state_dict["6.branch2.1.running_var"]   = state_dict["stage2.2.branch2.1.running_var"]
            model_state_dict["6.branch2.1.weight"]        = state_dict["stage2.2.branch2.1.weight"]
            model_state_dict["6.branch2.3.weight"]        = state_dict["stage2.2.branch2.3.weight"]
            model_state_dict["6.branch2.4.bias"]          = state_dict["stage2.2.branch2.4.bias"]
            model_state_dict["6.branch2.4.running_mean"]  = state_dict["stage2.2.branch2.4.running_mean"]
            model_state_dict["6.branch2.4.running_var"]   = state_dict["stage2.2.branch2.4.running_var"]
            model_state_dict["6.branch2.4.weight"]        = state_dict["stage2.2.branch2.4.weight"]
            model_state_dict["6.branch2.5.weight"]        = state_dict["stage2.2.branch2.5.weight"]
            model_state_dict["6.branch2.6.bias"]          = state_dict["stage2.2.branch2.6.bias"]
            model_state_dict["6.branch2.6.running_mean"]  = state_dict["stage2.2.branch2.6.running_mean"]
            model_state_dict["6.branch2.6.running_var"]   = state_dict["stage2.2.branch2.6.running_var"]
            model_state_dict["6.branch2.6.weight"]        = state_dict["stage2.2.branch2.6.weight"]
            model_state_dict["7.branch2.0.weight"]        = state_dict["stage2.3.branch2.0.weight"]
            model_state_dict["7.branch2.1.bias"]          = state_dict["stage2.3.branch2.1.bias"]
            model_state_dict["7.branch2.1.running_mean"]  = state_dict["stage2.3.branch2.1.running_mean"]
            model_state_dict["7.branch2.1.running_var"]   = state_dict["stage2.3.branch2.1.running_var"]
            model_state_dict["7.branch2.1.weight"]        = state_dict["stage2.3.branch2.1.weight"]
            model_state_dict["7.branch2.3.weight"]        = state_dict["stage2.3.branch2.3.weight"]
            model_state_dict["7.branch2.4.bias"]          = state_dict["stage2.3.branch2.4.bias"]
            model_state_dict["7.branch2.4.running_mean"]  = state_dict["stage2.3.branch2.4.running_mean"]
            model_state_dict["7.branch2.4.running_var"]   = state_dict["stage2.3.branch2.4.running_var"]
            model_state_dict["7.branch2.4.weight"]        = state_dict["stage2.3.branch2.4.weight"]
            model_state_dict["7.branch2.5.weight"]        = state_dict["stage2.3.branch2.5.weight"]
            model_state_dict["7.branch2.6.bias"]          = state_dict["stage2.3.branch2.6.bias"]
            model_state_dict["7.branch2.6.running_mean"]  = state_dict["stage2.3.branch2.6.running_mean"]
            model_state_dict["7.branch2.6.running_var"]   = state_dict["stage2.3.branch2.6.running_var"]
            model_state_dict["7.branch2.6.weight"]        = state_dict["stage2.3.branch2.6.weight"]
            model_state_dict["8.branch1.0.weight"]        = state_dict["stage3.0.branch1.0.weight"]
            model_state_dict["8.branch1.1.bias"]          = state_dict["stage3.0.branch1.1.bias"]
            model_state_dict["8.branch1.1.running_mean"]  = state_dict["stage3.0.branch1.1.running_mean"]
            model_state_dict["8.branch1.1.running_var"]   = state_dict["stage3.0.branch1.1.running_var"]
            model_state_dict["8.branch1.1.weight"]        = state_dict["stage3.0.branch1.1.weight"]
            model_state_dict["8.branch1.2.weight"]        = state_dict["stage3.0.branch1.2.weight"]
            model_state_dict["8.branch1.3.bias"]          = state_dict["stage3.0.branch1.3.bias"]
            model_state_dict["8.branch1.3.running_mean"]  = state_dict["stage3.0.branch1.3.running_mean"]
            model_state_dict["8.branch1.3.running_var"]   = state_dict["stage3.0.branch1.3.running_var"]
            model_state_dict["8.branch1.3.weight"]        = state_dict["stage3.0.branch1.3.weight"]
            model_state_dict["8.branch2.0.weight"]        = state_dict["stage3.0.branch2.0.weight"]
            model_state_dict["8.branch2.1.bias"]          = state_dict["stage3.0.branch2.1.bias"]
            model_state_dict["8.branch2.1.running_mean"]  = state_dict["stage3.0.branch2.1.running_mean"]
            model_state_dict["8.branch2.1.running_var"]   = state_dict["stage3.0.branch2.1.running_var"]
            model_state_dict["8.branch2.1.weight"]        = state_dict["stage3.0.branch2.1.weight"]
            model_state_dict["8.branch2.3.weight"]        = state_dict["stage3.0.branch2.3.weight"]
            model_state_dict["8.branch2.4.bias"]          = state_dict["stage3.0.branch2.4.bias"]
            model_state_dict["8.branch2.4.running_mean"]  = state_dict["stage3.0.branch2.4.running_mean"]
            model_state_dict["8.branch2.4.running_var"]   = state_dict["stage3.0.branch2.4.running_var"]
            model_state_dict["8.branch2.4.weight"]        = state_dict["stage3.0.branch2.4.weight"]
            model_state_dict["8.branch2.5.weight"]        = state_dict["stage3.0.branch2.5.weight"]
            model_state_dict["8.branch2.6.bias"]          = state_dict["stage3.0.branch2.6.bias"]
            model_state_dict["8.branch2.6.running_mean"]  = state_dict["stage3.0.branch2.6.running_mean"]
            model_state_dict["8.branch2.6.running_var"]   = state_dict["stage3.0.branch2.6.running_var"]
            model_state_dict["8.branch2.6.weight"]        = state_dict["stage3.0.branch2.6.weight"]
            model_state_dict["9.branch2.0.weight"]        = state_dict["stage3.1.branch2.0.weight"]
            model_state_dict["9.branch2.1.bias"]          = state_dict["stage3.1.branch2.1.bias"]
            model_state_dict["9.branch2.1.running_mean"]  = state_dict["stage3.1.branch2.1.running_mean"]
            model_state_dict["9.branch2.1.running_var"]   = state_dict["stage3.1.branch2.1.running_var"]
            model_state_dict["9.branch2.1.weight"]        = state_dict["stage3.1.branch2.1.weight"]
            model_state_dict["9.branch2.3.weight"]        = state_dict["stage3.1.branch2.3.weight"]
            model_state_dict["9.branch2.4.bias"]          = state_dict["stage3.1.branch2.4.bias"]
            model_state_dict["9.branch2.4.running_mean"]  = state_dict["stage3.1.branch2.4.running_mean"]
            model_state_dict["9.branch2.4.running_var"]   = state_dict["stage3.1.branch2.4.running_var"]
            model_state_dict["9.branch2.4.weight"]        = state_dict["stage3.1.branch2.4.weight"]
            model_state_dict["9.branch2.5.weight"]        = state_dict["stage3.1.branch2.5.weight"]
            model_state_dict["9.branch2.6.bias"]          = state_dict["stage3.1.branch2.6.bias"]
            model_state_dict["9.branch2.6.running_mean"]  = state_dict["stage3.1.branch2.6.running_mean"]
            model_state_dict["9.branch2.6.running_var"]   = state_dict["stage3.1.branch2.6.running_var"]
            model_state_dict["9.branch2.6.weight"]        = state_dict["stage3.1.branch2.6.weight"]
            model_state_dict["10.branch2.0.weight"]       = state_dict["stage3.2.branch2.0.weight"]
            model_state_dict["10.branch2.1.bias"]         = state_dict["stage3.2.branch2.1.bias"]
            model_state_dict["10.branch2.1.running_mean"] = state_dict["stage3.2.branch2.1.running_mean"]
            model_state_dict["10.branch2.1.running_var"]  = state_dict["stage3.2.branch2.1.running_var"]
            model_state_dict["10.branch2.1.weight"]       = state_dict["stage3.2.branch2.1.weight"]
            model_state_dict["10.branch2.3.weight"]       = state_dict["stage3.2.branch2.3.weight"]
            model_state_dict["10.branch2.4.bias"]         = state_dict["stage3.2.branch2.4.bias"]
            model_state_dict["10.branch2.4.running_mean"] = state_dict["stage3.2.branch2.4.running_mean"]
            model_state_dict["10.branch2.4.running_var"]  = state_dict["stage3.2.branch2.4.running_var"]
            model_state_dict["10.branch2.4.weight"]       = state_dict["stage3.2.branch2.4.weight"]
            model_state_dict["10.branch2.5.weight"]       = state_dict["stage3.2.branch2.5.weight"]
            model_state_dict["10.branch2.6.bias"]         = state_dict["stage3.2.branch2.6.bias"]
            model_state_dict["10.branch2.6.running_mean"] = state_dict["stage3.2.branch2.6.running_mean"]
            model_state_dict["10.branch2.6.running_var"]  = state_dict["stage3.2.branch2.6.running_var"]
            model_state_dict["10.branch2.6.weight"]       = state_dict["stage3.2.branch2.6.weight"]
            model_state_dict["11.branch2.0.weight"]       = state_dict["stage3.3.branch2.0.weight"]
            model_state_dict["11.branch2.1.bias"]         = state_dict["stage3.3.branch2.1.bias"]
            model_state_dict["11.branch2.1.running_mean"] = state_dict["stage3.3.branch2.1.running_mean"]
            model_state_dict["11.branch2.1.running_var"]  = state_dict["stage3.3.branch2.1.running_var"]
            model_state_dict["11.branch2.1.weight"]       = state_dict["stage3.3.branch2.1.weight"]
            model_state_dict["11.branch2.3.weight"]       = state_dict["stage3.3.branch2.3.weight"]
            model_state_dict["11.branch2.4.bias"]         = state_dict["stage3.3.branch2.4.bias"]
            model_state_dict["11.branch2.4.running_mean"] = state_dict["stage3.3.branch2.4.running_mean"]
            model_state_dict["11.branch2.4.running_var"]  = state_dict["stage3.3.branch2.4.running_var"]
            model_state_dict["11.branch2.4.weight"]       = state_dict["stage3.3.branch2.4.weight"]
            model_state_dict["11.branch2.5.weight"]       = state_dict["stage3.3.branch2.5.weight"]
            model_state_dict["11.branch2.6.bias"]         = state_dict["stage3.3.branch2.6.bias"]
            model_state_dict["11.branch2.6.running_mean"] = state_dict["stage3.3.branch2.6.running_mean"]
            model_state_dict["11.branch2.6.running_var"]  = state_dict["stage3.3.branch2.6.running_var"]
            model_state_dict["11.branch2.6.weight"]       = state_dict["stage3.3.branch2.6.weight"]
            model_state_dict["12.branch2.0.weight"]       = state_dict["stage3.4.branch2.0.weight"]
            model_state_dict["12.branch2.1.bias"]         = state_dict["stage3.4.branch2.1.bias"]
            model_state_dict["12.branch2.1.running_mean"] = state_dict["stage3.4.branch2.1.running_mean"]
            model_state_dict["12.branch2.1.running_var"]  = state_dict["stage3.4.branch2.1.running_var"]
            model_state_dict["12.branch2.1.weight"]       = state_dict["stage3.4.branch2.1.weight"]
            model_state_dict["12.branch2.3.weight"]       = state_dict["stage3.4.branch2.3.weight"]
            model_state_dict["12.branch2.4.bias"]         = state_dict["stage3.4.branch2.4.bias"]
            model_state_dict["12.branch2.4.running_mean"] = state_dict["stage3.4.branch2.4.running_mean"]
            model_state_dict["12.branch2.4.running_var"]  = state_dict["stage3.4.branch2.4.running_var"]
            model_state_dict["12.branch2.4.weight"]       = state_dict["stage3.4.branch2.4.weight"]
            model_state_dict["12.branch2.5.weight"]       = state_dict["stage3.4.branch2.5.weight"]
            model_state_dict["12.branch2.6.bias"]         = state_dict["stage3.4.branch2.6.bias"]
            model_state_dict["12.branch2.6.running_mean"] = state_dict["stage3.4.branch2.6.running_mean"]
            model_state_dict["12.branch2.6.running_var"]  = state_dict["stage3.4.branch2.6.running_var"]
            model_state_dict["12.branch2.6.weight"]       = state_dict["stage3.4.branch2.6.weight"]
            model_state_dict["13.branch2.0.weight"]       = state_dict["stage3.5.branch2.0.weight"]
            model_state_dict["13.branch2.1.bias"]         = state_dict["stage3.5.branch2.1.bias"]
            model_state_dict["13.branch2.1.running_mean"] = state_dict["stage3.5.branch2.1.running_mean"]
            model_state_dict["13.branch2.1.running_var"]  = state_dict["stage3.5.branch2.1.running_var"]
            model_state_dict["13.branch2.1.weight"]       = state_dict["stage3.5.branch2.1.weight"]
            model_state_dict["13.branch2.3.weight"]       = state_dict["stage3.5.branch2.3.weight"]
            model_state_dict["13.branch2.4.bias"]         = state_dict["stage3.5.branch2.4.bias"]
            model_state_dict["13.branch2.4.running_mean"] = state_dict["stage3.5.branch2.4.running_mean"]
            model_state_dict["13.branch2.4.running_var"]  = state_dict["stage3.5.branch2.4.running_var"]
            model_state_dict["13.branch2.4.weight"]       = state_dict["stage3.5.branch2.4.weight"]
            model_state_dict["13.branch2.5.weight"]       = state_dict["stage3.5.branch2.5.weight"]
            model_state_dict["13.branch2.6.bias"]         = state_dict["stage3.5.branch2.6.bias"]
            model_state_dict["13.branch2.6.running_mean"] = state_dict["stage3.5.branch2.6.running_mean"]
            model_state_dict["13.branch2.6.running_var"]  = state_dict["stage3.5.branch2.6.running_var"]
            model_state_dict["13.branch2.6.weight"]       = state_dict["stage3.5.branch2.6.weight"]
            model_state_dict["14.branch2.0.weight"]       = state_dict["stage3.6.branch2.0.weight"]
            model_state_dict["14.branch2.1.bias"]         = state_dict["stage3.6.branch2.1.bias"]
            model_state_dict["14.branch2.1.running_mean"] = state_dict["stage3.6.branch2.1.running_mean"]
            model_state_dict["14.branch2.1.running_var"]  = state_dict["stage3.6.branch2.1.running_var"]
            model_state_dict["14.branch2.1.weight"]       = state_dict["stage3.6.branch2.1.weight"]
            model_state_dict["14.branch2.3.weight"]       = state_dict["stage3.6.branch2.3.weight"]
            model_state_dict["14.branch2.4.bias"]         = state_dict["stage3.6.branch2.4.bias"]
            model_state_dict["14.branch2.4.running_mean"] = state_dict["stage3.6.branch2.4.running_mean"]
            model_state_dict["14.branch2.4.running_var"]  = state_dict["stage3.6.branch2.4.running_var"]
            model_state_dict["14.branch2.4.weight"]       = state_dict["stage3.6.branch2.4.weight"]
            model_state_dict["14.branch2.5.weight"]       = state_dict["stage3.6.branch2.5.weight"]
            model_state_dict["14.branch2.6.bias"]         = state_dict["stage3.6.branch2.6.bias"]
            model_state_dict["14.branch2.6.running_mean"] = state_dict["stage3.6.branch2.6.running_mean"]
            model_state_dict["14.branch2.6.running_var"]  = state_dict["stage3.6.branch2.6.running_var"]
            model_state_dict["14.branch2.6.weight"]       = state_dict["stage3.6.branch2.6.weight"]
            model_state_dict["15.branch2.0.weight"]       = state_dict["stage3.7.branch2.0.weight"]
            model_state_dict["15.branch2.1.bias"]         = state_dict["stage3.7.branch2.1.bias"]
            model_state_dict["15.branch2.1.running_mean"] = state_dict["stage3.7.branch2.1.running_mean"]
            model_state_dict["15.branch2.1.running_var"]  = state_dict["stage3.7.branch2.1.running_var"]
            model_state_dict["15.branch2.1.weight"]       = state_dict["stage3.7.branch2.1.weight"]
            model_state_dict["15.branch2.3.weight"]       = state_dict["stage3.7.branch2.3.weight"]
            model_state_dict["15.branch2.4.bias"]         = state_dict["stage3.7.branch2.4.bias"]
            model_state_dict["15.branch2.4.running_mean"] = state_dict["stage3.7.branch2.4.running_mean"]
            model_state_dict["15.branch2.4.running_var"]  = state_dict["stage3.7.branch2.4.running_var"]
            model_state_dict["15.branch2.4.weight"]       = state_dict["stage3.7.branch2.4.weight"]
            model_state_dict["15.branch2.5.weight"]       = state_dict["stage3.7.branch2.5.weight"]
            model_state_dict["15.branch2.6.bias"]         = state_dict["stage3.7.branch2.6.bias"]
            model_state_dict["15.branch2.6.running_mean"] = state_dict["stage3.7.branch2.6.running_mean"]
            model_state_dict["15.branch2.6.running_var"]  = state_dict["stage3.7.branch2.6.running_var"]
            model_state_dict["15.branch2.6.weight"]       = state_dict["stage3.7.branch2.6.weight"]
            model_state_dict["16.branch1.0.weight"]       = state_dict["stage4.0.branch1.0.weight"]
            model_state_dict["16.branch1.1.bias"]         = state_dict["stage4.0.branch1.1.bias"]
            model_state_dict["16.branch1.1.running_mean"] = state_dict["stage4.0.branch1.1.running_mean"]
            model_state_dict["16.branch1.1.running_var"]  = state_dict["stage4.0.branch1.1.running_var"]
            model_state_dict["16.branch1.1.weight"]       = state_dict["stage4.0.branch1.1.weight"]
            model_state_dict["16.branch1.2.weight"]       = state_dict["stage4.0.branch1.2.weight"]
            model_state_dict["16.branch1.3.bias"]         = state_dict["stage4.0.branch1.3.bias"]
            model_state_dict["16.branch1.3.running_mean"] = state_dict["stage4.0.branch1.3.running_mean"]
            model_state_dict["16.branch1.3.running_var"]  = state_dict["stage4.0.branch1.3.running_var"]
            model_state_dict["16.branch1.3.weight"]       = state_dict["stage4.0.branch1.3.weight"]
            model_state_dict["16.branch2.0.weight"]       = state_dict["stage4.0.branch2.0.weight"]
            model_state_dict["16.branch2.1.bias"]         = state_dict["stage4.0.branch2.1.bias"]
            model_state_dict["16.branch2.1.running_mean"] = state_dict["stage4.0.branch2.1.running_mean"]
            model_state_dict["16.branch2.1.running_var"]  = state_dict["stage4.0.branch2.1.running_var"]
            model_state_dict["16.branch2.1.weight"]       = state_dict["stage4.0.branch2.1.weight"]
            model_state_dict["16.branch2.3.weight"]       = state_dict["stage4.0.branch2.3.weight"]
            model_state_dict["16.branch2.4.bias"]         = state_dict["stage4.0.branch2.4.bias"]
            model_state_dict["16.branch2.4.running_mean"] = state_dict["stage4.0.branch2.4.running_mean"]
            model_state_dict["16.branch2.4.running_var"]  = state_dict["stage4.0.branch2.4.running_var"]
            model_state_dict["16.branch2.4.weight"]       = state_dict["stage4.0.branch2.4.weight"]
            model_state_dict["16.branch2.5.weight"]       = state_dict["stage4.0.branch2.5.weight"]
            model_state_dict["16.branch2.6.bias"]         = state_dict["stage4.0.branch2.6.bias"]
            model_state_dict["16.branch2.6.running_mean"] = state_dict["stage4.0.branch2.6.running_mean"]
            model_state_dict["16.branch2.6.running_var"]  = state_dict["stage4.0.branch2.6.running_var"]
            model_state_dict["16.branch2.6.weight"]       = state_dict["stage4.0.branch2.6.weight"]
            model_state_dict["17.branch2.0.weight"]       = state_dict["stage4.1.branch2.0.weight"]
            model_state_dict["17.branch2.1.bias"]         = state_dict["stage4.1.branch2.1.bias"]
            model_state_dict["17.branch2.1.running_mean"] = state_dict["stage4.1.branch2.1.running_mean"]
            model_state_dict["17.branch2.1.running_var"]  = state_dict["stage4.1.branch2.1.running_var"]
            model_state_dict["17.branch2.1.weight"]       = state_dict["stage4.1.branch2.1.weight"]
            model_state_dict["17.branch2.3.weight"]       = state_dict["stage4.1.branch2.3.weight"]
            model_state_dict["17.branch2.4.bias"]         = state_dict["stage4.1.branch2.4.bias"]
            model_state_dict["17.branch2.4.running_mean"] = state_dict["stage4.1.branch2.4.running_mean"]
            model_state_dict["17.branch2.4.running_var"]  = state_dict["stage4.1.branch2.4.running_var"]
            model_state_dict["17.branch2.4.weight"]       = state_dict["stage4.1.branch2.4.weight"]
            model_state_dict["17.branch2.5.weight"]       = state_dict["stage4.1.branch2.5.weight"]
            model_state_dict["17.branch2.6.bias"]         = state_dict["stage4.1.branch2.6.bias"]
            model_state_dict["17.branch2.6.running_mean"] = state_dict["stage4.1.branch2.6.running_mean"]
            model_state_dict["17.branch2.6.running_var"]  = state_dict["stage4.1.branch2.6.running_var"]
            model_state_dict["17.branch2.6.weight"]       = state_dict["stage4.1.branch2.6.weight"]
            model_state_dict["18.branch2.0.weight"]       = state_dict["stage4.2.branch2.0.weight"]
            model_state_dict["18.branch2.1.bias"]         = state_dict["stage4.2.branch2.1.bias"]
            model_state_dict["18.branch2.1.running_mean"] = state_dict["stage4.2.branch2.1.running_mean"]
            model_state_dict["18.branch2.1.running_var"]  = state_dict["stage4.2.branch2.1.running_var"]
            model_state_dict["18.branch2.1.weight"]       = state_dict["stage4.2.branch2.1.weight"]
            model_state_dict["18.branch2.3.weight"]       = state_dict["stage4.2.branch2.3.weight"]
            model_state_dict["18.branch2.4.bias"]         = state_dict["stage4.2.branch2.4.bias"]
            model_state_dict["18.branch2.4.running_mean"] = state_dict["stage4.2.branch2.4.running_mean"]
            model_state_dict["18.branch2.4.running_var"]  = state_dict["stage4.2.branch2.4.running_var"]
            model_state_dict["18.branch2.4.weight"]       = state_dict["stage4.2.branch2.4.weight"]
            model_state_dict["18.branch2.5.weight"]       = state_dict["stage4.2.branch2.5.weight"]
            model_state_dict["18.branch2.6.bias"]         = state_dict["stage4.2.branch2.6.bias"]
            model_state_dict["18.branch2.6.running_mean"] = state_dict["stage4.2.branch2.6.running_mean"]
            model_state_dict["18.branch2.6.running_var"]  = state_dict["stage4.2.branch2.6.running_var"]
            model_state_dict["18.branch2.6.weight"]       = state_dict["stage4.2.branch2.6.weight"]
            model_state_dict["19.branch2.0.weight"]       = state_dict["stage4.3.branch2.0.weight"]
            model_state_dict["19.branch2.1.bias"]         = state_dict["stage4.3.branch2.1.bias"]
            model_state_dict["19.branch2.1.running_mean"] = state_dict["stage4.3.branch2.1.running_mean"]
            model_state_dict["19.branch2.1.running_var"]  = state_dict["stage4.3.branch2.1.running_var"]
            model_state_dict["19.branch2.1.weight"]       = state_dict["stage4.3.branch2.1.weight"]
            model_state_dict["19.branch2.3.weight"]       = state_dict["stage4.3.branch2.3.weight"]
            model_state_dict["19.branch2.4.bias"]         = state_dict["stage4.3.branch2.4.bias"]
            model_state_dict["19.branch2.4.running_mean"] = state_dict["stage4.3.branch2.4.running_mean"]
            model_state_dict["19.branch2.4.running_var"]  = state_dict["stage4.3.branch2.4.running_var"]
            model_state_dict["19.branch2.4.weight"]       = state_dict["stage4.3.branch2.4.weight"]
            model_state_dict["19.branch2.5.weight"]       = state_dict["stage4.3.branch2.5.weight"]
            model_state_dict["19.branch2.6.bias"]         = state_dict["stage4.3.branch2.6.bias"]
            model_state_dict["19.branch2.6.running_mean"] = state_dict["stage4.3.branch2.6.running_mean"]
            model_state_dict["19.branch2.6.running_var"]  = state_dict["stage4.3.branch2.6.running_var"]
            model_state_dict["19.branch2.6.weight"]       = state_dict["stage4.3.branch2.6.weight"]
            model_state_dict["20.weight"]                 = state_dict["conv5.0.weight"]
            model_state_dict["21.bias"]                   = state_dict["conv5.1.bias"]
            model_state_dict["21.running_mean"]           = state_dict["conv5.1.running_mean"]
            model_state_dict["21.running_var"]            = state_dict["conv5.1.running_var"]
            model_state_dict["21.weight"]                 = state_dict["conv5.1.weight"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["23.linear.bias"]   = state_dict["fc.bias"]
                model_state_dict["23.linear.weight"] = state_dict["fc.weight"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()


@MODELS.register(name="shufflenet_v2_x1.0")
class ShuffleNetV2_x1_0(ShuffleNetV2):
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
            path        = "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
            filename    = "shufflenet_v2-x1.0-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "shufflenet_v2_x1.0.yaml",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "shufflenet_v2",
        fullname   : str  | None         = "shufflenet_v2-x1.0",
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
        cfg = cfg or "shufflenet_v2-x1.0"
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
            pretrained  = ShuffleNetV2_x1_0.init_pretrained(pretrained),
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
            model_state_dict["0.weight"]                  = state_dict["conv1.0.weight"]
            model_state_dict["1.bias"]                    = state_dict["conv1.1.bias"]
            model_state_dict["1.running_mean"]            = state_dict["conv1.1.running_mean"]
            model_state_dict["1.running_var"]             = state_dict["conv1.1.running_var"]
            model_state_dict["1.weight"]                  = state_dict["conv1.1.weight"]
            model_state_dict["4.branch1.0.weight"]        = state_dict["stage2.0.branch1.0.weight"]
            model_state_dict["4.branch1.1.bias"]          = state_dict["stage2.0.branch1.1.bias"]
            model_state_dict["4.branch1.1.running_mean"]  = state_dict["stage2.0.branch1.1.running_mean"]
            model_state_dict["4.branch1.1.running_var"]   = state_dict["stage2.0.branch1.1.running_var"]
            model_state_dict["4.branch1.1.weight"]        = state_dict["stage2.0.branch1.1.weight"]
            model_state_dict["4.branch1.2.weight"]        = state_dict["stage2.0.branch1.2.weight"]
            model_state_dict["4.branch1.3.bias"]          = state_dict["stage2.0.branch1.3.bias"]
            model_state_dict["4.branch1.3.running_mean"]  = state_dict["stage2.0.branch1.3.running_mean"]
            model_state_dict["4.branch1.3.running_var"]   = state_dict["stage2.0.branch1.3.running_var"]
            model_state_dict["4.branch1.3.weight"]        = state_dict["stage2.0.branch1.3.weight"]
            model_state_dict["4.branch2.0.weight"]        = state_dict["stage2.0.branch2.0.weight"]
            model_state_dict["4.branch2.1.bias"]          = state_dict["stage2.0.branch2.1.bias"]
            model_state_dict["4.branch2.1.running_mean"]  = state_dict["stage2.0.branch2.1.running_mean"]
            model_state_dict["4.branch2.1.running_var"]   = state_dict["stage2.0.branch2.1.running_var"]
            model_state_dict["4.branch2.1.weight"]        = state_dict["stage2.0.branch2.1.weight"]
            model_state_dict["4.branch2.3.weight"]        = state_dict["stage2.0.branch2.3.weight"]
            model_state_dict["4.branch2.4.bias"]          = state_dict["stage2.0.branch2.4.bias"]
            model_state_dict["4.branch2.4.running_mean"]  = state_dict["stage2.0.branch2.4.running_mean"]
            model_state_dict["4.branch2.4.running_var"]   = state_dict["stage2.0.branch2.4.running_var"]
            model_state_dict["4.branch2.4.weight"]        = state_dict["stage2.0.branch2.4.weight"]
            model_state_dict["4.branch2.5.weight"]        = state_dict["stage2.0.branch2.5.weight"]
            model_state_dict["4.branch2.6.bias"]          = state_dict["stage2.0.branch2.6.bias"]
            model_state_dict["4.branch2.6.running_mean"]  = state_dict["stage2.0.branch2.6.running_mean"]
            model_state_dict["4.branch2.6.running_var"]   = state_dict["stage2.0.branch2.6.running_var"]
            model_state_dict["4.branch2.6.weight"]        = state_dict["stage2.0.branch2.6.weight"]
            model_state_dict["5.branch2.0.weight"]        = state_dict["stage2.1.branch2.0.weight"]
            model_state_dict["5.branch2.1.bias"]          = state_dict["stage2.1.branch2.1.bias"]
            model_state_dict["5.branch2.1.running_mean"]  = state_dict["stage2.1.branch2.1.running_mean"]
            model_state_dict["5.branch2.1.running_var"]   = state_dict["stage2.1.branch2.1.running_var"]
            model_state_dict["5.branch2.1.weight"]        = state_dict["stage2.1.branch2.1.weight"]
            model_state_dict["5.branch2.3.weight"]        = state_dict["stage2.1.branch2.3.weight"]
            model_state_dict["5.branch2.4.bias"]          = state_dict["stage2.1.branch2.4.bias"]
            model_state_dict["5.branch2.4.running_mean"]  = state_dict["stage2.1.branch2.4.running_mean"]
            model_state_dict["5.branch2.4.running_var"]   = state_dict["stage2.1.branch2.4.running_var"]
            model_state_dict["5.branch2.4.weight"]        = state_dict["stage2.1.branch2.4.weight"]
            model_state_dict["5.branch2.5.weight"]        = state_dict["stage2.1.branch2.5.weight"]
            model_state_dict["5.branch2.6.bias"]          = state_dict["stage2.1.branch2.6.bias"]
            model_state_dict["5.branch2.6.running_mean"]  = state_dict["stage2.1.branch2.6.running_mean"]
            model_state_dict["5.branch2.6.running_var"]   = state_dict["stage2.1.branch2.6.running_var"]
            model_state_dict["5.branch2.6.weight"]        = state_dict["stage2.1.branch2.6.weight"]
            model_state_dict["6.branch2.0.weight"]        = state_dict["stage2.2.branch2.0.weight"]
            model_state_dict["6.branch2.1.bias"]          = state_dict["stage2.2.branch2.1.bias"]
            model_state_dict["6.branch2.1.running_mean"]  = state_dict["stage2.2.branch2.1.running_mean"]
            model_state_dict["6.branch2.1.running_var"]   = state_dict["stage2.2.branch2.1.running_var"]
            model_state_dict["6.branch2.1.weight"]        = state_dict["stage2.2.branch2.1.weight"]
            model_state_dict["6.branch2.3.weight"]        = state_dict["stage2.2.branch2.3.weight"]
            model_state_dict["6.branch2.4.bias"]          = state_dict["stage2.2.branch2.4.bias"]
            model_state_dict["6.branch2.4.running_mean"]  = state_dict["stage2.2.branch2.4.running_mean"]
            model_state_dict["6.branch2.4.running_var"]   = state_dict["stage2.2.branch2.4.running_var"]
            model_state_dict["6.branch2.4.weight"]        = state_dict["stage2.2.branch2.4.weight"]
            model_state_dict["6.branch2.5.weight"]        = state_dict["stage2.2.branch2.5.weight"]
            model_state_dict["6.branch2.6.bias"]          = state_dict["stage2.2.branch2.6.bias"]
            model_state_dict["6.branch2.6.running_mean"]  = state_dict["stage2.2.branch2.6.running_mean"]
            model_state_dict["6.branch2.6.running_var"]   = state_dict["stage2.2.branch2.6.running_var"]
            model_state_dict["6.branch2.6.weight"]        = state_dict["stage2.2.branch2.6.weight"]
            model_state_dict["7.branch2.0.weight"]        = state_dict["stage2.3.branch2.0.weight"]
            model_state_dict["7.branch2.1.bias"]          = state_dict["stage2.3.branch2.1.bias"]
            model_state_dict["7.branch2.1.running_mean"]  = state_dict["stage2.3.branch2.1.running_mean"]
            model_state_dict["7.branch2.1.running_var"]   = state_dict["stage2.3.branch2.1.running_var"]
            model_state_dict["7.branch2.1.weight"]        = state_dict["stage2.3.branch2.1.weight"]
            model_state_dict["7.branch2.3.weight"]        = state_dict["stage2.3.branch2.3.weight"]
            model_state_dict["7.branch2.4.bias"]          = state_dict["stage2.3.branch2.4.bias"]
            model_state_dict["7.branch2.4.running_mean"]  = state_dict["stage2.3.branch2.4.running_mean"]
            model_state_dict["7.branch2.4.running_var"]   = state_dict["stage2.3.branch2.4.running_var"]
            model_state_dict["7.branch2.4.weight"]        = state_dict["stage2.3.branch2.4.weight"]
            model_state_dict["7.branch2.5.weight"]        = state_dict["stage2.3.branch2.5.weight"]
            model_state_dict["7.branch2.6.bias"]          = state_dict["stage2.3.branch2.6.bias"]
            model_state_dict["7.branch2.6.running_mean"]  = state_dict["stage2.3.branch2.6.running_mean"]
            model_state_dict["7.branch2.6.running_var"]   = state_dict["stage2.3.branch2.6.running_var"]
            model_state_dict["7.branch2.6.weight"]        = state_dict["stage2.3.branch2.6.weight"]
            model_state_dict["8.branch1.0.weight"]        = state_dict["stage3.0.branch1.0.weight"]
            model_state_dict["8.branch1.1.bias"]          = state_dict["stage3.0.branch1.1.bias"]
            model_state_dict["8.branch1.1.running_mean"]  = state_dict["stage3.0.branch1.1.running_mean"]
            model_state_dict["8.branch1.1.running_var"]   = state_dict["stage3.0.branch1.1.running_var"]
            model_state_dict["8.branch1.1.weight"]        = state_dict["stage3.0.branch1.1.weight"]
            model_state_dict["8.branch1.2.weight"]        = state_dict["stage3.0.branch1.2.weight"]
            model_state_dict["8.branch1.3.bias"]          = state_dict["stage3.0.branch1.3.bias"]
            model_state_dict["8.branch1.3.running_mean"]  = state_dict["stage3.0.branch1.3.running_mean"]
            model_state_dict["8.branch1.3.running_var"]   = state_dict["stage3.0.branch1.3.running_var"]
            model_state_dict["8.branch1.3.weight"]        = state_dict["stage3.0.branch1.3.weight"]
            model_state_dict["8.branch2.0.weight"]        = state_dict["stage3.0.branch2.0.weight"]
            model_state_dict["8.branch2.1.bias"]          = state_dict["stage3.0.branch2.1.bias"]
            model_state_dict["8.branch2.1.running_mean"]  = state_dict["stage3.0.branch2.1.running_mean"]
            model_state_dict["8.branch2.1.running_var"]   = state_dict["stage3.0.branch2.1.running_var"]
            model_state_dict["8.branch2.1.weight"]        = state_dict["stage3.0.branch2.1.weight"]
            model_state_dict["8.branch2.3.weight"]        = state_dict["stage3.0.branch2.3.weight"]
            model_state_dict["8.branch2.4.bias"]          = state_dict["stage3.0.branch2.4.bias"]
            model_state_dict["8.branch2.4.running_mean"]  = state_dict["stage3.0.branch2.4.running_mean"]
            model_state_dict["8.branch2.4.running_var"]   = state_dict["stage3.0.branch2.4.running_var"]
            model_state_dict["8.branch2.4.weight"]        = state_dict["stage3.0.branch2.4.weight"]
            model_state_dict["8.branch2.5.weight"]        = state_dict["stage3.0.branch2.5.weight"]
            model_state_dict["8.branch2.6.bias"]          = state_dict["stage3.0.branch2.6.bias"]
            model_state_dict["8.branch2.6.running_mean"]  = state_dict["stage3.0.branch2.6.running_mean"]
            model_state_dict["8.branch2.6.running_var"]   = state_dict["stage3.0.branch2.6.running_var"]
            model_state_dict["8.branch2.6.weight"]        = state_dict["stage3.0.branch2.6.weight"]
            model_state_dict["9.branch2.0.weight"]        = state_dict["stage3.1.branch2.0.weight"]
            model_state_dict["9.branch2.1.bias"]          = state_dict["stage3.1.branch2.1.bias"]
            model_state_dict["9.branch2.1.running_mean"]  = state_dict["stage3.1.branch2.1.running_mean"]
            model_state_dict["9.branch2.1.running_var"]   = state_dict["stage3.1.branch2.1.running_var"]
            model_state_dict["9.branch2.1.weight"]        = state_dict["stage3.1.branch2.1.weight"]
            model_state_dict["9.branch2.3.weight"]        = state_dict["stage3.1.branch2.3.weight"]
            model_state_dict["9.branch2.4.bias"]          = state_dict["stage3.1.branch2.4.bias"]
            model_state_dict["9.branch2.4.running_mean"]  = state_dict["stage3.1.branch2.4.running_mean"]
            model_state_dict["9.branch2.4.running_var"]   = state_dict["stage3.1.branch2.4.running_var"]
            model_state_dict["9.branch2.4.weight"]        = state_dict["stage3.1.branch2.4.weight"]
            model_state_dict["9.branch2.5.weight"]        = state_dict["stage3.1.branch2.5.weight"]
            model_state_dict["9.branch2.6.bias"]          = state_dict["stage3.1.branch2.6.bias"]
            model_state_dict["9.branch2.6.running_mean"]  = state_dict["stage3.1.branch2.6.running_mean"]
            model_state_dict["9.branch2.6.running_var"]   = state_dict["stage3.1.branch2.6.running_var"]
            model_state_dict["9.branch2.6.weight"]        = state_dict["stage3.1.branch2.6.weight"]
            model_state_dict["10.branch2.0.weight"]       = state_dict["stage3.2.branch2.0.weight"]
            model_state_dict["10.branch2.1.bias"]         = state_dict["stage3.2.branch2.1.bias"]
            model_state_dict["10.branch2.1.running_mean"] = state_dict["stage3.2.branch2.1.running_mean"]
            model_state_dict["10.branch2.1.running_var"]  = state_dict["stage3.2.branch2.1.running_var"]
            model_state_dict["10.branch2.1.weight"]       = state_dict["stage3.2.branch2.1.weight"]
            model_state_dict["10.branch2.3.weight"]       = state_dict["stage3.2.branch2.3.weight"]
            model_state_dict["10.branch2.4.bias"]         = state_dict["stage3.2.branch2.4.bias"]
            model_state_dict["10.branch2.4.running_mean"] = state_dict["stage3.2.branch2.4.running_mean"]
            model_state_dict["10.branch2.4.running_var"]  = state_dict["stage3.2.branch2.4.running_var"]
            model_state_dict["10.branch2.4.weight"]       = state_dict["stage3.2.branch2.4.weight"]
            model_state_dict["10.branch2.5.weight"]       = state_dict["stage3.2.branch2.5.weight"]
            model_state_dict["10.branch2.6.bias"]         = state_dict["stage3.2.branch2.6.bias"]
            model_state_dict["10.branch2.6.running_mean"] = state_dict["stage3.2.branch2.6.running_mean"]
            model_state_dict["10.branch2.6.running_var"]  = state_dict["stage3.2.branch2.6.running_var"]
            model_state_dict["10.branch2.6.weight"]       = state_dict["stage3.2.branch2.6.weight"]
            model_state_dict["11.branch2.0.weight"]       = state_dict["stage3.3.branch2.0.weight"]
            model_state_dict["11.branch2.1.bias"]         = state_dict["stage3.3.branch2.1.bias"]
            model_state_dict["11.branch2.1.running_mean"] = state_dict["stage3.3.branch2.1.running_mean"]
            model_state_dict["11.branch2.1.running_var"]  = state_dict["stage3.3.branch2.1.running_var"]
            model_state_dict["11.branch2.1.weight"]       = state_dict["stage3.3.branch2.1.weight"]
            model_state_dict["11.branch2.3.weight"]       = state_dict["stage3.3.branch2.3.weight"]
            model_state_dict["11.branch2.4.bias"]         = state_dict["stage3.3.branch2.4.bias"]
            model_state_dict["11.branch2.4.running_mean"] = state_dict["stage3.3.branch2.4.running_mean"]
            model_state_dict["11.branch2.4.running_var"]  = state_dict["stage3.3.branch2.4.running_var"]
            model_state_dict["11.branch2.4.weight"]       = state_dict["stage3.3.branch2.4.weight"]
            model_state_dict["11.branch2.5.weight"]       = state_dict["stage3.3.branch2.5.weight"]
            model_state_dict["11.branch2.6.bias"]         = state_dict["stage3.3.branch2.6.bias"]
            model_state_dict["11.branch2.6.running_mean"] = state_dict["stage3.3.branch2.6.running_mean"]
            model_state_dict["11.branch2.6.running_var"]  = state_dict["stage3.3.branch2.6.running_var"]
            model_state_dict["11.branch2.6.weight"]       = state_dict["stage3.3.branch2.6.weight"]
            model_state_dict["12.branch2.0.weight"]       = state_dict["stage3.4.branch2.0.weight"]
            model_state_dict["12.branch2.1.bias"]         = state_dict["stage3.4.branch2.1.bias"]
            model_state_dict["12.branch2.1.running_mean"] = state_dict["stage3.4.branch2.1.running_mean"]
            model_state_dict["12.branch2.1.running_var"]  = state_dict["stage3.4.branch2.1.running_var"]
            model_state_dict["12.branch2.1.weight"]       = state_dict["stage3.4.branch2.1.weight"]
            model_state_dict["12.branch2.3.weight"]       = state_dict["stage3.4.branch2.3.weight"]
            model_state_dict["12.branch2.4.bias"]         = state_dict["stage3.4.branch2.4.bias"]
            model_state_dict["12.branch2.4.running_mean"] = state_dict["stage3.4.branch2.4.running_mean"]
            model_state_dict["12.branch2.4.running_var"]  = state_dict["stage3.4.branch2.4.running_var"]
            model_state_dict["12.branch2.4.weight"]       = state_dict["stage3.4.branch2.4.weight"]
            model_state_dict["12.branch2.5.weight"]       = state_dict["stage3.4.branch2.5.weight"]
            model_state_dict["12.branch2.6.bias"]         = state_dict["stage3.4.branch2.6.bias"]
            model_state_dict["12.branch2.6.running_mean"] = state_dict["stage3.4.branch2.6.running_mean"]
            model_state_dict["12.branch2.6.running_var"]  = state_dict["stage3.4.branch2.6.running_var"]
            model_state_dict["12.branch2.6.weight"]       = state_dict["stage3.4.branch2.6.weight"]
            model_state_dict["13.branch2.0.weight"]       = state_dict["stage3.5.branch2.0.weight"]
            model_state_dict["13.branch2.1.bias"]         = state_dict["stage3.5.branch2.1.bias"]
            model_state_dict["13.branch2.1.running_mean"] = state_dict["stage3.5.branch2.1.running_mean"]
            model_state_dict["13.branch2.1.running_var"]  = state_dict["stage3.5.branch2.1.running_var"]
            model_state_dict["13.branch2.1.weight"]       = state_dict["stage3.5.branch2.1.weight"]
            model_state_dict["13.branch2.3.weight"]       = state_dict["stage3.5.branch2.3.weight"]
            model_state_dict["13.branch2.4.bias"]         = state_dict["stage3.5.branch2.4.bias"]
            model_state_dict["13.branch2.4.running_mean"] = state_dict["stage3.5.branch2.4.running_mean"]
            model_state_dict["13.branch2.4.running_var"]  = state_dict["stage3.5.branch2.4.running_var"]
            model_state_dict["13.branch2.4.weight"]       = state_dict["stage3.5.branch2.4.weight"]
            model_state_dict["13.branch2.5.weight"]       = state_dict["stage3.5.branch2.5.weight"]
            model_state_dict["13.branch2.6.bias"]         = state_dict["stage3.5.branch2.6.bias"]
            model_state_dict["13.branch2.6.running_mean"] = state_dict["stage3.5.branch2.6.running_mean"]
            model_state_dict["13.branch2.6.running_var"]  = state_dict["stage3.5.branch2.6.running_var"]
            model_state_dict["13.branch2.6.weight"]       = state_dict["stage3.5.branch2.6.weight"]
            model_state_dict["14.branch2.0.weight"]       = state_dict["stage3.6.branch2.0.weight"]
            model_state_dict["14.branch2.1.bias"]         = state_dict["stage3.6.branch2.1.bias"]
            model_state_dict["14.branch2.1.running_mean"] = state_dict["stage3.6.branch2.1.running_mean"]
            model_state_dict["14.branch2.1.running_var"]  = state_dict["stage3.6.branch2.1.running_var"]
            model_state_dict["14.branch2.1.weight"]       = state_dict["stage3.6.branch2.1.weight"]
            model_state_dict["14.branch2.3.weight"]       = state_dict["stage3.6.branch2.3.weight"]
            model_state_dict["14.branch2.4.bias"]         = state_dict["stage3.6.branch2.4.bias"]
            model_state_dict["14.branch2.4.running_mean"] = state_dict["stage3.6.branch2.4.running_mean"]
            model_state_dict["14.branch2.4.running_var"]  = state_dict["stage3.6.branch2.4.running_var"]
            model_state_dict["14.branch2.4.weight"]       = state_dict["stage3.6.branch2.4.weight"]
            model_state_dict["14.branch2.5.weight"]       = state_dict["stage3.6.branch2.5.weight"]
            model_state_dict["14.branch2.6.bias"]         = state_dict["stage3.6.branch2.6.bias"]
            model_state_dict["14.branch2.6.running_mean"] = state_dict["stage3.6.branch2.6.running_mean"]
            model_state_dict["14.branch2.6.running_var"]  = state_dict["stage3.6.branch2.6.running_var"]
            model_state_dict["14.branch2.6.weight"]       = state_dict["stage3.6.branch2.6.weight"]
            model_state_dict["15.branch2.0.weight"]       = state_dict["stage3.7.branch2.0.weight"]
            model_state_dict["15.branch2.1.bias"]         = state_dict["stage3.7.branch2.1.bias"]
            model_state_dict["15.branch2.1.running_mean"] = state_dict["stage3.7.branch2.1.running_mean"]
            model_state_dict["15.branch2.1.running_var"]  = state_dict["stage3.7.branch2.1.running_var"]
            model_state_dict["15.branch2.1.weight"]       = state_dict["stage3.7.branch2.1.weight"]
            model_state_dict["15.branch2.3.weight"]       = state_dict["stage3.7.branch2.3.weight"]
            model_state_dict["15.branch2.4.bias"]         = state_dict["stage3.7.branch2.4.bias"]
            model_state_dict["15.branch2.4.running_mean"] = state_dict["stage3.7.branch2.4.running_mean"]
            model_state_dict["15.branch2.4.running_var"]  = state_dict["stage3.7.branch2.4.running_var"]
            model_state_dict["15.branch2.4.weight"]       = state_dict["stage3.7.branch2.4.weight"]
            model_state_dict["15.branch2.5.weight"]       = state_dict["stage3.7.branch2.5.weight"]
            model_state_dict["15.branch2.6.bias"]         = state_dict["stage3.7.branch2.6.bias"]
            model_state_dict["15.branch2.6.running_mean"] = state_dict["stage3.7.branch2.6.running_mean"]
            model_state_dict["15.branch2.6.running_var"]  = state_dict["stage3.7.branch2.6.running_var"]
            model_state_dict["15.branch2.6.weight"]       = state_dict["stage3.7.branch2.6.weight"]
            model_state_dict["16.branch1.0.weight"]       = state_dict["stage4.0.branch1.0.weight"]
            model_state_dict["16.branch1.1.bias"]         = state_dict["stage4.0.branch1.1.bias"]
            model_state_dict["16.branch1.1.running_mean"] = state_dict["stage4.0.branch1.1.running_mean"]
            model_state_dict["16.branch1.1.running_var"]  = state_dict["stage4.0.branch1.1.running_var"]
            model_state_dict["16.branch1.1.weight"]       = state_dict["stage4.0.branch1.1.weight"]
            model_state_dict["16.branch1.2.weight"]       = state_dict["stage4.0.branch1.2.weight"]
            model_state_dict["16.branch1.3.bias"]         = state_dict["stage4.0.branch1.3.bias"]
            model_state_dict["16.branch1.3.running_mean"] = state_dict["stage4.0.branch1.3.running_mean"]
            model_state_dict["16.branch1.3.running_var"]  = state_dict["stage4.0.branch1.3.running_var"]
            model_state_dict["16.branch1.3.weight"]       = state_dict["stage4.0.branch1.3.weight"]
            model_state_dict["16.branch2.0.weight"]       = state_dict["stage4.0.branch2.0.weight"]
            model_state_dict["16.branch2.1.bias"]         = state_dict["stage4.0.branch2.1.bias"]
            model_state_dict["16.branch2.1.running_mean"] = state_dict["stage4.0.branch2.1.running_mean"]
            model_state_dict["16.branch2.1.running_var"]  = state_dict["stage4.0.branch2.1.running_var"]
            model_state_dict["16.branch2.1.weight"]       = state_dict["stage4.0.branch2.1.weight"]
            model_state_dict["16.branch2.3.weight"]       = state_dict["stage4.0.branch2.3.weight"]
            model_state_dict["16.branch2.4.bias"]         = state_dict["stage4.0.branch2.4.bias"]
            model_state_dict["16.branch2.4.running_mean"] = state_dict["stage4.0.branch2.4.running_mean"]
            model_state_dict["16.branch2.4.running_var"]  = state_dict["stage4.0.branch2.4.running_var"]
            model_state_dict["16.branch2.4.weight"]       = state_dict["stage4.0.branch2.4.weight"]
            model_state_dict["16.branch2.5.weight"]       = state_dict["stage4.0.branch2.5.weight"]
            model_state_dict["16.branch2.6.bias"]         = state_dict["stage4.0.branch2.6.bias"]
            model_state_dict["16.branch2.6.running_mean"] = state_dict["stage4.0.branch2.6.running_mean"]
            model_state_dict["16.branch2.6.running_var"]  = state_dict["stage4.0.branch2.6.running_var"]
            model_state_dict["16.branch2.6.weight"]       = state_dict["stage4.0.branch2.6.weight"]
            model_state_dict["17.branch2.0.weight"]       = state_dict["stage4.1.branch2.0.weight"]
            model_state_dict["17.branch2.1.bias"]         = state_dict["stage4.1.branch2.1.bias"]
            model_state_dict["17.branch2.1.running_mean"] = state_dict["stage4.1.branch2.1.running_mean"]
            model_state_dict["17.branch2.1.running_var"]  = state_dict["stage4.1.branch2.1.running_var"]
            model_state_dict["17.branch2.1.weight"]       = state_dict["stage4.1.branch2.1.weight"]
            model_state_dict["17.branch2.3.weight"]       = state_dict["stage4.1.branch2.3.weight"]
            model_state_dict["17.branch2.4.bias"]         = state_dict["stage4.1.branch2.4.bias"]
            model_state_dict["17.branch2.4.running_mean"] = state_dict["stage4.1.branch2.4.running_mean"]
            model_state_dict["17.branch2.4.running_var"]  = state_dict["stage4.1.branch2.4.running_var"]
            model_state_dict["17.branch2.4.weight"]       = state_dict["stage4.1.branch2.4.weight"]
            model_state_dict["17.branch2.5.weight"]       = state_dict["stage4.1.branch2.5.weight"]
            model_state_dict["17.branch2.6.bias"]         = state_dict["stage4.1.branch2.6.bias"]
            model_state_dict["17.branch2.6.running_mean"] = state_dict["stage4.1.branch2.6.running_mean"]
            model_state_dict["17.branch2.6.running_var"]  = state_dict["stage4.1.branch2.6.running_var"]
            model_state_dict["17.branch2.6.weight"]       = state_dict["stage4.1.branch2.6.weight"]
            model_state_dict["18.branch2.0.weight"]       = state_dict["stage4.2.branch2.0.weight"]
            model_state_dict["18.branch2.1.bias"]         = state_dict["stage4.2.branch2.1.bias"]
            model_state_dict["18.branch2.1.running_mean"] = state_dict["stage4.2.branch2.1.running_mean"]
            model_state_dict["18.branch2.1.running_var"]  = state_dict["stage4.2.branch2.1.running_var"]
            model_state_dict["18.branch2.1.weight"]       = state_dict["stage4.2.branch2.1.weight"]
            model_state_dict["18.branch2.3.weight"]       = state_dict["stage4.2.branch2.3.weight"]
            model_state_dict["18.branch2.4.bias"]         = state_dict["stage4.2.branch2.4.bias"]
            model_state_dict["18.branch2.4.running_mean"] = state_dict["stage4.2.branch2.4.running_mean"]
            model_state_dict["18.branch2.4.running_var"]  = state_dict["stage4.2.branch2.4.running_var"]
            model_state_dict["18.branch2.4.weight"]       = state_dict["stage4.2.branch2.4.weight"]
            model_state_dict["18.branch2.5.weight"]       = state_dict["stage4.2.branch2.5.weight"]
            model_state_dict["18.branch2.6.bias"]         = state_dict["stage4.2.branch2.6.bias"]
            model_state_dict["18.branch2.6.running_mean"] = state_dict["stage4.2.branch2.6.running_mean"]
            model_state_dict["18.branch2.6.running_var"]  = state_dict["stage4.2.branch2.6.running_var"]
            model_state_dict["18.branch2.6.weight"]       = state_dict["stage4.2.branch2.6.weight"]
            model_state_dict["19.branch2.0.weight"]       = state_dict["stage4.3.branch2.0.weight"]
            model_state_dict["19.branch2.1.bias"]         = state_dict["stage4.3.branch2.1.bias"]
            model_state_dict["19.branch2.1.running_mean"] = state_dict["stage4.3.branch2.1.running_mean"]
            model_state_dict["19.branch2.1.running_var"]  = state_dict["stage4.3.branch2.1.running_var"]
            model_state_dict["19.branch2.1.weight"]       = state_dict["stage4.3.branch2.1.weight"]
            model_state_dict["19.branch2.3.weight"]       = state_dict["stage4.3.branch2.3.weight"]
            model_state_dict["19.branch2.4.bias"]         = state_dict["stage4.3.branch2.4.bias"]
            model_state_dict["19.branch2.4.running_mean"] = state_dict["stage4.3.branch2.4.running_mean"]
            model_state_dict["19.branch2.4.running_var"]  = state_dict["stage4.3.branch2.4.running_var"]
            model_state_dict["19.branch2.4.weight"]       = state_dict["stage4.3.branch2.4.weight"]
            model_state_dict["19.branch2.5.weight"]       = state_dict["stage4.3.branch2.5.weight"]
            model_state_dict["19.branch2.6.bias"]         = state_dict["stage4.3.branch2.6.bias"]
            model_state_dict["19.branch2.6.running_mean"] = state_dict["stage4.3.branch2.6.running_mean"]
            model_state_dict["19.branch2.6.running_var"]  = state_dict["stage4.3.branch2.6.running_var"]
            model_state_dict["19.branch2.6.weight"]       = state_dict["stage4.3.branch2.6.weight"]
            model_state_dict["20.weight"]                 = state_dict["conv5.0.weight"]
            model_state_dict["21.bias"]                   = state_dict["conv5.1.bias"]
            model_state_dict["21.running_mean"]           = state_dict["conv5.1.running_mean"]
            model_state_dict["21.running_var"]            = state_dict["conv5.1.running_var"]
            model_state_dict["21.weight"]                 = state_dict["conv5.1.weight"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["23.linear.bias"]   = state_dict["fc.bias"]
                model_state_dict["23.linear.weight"] = state_dict["fc.weight"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()


@MODELS.register(name="shufflenet_v2_x1.5")
class ShuffleNetV2_x1_5(ShuffleNetV2):
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
            path        = "https://download.pytorch.org/models/shufflenetv2_x1_5-3c479a10.pth",
            filename    = "shufflenet_v2-x1.5-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "shufflenet_v2_x1.5.yaml",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "shufflenet_v2",
        fullname   : str  | None         = "shufflenet_v2-x1.5",
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
        cfg = cfg or "shufflenet_v2-x1.5"
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
            pretrained  = ShuffleNetV2_x1_5.init_pretrained(pretrained),
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
            model_state_dict["0.weight"]                  = state_dict["conv1.0.weight"]
            model_state_dict["1.bias"]                    = state_dict["conv1.1.bias"]
            model_state_dict["1.running_mean"]            = state_dict["conv1.1.running_mean"]
            model_state_dict["1.running_var"]             = state_dict["conv1.1.running_var"]
            model_state_dict["1.weight"]                  = state_dict["conv1.1.weight"]
            model_state_dict["4.branch1.0.weight"]        = state_dict["stage2.0.branch1.0.weight"]
            model_state_dict["4.branch1.1.bias"]          = state_dict["stage2.0.branch1.1.bias"]
            model_state_dict["4.branch1.1.running_mean"]  = state_dict["stage2.0.branch1.1.running_mean"]
            model_state_dict["4.branch1.1.running_var"]   = state_dict["stage2.0.branch1.1.running_var"]
            model_state_dict["4.branch1.1.weight"]        = state_dict["stage2.0.branch1.1.weight"]
            model_state_dict["4.branch1.2.weight"]        = state_dict["stage2.0.branch1.2.weight"]
            model_state_dict["4.branch1.3.bias"]          = state_dict["stage2.0.branch1.3.bias"]
            model_state_dict["4.branch1.3.running_mean"]  = state_dict["stage2.0.branch1.3.running_mean"]
            model_state_dict["4.branch1.3.running_var"]   = state_dict["stage2.0.branch1.3.running_var"]
            model_state_dict["4.branch1.3.weight"]        = state_dict["stage2.0.branch1.3.weight"]
            model_state_dict["4.branch2.0.weight"]        = state_dict["stage2.0.branch2.0.weight"]
            model_state_dict["4.branch2.1.bias"]          = state_dict["stage2.0.branch2.1.bias"]
            model_state_dict["4.branch2.1.running_mean"]  = state_dict["stage2.0.branch2.1.running_mean"]
            model_state_dict["4.branch2.1.running_var"]   = state_dict["stage2.0.branch2.1.running_var"]
            model_state_dict["4.branch2.1.weight"]        = state_dict["stage2.0.branch2.1.weight"]
            model_state_dict["4.branch2.3.weight"]        = state_dict["stage2.0.branch2.3.weight"]
            model_state_dict["4.branch2.4.bias"]          = state_dict["stage2.0.branch2.4.bias"]
            model_state_dict["4.branch2.4.running_mean"]  = state_dict["stage2.0.branch2.4.running_mean"]
            model_state_dict["4.branch2.4.running_var"]   = state_dict["stage2.0.branch2.4.running_var"]
            model_state_dict["4.branch2.4.weight"]        = state_dict["stage2.0.branch2.4.weight"]
            model_state_dict["4.branch2.5.weight"]        = state_dict["stage2.0.branch2.5.weight"]
            model_state_dict["4.branch2.6.bias"]          = state_dict["stage2.0.branch2.6.bias"]
            model_state_dict["4.branch2.6.running_mean"]  = state_dict["stage2.0.branch2.6.running_mean"]
            model_state_dict["4.branch2.6.running_var"]   = state_dict["stage2.0.branch2.6.running_var"]
            model_state_dict["4.branch2.6.weight"]        = state_dict["stage2.0.branch2.6.weight"]
            model_state_dict["5.branch2.0.weight"]        = state_dict["stage2.1.branch2.0.weight"]
            model_state_dict["5.branch2.1.bias"]          = state_dict["stage2.1.branch2.1.bias"]
            model_state_dict["5.branch2.1.running_mean"]  = state_dict["stage2.1.branch2.1.running_mean"]
            model_state_dict["5.branch2.1.running_var"]   = state_dict["stage2.1.branch2.1.running_var"]
            model_state_dict["5.branch2.1.weight"]        = state_dict["stage2.1.branch2.1.weight"]
            model_state_dict["5.branch2.3.weight"]        = state_dict["stage2.1.branch2.3.weight"]
            model_state_dict["5.branch2.4.bias"]          = state_dict["stage2.1.branch2.4.bias"]
            model_state_dict["5.branch2.4.running_mean"]  = state_dict["stage2.1.branch2.4.running_mean"]
            model_state_dict["5.branch2.4.running_var"]   = state_dict["stage2.1.branch2.4.running_var"]
            model_state_dict["5.branch2.4.weight"]        = state_dict["stage2.1.branch2.4.weight"]
            model_state_dict["5.branch2.5.weight"]        = state_dict["stage2.1.branch2.5.weight"]
            model_state_dict["5.branch2.6.bias"]          = state_dict["stage2.1.branch2.6.bias"]
            model_state_dict["5.branch2.6.running_mean"]  = state_dict["stage2.1.branch2.6.running_mean"]
            model_state_dict["5.branch2.6.running_var"]   = state_dict["stage2.1.branch2.6.running_var"]
            model_state_dict["5.branch2.6.weight"]        = state_dict["stage2.1.branch2.6.weight"]
            model_state_dict["6.branch2.0.weight"]        = state_dict["stage2.2.branch2.0.weight"]
            model_state_dict["6.branch2.1.bias"]          = state_dict["stage2.2.branch2.1.bias"]
            model_state_dict["6.branch2.1.running_mean"]  = state_dict["stage2.2.branch2.1.running_mean"]
            model_state_dict["6.branch2.1.running_var"]   = state_dict["stage2.2.branch2.1.running_var"]
            model_state_dict["6.branch2.1.weight"]        = state_dict["stage2.2.branch2.1.weight"]
            model_state_dict["6.branch2.3.weight"]        = state_dict["stage2.2.branch2.3.weight"]
            model_state_dict["6.branch2.4.bias"]          = state_dict["stage2.2.branch2.4.bias"]
            model_state_dict["6.branch2.4.running_mean"]  = state_dict["stage2.2.branch2.4.running_mean"]
            model_state_dict["6.branch2.4.running_var"]   = state_dict["stage2.2.branch2.4.running_var"]
            model_state_dict["6.branch2.4.weight"]        = state_dict["stage2.2.branch2.4.weight"]
            model_state_dict["6.branch2.5.weight"]        = state_dict["stage2.2.branch2.5.weight"]
            model_state_dict["6.branch2.6.bias"]          = state_dict["stage2.2.branch2.6.bias"]
            model_state_dict["6.branch2.6.running_mean"]  = state_dict["stage2.2.branch2.6.running_mean"]
            model_state_dict["6.branch2.6.running_var"]   = state_dict["stage2.2.branch2.6.running_var"]
            model_state_dict["6.branch2.6.weight"]        = state_dict["stage2.2.branch2.6.weight"]
            model_state_dict["7.branch2.0.weight"]        = state_dict["stage2.3.branch2.0.weight"]
            model_state_dict["7.branch2.1.bias"]          = state_dict["stage2.3.branch2.1.bias"]
            model_state_dict["7.branch2.1.running_mean"]  = state_dict["stage2.3.branch2.1.running_mean"]
            model_state_dict["7.branch2.1.running_var"]   = state_dict["stage2.3.branch2.1.running_var"]
            model_state_dict["7.branch2.1.weight"]        = state_dict["stage2.3.branch2.1.weight"]
            model_state_dict["7.branch2.3.weight"]        = state_dict["stage2.3.branch2.3.weight"]
            model_state_dict["7.branch2.4.bias"]          = state_dict["stage2.3.branch2.4.bias"]
            model_state_dict["7.branch2.4.running_mean"]  = state_dict["stage2.3.branch2.4.running_mean"]
            model_state_dict["7.branch2.4.running_var"]   = state_dict["stage2.3.branch2.4.running_var"]
            model_state_dict["7.branch2.4.weight"]        = state_dict["stage2.3.branch2.4.weight"]
            model_state_dict["7.branch2.5.weight"]        = state_dict["stage2.3.branch2.5.weight"]
            model_state_dict["7.branch2.6.bias"]          = state_dict["stage2.3.branch2.6.bias"]
            model_state_dict["7.branch2.6.running_mean"]  = state_dict["stage2.3.branch2.6.running_mean"]
            model_state_dict["7.branch2.6.running_var"]   = state_dict["stage2.3.branch2.6.running_var"]
            model_state_dict["7.branch2.6.weight"]        = state_dict["stage2.3.branch2.6.weight"]
            model_state_dict["8.branch1.0.weight"]        = state_dict["stage3.0.branch1.0.weight"]
            model_state_dict["8.branch1.1.bias"]          = state_dict["stage3.0.branch1.1.bias"]
            model_state_dict["8.branch1.1.running_mean"]  = state_dict["stage3.0.branch1.1.running_mean"]
            model_state_dict["8.branch1.1.running_var"]   = state_dict["stage3.0.branch1.1.running_var"]
            model_state_dict["8.branch1.1.weight"]        = state_dict["stage3.0.branch1.1.weight"]
            model_state_dict["8.branch1.2.weight"]        = state_dict["stage3.0.branch1.2.weight"]
            model_state_dict["8.branch1.3.bias"]          = state_dict["stage3.0.branch1.3.bias"]
            model_state_dict["8.branch1.3.running_mean"]  = state_dict["stage3.0.branch1.3.running_mean"]
            model_state_dict["8.branch1.3.running_var"]   = state_dict["stage3.0.branch1.3.running_var"]
            model_state_dict["8.branch1.3.weight"]        = state_dict["stage3.0.branch1.3.weight"]
            model_state_dict["8.branch2.0.weight"]        = state_dict["stage3.0.branch2.0.weight"]
            model_state_dict["8.branch2.1.bias"]          = state_dict["stage3.0.branch2.1.bias"]
            model_state_dict["8.branch2.1.running_mean"]  = state_dict["stage3.0.branch2.1.running_mean"]
            model_state_dict["8.branch2.1.running_var"]   = state_dict["stage3.0.branch2.1.running_var"]
            model_state_dict["8.branch2.1.weight"]        = state_dict["stage3.0.branch2.1.weight"]
            model_state_dict["8.branch2.3.weight"]        = state_dict["stage3.0.branch2.3.weight"]
            model_state_dict["8.branch2.4.bias"]          = state_dict["stage3.0.branch2.4.bias"]
            model_state_dict["8.branch2.4.running_mean"]  = state_dict["stage3.0.branch2.4.running_mean"]
            model_state_dict["8.branch2.4.running_var"]   = state_dict["stage3.0.branch2.4.running_var"]
            model_state_dict["8.branch2.4.weight"]        = state_dict["stage3.0.branch2.4.weight"]
            model_state_dict["8.branch2.5.weight"]        = state_dict["stage3.0.branch2.5.weight"]
            model_state_dict["8.branch2.6.bias"]          = state_dict["stage3.0.branch2.6.bias"]
            model_state_dict["8.branch2.6.running_mean"]  = state_dict["stage3.0.branch2.6.running_mean"]
            model_state_dict["8.branch2.6.running_var"]   = state_dict["stage3.0.branch2.6.running_var"]
            model_state_dict["8.branch2.6.weight"]        = state_dict["stage3.0.branch2.6.weight"]
            model_state_dict["9.branch2.0.weight"]        = state_dict["stage3.1.branch2.0.weight"]
            model_state_dict["9.branch2.1.bias"]          = state_dict["stage3.1.branch2.1.bias"]
            model_state_dict["9.branch2.1.running_mean"]  = state_dict["stage3.1.branch2.1.running_mean"]
            model_state_dict["9.branch2.1.running_var"]   = state_dict["stage3.1.branch2.1.running_var"]
            model_state_dict["9.branch2.1.weight"]        = state_dict["stage3.1.branch2.1.weight"]
            model_state_dict["9.branch2.3.weight"]        = state_dict["stage3.1.branch2.3.weight"]
            model_state_dict["9.branch2.4.bias"]          = state_dict["stage3.1.branch2.4.bias"]
            model_state_dict["9.branch2.4.running_mean"]  = state_dict["stage3.1.branch2.4.running_mean"]
            model_state_dict["9.branch2.4.running_var"]   = state_dict["stage3.1.branch2.4.running_var"]
            model_state_dict["9.branch2.4.weight"]        = state_dict["stage3.1.branch2.4.weight"]
            model_state_dict["9.branch2.5.weight"]        = state_dict["stage3.1.branch2.5.weight"]
            model_state_dict["9.branch2.6.bias"]          = state_dict["stage3.1.branch2.6.bias"]
            model_state_dict["9.branch2.6.running_mean"]  = state_dict["stage3.1.branch2.6.running_mean"]
            model_state_dict["9.branch2.6.running_var"]   = state_dict["stage3.1.branch2.6.running_var"]
            model_state_dict["9.branch2.6.weight"]        = state_dict["stage3.1.branch2.6.weight"]
            model_state_dict["10.branch2.0.weight"]       = state_dict["stage3.2.branch2.0.weight"]
            model_state_dict["10.branch2.1.bias"]         = state_dict["stage3.2.branch2.1.bias"]
            model_state_dict["10.branch2.1.running_mean"] = state_dict["stage3.2.branch2.1.running_mean"]
            model_state_dict["10.branch2.1.running_var"]  = state_dict["stage3.2.branch2.1.running_var"]
            model_state_dict["10.branch2.1.weight"]       = state_dict["stage3.2.branch2.1.weight"]
            model_state_dict["10.branch2.3.weight"]       = state_dict["stage3.2.branch2.3.weight"]
            model_state_dict["10.branch2.4.bias"]         = state_dict["stage3.2.branch2.4.bias"]
            model_state_dict["10.branch2.4.running_mean"] = state_dict["stage3.2.branch2.4.running_mean"]
            model_state_dict["10.branch2.4.running_var"]  = state_dict["stage3.2.branch2.4.running_var"]
            model_state_dict["10.branch2.4.weight"]       = state_dict["stage3.2.branch2.4.weight"]
            model_state_dict["10.branch2.5.weight"]       = state_dict["stage3.2.branch2.5.weight"]
            model_state_dict["10.branch2.6.bias"]         = state_dict["stage3.2.branch2.6.bias"]
            model_state_dict["10.branch2.6.running_mean"] = state_dict["stage3.2.branch2.6.running_mean"]
            model_state_dict["10.branch2.6.running_var"]  = state_dict["stage3.2.branch2.6.running_var"]
            model_state_dict["10.branch2.6.weight"]       = state_dict["stage3.2.branch2.6.weight"]
            model_state_dict["11.branch2.0.weight"]       = state_dict["stage3.3.branch2.0.weight"]
            model_state_dict["11.branch2.1.bias"]         = state_dict["stage3.3.branch2.1.bias"]
            model_state_dict["11.branch2.1.running_mean"] = state_dict["stage3.3.branch2.1.running_mean"]
            model_state_dict["11.branch2.1.running_var"]  = state_dict["stage3.3.branch2.1.running_var"]
            model_state_dict["11.branch2.1.weight"]       = state_dict["stage3.3.branch2.1.weight"]
            model_state_dict["11.branch2.3.weight"]       = state_dict["stage3.3.branch2.3.weight"]
            model_state_dict["11.branch2.4.bias"]         = state_dict["stage3.3.branch2.4.bias"]
            model_state_dict["11.branch2.4.running_mean"] = state_dict["stage3.3.branch2.4.running_mean"]
            model_state_dict["11.branch2.4.running_var"]  = state_dict["stage3.3.branch2.4.running_var"]
            model_state_dict["11.branch2.4.weight"]       = state_dict["stage3.3.branch2.4.weight"]
            model_state_dict["11.branch2.5.weight"]       = state_dict["stage3.3.branch2.5.weight"]
            model_state_dict["11.branch2.6.bias"]         = state_dict["stage3.3.branch2.6.bias"]
            model_state_dict["11.branch2.6.running_mean"] = state_dict["stage3.3.branch2.6.running_mean"]
            model_state_dict["11.branch2.6.running_var"]  = state_dict["stage3.3.branch2.6.running_var"]
            model_state_dict["11.branch2.6.weight"]       = state_dict["stage3.3.branch2.6.weight"]
            model_state_dict["12.branch2.0.weight"]       = state_dict["stage3.4.branch2.0.weight"]
            model_state_dict["12.branch2.1.bias"]         = state_dict["stage3.4.branch2.1.bias"]
            model_state_dict["12.branch2.1.running_mean"] = state_dict["stage3.4.branch2.1.running_mean"]
            model_state_dict["12.branch2.1.running_var"]  = state_dict["stage3.4.branch2.1.running_var"]
            model_state_dict["12.branch2.1.weight"]       = state_dict["stage3.4.branch2.1.weight"]
            model_state_dict["12.branch2.3.weight"]       = state_dict["stage3.4.branch2.3.weight"]
            model_state_dict["12.branch2.4.bias"]         = state_dict["stage3.4.branch2.4.bias"]
            model_state_dict["12.branch2.4.running_mean"] = state_dict["stage3.4.branch2.4.running_mean"]
            model_state_dict["12.branch2.4.running_var"]  = state_dict["stage3.4.branch2.4.running_var"]
            model_state_dict["12.branch2.4.weight"]       = state_dict["stage3.4.branch2.4.weight"]
            model_state_dict["12.branch2.5.weight"]       = state_dict["stage3.4.branch2.5.weight"]
            model_state_dict["12.branch2.6.bias"]         = state_dict["stage3.4.branch2.6.bias"]
            model_state_dict["12.branch2.6.running_mean"] = state_dict["stage3.4.branch2.6.running_mean"]
            model_state_dict["12.branch2.6.running_var"]  = state_dict["stage3.4.branch2.6.running_var"]
            model_state_dict["12.branch2.6.weight"]       = state_dict["stage3.4.branch2.6.weight"]
            model_state_dict["13.branch2.0.weight"]       = state_dict["stage3.5.branch2.0.weight"]
            model_state_dict["13.branch2.1.bias"]         = state_dict["stage3.5.branch2.1.bias"]
            model_state_dict["13.branch2.1.running_mean"] = state_dict["stage3.5.branch2.1.running_mean"]
            model_state_dict["13.branch2.1.running_var"]  = state_dict["stage3.5.branch2.1.running_var"]
            model_state_dict["13.branch2.1.weight"]       = state_dict["stage3.5.branch2.1.weight"]
            model_state_dict["13.branch2.3.weight"]       = state_dict["stage3.5.branch2.3.weight"]
            model_state_dict["13.branch2.4.bias"]         = state_dict["stage3.5.branch2.4.bias"]
            model_state_dict["13.branch2.4.running_mean"] = state_dict["stage3.5.branch2.4.running_mean"]
            model_state_dict["13.branch2.4.running_var"]  = state_dict["stage3.5.branch2.4.running_var"]
            model_state_dict["13.branch2.4.weight"]       = state_dict["stage3.5.branch2.4.weight"]
            model_state_dict["13.branch2.5.weight"]       = state_dict["stage3.5.branch2.5.weight"]
            model_state_dict["13.branch2.6.bias"]         = state_dict["stage3.5.branch2.6.bias"]
            model_state_dict["13.branch2.6.running_mean"] = state_dict["stage3.5.branch2.6.running_mean"]
            model_state_dict["13.branch2.6.running_var"]  = state_dict["stage3.5.branch2.6.running_var"]
            model_state_dict["13.branch2.6.weight"]       = state_dict["stage3.5.branch2.6.weight"]
            model_state_dict["14.branch2.0.weight"]       = state_dict["stage3.6.branch2.0.weight"]
            model_state_dict["14.branch2.1.bias"]         = state_dict["stage3.6.branch2.1.bias"]
            model_state_dict["14.branch2.1.running_mean"] = state_dict["stage3.6.branch2.1.running_mean"]
            model_state_dict["14.branch2.1.running_var"]  = state_dict["stage3.6.branch2.1.running_var"]
            model_state_dict["14.branch2.1.weight"]       = state_dict["stage3.6.branch2.1.weight"]
            model_state_dict["14.branch2.3.weight"]       = state_dict["stage3.6.branch2.3.weight"]
            model_state_dict["14.branch2.4.bias"]         = state_dict["stage3.6.branch2.4.bias"]
            model_state_dict["14.branch2.4.running_mean"] = state_dict["stage3.6.branch2.4.running_mean"]
            model_state_dict["14.branch2.4.running_var"]  = state_dict["stage3.6.branch2.4.running_var"]
            model_state_dict["14.branch2.4.weight"]       = state_dict["stage3.6.branch2.4.weight"]
            model_state_dict["14.branch2.5.weight"]       = state_dict["stage3.6.branch2.5.weight"]
            model_state_dict["14.branch2.6.bias"]         = state_dict["stage3.6.branch2.6.bias"]
            model_state_dict["14.branch2.6.running_mean"] = state_dict["stage3.6.branch2.6.running_mean"]
            model_state_dict["14.branch2.6.running_var"]  = state_dict["stage3.6.branch2.6.running_var"]
            model_state_dict["14.branch2.6.weight"]       = state_dict["stage3.6.branch2.6.weight"]
            model_state_dict["15.branch2.0.weight"]       = state_dict["stage3.7.branch2.0.weight"]
            model_state_dict["15.branch2.1.bias"]         = state_dict["stage3.7.branch2.1.bias"]
            model_state_dict["15.branch2.1.running_mean"] = state_dict["stage3.7.branch2.1.running_mean"]
            model_state_dict["15.branch2.1.running_var"]  = state_dict["stage3.7.branch2.1.running_var"]
            model_state_dict["15.branch2.1.weight"]       = state_dict["stage3.7.branch2.1.weight"]
            model_state_dict["15.branch2.3.weight"]       = state_dict["stage3.7.branch2.3.weight"]
            model_state_dict["15.branch2.4.bias"]         = state_dict["stage3.7.branch2.4.bias"]
            model_state_dict["15.branch2.4.running_mean"] = state_dict["stage3.7.branch2.4.running_mean"]
            model_state_dict["15.branch2.4.running_var"]  = state_dict["stage3.7.branch2.4.running_var"]
            model_state_dict["15.branch2.4.weight"]       = state_dict["stage3.7.branch2.4.weight"]
            model_state_dict["15.branch2.5.weight"]       = state_dict["stage3.7.branch2.5.weight"]
            model_state_dict["15.branch2.6.bias"]         = state_dict["stage3.7.branch2.6.bias"]
            model_state_dict["15.branch2.6.running_mean"] = state_dict["stage3.7.branch2.6.running_mean"]
            model_state_dict["15.branch2.6.running_var"]  = state_dict["stage3.7.branch2.6.running_var"]
            model_state_dict["15.branch2.6.weight"]       = state_dict["stage3.7.branch2.6.weight"]
            model_state_dict["16.branch1.0.weight"]       = state_dict["stage4.0.branch1.0.weight"]
            model_state_dict["16.branch1.1.bias"]         = state_dict["stage4.0.branch1.1.bias"]
            model_state_dict["16.branch1.1.running_mean"] = state_dict["stage4.0.branch1.1.running_mean"]
            model_state_dict["16.branch1.1.running_var"]  = state_dict["stage4.0.branch1.1.running_var"]
            model_state_dict["16.branch1.1.weight"]       = state_dict["stage4.0.branch1.1.weight"]
            model_state_dict["16.branch1.2.weight"]       = state_dict["stage4.0.branch1.2.weight"]
            model_state_dict["16.branch1.3.bias"]         = state_dict["stage4.0.branch1.3.bias"]
            model_state_dict["16.branch1.3.running_mean"] = state_dict["stage4.0.branch1.3.running_mean"]
            model_state_dict["16.branch1.3.running_var"]  = state_dict["stage4.0.branch1.3.running_var"]
            model_state_dict["16.branch1.3.weight"]       = state_dict["stage4.0.branch1.3.weight"]
            model_state_dict["16.branch2.0.weight"]       = state_dict["stage4.0.branch2.0.weight"]
            model_state_dict["16.branch2.1.bias"]         = state_dict["stage4.0.branch2.1.bias"]
            model_state_dict["16.branch2.1.running_mean"] = state_dict["stage4.0.branch2.1.running_mean"]
            model_state_dict["16.branch2.1.running_var"]  = state_dict["stage4.0.branch2.1.running_var"]
            model_state_dict["16.branch2.1.weight"]       = state_dict["stage4.0.branch2.1.weight"]
            model_state_dict["16.branch2.3.weight"]       = state_dict["stage4.0.branch2.3.weight"]
            model_state_dict["16.branch2.4.bias"]         = state_dict["stage4.0.branch2.4.bias"]
            model_state_dict["16.branch2.4.running_mean"] = state_dict["stage4.0.branch2.4.running_mean"]
            model_state_dict["16.branch2.4.running_var"]  = state_dict["stage4.0.branch2.4.running_var"]
            model_state_dict["16.branch2.4.weight"]       = state_dict["stage4.0.branch2.4.weight"]
            model_state_dict["16.branch2.5.weight"]       = state_dict["stage4.0.branch2.5.weight"]
            model_state_dict["16.branch2.6.bias"]         = state_dict["stage4.0.branch2.6.bias"]
            model_state_dict["16.branch2.6.running_mean"] = state_dict["stage4.0.branch2.6.running_mean"]
            model_state_dict["16.branch2.6.running_var"]  = state_dict["stage4.0.branch2.6.running_var"]
            model_state_dict["16.branch2.6.weight"]       = state_dict["stage4.0.branch2.6.weight"]
            model_state_dict["17.branch2.0.weight"]       = state_dict["stage4.1.branch2.0.weight"]
            model_state_dict["17.branch2.1.bias"]         = state_dict["stage4.1.branch2.1.bias"]
            model_state_dict["17.branch2.1.running_mean"] = state_dict["stage4.1.branch2.1.running_mean"]
            model_state_dict["17.branch2.1.running_var"]  = state_dict["stage4.1.branch2.1.running_var"]
            model_state_dict["17.branch2.1.weight"]       = state_dict["stage4.1.branch2.1.weight"]
            model_state_dict["17.branch2.3.weight"]       = state_dict["stage4.1.branch2.3.weight"]
            model_state_dict["17.branch2.4.bias"]         = state_dict["stage4.1.branch2.4.bias"]
            model_state_dict["17.branch2.4.running_mean"] = state_dict["stage4.1.branch2.4.running_mean"]
            model_state_dict["17.branch2.4.running_var"]  = state_dict["stage4.1.branch2.4.running_var"]
            model_state_dict["17.branch2.4.weight"]       = state_dict["stage4.1.branch2.4.weight"]
            model_state_dict["17.branch2.5.weight"]       = state_dict["stage4.1.branch2.5.weight"]
            model_state_dict["17.branch2.6.bias"]         = state_dict["stage4.1.branch2.6.bias"]
            model_state_dict["17.branch2.6.running_mean"] = state_dict["stage4.1.branch2.6.running_mean"]
            model_state_dict["17.branch2.6.running_var"]  = state_dict["stage4.1.branch2.6.running_var"]
            model_state_dict["17.branch2.6.weight"]       = state_dict["stage4.1.branch2.6.weight"]
            model_state_dict["18.branch2.0.weight"]       = state_dict["stage4.2.branch2.0.weight"]
            model_state_dict["18.branch2.1.bias"]         = state_dict["stage4.2.branch2.1.bias"]
            model_state_dict["18.branch2.1.running_mean"] = state_dict["stage4.2.branch2.1.running_mean"]
            model_state_dict["18.branch2.1.running_var"]  = state_dict["stage4.2.branch2.1.running_var"]
            model_state_dict["18.branch2.1.weight"]       = state_dict["stage4.2.branch2.1.weight"]
            model_state_dict["18.branch2.3.weight"]       = state_dict["stage4.2.branch2.3.weight"]
            model_state_dict["18.branch2.4.bias"]         = state_dict["stage4.2.branch2.4.bias"]
            model_state_dict["18.branch2.4.running_mean"] = state_dict["stage4.2.branch2.4.running_mean"]
            model_state_dict["18.branch2.4.running_var"]  = state_dict["stage4.2.branch2.4.running_var"]
            model_state_dict["18.branch2.4.weight"]       = state_dict["stage4.2.branch2.4.weight"]
            model_state_dict["18.branch2.5.weight"]       = state_dict["stage4.2.branch2.5.weight"]
            model_state_dict["18.branch2.6.bias"]         = state_dict["stage4.2.branch2.6.bias"]
            model_state_dict["18.branch2.6.running_mean"] = state_dict["stage4.2.branch2.6.running_mean"]
            model_state_dict["18.branch2.6.running_var"]  = state_dict["stage4.2.branch2.6.running_var"]
            model_state_dict["18.branch2.6.weight"]       = state_dict["stage4.2.branch2.6.weight"]
            model_state_dict["19.branch2.0.weight"]       = state_dict["stage4.3.branch2.0.weight"]
            model_state_dict["19.branch2.1.bias"]         = state_dict["stage4.3.branch2.1.bias"]
            model_state_dict["19.branch2.1.running_mean"] = state_dict["stage4.3.branch2.1.running_mean"]
            model_state_dict["19.branch2.1.running_var"]  = state_dict["stage4.3.branch2.1.running_var"]
            model_state_dict["19.branch2.1.weight"]       = state_dict["stage4.3.branch2.1.weight"]
            model_state_dict["19.branch2.3.weight"]       = state_dict["stage4.3.branch2.3.weight"]
            model_state_dict["19.branch2.4.bias"]         = state_dict["stage4.3.branch2.4.bias"]
            model_state_dict["19.branch2.4.running_mean"] = state_dict["stage4.3.branch2.4.running_mean"]
            model_state_dict["19.branch2.4.running_var"]  = state_dict["stage4.3.branch2.4.running_var"]
            model_state_dict["19.branch2.4.weight"]       = state_dict["stage4.3.branch2.4.weight"]
            model_state_dict["19.branch2.5.weight"]       = state_dict["stage4.3.branch2.5.weight"]
            model_state_dict["19.branch2.6.bias"]         = state_dict["stage4.3.branch2.6.bias"]
            model_state_dict["19.branch2.6.running_mean"] = state_dict["stage4.3.branch2.6.running_mean"]
            model_state_dict["19.branch2.6.running_var"]  = state_dict["stage4.3.branch2.6.running_var"]
            model_state_dict["19.branch2.6.weight"]       = state_dict["stage4.3.branch2.6.weight"]
            model_state_dict["20.weight"]                 = state_dict["conv5.0.weight"]
            model_state_dict["21.bias"]                   = state_dict["conv5.1.bias"]
            model_state_dict["21.running_mean"]           = state_dict["conv5.1.running_mean"]
            model_state_dict["21.running_var"]            = state_dict["conv5.1.running_var"]
            model_state_dict["21.weight"]                 = state_dict["conv5.1.weight"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["23.linear.bias"]   = state_dict["fc.bias"]
                model_state_dict["23.linear.weight"] = state_dict["fc.weight"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()


@MODELS.register(name="shufflenet_v2_x2.0")
class ShuffleNetV2_x2_0(ShuffleNetV2):
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
            path        = "https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth",
            filename    = "shufflenet_v2-x2.0-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "shufflenet_v2_x2.0.yaml",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "shufflenet_v2",
        fullname   : str  | None         = "shufflenet_v2-x2.0",
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
        cfg = cfg or "shufflenet_v2-x2.0"
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
            pretrained  = ShuffleNetV2_x2_0.init_pretrained(pretrained),
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
            model_state_dict["0.weight"]                  = state_dict["conv1.0.weight"]
            model_state_dict["1.bias"]                    = state_dict["conv1.1.bias"]
            model_state_dict["1.running_mean"]            = state_dict["conv1.1.running_mean"]
            model_state_dict["1.running_var"]             = state_dict["conv1.1.running_var"]
            model_state_dict["1.weight"]                  = state_dict["conv1.1.weight"]
            model_state_dict["4.branch1.0.weight"]        = state_dict["stage2.0.branch1.0.weight"]
            model_state_dict["4.branch1.1.bias"]          = state_dict["stage2.0.branch1.1.bias"]
            model_state_dict["4.branch1.1.running_mean"]  = state_dict["stage2.0.branch1.1.running_mean"]
            model_state_dict["4.branch1.1.running_var"]   = state_dict["stage2.0.branch1.1.running_var"]
            model_state_dict["4.branch1.1.weight"]        = state_dict["stage2.0.branch1.1.weight"]
            model_state_dict["4.branch1.2.weight"]        = state_dict["stage2.0.branch1.2.weight"]
            model_state_dict["4.branch1.3.bias"]          = state_dict["stage2.0.branch1.3.bias"]
            model_state_dict["4.branch1.3.running_mean"]  = state_dict["stage2.0.branch1.3.running_mean"]
            model_state_dict["4.branch1.3.running_var"]   = state_dict["stage2.0.branch1.3.running_var"]
            model_state_dict["4.branch1.3.weight"]        = state_dict["stage2.0.branch1.3.weight"]
            model_state_dict["4.branch2.0.weight"]        = state_dict["stage2.0.branch2.0.weight"]
            model_state_dict["4.branch2.1.bias"]          = state_dict["stage2.0.branch2.1.bias"]
            model_state_dict["4.branch2.1.running_mean"]  = state_dict["stage2.0.branch2.1.running_mean"]
            model_state_dict["4.branch2.1.running_var"]   = state_dict["stage2.0.branch2.1.running_var"]
            model_state_dict["4.branch2.1.weight"]        = state_dict["stage2.0.branch2.1.weight"]
            model_state_dict["4.branch2.3.weight"]        = state_dict["stage2.0.branch2.3.weight"]
            model_state_dict["4.branch2.4.bias"]          = state_dict["stage2.0.branch2.4.bias"]
            model_state_dict["4.branch2.4.running_mean"]  = state_dict["stage2.0.branch2.4.running_mean"]
            model_state_dict["4.branch2.4.running_var"]   = state_dict["stage2.0.branch2.4.running_var"]
            model_state_dict["4.branch2.4.weight"]        = state_dict["stage2.0.branch2.4.weight"]
            model_state_dict["4.branch2.5.weight"]        = state_dict["stage2.0.branch2.5.weight"]
            model_state_dict["4.branch2.6.bias"]          = state_dict["stage2.0.branch2.6.bias"]
            model_state_dict["4.branch2.6.running_mean"]  = state_dict["stage2.0.branch2.6.running_mean"]
            model_state_dict["4.branch2.6.running_var"]   = state_dict["stage2.0.branch2.6.running_var"]
            model_state_dict["4.branch2.6.weight"]        = state_dict["stage2.0.branch2.6.weight"]
            model_state_dict["5.branch2.0.weight"]        = state_dict["stage2.1.branch2.0.weight"]
            model_state_dict["5.branch2.1.bias"]          = state_dict["stage2.1.branch2.1.bias"]
            model_state_dict["5.branch2.1.running_mean"]  = state_dict["stage2.1.branch2.1.running_mean"]
            model_state_dict["5.branch2.1.running_var"]   = state_dict["stage2.1.branch2.1.running_var"]
            model_state_dict["5.branch2.1.weight"]        = state_dict["stage2.1.branch2.1.weight"]
            model_state_dict["5.branch2.3.weight"]        = state_dict["stage2.1.branch2.3.weight"]
            model_state_dict["5.branch2.4.bias"]          = state_dict["stage2.1.branch2.4.bias"]
            model_state_dict["5.branch2.4.running_mean"]  = state_dict["stage2.1.branch2.4.running_mean"]
            model_state_dict["5.branch2.4.running_var"]   = state_dict["stage2.1.branch2.4.running_var"]
            model_state_dict["5.branch2.4.weight"]        = state_dict["stage2.1.branch2.4.weight"]
            model_state_dict["5.branch2.5.weight"]        = state_dict["stage2.1.branch2.5.weight"]
            model_state_dict["5.branch2.6.bias"]          = state_dict["stage2.1.branch2.6.bias"]
            model_state_dict["5.branch2.6.running_mean"]  = state_dict["stage2.1.branch2.6.running_mean"]
            model_state_dict["5.branch2.6.running_var"]   = state_dict["stage2.1.branch2.6.running_var"]
            model_state_dict["5.branch2.6.weight"]        = state_dict["stage2.1.branch2.6.weight"]
            model_state_dict["6.branch2.0.weight"]        = state_dict["stage2.2.branch2.0.weight"]
            model_state_dict["6.branch2.1.bias"]          = state_dict["stage2.2.branch2.1.bias"]
            model_state_dict["6.branch2.1.running_mean"]  = state_dict["stage2.2.branch2.1.running_mean"]
            model_state_dict["6.branch2.1.running_var"]   = state_dict["stage2.2.branch2.1.running_var"]
            model_state_dict["6.branch2.1.weight"]        = state_dict["stage2.2.branch2.1.weight"]
            model_state_dict["6.branch2.3.weight"]        = state_dict["stage2.2.branch2.3.weight"]
            model_state_dict["6.branch2.4.bias"]          = state_dict["stage2.2.branch2.4.bias"]
            model_state_dict["6.branch2.4.running_mean"]  = state_dict["stage2.2.branch2.4.running_mean"]
            model_state_dict["6.branch2.4.running_var"]   = state_dict["stage2.2.branch2.4.running_var"]
            model_state_dict["6.branch2.4.weight"]        = state_dict["stage2.2.branch2.4.weight"]
            model_state_dict["6.branch2.5.weight"]        = state_dict["stage2.2.branch2.5.weight"]
            model_state_dict["6.branch2.6.bias"]          = state_dict["stage2.2.branch2.6.bias"]
            model_state_dict["6.branch2.6.running_mean"]  = state_dict["stage2.2.branch2.6.running_mean"]
            model_state_dict["6.branch2.6.running_var"]   = state_dict["stage2.2.branch2.6.running_var"]
            model_state_dict["6.branch2.6.weight"]        = state_dict["stage2.2.branch2.6.weight"]
            model_state_dict["7.branch2.0.weight"]        = state_dict["stage2.3.branch2.0.weight"]
            model_state_dict["7.branch2.1.bias"]          = state_dict["stage2.3.branch2.1.bias"]
            model_state_dict["7.branch2.1.running_mean"]  = state_dict["stage2.3.branch2.1.running_mean"]
            model_state_dict["7.branch2.1.running_var"]   = state_dict["stage2.3.branch2.1.running_var"]
            model_state_dict["7.branch2.1.weight"]        = state_dict["stage2.3.branch2.1.weight"]
            model_state_dict["7.branch2.3.weight"]        = state_dict["stage2.3.branch2.3.weight"]
            model_state_dict["7.branch2.4.bias"]          = state_dict["stage2.3.branch2.4.bias"]
            model_state_dict["7.branch2.4.running_mean"]  = state_dict["stage2.3.branch2.4.running_mean"]
            model_state_dict["7.branch2.4.running_var"]   = state_dict["stage2.3.branch2.4.running_var"]
            model_state_dict["7.branch2.4.weight"]        = state_dict["stage2.3.branch2.4.weight"]
            model_state_dict["7.branch2.5.weight"]        = state_dict["stage2.3.branch2.5.weight"]
            model_state_dict["7.branch2.6.bias"]          = state_dict["stage2.3.branch2.6.bias"]
            model_state_dict["7.branch2.6.running_mean"]  = state_dict["stage2.3.branch2.6.running_mean"]
            model_state_dict["7.branch2.6.running_var"]   = state_dict["stage2.3.branch2.6.running_var"]
            model_state_dict["7.branch2.6.weight"]        = state_dict["stage2.3.branch2.6.weight"]
            model_state_dict["8.branch1.0.weight"]        = state_dict["stage3.0.branch1.0.weight"]
            model_state_dict["8.branch1.1.bias"]          = state_dict["stage3.0.branch1.1.bias"]
            model_state_dict["8.branch1.1.running_mean"]  = state_dict["stage3.0.branch1.1.running_mean"]
            model_state_dict["8.branch1.1.running_var"]   = state_dict["stage3.0.branch1.1.running_var"]
            model_state_dict["8.branch1.1.weight"]        = state_dict["stage3.0.branch1.1.weight"]
            model_state_dict["8.branch1.2.weight"]        = state_dict["stage3.0.branch1.2.weight"]
            model_state_dict["8.branch1.3.bias"]          = state_dict["stage3.0.branch1.3.bias"]
            model_state_dict["8.branch1.3.running_mean"]  = state_dict["stage3.0.branch1.3.running_mean"]
            model_state_dict["8.branch1.3.running_var"]   = state_dict["stage3.0.branch1.3.running_var"]
            model_state_dict["8.branch1.3.weight"]        = state_dict["stage3.0.branch1.3.weight"]
            model_state_dict["8.branch2.0.weight"]        = state_dict["stage3.0.branch2.0.weight"]
            model_state_dict["8.branch2.1.bias"]          = state_dict["stage3.0.branch2.1.bias"]
            model_state_dict["8.branch2.1.running_mean"]  = state_dict["stage3.0.branch2.1.running_mean"]
            model_state_dict["8.branch2.1.running_var"]   = state_dict["stage3.0.branch2.1.running_var"]
            model_state_dict["8.branch2.1.weight"]        = state_dict["stage3.0.branch2.1.weight"]
            model_state_dict["8.branch2.3.weight"]        = state_dict["stage3.0.branch2.3.weight"]
            model_state_dict["8.branch2.4.bias"]          = state_dict["stage3.0.branch2.4.bias"]
            model_state_dict["8.branch2.4.running_mean"]  = state_dict["stage3.0.branch2.4.running_mean"]
            model_state_dict["8.branch2.4.running_var"]   = state_dict["stage3.0.branch2.4.running_var"]
            model_state_dict["8.branch2.4.weight"]        = state_dict["stage3.0.branch2.4.weight"]
            model_state_dict["8.branch2.5.weight"]        = state_dict["stage3.0.branch2.5.weight"]
            model_state_dict["8.branch2.6.bias"]          = state_dict["stage3.0.branch2.6.bias"]
            model_state_dict["8.branch2.6.running_mean"]  = state_dict["stage3.0.branch2.6.running_mean"]
            model_state_dict["8.branch2.6.running_var"]   = state_dict["stage3.0.branch2.6.running_var"]
            model_state_dict["8.branch2.6.weight"]        = state_dict["stage3.0.branch2.6.weight"]
            model_state_dict["9.branch2.0.weight"]        = state_dict["stage3.1.branch2.0.weight"]
            model_state_dict["9.branch2.1.bias"]          = state_dict["stage3.1.branch2.1.bias"]
            model_state_dict["9.branch2.1.running_mean"]  = state_dict["stage3.1.branch2.1.running_mean"]
            model_state_dict["9.branch2.1.running_var"]   = state_dict["stage3.1.branch2.1.running_var"]
            model_state_dict["9.branch2.1.weight"]        = state_dict["stage3.1.branch2.1.weight"]
            model_state_dict["9.branch2.3.weight"]        = state_dict["stage3.1.branch2.3.weight"]
            model_state_dict["9.branch2.4.bias"]          = state_dict["stage3.1.branch2.4.bias"]
            model_state_dict["9.branch2.4.running_mean"]  = state_dict["stage3.1.branch2.4.running_mean"]
            model_state_dict["9.branch2.4.running_var"]   = state_dict["stage3.1.branch2.4.running_var"]
            model_state_dict["9.branch2.4.weight"]        = state_dict["stage3.1.branch2.4.weight"]
            model_state_dict["9.branch2.5.weight"]        = state_dict["stage3.1.branch2.5.weight"]
            model_state_dict["9.branch2.6.bias"]          = state_dict["stage3.1.branch2.6.bias"]
            model_state_dict["9.branch2.6.running_mean"]  = state_dict["stage3.1.branch2.6.running_mean"]
            model_state_dict["9.branch2.6.running_var"]   = state_dict["stage3.1.branch2.6.running_var"]
            model_state_dict["9.branch2.6.weight"]        = state_dict["stage3.1.branch2.6.weight"]
            model_state_dict["10.branch2.0.weight"]       = state_dict["stage3.2.branch2.0.weight"]
            model_state_dict["10.branch2.1.bias"]         = state_dict["stage3.2.branch2.1.bias"]
            model_state_dict["10.branch2.1.running_mean"] = state_dict["stage3.2.branch2.1.running_mean"]
            model_state_dict["10.branch2.1.running_var"]  = state_dict["stage3.2.branch2.1.running_var"]
            model_state_dict["10.branch2.1.weight"]       = state_dict["stage3.2.branch2.1.weight"]
            model_state_dict["10.branch2.3.weight"]       = state_dict["stage3.2.branch2.3.weight"]
            model_state_dict["10.branch2.4.bias"]         = state_dict["stage3.2.branch2.4.bias"]
            model_state_dict["10.branch2.4.running_mean"] = state_dict["stage3.2.branch2.4.running_mean"]
            model_state_dict["10.branch2.4.running_var"]  = state_dict["stage3.2.branch2.4.running_var"]
            model_state_dict["10.branch2.4.weight"]       = state_dict["stage3.2.branch2.4.weight"]
            model_state_dict["10.branch2.5.weight"]       = state_dict["stage3.2.branch2.5.weight"]
            model_state_dict["10.branch2.6.bias"]         = state_dict["stage3.2.branch2.6.bias"]
            model_state_dict["10.branch2.6.running_mean"] = state_dict["stage3.2.branch2.6.running_mean"]
            model_state_dict["10.branch2.6.running_var"]  = state_dict["stage3.2.branch2.6.running_var"]
            model_state_dict["10.branch2.6.weight"]       = state_dict["stage3.2.branch2.6.weight"]
            model_state_dict["11.branch2.0.weight"]       = state_dict["stage3.3.branch2.0.weight"]
            model_state_dict["11.branch2.1.bias"]         = state_dict["stage3.3.branch2.1.bias"]
            model_state_dict["11.branch2.1.running_mean"] = state_dict["stage3.3.branch2.1.running_mean"]
            model_state_dict["11.branch2.1.running_var"]  = state_dict["stage3.3.branch2.1.running_var"]
            model_state_dict["11.branch2.1.weight"]       = state_dict["stage3.3.branch2.1.weight"]
            model_state_dict["11.branch2.3.weight"]       = state_dict["stage3.3.branch2.3.weight"]
            model_state_dict["11.branch2.4.bias"]         = state_dict["stage3.3.branch2.4.bias"]
            model_state_dict["11.branch2.4.running_mean"] = state_dict["stage3.3.branch2.4.running_mean"]
            model_state_dict["11.branch2.4.running_var"]  = state_dict["stage3.3.branch2.4.running_var"]
            model_state_dict["11.branch2.4.weight"]       = state_dict["stage3.3.branch2.4.weight"]
            model_state_dict["11.branch2.5.weight"]       = state_dict["stage3.3.branch2.5.weight"]
            model_state_dict["11.branch2.6.bias"]         = state_dict["stage3.3.branch2.6.bias"]
            model_state_dict["11.branch2.6.running_mean"] = state_dict["stage3.3.branch2.6.running_mean"]
            model_state_dict["11.branch2.6.running_var"]  = state_dict["stage3.3.branch2.6.running_var"]
            model_state_dict["11.branch2.6.weight"]       = state_dict["stage3.3.branch2.6.weight"]
            model_state_dict["12.branch2.0.weight"]       = state_dict["stage3.4.branch2.0.weight"]
            model_state_dict["12.branch2.1.bias"]         = state_dict["stage3.4.branch2.1.bias"]
            model_state_dict["12.branch2.1.running_mean"] = state_dict["stage3.4.branch2.1.running_mean"]
            model_state_dict["12.branch2.1.running_var"]  = state_dict["stage3.4.branch2.1.running_var"]
            model_state_dict["12.branch2.1.weight"]       = state_dict["stage3.4.branch2.1.weight"]
            model_state_dict["12.branch2.3.weight"]       = state_dict["stage3.4.branch2.3.weight"]
            model_state_dict["12.branch2.4.bias"]         = state_dict["stage3.4.branch2.4.bias"]
            model_state_dict["12.branch2.4.running_mean"] = state_dict["stage3.4.branch2.4.running_mean"]
            model_state_dict["12.branch2.4.running_var"]  = state_dict["stage3.4.branch2.4.running_var"]
            model_state_dict["12.branch2.4.weight"]       = state_dict["stage3.4.branch2.4.weight"]
            model_state_dict["12.branch2.5.weight"]       = state_dict["stage3.4.branch2.5.weight"]
            model_state_dict["12.branch2.6.bias"]         = state_dict["stage3.4.branch2.6.bias"]
            model_state_dict["12.branch2.6.running_mean"] = state_dict["stage3.4.branch2.6.running_mean"]
            model_state_dict["12.branch2.6.running_var"]  = state_dict["stage3.4.branch2.6.running_var"]
            model_state_dict["12.branch2.6.weight"]       = state_dict["stage3.4.branch2.6.weight"]
            model_state_dict["13.branch2.0.weight"]       = state_dict["stage3.5.branch2.0.weight"]
            model_state_dict["13.branch2.1.bias"]         = state_dict["stage3.5.branch2.1.bias"]
            model_state_dict["13.branch2.1.running_mean"] = state_dict["stage3.5.branch2.1.running_mean"]
            model_state_dict["13.branch2.1.running_var"]  = state_dict["stage3.5.branch2.1.running_var"]
            model_state_dict["13.branch2.1.weight"]       = state_dict["stage3.5.branch2.1.weight"]
            model_state_dict["13.branch2.3.weight"]       = state_dict["stage3.5.branch2.3.weight"]
            model_state_dict["13.branch2.4.bias"]         = state_dict["stage3.5.branch2.4.bias"]
            model_state_dict["13.branch2.4.running_mean"] = state_dict["stage3.5.branch2.4.running_mean"]
            model_state_dict["13.branch2.4.running_var"]  = state_dict["stage3.5.branch2.4.running_var"]
            model_state_dict["13.branch2.4.weight"]       = state_dict["stage3.5.branch2.4.weight"]
            model_state_dict["13.branch2.5.weight"]       = state_dict["stage3.5.branch2.5.weight"]
            model_state_dict["13.branch2.6.bias"]         = state_dict["stage3.5.branch2.6.bias"]
            model_state_dict["13.branch2.6.running_mean"] = state_dict["stage3.5.branch2.6.running_mean"]
            model_state_dict["13.branch2.6.running_var"]  = state_dict["stage3.5.branch2.6.running_var"]
            model_state_dict["13.branch2.6.weight"]       = state_dict["stage3.5.branch2.6.weight"]
            model_state_dict["14.branch2.0.weight"]       = state_dict["stage3.6.branch2.0.weight"]
            model_state_dict["14.branch2.1.bias"]         = state_dict["stage3.6.branch2.1.bias"]
            model_state_dict["14.branch2.1.running_mean"] = state_dict["stage3.6.branch2.1.running_mean"]
            model_state_dict["14.branch2.1.running_var"]  = state_dict["stage3.6.branch2.1.running_var"]
            model_state_dict["14.branch2.1.weight"]       = state_dict["stage3.6.branch2.1.weight"]
            model_state_dict["14.branch2.3.weight"]       = state_dict["stage3.6.branch2.3.weight"]
            model_state_dict["14.branch2.4.bias"]         = state_dict["stage3.6.branch2.4.bias"]
            model_state_dict["14.branch2.4.running_mean"] = state_dict["stage3.6.branch2.4.running_mean"]
            model_state_dict["14.branch2.4.running_var"]  = state_dict["stage3.6.branch2.4.running_var"]
            model_state_dict["14.branch2.4.weight"]       = state_dict["stage3.6.branch2.4.weight"]
            model_state_dict["14.branch2.5.weight"]       = state_dict["stage3.6.branch2.5.weight"]
            model_state_dict["14.branch2.6.bias"]         = state_dict["stage3.6.branch2.6.bias"]
            model_state_dict["14.branch2.6.running_mean"] = state_dict["stage3.6.branch2.6.running_mean"]
            model_state_dict["14.branch2.6.running_var"]  = state_dict["stage3.6.branch2.6.running_var"]
            model_state_dict["14.branch2.6.weight"]       = state_dict["stage3.6.branch2.6.weight"]
            model_state_dict["15.branch2.0.weight"]       = state_dict["stage3.7.branch2.0.weight"]
            model_state_dict["15.branch2.1.bias"]         = state_dict["stage3.7.branch2.1.bias"]
            model_state_dict["15.branch2.1.running_mean"] = state_dict["stage3.7.branch2.1.running_mean"]
            model_state_dict["15.branch2.1.running_var"]  = state_dict["stage3.7.branch2.1.running_var"]
            model_state_dict["15.branch2.1.weight"]       = state_dict["stage3.7.branch2.1.weight"]
            model_state_dict["15.branch2.3.weight"]       = state_dict["stage3.7.branch2.3.weight"]
            model_state_dict["15.branch2.4.bias"]         = state_dict["stage3.7.branch2.4.bias"]
            model_state_dict["15.branch2.4.running_mean"] = state_dict["stage3.7.branch2.4.running_mean"]
            model_state_dict["15.branch2.4.running_var"]  = state_dict["stage3.7.branch2.4.running_var"]
            model_state_dict["15.branch2.4.weight"]       = state_dict["stage3.7.branch2.4.weight"]
            model_state_dict["15.branch2.5.weight"]       = state_dict["stage3.7.branch2.5.weight"]
            model_state_dict["15.branch2.6.bias"]         = state_dict["stage3.7.branch2.6.bias"]
            model_state_dict["15.branch2.6.running_mean"] = state_dict["stage3.7.branch2.6.running_mean"]
            model_state_dict["15.branch2.6.running_var"]  = state_dict["stage3.7.branch2.6.running_var"]
            model_state_dict["15.branch2.6.weight"]       = state_dict["stage3.7.branch2.6.weight"]
            model_state_dict["16.branch1.0.weight"]       = state_dict["stage4.0.branch1.0.weight"]
            model_state_dict["16.branch1.1.bias"]         = state_dict["stage4.0.branch1.1.bias"]
            model_state_dict["16.branch1.1.running_mean"] = state_dict["stage4.0.branch1.1.running_mean"]
            model_state_dict["16.branch1.1.running_var"]  = state_dict["stage4.0.branch1.1.running_var"]
            model_state_dict["16.branch1.1.weight"]       = state_dict["stage4.0.branch1.1.weight"]
            model_state_dict["16.branch1.2.weight"]       = state_dict["stage4.0.branch1.2.weight"]
            model_state_dict["16.branch1.3.bias"]         = state_dict["stage4.0.branch1.3.bias"]
            model_state_dict["16.branch1.3.running_mean"] = state_dict["stage4.0.branch1.3.running_mean"]
            model_state_dict["16.branch1.3.running_var"]  = state_dict["stage4.0.branch1.3.running_var"]
            model_state_dict["16.branch1.3.weight"]       = state_dict["stage4.0.branch1.3.weight"]
            model_state_dict["16.branch2.0.weight"]       = state_dict["stage4.0.branch2.0.weight"]
            model_state_dict["16.branch2.1.bias"]         = state_dict["stage4.0.branch2.1.bias"]
            model_state_dict["16.branch2.1.running_mean"] = state_dict["stage4.0.branch2.1.running_mean"]
            model_state_dict["16.branch2.1.running_var"]  = state_dict["stage4.0.branch2.1.running_var"]
            model_state_dict["16.branch2.1.weight"]       = state_dict["stage4.0.branch2.1.weight"]
            model_state_dict["16.branch2.3.weight"]       = state_dict["stage4.0.branch2.3.weight"]
            model_state_dict["16.branch2.4.bias"]         = state_dict["stage4.0.branch2.4.bias"]
            model_state_dict["16.branch2.4.running_mean"] = state_dict["stage4.0.branch2.4.running_mean"]
            model_state_dict["16.branch2.4.running_var"]  = state_dict["stage4.0.branch2.4.running_var"]
            model_state_dict["16.branch2.4.weight"]       = state_dict["stage4.0.branch2.4.weight"]
            model_state_dict["16.branch2.5.weight"]       = state_dict["stage4.0.branch2.5.weight"]
            model_state_dict["16.branch2.6.bias"]         = state_dict["stage4.0.branch2.6.bias"]
            model_state_dict["16.branch2.6.running_mean"] = state_dict["stage4.0.branch2.6.running_mean"]
            model_state_dict["16.branch2.6.running_var"]  = state_dict["stage4.0.branch2.6.running_var"]
            model_state_dict["16.branch2.6.weight"]       = state_dict["stage4.0.branch2.6.weight"]
            model_state_dict["17.branch2.0.weight"]       = state_dict["stage4.1.branch2.0.weight"]
            model_state_dict["17.branch2.1.bias"]         = state_dict["stage4.1.branch2.1.bias"]
            model_state_dict["17.branch2.1.running_mean"] = state_dict["stage4.1.branch2.1.running_mean"]
            model_state_dict["17.branch2.1.running_var"]  = state_dict["stage4.1.branch2.1.running_var"]
            model_state_dict["17.branch2.1.weight"]       = state_dict["stage4.1.branch2.1.weight"]
            model_state_dict["17.branch2.3.weight"]       = state_dict["stage4.1.branch2.3.weight"]
            model_state_dict["17.branch2.4.bias"]         = state_dict["stage4.1.branch2.4.bias"]
            model_state_dict["17.branch2.4.running_mean"] = state_dict["stage4.1.branch2.4.running_mean"]
            model_state_dict["17.branch2.4.running_var"]  = state_dict["stage4.1.branch2.4.running_var"]
            model_state_dict["17.branch2.4.weight"]       = state_dict["stage4.1.branch2.4.weight"]
            model_state_dict["17.branch2.5.weight"]       = state_dict["stage4.1.branch2.5.weight"]
            model_state_dict["17.branch2.6.bias"]         = state_dict["stage4.1.branch2.6.bias"]
            model_state_dict["17.branch2.6.running_mean"] = state_dict["stage4.1.branch2.6.running_mean"]
            model_state_dict["17.branch2.6.running_var"]  = state_dict["stage4.1.branch2.6.running_var"]
            model_state_dict["17.branch2.6.weight"]       = state_dict["stage4.1.branch2.6.weight"]
            model_state_dict["18.branch2.0.weight"]       = state_dict["stage4.2.branch2.0.weight"]
            model_state_dict["18.branch2.1.bias"]         = state_dict["stage4.2.branch2.1.bias"]
            model_state_dict["18.branch2.1.running_mean"] = state_dict["stage4.2.branch2.1.running_mean"]
            model_state_dict["18.branch2.1.running_var"]  = state_dict["stage4.2.branch2.1.running_var"]
            model_state_dict["18.branch2.1.weight"]       = state_dict["stage4.2.branch2.1.weight"]
            model_state_dict["18.branch2.3.weight"]       = state_dict["stage4.2.branch2.3.weight"]
            model_state_dict["18.branch2.4.bias"]         = state_dict["stage4.2.branch2.4.bias"]
            model_state_dict["18.branch2.4.running_mean"] = state_dict["stage4.2.branch2.4.running_mean"]
            model_state_dict["18.branch2.4.running_var"]  = state_dict["stage4.2.branch2.4.running_var"]
            model_state_dict["18.branch2.4.weight"]       = state_dict["stage4.2.branch2.4.weight"]
            model_state_dict["18.branch2.5.weight"]       = state_dict["stage4.2.branch2.5.weight"]
            model_state_dict["18.branch2.6.bias"]         = state_dict["stage4.2.branch2.6.bias"]
            model_state_dict["18.branch2.6.running_mean"] = state_dict["stage4.2.branch2.6.running_mean"]
            model_state_dict["18.branch2.6.running_var"]  = state_dict["stage4.2.branch2.6.running_var"]
            model_state_dict["18.branch2.6.weight"]       = state_dict["stage4.2.branch2.6.weight"]
            model_state_dict["19.branch2.0.weight"]       = state_dict["stage4.3.branch2.0.weight"]
            model_state_dict["19.branch2.1.bias"]         = state_dict["stage4.3.branch2.1.bias"]
            model_state_dict["19.branch2.1.running_mean"] = state_dict["stage4.3.branch2.1.running_mean"]
            model_state_dict["19.branch2.1.running_var"]  = state_dict["stage4.3.branch2.1.running_var"]
            model_state_dict["19.branch2.1.weight"]       = state_dict["stage4.3.branch2.1.weight"]
            model_state_dict["19.branch2.3.weight"]       = state_dict["stage4.3.branch2.3.weight"]
            model_state_dict["19.branch2.4.bias"]         = state_dict["stage4.3.branch2.4.bias"]
            model_state_dict["19.branch2.4.running_mean"] = state_dict["stage4.3.branch2.4.running_mean"]
            model_state_dict["19.branch2.4.running_var"]  = state_dict["stage4.3.branch2.4.running_var"]
            model_state_dict["19.branch2.4.weight"]       = state_dict["stage4.3.branch2.4.weight"]
            model_state_dict["19.branch2.5.weight"]       = state_dict["stage4.3.branch2.5.weight"]
            model_state_dict["19.branch2.6.bias"]         = state_dict["stage4.3.branch2.6.bias"]
            model_state_dict["19.branch2.6.running_mean"] = state_dict["stage4.3.branch2.6.running_mean"]
            model_state_dict["19.branch2.6.running_var"]  = state_dict["stage4.3.branch2.6.running_var"]
            model_state_dict["19.branch2.6.weight"]       = state_dict["stage4.3.branch2.6.weight"]
            model_state_dict["20.weight"]                 = state_dict["conv5.0.weight"]
            model_state_dict["21.bias"]                   = state_dict["conv5.1.bias"]
            model_state_dict["21.running_mean"]           = state_dict["conv5.1.running_mean"]
            model_state_dict["21.running_var"]            = state_dict["conv5.1.running_var"]
            model_state_dict["21.weight"]                 = state_dict["conv5.1.weight"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["23.linear.bias"]   = state_dict["fc.bias"]
                model_state_dict["23.linear.weight"] = state_dict["fc.weight"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()
