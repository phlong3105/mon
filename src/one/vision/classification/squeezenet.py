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
    "squeezenet-1.0": {
        "channels": 3,
        "backbone": [
            # [from, number, module,    args(out_channels, ...)]
            [-1,     1,      Conv2d,    [96, 7, 2]],                 # 0
            [-1,     1,      ReLU,      [True]],                     # 1
            [-1,     1,      MaxPool2d, [3, 2, 0, 1, False, True]],  # 2
            [-1,     1,      Fire,      [96,  16, 64,  64]],         # 3
            [-1,     1,      Fire,      [128, 16, 64,  64]],         # 4
            [-1,     1,      Fire,      [128, 32, 128, 128]],        # 5
            [-1,     1,      MaxPool2d, [3, 2, 0, 1, False, True]],  # 6
            [-1,     1,      Fire,      [256, 32, 128, 128]],        # 7
            [-1,     1,      Fire,      [256, 48, 192, 192]],        # 8
            [-1,     1,      Fire,      [384, 48, 192, 192]],        # 9
            [-1,     1,      Fire,      [384, 64, 256, 256]],        # 10
            [-1,     1,      MaxPool2d, [3, 2, 0, 1, False, True]],  # 11
            [-1,     1,      Fire,      [512, 64, 256, 256]],        # 12
        ],
        "head": [
            [-1,     1,      SqueezeNetClassifier, [512]],           # 13
        ]
    },
    "squeezenet-1.1": {
        "channels": 3,
        "backbone": [
            # [from, number, module,    args(out_channels, ...)]
            [-1,     1,      Conv2d,    [64, 3, 2]],                 # 0
            [-1,     1,      ReLU,      [True]],                     # 1
            [-1,     1,      MaxPool2d, [3, 2, 0, 1, False, True]],  # 2
            [-1,     1,      Fire,      [64,  16, 64,  64]],         # 3
            [-1,     1,      Fire,      [128, 16, 64,  64]],         # 4
            [-1,     1,      MaxPool2d, [3, 2, 0, 1, False, True]],  # 5
            [-1,     1,      Fire,      [128, 32, 128, 128]],        # 6
            [-1,     1,      Fire,      [256, 32, 128, 128]],        # 7
            [-1,     1,      MaxPool2d, [3, 2, 0, 1, False, True]],  # 8
            [-1,     1,      Fire,      [256, 48, 192, 192]],        # 9
            [-1,     1,      Fire,      [384, 48, 192, 192]],        # 10
            [-1,     1,      Fire,      [384, 64, 256, 256]],        # 11
            [-1,     1,      Fire,      [512, 64, 256, 256]],        # 12
        ],
        "head": [
            [-1,     1,      SqueezeNetClassifier, [512]],           # 13
        ]
    },
}


@MODELS.register(name="squeezenet")
class SqueezeNet(ImageClassificationModel):
    """
    
    Args:
        cfg (dict | Path_ | None): Model's layers configuration. It can be an
            external .yaml path or a dictionary. Defaults to None means you
            should define each layer manually in `self.parse_model()` method.
        root (Path_): The root directory of the model. Defaults to RUNS_DIR.
        project (str | None): Project name. Defaults to None.
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
        cfg        : dict | Path_ | None = "squeezenet_1.0",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "squeezenet",
        fullname   : str          | None = "squeezenet_1.0",
        channels   : int                 = 3,
        num_classes: int          | None = None,
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
        cfg = cfg or "squeezenet_1.0"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg
        
        super().__init__(
            cfg         = cfg,
            root        = root,
            project     = project,
            name        = name,
            fullname    = fullname,
            channels    = channels,
            num_classes = num_classes,
            classlabels = classlabels,
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
   

@MODELS.register(name="squeezenet-1.0")
class SqueezeNet_1_0(SqueezeNet):
    """
    
    Args:
        cfg (dict | Path_ | None): Model's layers configuration. It can be an
            external .yaml path or a dictionary. Defaults to None means you
            should define each layer manually in `self.parse_model()` method.
        root (Path_): The root directory of the model. Defaults to RUNS_DIR.
        project (str | None): Project name. Defaults to None.
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
            path        = "https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth",
            filename    = "squeezenet-1.0-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "squeezenet-1.0.yaml",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "squeezenet",
        fullname   : str          | None = "squeezenet-1.0",
        channels   : int                 = 3,
        num_classes: int          | None = None,
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
        cfg = cfg or "squeezenet-1.0"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg
        
        super().__init__(
            cfg         = cfg,
            root        = root,
            project     = project,
            name        = name,
            fullname    = fullname,
            channels    = channels,
            num_classes = num_classes,
            classlabels = classlabels,
            pretrained  = SqueezeNet_1_0.init_pretrained(pretrained),
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
            model_state_dict["0.bias"]              = state_dict["features.0.bias"]
            model_state_dict["0.weight"]            = state_dict["features.0.weight"]
            model_state_dict["3.expand1x1.bias"]    = state_dict["features.3.expand1x1.bias"]
            model_state_dict["3.expand1x1.weight"]  = state_dict["features.3.expand1x1.weight"]
            model_state_dict["3.expand3x3.bias"]    = state_dict["features.3.expand3x3.bias"]
            model_state_dict["3.expand3x3.weight"]  = state_dict["features.3.expand3x3.weight"]
            model_state_dict["3.squeeze.bias"]      = state_dict["features.3.squeeze.bias"]
            model_state_dict["3.squeeze.weight"]    = state_dict["features.3.squeeze.weight"]
            model_state_dict["4.expand1x1.bias"]    = state_dict["features.4.expand1x1.bias"]
            model_state_dict["4.expand1x1.weight"]  = state_dict["features.4.expand1x1.weight"]
            model_state_dict["4.expand3x3.bias"]    = state_dict["features.4.expand3x3.bias"]
            model_state_dict["4.expand3x3.weight"]  = state_dict["features.4.expand3x3.weight"]
            model_state_dict["4.squeeze.bias"]      = state_dict["features.4.squeeze.bias"]
            model_state_dict["4.squeeze.weight"]    = state_dict["features.4.squeeze.weight"]
            model_state_dict["5.expand1x1.bias"]    = state_dict["features.5.expand1x1.bias"]
            model_state_dict["5.expand1x1.weight"]  = state_dict["features.5.expand1x1.weight"]
            model_state_dict["5.expand3x3.bias"]    = state_dict["features.5.expand3x3.bias"]
            model_state_dict["5.expand3x3.weight"]  = state_dict["features.5.expand3x3.weight"]
            model_state_dict["5.squeeze.bias"]      = state_dict["features.5.squeeze.bias"]
            model_state_dict["5.squeeze.weight"]    = state_dict["features.5.squeeze.weight"]
            model_state_dict["7.expand1x1.bias"]    = state_dict["features.7.expand1x1.bias"]
            model_state_dict["7.expand1x1.weight"]  = state_dict["features.7.expand1x1.weight"]
            model_state_dict["7.expand3x3.bias"]    = state_dict["features.7.expand3x3.bias"]
            model_state_dict["7.expand3x3.weight"]  = state_dict["features.7.expand3x3.weight"]
            model_state_dict["7.squeeze.bias"]      = state_dict["features.7.squeeze.bias"]
            model_state_dict["7.squeeze.weight"]    = state_dict["features.7.squeeze.weight"]
            model_state_dict["8.expand1x1.bias"]    = state_dict["features.8.expand1x1.bias"]
            model_state_dict["8.expand1x1.weight"]  = state_dict["features.8.expand1x1.weight"]
            model_state_dict["8.expand3x3.bias"]    = state_dict["features.8.expand3x3.bias"]
            model_state_dict["8.expand3x3.weight"]  = state_dict["features.8.expand3x3.weight"]
            model_state_dict["8.squeeze.bias"]      = state_dict["features.8.squeeze.bias"]
            model_state_dict["8.squeeze.weight"]    = state_dict["features.8.squeeze.weight"]
            model_state_dict["9.expand1x1.bias"]    = state_dict["features.9.expand1x1.bias"]
            model_state_dict["9.expand1x1.weight"]  = state_dict["features.9.expand1x1.weight"]
            model_state_dict["9.expand3x3.bias"]    = state_dict["features.9.expand3x3.bias"]
            model_state_dict["9.expand3x3.weight"]  = state_dict["features.9.expand3x3.weight"]
            model_state_dict["9.squeeze.bias"]      = state_dict["features.9.squeeze.bias"]
            model_state_dict["9.squeeze.weight"]    = state_dict["features.9.squeeze.weight"]
            model_state_dict["10.expand1x1.bias"]   = state_dict["features.10.expand1x1.bias"]
            model_state_dict["10.expand1x1.weight"] = state_dict["features.10.expand1x1.weight"]
            model_state_dict["10.expand3x3.bias"]   = state_dict["features.10.expand3x3.bias"]
            model_state_dict["10.expand3x3.weight"] = state_dict["features.10.expand3x3.weight"]
            model_state_dict["10.squeeze.bias"]     = state_dict["features.10.squeeze.bias"]
            model_state_dict["10.squeeze.weight"]   = state_dict["features.10.squeeze.weight"]
            model_state_dict["12.expand1x1.bias"]   = state_dict["features.12.expand1x1.bias"]
            model_state_dict["12.expand1x1.weight"] = state_dict["features.12.expand1x1.weight"]
            model_state_dict["12.expand3x3.bias"]   = state_dict["features.12.expand3x3.bias"]
            model_state_dict["12.expand3x3.weight"] = state_dict["features.12.expand3x3.weight"]
            model_state_dict["12.squeeze.bias"]     = state_dict["features.12.squeeze.bias"]
            model_state_dict["12.squeeze.weight"]   = state_dict["features.12.squeeze.weight"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["13.conv.bias"]   = state_dict["classifier.1.bias"]
                model_state_dict["13.conv.weight"] = state_dict["classifier.1.weight"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()


@MODELS.register(name="squeezenet-1.1")
class SqueezeNet_1_1(SqueezeNet):
    """
    
    Args:
        cfg (dict | Path_ | None): Model's layers configuration. It can be an
            external .yaml path or a dictionary. Defaults to None means you
            should define each layer manually in `self.parse_model()` method.
        root (Path_): The root directory of the model. Defaults to RUNS_DIR.
        project (str | None): Project name. Defaults to None.
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
            path        = "https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",
            filename    = "squeezenet-1.1-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "squeezenet-1.1.yaml",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "squeezenet",
        fullname   : str          | None = "squeezenet-1.1",
        channels   : int                 = 3,
        num_classes: int          | None = None,
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
        cfg = cfg or "squeezenet-1.1"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg
        
        super().__init__(
            cfg         = cfg,
            root        = root,
            project     = project,
            name        = name,
            fullname    = fullname,
            channels    = channels,
            num_classes = num_classes,
            classlabels = classlabels,
            pretrained  = SqueezeNet_1_1.init_pretrained(pretrained),
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
            model_state_dict["0.bias"]              = state_dict["features.0.bias"]
            model_state_dict["0.weight"]            = state_dict["features.0.weight"]
            model_state_dict["3.expand1x1.bias"]    = state_dict["features.3.expand1x1.bias"]
            model_state_dict["3.expand1x1.weight"]  = state_dict["features.3.expand1x1.weight"]
            model_state_dict["3.expand3x3.bias"]    = state_dict["features.3.expand3x3.bias"]
            model_state_dict["3.expand3x3.weight"]  = state_dict["features.3.expand3x3.weight"]
            model_state_dict["3.squeeze.bias"]      = state_dict["features.3.squeeze.bias"]
            model_state_dict["3.squeeze.weight"]    = state_dict["features.3.squeeze.weight"]
            model_state_dict["4.expand1x1.bias"]    = state_dict["features.4.expand1x1.bias"]
            model_state_dict["4.expand1x1.weight"]  = state_dict["features.4.expand1x1.weight"]
            model_state_dict["4.expand3x3.bias"]    = state_dict["features.4.expand3x3.bias"]
            model_state_dict["4.expand3x3.weight"]  = state_dict["features.4.expand3x3.weight"]
            model_state_dict["4.squeeze.bias"]      = state_dict["features.4.squeeze.bias"]
            model_state_dict["4.squeeze.weight"]    = state_dict["features.4.squeeze.weight"]
            model_state_dict["6.expand1x1.bias"]    = state_dict["features.6.expand1x1.bias"]
            model_state_dict["6.expand1x1.weight"]  = state_dict["features.6.expand1x1.weight"]
            model_state_dict["6.expand3x3.bias"]    = state_dict["features.6.expand3x3.bias"]
            model_state_dict["6.expand3x3.weight"]  = state_dict["features.6.expand3x3.weight"]
            model_state_dict["6.squeeze.bias"]      = state_dict["features.6.squeeze.bias"]
            model_state_dict["6.squeeze.weight"]    = state_dict["features.6.squeeze.weight"]
            model_state_dict["7.expand1x1.bias"]    = state_dict["features.7.expand1x1.bias"]
            model_state_dict["7.expand1x1.weight"]  = state_dict["features.7.expand1x1.weight"]
            model_state_dict["7.expand3x3.bias"]    = state_dict["features.7.expand3x3.bias"]
            model_state_dict["7.expand3x3.weight"]  = state_dict["features.7.expand3x3.weight"]
            model_state_dict["7.squeeze.bias"]      = state_dict["features.7.squeeze.bias"]
            model_state_dict["7.squeeze.weight"]    = state_dict["features.7.squeeze.weight"]
            model_state_dict["9.expand1x1.bias"]    = state_dict["features.9.expand1x1.bias"]
            model_state_dict["9.expand1x1.weight"]  = state_dict["features.9.expand1x1.weight"]
            model_state_dict["9.expand3x3.bias"]    = state_dict["features.9.expand3x3.bias"]
            model_state_dict["9.expand3x3.weight"]  = state_dict["features.9.expand3x3.weight"]
            model_state_dict["9.squeeze.bias"]      = state_dict["features.9.squeeze.bias"]
            model_state_dict["9.squeeze.weight"]    = state_dict["features.9.squeeze.weight"]
            model_state_dict["10.expand1x1.bias"]   = state_dict["features.10.expand1x1.bias"]
            model_state_dict["10.expand1x1.weight"] = state_dict["features.10.expand1x1.weight"]
            model_state_dict["10.expand3x3.bias"]   = state_dict["features.10.expand3x3.bias"]
            model_state_dict["10.expand3x3.weight"] = state_dict["features.10.expand3x3.weight"]
            model_state_dict["10.squeeze.bias"]     = state_dict["features.10.squeeze.bias"]
            model_state_dict["10.squeeze.weight"]   = state_dict["features.10.squeeze.weight"]
            model_state_dict["11.expand1x1.bias"]   = state_dict["features.11.expand1x1.bias"]
            model_state_dict["11.expand1x1.weight"] = state_dict["features.11.expand1x1.weight"]
            model_state_dict["11.expand3x3.bias"]   = state_dict["features.11.expand3x3.bias"]
            model_state_dict["11.expand3x3.weight"] = state_dict["features.11.expand3x3.weight"]
            model_state_dict["11.squeeze.bias"]     = state_dict["features.11.squeeze.bias"]
            model_state_dict["11.squeeze.weight"]   = state_dict["features.11.squeeze.weight"]
            model_state_dict["12.expand1x1.bias"]   = state_dict["features.12.expand1x1.bias"]
            model_state_dict["12.expand1x1.weight"] = state_dict["features.12.expand1x1.weight"]
            model_state_dict["12.expand3x3.bias"]   = state_dict["features.12.expand3x3.bias"]
            model_state_dict["12.expand3x3.weight"] = state_dict["features.12.expand3x3.weight"]
            model_state_dict["12.squeeze.bias"]     = state_dict["features.12.squeeze.bias"]
            model_state_dict["12.squeeze.weight"]   = state_dict["features.12.squeeze.weight"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["13.conv.bias"]   = state_dict["classifier.1.bias"]
                model_state_dict["13.conv.weight"] = state_dict["classifier.1.weight"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()
