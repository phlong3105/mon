#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from torchvision.ops import Conv2dNormActivation

from one.nn import *

CURRENT_DIR = Path(__file__).resolve().parent
CFG_DIR     = CURRENT_DIR / "cfg"


# H1: - Model ------------------------------------------------------------------

cfgs = {
    "convnext-base": {
        "channels": 3,
        "backbone": [
            # [from, number, module,               args(out_channels, ...)]
            [-1,     1,      Conv2dNormActivation, [128,  4, 4, 0, 1, LayerNorm2d, None, 1, True, True]],  # 0 total_stage_blocks = 36
            [-1,     1,      ConvNeXtBlock,        [128,  1e-6, 0.5, 3,  0,  36]],                         # 1
            [-1,     1,      LayerNorm2d,          [128]],                                                 # 2
            [-1,     1,      Conv2d,               [256, 2, 2]],                                           # 3
            [-1,     1,      ConvNeXtBlock,        [256,  1e-6, 0.5, 3,  3,  36]],                         # 4
            [-1,     1,      LayerNorm2d,          [256]],                                                 # 5
            [-1,     1,      Conv2d,               [512, 2, 2]],                                           # 6
            [-1,     1,      ConvNeXtBlock,        [512,  1e-6, 0.5, 27, 6,  36]],                         # 7
            [-1,     1,      LayerNorm2d,          [512]],                                                 # 8
            [-1,     1,      Conv2d,               [1024, 2, 2]],                                          # 9
            [-1,     1,      ConvNeXtBlock,        [1024, 1e-6, 0.5, 3,  33, 36]],                         # 10
            [-1,     1,      AdaptiveAvgPool2d,    [1]],                                                   # 11
        ],
        "head": [
            [-1,     1,      ConvNeXtClassifier,   [1024, LayerNorm2d]],                                   # 12
        ]
    },
    "convnext-tiny": {
        "channels": 3,
        "backbone": [
            # [from, number, module,               args(out_channels, ...)]
            [-1,     1,      Conv2dNormActivation, [96,  4, 4, 0, 1, LayerNorm2d, None, 1, True, True]],  # 0 total_stage_blocks = 18
            [-1,     1,      ConvNeXtBlock,        [96,  1e-6, 0.1, 3, 0,  18]],                          # 1
            [-1,     1,      LayerNorm2d,          [96]],                                                 # 2
            [-1,     1,      Conv2d,               [192, 2, 2]],                                          # 3
            [-1,     1,      ConvNeXtBlock,        [192, 1e-6, 0.1, 3, 3,  18]],                          # 4
            [-1,     1,      LayerNorm2d,          [192]],                                                # 5
            [-1,     1,      Conv2d,               [384, 2, 2]],                                          # 6
            [-1,     1,      ConvNeXtBlock,        [384, 1e-6, 0.1, 9, 6,  18]],                          # 7
            [-1,     1,      LayerNorm2d,          [384]],                                                # 8
            [-1,     1,      Conv2d,               [768, 2, 2]],                                          # 9
            [-1,     1,      ConvNeXtBlock,        [768, 1e-6, 0.1, 3, 15, 18]],                          # 10
            [-1,     1,      AdaptiveAvgPool2d,    [1]],                                                  # 11
        ],
        "head": [
            [-1,     1,      ConvNeXtClassifier,   [768, LayerNorm2d]],                                   # 12
        ]
    },
    "convnext-small": {
        "channels": 3,
        "backbone": [
            # [from, number, module,               args(out_channels, ...)]
            [-1,     1,      Conv2dNormActivation, [96,  4, 4, 0, 1, LayerNorm2d, None, 1, True, True]],  # 0 total_stage_blocks = 36
            [-1,     1,      ConvNeXtBlock,        [96,  1e-6, 0.4, 3,  0,  36]],                         # 1
            [-1,     1,      LayerNorm2d,          [96]],                                                 # 2
            [-1,     1,      Conv2d,               [192, 2, 2]],                                          # 3
            [-1,     1,      ConvNeXtBlock,        [192, 1e-6, 0.4, 3,  3,  36]],                         # 4
            [-1,     1,      LayerNorm2d,          [192]],                                                # 5
            [-1,     1,      Conv2d,               [384, 2, 2]],                                          # 6
            [-1,     1,      ConvNeXtBlock,        [384, 1e-6, 0.4, 27, 6,  36]],                         # 7
            [-1,     1,      LayerNorm2d,          [384]],                                                # 8
            [-1,     1,      Conv2d,               [768, 2, 2]],                                          # 9
            [-1,     1,      ConvNeXtBlock,        [768, 1e-6, 0.4, 3,  33, 36]],                         # 10
            [-1,     1,      AdaptiveAvgPool2d,    [1]],                                                  # 11
        ],
        "head": [
            [-1,     1,      ConvNeXtClassifier,   [768, LayerNorm2d]],                                   # 12
        ]
    },
    "convnext-large": {
        "channels": 3,
        "backbone": [
            # [from, number, module,               args(out_channels, ...)]
            [-1,     1,      Conv2dNormActivation, [192, 4, 4, 0, 1, LayerNorm2d, None, 1, True, True]],  # 0 total_stage_blocks = 36
            [-1,     1,      ConvNeXtBlock,        [192, 1e-6, 0.5, 3,  0,  36]],                         # 1
            [-1,     1,      LayerNorm2d,          [192]],                                                # 2
            [-1,     1,      Conv2d,               [384, 2, 2]],                                          # 3
            [-1,     1,      ConvNeXtBlock,        [384, 1e-6, 0.5, 3,  3,  36]],                         # 4
            [-1,     1,      LayerNorm2d,          [384]],                                                # 5
            [-1,     1,      Conv2d,               [768, 2, 2]],                                          # 6
            [-1,     1,      ConvNeXtBlock,        [768, 1e-6, 0.5, 27, 6,  36]],                         # 7
            [-1,     1,      LayerNorm2d,          [768]],                                                # 8
            [-1,     1,      Conv2d,               [1536, 2, 2]],                                         # 9
            [-1,     1,      ConvNeXtBlock,        [1536, 1e-6, 0.5, 3,  33, 36]],                        # 10
            [-1,     1,      AdaptiveAvgPool2d,    [1]],                                                  # 11
        ],
        "head": [
            [-1,     1,      ConvNeXtClassifier,   [1536, LayerNorm2d]],                                  # 12
        ]
    },
}


@MODELS.register(name="convnext")
class ConvNeXt(ImageClassificationModel):
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
        cfg        : dict | Path_ | None = "convnext-base.yaml",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "convnext",
        fullname   : str  | None         = "convnext-base",
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
        cfg = cfg or "convnext-base"
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
                nn.init.trunc_normal_(m.conv.weight, std=0.02)
                if m.conv.bias is not None:
                    nn.init.zeros_(m.conv.bias)
            else:
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        elif classname.find("Linear") != -1:
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                

@MODELS.register(name="convnext-base")
class ConvNeXtBase(ConvNeXt):
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
            path        = "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
            filename    = "convnext-base-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "convnext-base.yaml",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "convnext",
        fullname   : str  | None         = "convnext-base",
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
        cfg = cfg or "convnext-base"
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
            pretrained  = ConvNeXtBase.init_pretrained(pretrained),
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
            model_state_dict["0.0.bias"]                  = state_dict["features.0.0.bias"]
            model_state_dict["0.0.weight"]                = state_dict["features.0.0.weight"]
            model_state_dict["0.1.bias"]                  = state_dict["features.0.1.bias"]
            model_state_dict["0.1.weight"]                = state_dict["features.0.1.weight"]
            model_state_dict["1.block.0.block.0.bias"]    = state_dict["features.1.0.block.0.bias"]
            model_state_dict["1.block.0.block.0.weight"]  = state_dict["features.1.0.block.0.weight"]
            model_state_dict["1.block.0.block.2.bias"]    = state_dict["features.1.0.block.2.bias"]
            model_state_dict["1.block.0.block.2.weight"]  = state_dict["features.1.0.block.2.weight"]
            model_state_dict["1.block.0.block.3.bias"]    = state_dict["features.1.0.block.3.bias"]
            model_state_dict["1.block.0.block.3.weight"]  = state_dict["features.1.0.block.3.weight"]
            model_state_dict["1.block.0.block.5.bias"]    = state_dict["features.1.0.block.5.bias"]
            model_state_dict["1.block.0.block.5.weight"]  = state_dict["features.1.0.block.5.weight"]
            model_state_dict["1.block.0.layer_scale"]     = state_dict["features.1.0.layer_scale"]
            model_state_dict["1.block.1.block.0.bias"]    = state_dict["features.1.1.block.0.bias"]
            model_state_dict["1.block.1.block.0.weight"]  = state_dict["features.1.1.block.0.weight"]
            model_state_dict["1.block.1.block.2.bias"]    = state_dict["features.1.1.block.2.bias"]
            model_state_dict["1.block.1.block.2.weight"]  = state_dict["features.1.1.block.2.weight"]
            model_state_dict["1.block.1.block.3.bias"]    = state_dict["features.1.1.block.3.bias"]
            model_state_dict["1.block.1.block.3.weight"]  = state_dict["features.1.1.block.3.weight"]
            model_state_dict["1.block.1.block.5.bias"]    = state_dict["features.1.1.block.5.bias"]
            model_state_dict["1.block.1.block.5.weight"]  = state_dict["features.1.1.block.5.weight"]
            model_state_dict["1.block.1.layer_scale"]     = state_dict["features.1.1.layer_scale"]
            model_state_dict["1.block.2.block.0.bias"]    = state_dict["features.1.2.block.0.bias"]
            model_state_dict["1.block.2.block.0.weight"]  = state_dict["features.1.2.block.0.weight"]
            model_state_dict["1.block.2.block.2.bias"]    = state_dict["features.1.2.block.2.bias"]
            model_state_dict["1.block.2.block.2.weight"]  = state_dict["features.1.2.block.2.weight"]
            model_state_dict["1.block.2.block.3.bias"]    = state_dict["features.1.2.block.3.bias"]
            model_state_dict["1.block.2.block.3.weight"]  = state_dict["features.1.2.block.3.weight"]
            model_state_dict["1.block.2.block.5.bias"]    = state_dict["features.1.2.block.5.bias"]
            model_state_dict["1.block.2.block.5.weight"]  = state_dict["features.1.2.block.5.weight"]
            model_state_dict["1.block.2.layer_scale"]     = state_dict["features.1.2.layer_scale"]
            model_state_dict["2.bias"]                    = state_dict["features.2.0.bias"]
            model_state_dict["2.weight"]                  = state_dict["features.2.0.weight"]
            model_state_dict["3.bias"]                    = state_dict["features.2.1.bias"]
            model_state_dict["3.weight"]                  = state_dict["features.2.1.weight"]
            model_state_dict["4.block.0.block.0.bias"]    = state_dict["features.3.0.block.0.bias"]
            model_state_dict["4.block.0.block.0.weight"]  = state_dict["features.3.0.block.0.weight"]
            model_state_dict["4.block.0.block.2.bias"]    = state_dict["features.3.0.block.2.bias"]
            model_state_dict["4.block.0.block.2.weight"]  = state_dict["features.3.0.block.2.weight"]
            model_state_dict["4.block.0.block.3.bias"]    = state_dict["features.3.0.block.3.bias"]
            model_state_dict["4.block.0.block.3.weight"]  = state_dict["features.3.0.block.3.weight"]
            model_state_dict["4.block.0.block.5.bias"]    = state_dict["features.3.0.block.5.bias"]
            model_state_dict["4.block.0.block.5.weight"]  = state_dict["features.3.0.block.5.weight"]
            model_state_dict["4.block.0.layer_scale"]     = state_dict["features.3.0.layer_scale"]
            model_state_dict["4.block.1.block.0.bias"]    = state_dict["features.3.1.block.0.bias"]
            model_state_dict["4.block.1.block.0.weight"]  = state_dict["features.3.1.block.0.weight"]
            model_state_dict["4.block.1.block.2.bias"]    = state_dict["features.3.1.block.2.bias"]
            model_state_dict["4.block.1.block.2.weight"]  = state_dict["features.3.1.block.2.weight"]
            model_state_dict["4.block.1.block.3.bias"]    = state_dict["features.3.1.block.3.bias"]
            model_state_dict["4.block.1.block.3.weight"]  = state_dict["features.3.1.block.3.weight"]
            model_state_dict["4.block.1.block.5.bias"]    = state_dict["features.3.1.block.5.bias"]
            model_state_dict["4.block.1.block.5.weight"]  = state_dict["features.3.1.block.5.weight"]
            model_state_dict["4.block.1.layer_scale"]     = state_dict["features.3.1.layer_scale"]
            model_state_dict["4.block.2.block.0.bias"]    = state_dict["features.3.2.block.0.bias"]
            model_state_dict["4.block.2.block.0.weight"]  = state_dict["features.3.2.block.0.weight"]
            model_state_dict["4.block.2.block.2.bias"]    = state_dict["features.3.2.block.2.bias"]
            model_state_dict["4.block.2.block.2.weight"]  = state_dict["features.3.2.block.2.weight"]
            model_state_dict["4.block.2.block.3.bias"]    = state_dict["features.3.2.block.3.bias"]
            model_state_dict["4.block.2.block.3.weight"]  = state_dict["features.3.2.block.3.weight"]
            model_state_dict["4.block.2.block.5.bias"]    = state_dict["features.3.2.block.5.bias"]
            model_state_dict["4.block.2.block.5.weight"]  = state_dict["features.3.2.block.5.weight"]
            model_state_dict["4.block.2.layer_scale"]     = state_dict["features.3.2.layer_scale"]
            model_state_dict["5.bias"]                    = state_dict["features.4.0.bias"]
            model_state_dict["5.weight"]                  = state_dict["features.4.0.weight"]
            model_state_dict["6.bias"]                    = state_dict["features.4.1.bias"]
            model_state_dict["6.weight"]                  = state_dict["features.4.1.weight"]
            model_state_dict["7.block.0.block.0.bias"]    = state_dict["features.5.0.block.0.bias"]
            model_state_dict["7.block.0.block.0.weight"]  = state_dict["features.5.0.block.0.weight"]
            model_state_dict["7.block.0.block.2.bias"]    = state_dict["features.5.0.block.2.bias"]
            model_state_dict["7.block.0.block.2.weight"]  = state_dict["features.5.0.block.2.weight"]
            model_state_dict["7.block.0.block.3.bias"]    = state_dict["features.5.0.block.3.bias"]
            model_state_dict["7.block.0.block.3.weight"]  = state_dict["features.5.0.block.3.weight"]
            model_state_dict["7.block.0.block.5.bias"]    = state_dict["features.5.0.block.5.bias"]
            model_state_dict["7.block.0.block.5.weight"]  = state_dict["features.5.0.block.5.weight"]
            model_state_dict["7.block.0.layer_scale"]     = state_dict["features.5.0.layer_scale"]
            model_state_dict["7.block.1.block.0.bias"]    = state_dict["features.5.1.block.0.bias"]
            model_state_dict["7.block.1.block.0.weight"]  = state_dict["features.5.1.block.0.weight"]
            model_state_dict["7.block.1.block.2.bias"]    = state_dict["features.5.1.block.2.bias"]
            model_state_dict["7.block.1.block.2.weight"]  = state_dict["features.5.1.block.2.weight"]
            model_state_dict["7.block.1.block.3.bias"]    = state_dict["features.5.1.block.3.bias"]
            model_state_dict["7.block.1.block.3.weight"]  = state_dict["features.5.1.block.3.weight"]
            model_state_dict["7.block.1.block.5.bias"]    = state_dict["features.5.1.block.5.bias"]
            model_state_dict["7.block.1.block.5.weight"]  = state_dict["features.5.1.block.5.weight"]
            model_state_dict["7.block.1.layer_scale"]     = state_dict["features.5.1.layer_scale"]
            model_state_dict["7.block.10.block.0.bias"]   = state_dict["features.5.10.block.0.bias"]
            model_state_dict["7.block.10.block.0.weight"] = state_dict["features.5.10.block.0.weight"]
            model_state_dict["7.block.10.block.2.bias"]   = state_dict["features.5.10.block.2.bias"]
            model_state_dict["7.block.10.block.2.weight"] = state_dict["features.5.10.block.2.weight"]
            model_state_dict["7.block.10.block.3.bias"]   = state_dict["features.5.10.block.3.bias"]
            model_state_dict["7.block.10.block.3.weight"] = state_dict["features.5.10.block.3.weight"]
            model_state_dict["7.block.10.block.5.bias"]   = state_dict["features.5.10.block.5.bias"]
            model_state_dict["7.block.10.block.5.weight"] = state_dict["features.5.10.block.5.weight"]
            model_state_dict["7.block.10.layer_scale"]    = state_dict["features.5.10.layer_scale"]
            model_state_dict["7.block.11.block.0.bias"]   = state_dict["features.5.11.block.0.bias"]
            model_state_dict["7.block.11.block.0.weight"] = state_dict["features.5.11.block.0.weight"]
            model_state_dict["7.block.11.block.2.bias"]   = state_dict["features.5.11.block.2.bias"]
            model_state_dict["7.block.11.block.2.weight"] = state_dict["features.5.11.block.2.weight"]
            model_state_dict["7.block.11.block.3.bias"]   = state_dict["features.5.11.block.3.bias"]
            model_state_dict["7.block.11.block.3.weight"] = state_dict["features.5.11.block.3.weight"]
            model_state_dict["7.block.11.block.5.bias"]   = state_dict["features.5.11.block.5.bias"]
            model_state_dict["7.block.11.block.5.weight"] = state_dict["features.5.11.block.5.weight"]
            model_state_dict["7.block.11.layer_scale"]    = state_dict["features.5.11.layer_scale"]
            model_state_dict["7.block.12.block.0.bias"]   = state_dict["features.5.12.block.0.bias"]
            model_state_dict["7.block.12.block.0.weight"] = state_dict["features.5.12.block.0.weight"]
            model_state_dict["7.block.12.block.2.bias"]   = state_dict["features.5.12.block.2.bias"]
            model_state_dict["7.block.12.block.2.weight"] = state_dict["features.5.12.block.2.weight"]
            model_state_dict["7.block.12.block.3.bias"]   = state_dict["features.5.12.block.3.bias"]
            model_state_dict["7.block.12.block.3.weight"] = state_dict["features.5.12.block.3.weight"]
            model_state_dict["7.block.12.block.5.bias"]   = state_dict["features.5.12.block.5.bias"]
            model_state_dict["7.block.12.block.5.weight"] = state_dict["features.5.12.block.5.weight"]
            model_state_dict["7.block.12.layer_scale"]    = state_dict["features.5.12.layer_scale"]
            model_state_dict["7.block.13.block.0.bias"]   = state_dict["features.5.13.block.0.bias"]
            model_state_dict["7.block.13.block.0.weight"] = state_dict["features.5.13.block.0.weight"]
            model_state_dict["7.block.13.block.2.bias"]   = state_dict["features.5.13.block.2.bias"]
            model_state_dict["7.block.13.block.2.weight"] = state_dict["features.5.13.block.2.weight"]
            model_state_dict["7.block.13.block.3.bias"]   = state_dict["features.5.13.block.3.bias"]
            model_state_dict["7.block.13.block.3.weight"] = state_dict["features.5.13.block.3.weight"]
            model_state_dict["7.block.13.block.5.bias"]   = state_dict["features.5.13.block.5.bias"]
            model_state_dict["7.block.13.block.5.weight"] = state_dict["features.5.13.block.5.weight"]
            model_state_dict["7.block.13.layer_scale"]    = state_dict["features.5.13.layer_scale"]
            model_state_dict["7.block.14.block.0.bias"]   = state_dict["features.5.14.block.0.bias"]
            model_state_dict["7.block.14.block.0.weight"] = state_dict["features.5.14.block.0.weight"]
            model_state_dict["7.block.14.block.2.bias"]   = state_dict["features.5.14.block.2.bias"]
            model_state_dict["7.block.14.block.2.weight"] = state_dict["features.5.14.block.2.weight"]
            model_state_dict["7.block.14.block.3.bias"]   = state_dict["features.5.14.block.3.bias"]
            model_state_dict["7.block.14.block.3.weight"] = state_dict["features.5.14.block.3.weight"]
            model_state_dict["7.block.14.block.5.bias"]   = state_dict["features.5.14.block.5.bias"]
            model_state_dict["7.block.14.block.5.weight"] = state_dict["features.5.14.block.5.weight"]
            model_state_dict["7.block.14.layer_scale"]    = state_dict["features.5.14.layer_scale"]
            model_state_dict["7.block.15.block.0.bias"]   = state_dict["features.5.15.block.0.bias"]
            model_state_dict["7.block.15.block.0.weight"] = state_dict["features.5.15.block.0.weight"]
            model_state_dict["7.block.15.block.2.bias"]   = state_dict["features.5.15.block.2.bias"]
            model_state_dict["7.block.15.block.2.weight"] = state_dict["features.5.15.block.2.weight"]
            model_state_dict["7.block.15.block.3.bias"]   = state_dict["features.5.15.block.3.bias"]
            model_state_dict["7.block.15.block.3.weight"] = state_dict["features.5.15.block.3.weight"]
            model_state_dict["7.block.15.block.5.bias"]   = state_dict["features.5.15.block.5.bias"]
            model_state_dict["7.block.15.block.5.weight"] = state_dict["features.5.15.block.5.weight"]
            model_state_dict["7.block.15.layer_scale"]    = state_dict["features.5.15.layer_scale"]
            model_state_dict["7.block.16.block.0.bias"]   = state_dict["features.5.16.block.0.bias"]
            model_state_dict["7.block.16.block.0.weight"] = state_dict["features.5.16.block.0.weight"]
            model_state_dict["7.block.16.block.2.bias"]   = state_dict["features.5.16.block.2.bias"]
            model_state_dict["7.block.16.block.2.weight"] = state_dict["features.5.16.block.2.weight"]
            model_state_dict["7.block.16.block.3.bias"]   = state_dict["features.5.16.block.3.bias"]
            model_state_dict["7.block.16.block.3.weight"] = state_dict["features.5.16.block.3.weight"]
            model_state_dict["7.block.16.block.5.bias"]   = state_dict["features.5.16.block.5.bias"]
            model_state_dict["7.block.16.block.5.weight"] = state_dict["features.5.16.block.5.weight"]
            model_state_dict["7.block.16.layer_scale"]    = state_dict["features.5.16.layer_scale"]
            model_state_dict["7.block.17.block.0.bias"]   = state_dict["features.5.17.block.0.bias"]
            model_state_dict["7.block.17.block.0.weight"] = state_dict["features.5.17.block.0.weight"]
            model_state_dict["7.block.17.block.2.bias"]   = state_dict["features.5.17.block.2.bias"]
            model_state_dict["7.block.17.block.2.weight"] = state_dict["features.5.17.block.2.weight"]
            model_state_dict["7.block.17.block.3.bias"]   = state_dict["features.5.17.block.3.bias"]
            model_state_dict["7.block.17.block.3.weight"] = state_dict["features.5.17.block.3.weight"]
            model_state_dict["7.block.17.block.5.bias"]   = state_dict["features.5.17.block.5.bias"]
            model_state_dict["7.block.17.block.5.weight"] = state_dict["features.5.17.block.5.weight"]
            model_state_dict["7.block.17.layer_scale"]    = state_dict["features.5.17.layer_scale"]
            model_state_dict["7.block.18.block.0.bias"]   = state_dict["features.5.18.block.0.bias"]
            model_state_dict["7.block.18.block.0.weight"] = state_dict["features.5.18.block.0.weight"]
            model_state_dict["7.block.18.block.2.bias"]   = state_dict["features.5.18.block.2.bias"]
            model_state_dict["7.block.18.block.2.weight"] = state_dict["features.5.18.block.2.weight"]
            model_state_dict["7.block.18.block.3.bias"]   = state_dict["features.5.18.block.3.bias"]
            model_state_dict["7.block.18.block.3.weight"] = state_dict["features.5.18.block.3.weight"]
            model_state_dict["7.block.18.block.5.bias"]   = state_dict["features.5.18.block.5.bias"]
            model_state_dict["7.block.18.block.5.weight"] = state_dict["features.5.18.block.5.weight"]
            model_state_dict["7.block.18.layer_scale"]    = state_dict["features.5.18.layer_scale"]
            model_state_dict["7.block.19.block.0.bias"]   = state_dict["features.5.19.block.0.bias"]
            model_state_dict["7.block.19.block.0.weight"] = state_dict["features.5.19.block.0.weight"]
            model_state_dict["7.block.19.block.2.bias"]   = state_dict["features.5.19.block.2.bias"]
            model_state_dict["7.block.19.block.2.weight"] = state_dict["features.5.19.block.2.weight"]
            model_state_dict["7.block.19.block.3.bias"]   = state_dict["features.5.19.block.3.bias"]
            model_state_dict["7.block.19.block.3.weight"] = state_dict["features.5.19.block.3.weight"]
            model_state_dict["7.block.19.block.5.bias"]   = state_dict["features.5.19.block.5.bias"]
            model_state_dict["7.block.19.block.5.weight"] = state_dict["features.5.19.block.5.weight"]
            model_state_dict["7.block.19.layer_scale"]    = state_dict["features.5.19.layer_scale"]
            model_state_dict["7.block.2.block.0.bias"]    = state_dict["features.5.2.block.0.bias"]
            model_state_dict["7.block.2.block.0.weight"]  = state_dict["features.5.2.block.0.weight"]
            model_state_dict["7.block.2.block.2.bias"]    = state_dict["features.5.2.block.2.bias"]
            model_state_dict["7.block.2.block.2.weight"]  = state_dict["features.5.2.block.2.weight"]
            model_state_dict["7.block.2.block.3.bias"]    = state_dict["features.5.2.block.3.bias"]
            model_state_dict["7.block.2.block.3.weight"]  = state_dict["features.5.2.block.3.weight"]
            model_state_dict["7.block.2.block.5.bias"]    = state_dict["features.5.2.block.5.bias"]
            model_state_dict["7.block.2.block.5.weight"]  = state_dict["features.5.2.block.5.weight"]
            model_state_dict["7.block.2.layer_scale"]     = state_dict["features.5.2.layer_scale"]
            model_state_dict["7.block.20.block.0.bias"]   = state_dict["features.5.20.block.0.bias"]
            model_state_dict["7.block.20.block.0.weight"] = state_dict["features.5.20.block.0.weight"]
            model_state_dict["7.block.20.block.2.bias"]   = state_dict["features.5.20.block.2.bias"]
            model_state_dict["7.block.20.block.2.weight"] = state_dict["features.5.20.block.2.weight"]
            model_state_dict["7.block.20.block.3.bias"]   = state_dict["features.5.20.block.3.bias"]
            model_state_dict["7.block.20.block.3.weight"] = state_dict["features.5.20.block.3.weight"]
            model_state_dict["7.block.20.block.5.bias"]   = state_dict["features.5.20.block.5.bias"]
            model_state_dict["7.block.20.block.5.weight"] = state_dict["features.5.20.block.5.weight"]
            model_state_dict["7.block.20.layer_scale"]    = state_dict["features.5.20.layer_scale"]
            model_state_dict["7.block.21.block.0.bias"]   = state_dict["features.5.21.block.0.bias"]
            model_state_dict["7.block.21.block.0.weight"] = state_dict["features.5.21.block.0.weight"]
            model_state_dict["7.block.21.block.2.bias"]   = state_dict["features.5.21.block.2.bias"]
            model_state_dict["7.block.21.block.2.weight"] = state_dict["features.5.21.block.2.weight"]
            model_state_dict["7.block.21.block.3.bias"]   = state_dict["features.5.21.block.3.bias"]
            model_state_dict["7.block.21.block.3.weight"] = state_dict["features.5.21.block.3.weight"]
            model_state_dict["7.block.21.block.5.bias"]   = state_dict["features.5.21.block.5.bias"]
            model_state_dict["7.block.21.block.5.weight"] = state_dict["features.5.21.block.5.weight"]
            model_state_dict["7.block.21.layer_scale"]    = state_dict["features.5.21.layer_scale"]
            model_state_dict["7.block.22.block.0.bias"]   = state_dict["features.5.22.block.0.bias"]
            model_state_dict["7.block.22.block.0.weight"] = state_dict["features.5.22.block.0.weight"]
            model_state_dict["7.block.22.block.2.bias"]   = state_dict["features.5.22.block.2.bias"]
            model_state_dict["7.block.22.block.2.weight"] = state_dict["features.5.22.block.2.weight"]
            model_state_dict["7.block.22.block.3.bias"]   = state_dict["features.5.22.block.3.bias"]
            model_state_dict["7.block.22.block.3.weight"] = state_dict["features.5.22.block.3.weight"]
            model_state_dict["7.block.22.block.5.bias"]   = state_dict["features.5.22.block.5.bias"]
            model_state_dict["7.block.22.block.5.weight"] = state_dict["features.5.22.block.5.weight"]
            model_state_dict["7.block.22.layer_scale"]    = state_dict["features.5.22.layer_scale"]
            model_state_dict["7.block.23.block.0.bias"]   = state_dict["features.5.23.block.0.bias"]
            model_state_dict["7.block.23.block.0.weight"] = state_dict["features.5.23.block.0.weight"]
            model_state_dict["7.block.23.block.2.bias"]   = state_dict["features.5.23.block.2.bias"]
            model_state_dict["7.block.23.block.2.weight"] = state_dict["features.5.23.block.2.weight"]
            model_state_dict["7.block.23.block.3.bias"]   = state_dict["features.5.23.block.3.bias"]
            model_state_dict["7.block.23.block.3.weight"] = state_dict["features.5.23.block.3.weight"]
            model_state_dict["7.block.23.block.5.bias"]   = state_dict["features.5.23.block.5.bias"]
            model_state_dict["7.block.23.block.5.weight"] = state_dict["features.5.23.block.5.weight"]
            model_state_dict["7.block.23.layer_scale"]    = state_dict["features.5.23.layer_scale"]
            model_state_dict["7.block.24.block.0.bias"]   = state_dict["features.5.24.block.0.bias"]
            model_state_dict["7.block.24.block.0.weight"] = state_dict["features.5.24.block.0.weight"]
            model_state_dict["7.block.24.block.2.bias"]   = state_dict["features.5.24.block.2.bias"]
            model_state_dict["7.block.24.block.2.weight"] = state_dict["features.5.24.block.2.weight"]
            model_state_dict["7.block.24.block.3.bias"]   = state_dict["features.5.24.block.3.bias"]
            model_state_dict["7.block.24.block.3.weight"] = state_dict["features.5.24.block.3.weight"]
            model_state_dict["7.block.24.block.5.bias"]   = state_dict["features.5.24.block.5.bias"]
            model_state_dict["7.block.24.block.5.weight"] = state_dict["features.5.24.block.5.weight"]
            model_state_dict["7.block.24.layer_scale"]    = state_dict["features.5.24.layer_scale"]
            model_state_dict["7.block.25.block.0.bias"]   = state_dict["features.5.25.block.0.bias"]
            model_state_dict["7.block.25.block.0.weight"] = state_dict["features.5.25.block.0.weight"]
            model_state_dict["7.block.25.block.2.bias"]   = state_dict["features.5.25.block.2.bias"]
            model_state_dict["7.block.25.block.2.weight"] = state_dict["features.5.25.block.2.weight"]
            model_state_dict["7.block.25.block.3.bias"]   = state_dict["features.5.25.block.3.bias"]
            model_state_dict["7.block.25.block.3.weight"] = state_dict["features.5.25.block.3.weight"]
            model_state_dict["7.block.25.block.5.bias"]   = state_dict["features.5.25.block.5.bias"]
            model_state_dict["7.block.25.block.5.weight"] = state_dict["features.5.25.block.5.weight"]
            model_state_dict["7.block.25.layer_scale"]    = state_dict["features.5.25.layer_scale"]
            model_state_dict["7.block.26.block.0.bias"]   = state_dict["features.5.26.block.0.bias"]
            model_state_dict["7.block.26.block.0.weight"] = state_dict["features.5.26.block.0.weight"]
            model_state_dict["7.block.26.block.2.bias"]   = state_dict["features.5.26.block.2.bias"]
            model_state_dict["7.block.26.block.2.weight"] = state_dict["features.5.26.block.2.weight"]
            model_state_dict["7.block.26.block.3.bias"]   = state_dict["features.5.26.block.3.bias"]
            model_state_dict["7.block.26.block.3.weight"] = state_dict["features.5.26.block.3.weight"]
            model_state_dict["7.block.26.block.5.bias"]   = state_dict["features.5.26.block.5.bias"]
            model_state_dict["7.block.26.block.5.weight"] = state_dict["features.5.26.block.5.weight"]
            model_state_dict["7.block.26.layer_scale"]    = state_dict["features.5.26.layer_scale"]
            model_state_dict["7.block.3.block.0.bias"]    = state_dict["features.5.3.block.0.bias"]
            model_state_dict["7.block.3.block.0.weight"]  = state_dict["features.5.3.block.0.weight"]
            model_state_dict["7.block.3.block.2.bias"]    = state_dict["features.5.3.block.2.bias"]
            model_state_dict["7.block.3.block.2.weight"]  = state_dict["features.5.3.block.2.weight"]
            model_state_dict["7.block.3.block.3.bias"]    = state_dict["features.5.3.block.3.bias"]
            model_state_dict["7.block.3.block.3.weight"]  = state_dict["features.5.3.block.3.weight"]
            model_state_dict["7.block.3.block.5.bias"]    = state_dict["features.5.3.block.5.bias"]
            model_state_dict["7.block.3.block.5.weight"]  = state_dict["features.5.3.block.5.weight"]
            model_state_dict["7.block.3.layer_scale"]     = state_dict["features.5.3.layer_scale"]
            model_state_dict["7.block.4.block.0.bias"]    = state_dict["features.5.4.block.0.bias"]
            model_state_dict["7.block.4.block.0.weight"]  = state_dict["features.5.4.block.0.weight"]
            model_state_dict["7.block.4.block.2.bias"]    = state_dict["features.5.4.block.2.bias"]
            model_state_dict["7.block.4.block.2.weight"]  = state_dict["features.5.4.block.2.weight"]
            model_state_dict["7.block.4.block.3.bias"]    = state_dict["features.5.4.block.3.bias"]
            model_state_dict["7.block.4.block.3.weight"]  = state_dict["features.5.4.block.3.weight"]
            model_state_dict["7.block.4.block.5.bias"]    = state_dict["features.5.4.block.5.bias"]
            model_state_dict["7.block.4.block.5.weight"]  = state_dict["features.5.4.block.5.weight"]
            model_state_dict["7.block.4.layer_scale"]     = state_dict["features.5.4.layer_scale"]
            model_state_dict["7.block.5.block.0.bias"]    = state_dict["features.5.5.block.0.bias"]
            model_state_dict["7.block.5.block.0.weight"]  = state_dict["features.5.5.block.0.weight"]
            model_state_dict["7.block.5.block.2.bias"]    = state_dict["features.5.5.block.2.bias"]
            model_state_dict["7.block.5.block.2.weight"]  = state_dict["features.5.5.block.2.weight"]
            model_state_dict["7.block.5.block.3.bias"]    = state_dict["features.5.5.block.3.bias"]
            model_state_dict["7.block.5.block.3.weight"]  = state_dict["features.5.5.block.3.weight"]
            model_state_dict["7.block.5.block.5.bias"]    = state_dict["features.5.5.block.5.bias"]
            model_state_dict["7.block.5.block.5.weight"]  = state_dict["features.5.5.block.5.weight"]
            model_state_dict["7.block.5.layer_scale"]     = state_dict["features.5.5.layer_scale"]
            model_state_dict["7.block.6.block.0.bias"]    = state_dict["features.5.6.block.0.bias"]
            model_state_dict["7.block.6.block.0.weight"]  = state_dict["features.5.6.block.0.weight"]
            model_state_dict["7.block.6.block.2.bias"]    = state_dict["features.5.6.block.2.bias"]
            model_state_dict["7.block.6.block.2.weight"]  = state_dict["features.5.6.block.2.weight"]
            model_state_dict["7.block.6.block.3.bias"]    = state_dict["features.5.6.block.3.bias"]
            model_state_dict["7.block.6.block.3.weight"]  = state_dict["features.5.6.block.3.weight"]
            model_state_dict["7.block.6.block.5.bias"]    = state_dict["features.5.6.block.5.bias"]
            model_state_dict["7.block.6.block.5.weight"]  = state_dict["features.5.6.block.5.weight"]
            model_state_dict["7.block.6.layer_scale"]     = state_dict["features.5.6.layer_scale"]
            model_state_dict["7.block.7.block.0.bias"]    = state_dict["features.5.7.block.0.bias"]
            model_state_dict["7.block.7.block.0.weight"]  = state_dict["features.5.7.block.0.weight"]
            model_state_dict["7.block.7.block.2.bias"]    = state_dict["features.5.7.block.2.bias"]
            model_state_dict["7.block.7.block.2.weight"]  = state_dict["features.5.7.block.2.weight"]
            model_state_dict["7.block.7.block.3.bias"]    = state_dict["features.5.7.block.3.bias"]
            model_state_dict["7.block.7.block.3.weight"]  = state_dict["features.5.7.block.3.weight"]
            model_state_dict["7.block.7.block.5.bias"]    = state_dict["features.5.7.block.5.bias"]
            model_state_dict["7.block.7.block.5.weight"]  = state_dict["features.5.7.block.5.weight"]
            model_state_dict["7.block.7.layer_scale"]     = state_dict["features.5.7.layer_scale"]
            model_state_dict["7.block.8.block.0.bias"]    = state_dict["features.5.8.block.0.bias"]
            model_state_dict["7.block.8.block.0.weight"]  = state_dict["features.5.8.block.0.weight"]
            model_state_dict["7.block.8.block.2.bias"]    = state_dict["features.5.8.block.2.bias"]
            model_state_dict["7.block.8.block.2.weight"]  = state_dict["features.5.8.block.2.weight"]
            model_state_dict["7.block.8.block.3.bias"]    = state_dict["features.5.8.block.3.bias"]
            model_state_dict["7.block.8.block.3.weight"]  = state_dict["features.5.8.block.3.weight"]
            model_state_dict["7.block.8.block.5.bias"]    = state_dict["features.5.8.block.5.bias"]
            model_state_dict["7.block.8.block.5.weight"]  = state_dict["features.5.8.block.5.weight"]
            model_state_dict["7.block.8.layer_scale"]     = state_dict["features.5.8.layer_scale"]
            model_state_dict["7.block.9.block.0.bias"]    = state_dict["features.5.9.block.0.bias"]
            model_state_dict["7.block.9.block.0.weight"]  = state_dict["features.5.9.block.0.weight"]
            model_state_dict["7.block.9.block.2.bias"]    = state_dict["features.5.9.block.2.bias"]
            model_state_dict["7.block.9.block.2.weight"]  = state_dict["features.5.9.block.2.weight"]
            model_state_dict["7.block.9.block.3.bias"]    = state_dict["features.5.9.block.3.bias"]
            model_state_dict["7.block.9.block.3.weight"]  = state_dict["features.5.9.block.3.weight"]
            model_state_dict["7.block.9.block.5.bias"]    = state_dict["features.5.9.block.5.bias"]
            model_state_dict["7.block.9.block.5.weight"]  = state_dict["features.5.9.block.5.weight"]
            model_state_dict["7.block.9.layer_scale"]     = state_dict["features.5.9.layer_scale"]
            model_state_dict["8.bias"]                    = state_dict["features.6.0.bias"]
            model_state_dict["8.weight"]                  = state_dict["features.6.0.weight"]
            model_state_dict["9.bias"]                    = state_dict["features.6.1.bias"]
            model_state_dict["9.weight"]                  = state_dict["features.6.1.weight"]
            model_state_dict["10.block.0.block.0.bias"]   = state_dict["features.7.0.block.0.bias"]
            model_state_dict["10.block.0.block.0.weight"] = state_dict["features.7.0.block.0.weight"]
            model_state_dict["10.block.0.block.2.bias"]   = state_dict["features.7.0.block.2.bias"]
            model_state_dict["10.block.0.block.2.weight"] = state_dict["features.7.0.block.2.weight"]
            model_state_dict["10.block.0.block.3.bias"]   = state_dict["features.7.0.block.3.bias"]
            model_state_dict["10.block.0.block.3.weight"] = state_dict["features.7.0.block.3.weight"]
            model_state_dict["10.block.0.block.5.bias"]   = state_dict["features.7.0.block.5.bias"]
            model_state_dict["10.block.0.block.5.weight"] = state_dict["features.7.0.block.5.weight"]
            model_state_dict["10.block.0.layer_scale"]    = state_dict["features.7.0.layer_scale"]
            model_state_dict["10.block.1.block.0.bias"]   = state_dict["features.7.1.block.0.bias"]
            model_state_dict["10.block.1.block.0.weight"] = state_dict["features.7.1.block.0.weight"]
            model_state_dict["10.block.1.block.2.bias"]   = state_dict["features.7.1.block.2.bias"]
            model_state_dict["10.block.1.block.2.weight"] = state_dict["features.7.1.block.2.weight"]
            model_state_dict["10.block.1.block.3.bias"]   = state_dict["features.7.1.block.3.bias"]
            model_state_dict["10.block.1.block.3.weight"] = state_dict["features.7.1.block.3.weight"]
            model_state_dict["10.block.1.block.5.bias"]   = state_dict["features.7.1.block.5.bias"]
            model_state_dict["10.block.1.block.5.weight"] = state_dict["features.7.1.block.5.weight"]
            model_state_dict["10.block.1.layer_scale"]    = state_dict["features.7.1.layer_scale"]
            model_state_dict["10.block.2.block.0.bias"]   = state_dict["features.7.2.block.0.bias"]
            model_state_dict["10.block.2.block.0.weight"] = state_dict["features.7.2.block.0.weight"]
            model_state_dict["10.block.2.block.2.bias"]   = state_dict["features.7.2.block.2.bias"]
            model_state_dict["10.block.2.block.2.weight"] = state_dict["features.7.2.block.2.weight"]
            model_state_dict["10.block.2.block.3.bias"]   = state_dict["features.7.2.block.3.bias"]
            model_state_dict["10.block.2.block.3.weight"] = state_dict["features.7.2.block.3.weight"]
            model_state_dict["10.block.2.block.5.bias"]   = state_dict["features.7.2.block.5.bias"]
            model_state_dict["10.block.2.block.5.weight"] = state_dict["features.7.2.block.5.weight"]
            model_state_dict["10.block.2.layer_scale"]    = state_dict["features.7.2.layer_scale"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["12.norm.bias"]     = state_dict["classifier.0.bias"]
                model_state_dict["12.norm.weight"]   = state_dict["classifier.0.weight"]
                model_state_dict["12.linear.bias"]   = state_dict["classifier.2.bias"]
                model_state_dict["12.linear.weight"] = state_dict["classifier.2.weight"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()


@MODELS.register(name="convnext-tiny")
class ConvNeXtTiny(ConvNeXt):
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
            path        = "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
            filename    = "convnext-tiny-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "convnext-tiny.yaml",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "convnext",
        fullname   : str  | None         = "convnext-tiny",
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
        cfg = cfg or "convnext-tiny"
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
            pretrained  = ConvNeXtTiny.init_pretrained(pretrained),
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
            model_state_dict["0.0.bias"]                  = state_dict["features.0.0.bias"]
            model_state_dict["0.0.weight"]                = state_dict["features.0.0.weight"]
            model_state_dict["0.1.bias"]                  = state_dict["features.0.1.bias"]
            model_state_dict["0.1.weight"]                = state_dict["features.0.1.weight"]
            model_state_dict["1.block.0.block.0.bias"]    = state_dict["features.1.0.block.0.bias"]
            model_state_dict["1.block.0.block.0.weight"]  = state_dict["features.1.0.block.0.weight"]
            model_state_dict["1.block.0.block.2.bias"]    = state_dict["features.1.0.block.2.bias"]
            model_state_dict["1.block.0.block.2.weight"]  = state_dict["features.1.0.block.2.weight"]
            model_state_dict["1.block.0.block.3.bias"]    = state_dict["features.1.0.block.3.bias"]
            model_state_dict["1.block.0.block.3.weight"]  = state_dict["features.1.0.block.3.weight"]
            model_state_dict["1.block.0.block.5.bias"]    = state_dict["features.1.0.block.5.bias"]
            model_state_dict["1.block.0.block.5.weight"]  = state_dict["features.1.0.block.5.weight"]
            model_state_dict["1.block.0.layer_scale"]     = state_dict["features.1.0.layer_scale"]
            model_state_dict["1.block.1.block.0.bias"]    = state_dict["features.1.1.block.0.bias"]
            model_state_dict["1.block.1.block.0.weight"]  = state_dict["features.1.1.block.0.weight"]
            model_state_dict["1.block.1.block.2.bias"]    = state_dict["features.1.1.block.2.bias"]
            model_state_dict["1.block.1.block.2.weight"]  = state_dict["features.1.1.block.2.weight"]
            model_state_dict["1.block.1.block.3.bias"]    = state_dict["features.1.1.block.3.bias"]
            model_state_dict["1.block.1.block.3.weight"]  = state_dict["features.1.1.block.3.weight"]
            model_state_dict["1.block.1.block.5.bias"]    = state_dict["features.1.1.block.5.bias"]
            model_state_dict["1.block.1.block.5.weight"]  = state_dict["features.1.1.block.5.weight"]
            model_state_dict["1.block.1.layer_scale"]     = state_dict["features.1.1.layer_scale"]
            model_state_dict["1.block.2.block.0.bias"]    = state_dict["features.1.2.block.0.bias"]
            model_state_dict["1.block.2.block.0.weight"]  = state_dict["features.1.2.block.0.weight"]
            model_state_dict["1.block.2.block.2.bias"]    = state_dict["features.1.2.block.2.bias"]
            model_state_dict["1.block.2.block.2.weight"]  = state_dict["features.1.2.block.2.weight"]
            model_state_dict["1.block.2.block.3.bias"]    = state_dict["features.1.2.block.3.bias"]
            model_state_dict["1.block.2.block.3.weight"]  = state_dict["features.1.2.block.3.weight"]
            model_state_dict["1.block.2.block.5.bias"]    = state_dict["features.1.2.block.5.bias"]
            model_state_dict["1.block.2.block.5.weight"]  = state_dict["features.1.2.block.5.weight"]
            model_state_dict["1.block.2.layer_scale"]     = state_dict["features.1.2.layer_scale"]
            model_state_dict["2.bias"]                    = state_dict["features.2.0.bias"]
            model_state_dict["2.weight"]                  = state_dict["features.2.0.weight"]
            model_state_dict["3.bias"]                    = state_dict["features.2.1.bias"]
            model_state_dict["3.weight"]                  = state_dict["features.2.1.weight"]
            model_state_dict["4.block.0.block.0.bias"]    = state_dict["features.3.0.block.0.bias"]
            model_state_dict["4.block.0.block.0.weight"]  = state_dict["features.3.0.block.0.weight"]
            model_state_dict["4.block.0.block.2.bias"]    = state_dict["features.3.0.block.2.bias"]
            model_state_dict["4.block.0.block.2.weight"]  = state_dict["features.3.0.block.2.weight"]
            model_state_dict["4.block.0.block.3.bias"]    = state_dict["features.3.0.block.3.bias"]
            model_state_dict["4.block.0.block.3.weight"]  = state_dict["features.3.0.block.3.weight"]
            model_state_dict["4.block.0.block.5.bias"]    = state_dict["features.3.0.block.5.bias"]
            model_state_dict["4.block.0.block.5.weight"]  = state_dict["features.3.0.block.5.weight"]
            model_state_dict["4.block.0.layer_scale"]     = state_dict["features.3.0.layer_scale"]
            model_state_dict["4.block.1.block.0.bias"]    = state_dict["features.3.1.block.0.bias"]
            model_state_dict["4.block.1.block.0.weight"]  = state_dict["features.3.1.block.0.weight"]
            model_state_dict["4.block.1.block.2.bias"]    = state_dict["features.3.1.block.2.bias"]
            model_state_dict["4.block.1.block.2.weight"]  = state_dict["features.3.1.block.2.weight"]
            model_state_dict["4.block.1.block.3.bias"]    = state_dict["features.3.1.block.3.bias"]
            model_state_dict["4.block.1.block.3.weight"]  = state_dict["features.3.1.block.3.weight"]
            model_state_dict["4.block.1.block.5.bias"]    = state_dict["features.3.1.block.5.bias"]
            model_state_dict["4.block.1.block.5.weight"]  = state_dict["features.3.1.block.5.weight"]
            model_state_dict["4.block.1.layer_scale"]     = state_dict["features.3.1.layer_scale"]
            model_state_dict["4.block.2.block.0.bias"]    = state_dict["features.3.2.block.0.bias"]
            model_state_dict["4.block.2.block.0.weight"]  = state_dict["features.3.2.block.0.weight"]
            model_state_dict["4.block.2.block.2.bias"]    = state_dict["features.3.2.block.2.bias"]
            model_state_dict["4.block.2.block.2.weight"]  = state_dict["features.3.2.block.2.weight"]
            model_state_dict["4.block.2.block.3.bias"]    = state_dict["features.3.2.block.3.bias"]
            model_state_dict["4.block.2.block.3.weight"]  = state_dict["features.3.2.block.3.weight"]
            model_state_dict["4.block.2.block.5.bias"]    = state_dict["features.3.2.block.5.bias"]
            model_state_dict["4.block.2.block.5.weight"]  = state_dict["features.3.2.block.5.weight"]
            model_state_dict["4.block.2.layer_scale"]     = state_dict["features.3.2.layer_scale"]
            model_state_dict["5.bias"]                    = state_dict["features.4.0.bias"]
            model_state_dict["5.weight"]                  = state_dict["features.4.0.weight"]
            model_state_dict["6.bias"]                    = state_dict["features.4.1.bias"]
            model_state_dict["6.weight"]                  = state_dict["features.4.1.weight"]
            model_state_dict["7.block.0.block.0.bias"]    = state_dict["features.5.0.block.0.bias"]
            model_state_dict["7.block.0.block.0.weight"]  = state_dict["features.5.0.block.0.weight"]
            model_state_dict["7.block.0.block.2.bias"]    = state_dict["features.5.0.block.2.bias"]
            model_state_dict["7.block.0.block.2.weight"]  = state_dict["features.5.0.block.2.weight"]
            model_state_dict["7.block.0.block.3.bias"]    = state_dict["features.5.0.block.3.bias"]
            model_state_dict["7.block.0.block.3.weight"]  = state_dict["features.5.0.block.3.weight"]
            model_state_dict["7.block.0.block.5.bias"]    = state_dict["features.5.0.block.5.bias"]
            model_state_dict["7.block.0.block.5.weight"]  = state_dict["features.5.0.block.5.weight"]
            model_state_dict["7.block.0.layer_scale"]     = state_dict["features.5.0.layer_scale"]
            model_state_dict["7.block.1.block.0.bias"]    = state_dict["features.5.1.block.0.bias"]
            model_state_dict["7.block.1.block.0.weight"]  = state_dict["features.5.1.block.0.weight"]
            model_state_dict["7.block.1.block.2.bias"]    = state_dict["features.5.1.block.2.bias"]
            model_state_dict["7.block.1.block.2.weight"]  = state_dict["features.5.1.block.2.weight"]
            model_state_dict["7.block.1.block.3.bias"]    = state_dict["features.5.1.block.3.bias"]
            model_state_dict["7.block.1.block.3.weight"]  = state_dict["features.5.1.block.3.weight"]
            model_state_dict["7.block.1.block.5.bias"]    = state_dict["features.5.1.block.5.bias"]
            model_state_dict["7.block.1.block.5.weight"]  = state_dict["features.5.1.block.5.weight"]
            model_state_dict["7.block.1.layer_scale"]     = state_dict["features.5.1.layer_scale"]
            model_state_dict["7.block.2.block.0.bias"]    = state_dict["features.5.2.block.0.bias"]
            model_state_dict["7.block.2.block.0.weight"]  = state_dict["features.5.2.block.0.weight"]
            model_state_dict["7.block.2.block.2.bias"]    = state_dict["features.5.2.block.2.bias"]
            model_state_dict["7.block.2.block.2.weight"]  = state_dict["features.5.2.block.2.weight"]
            model_state_dict["7.block.2.block.3.bias"]    = state_dict["features.5.2.block.3.bias"]
            model_state_dict["7.block.2.block.3.weight"]  = state_dict["features.5.2.block.3.weight"]
            model_state_dict["7.block.2.block.5.bias"]    = state_dict["features.5.2.block.5.bias"]
            model_state_dict["7.block.2.block.5.weight"]  = state_dict["features.5.2.block.5.weight"]
            model_state_dict["7.block.2.layer_scale"]     = state_dict["features.5.2.layer_scale"]
            model_state_dict["7.block.3.block.0.bias"]    = state_dict["features.5.3.block.0.bias"]
            model_state_dict["7.block.3.block.0.weight"]  = state_dict["features.5.3.block.0.weight"]
            model_state_dict["7.block.3.block.2.bias"]    = state_dict["features.5.3.block.2.bias"]
            model_state_dict["7.block.3.block.2.weight"]  = state_dict["features.5.3.block.2.weight"]
            model_state_dict["7.block.3.block.3.bias"]    = state_dict["features.5.3.block.3.bias"]
            model_state_dict["7.block.3.block.3.weight"]  = state_dict["features.5.3.block.3.weight"]
            model_state_dict["7.block.3.block.5.bias"]    = state_dict["features.5.3.block.5.bias"]
            model_state_dict["7.block.3.block.5.weight"]  = state_dict["features.5.3.block.5.weight"]
            model_state_dict["7.block.3.layer_scale"]     = state_dict["features.5.3.layer_scale"]
            model_state_dict["7.block.4.block.0.bias"]    = state_dict["features.5.4.block.0.bias"]
            model_state_dict["7.block.4.block.0.weight"]  = state_dict["features.5.4.block.0.weight"]
            model_state_dict["7.block.4.block.2.bias"]    = state_dict["features.5.4.block.2.bias"]
            model_state_dict["7.block.4.block.2.weight"]  = state_dict["features.5.4.block.2.weight"]
            model_state_dict["7.block.4.block.3.bias"]    = state_dict["features.5.4.block.3.bias"]
            model_state_dict["7.block.4.block.3.weight"]  = state_dict["features.5.4.block.3.weight"]
            model_state_dict["7.block.4.block.5.bias"]    = state_dict["features.5.4.block.5.bias"]
            model_state_dict["7.block.4.block.5.weight"]  = state_dict["features.5.4.block.5.weight"]
            model_state_dict["7.block.4.layer_scale"]     = state_dict["features.5.4.layer_scale"]
            model_state_dict["7.block.5.block.0.bias"]    = state_dict["features.5.5.block.0.bias"]
            model_state_dict["7.block.5.block.0.weight"]  = state_dict["features.5.5.block.0.weight"]
            model_state_dict["7.block.5.block.2.bias"]    = state_dict["features.5.5.block.2.bias"]
            model_state_dict["7.block.5.block.2.weight"]  = state_dict["features.5.5.block.2.weight"]
            model_state_dict["7.block.5.block.3.bias"]    = state_dict["features.5.5.block.3.bias"]
            model_state_dict["7.block.5.block.3.weight"]  = state_dict["features.5.5.block.3.weight"]
            model_state_dict["7.block.5.block.5.bias"]    = state_dict["features.5.5.block.5.bias"]
            model_state_dict["7.block.5.block.5.weight"]  = state_dict["features.5.5.block.5.weight"]
            model_state_dict["7.block.5.layer_scale"]     = state_dict["features.5.5.layer_scale"]
            model_state_dict["7.block.6.block.0.bias"]    = state_dict["features.5.6.block.0.bias"]
            model_state_dict["7.block.6.block.0.weight"]  = state_dict["features.5.6.block.0.weight"]
            model_state_dict["7.block.6.block.2.bias"]    = state_dict["features.5.6.block.2.bias"]
            model_state_dict["7.block.6.block.2.weight"]  = state_dict["features.5.6.block.2.weight"]
            model_state_dict["7.block.6.block.3.bias"]    = state_dict["features.5.6.block.3.bias"]
            model_state_dict["7.block.6.block.3.weight"]  = state_dict["features.5.6.block.3.weight"]
            model_state_dict["7.block.6.block.5.bias"]    = state_dict["features.5.6.block.5.bias"]
            model_state_dict["7.block.6.block.5.weight"]  = state_dict["features.5.6.block.5.weight"]
            model_state_dict["7.block.6.layer_scale"]     = state_dict["features.5.6.layer_scale"]
            model_state_dict["7.block.7.block.0.bias"]    = state_dict["features.5.7.block.0.bias"]
            model_state_dict["7.block.7.block.0.weight"]  = state_dict["features.5.7.block.0.weight"]
            model_state_dict["7.block.7.block.2.bias"]    = state_dict["features.5.7.block.2.bias"]
            model_state_dict["7.block.7.block.2.weight"]  = state_dict["features.5.7.block.2.weight"]
            model_state_dict["7.block.7.block.3.bias"]    = state_dict["features.5.7.block.3.bias"]
            model_state_dict["7.block.7.block.3.weight"]  = state_dict["features.5.7.block.3.weight"]
            model_state_dict["7.block.7.block.5.bias"]    = state_dict["features.5.7.block.5.bias"]
            model_state_dict["7.block.7.block.5.weight"]  = state_dict["features.5.7.block.5.weight"]
            model_state_dict["7.block.7.layer_scale"]     = state_dict["features.5.7.layer_scale"]
            model_state_dict["7.block.8.block.0.bias"]    = state_dict["features.5.8.block.0.bias"]
            model_state_dict["7.block.8.block.0.weight"]  = state_dict["features.5.8.block.0.weight"]
            model_state_dict["7.block.8.block.2.bias"]    = state_dict["features.5.8.block.2.bias"]
            model_state_dict["7.block.8.block.2.weight"]  = state_dict["features.5.8.block.2.weight"]
            model_state_dict["7.block.8.block.3.bias"]    = state_dict["features.5.8.block.3.bias"]
            model_state_dict["7.block.8.block.3.weight"]  = state_dict["features.5.8.block.3.weight"]
            model_state_dict["7.block.8.block.5.bias"]    = state_dict["features.5.8.block.5.bias"]
            model_state_dict["7.block.8.block.5.weight"]  = state_dict["features.5.8.block.5.weight"]
            model_state_dict["7.block.8.layer_scale"]     = state_dict["features.5.8.layer_scale"]
            model_state_dict["8.bias"]                    = state_dict["features.6.0.bias"]
            model_state_dict["8.weight"]                  = state_dict["features.6.0.weight"]
            model_state_dict["9.bias"]                    = state_dict["features.6.1.bias"]
            model_state_dict["9.weight"]                  = state_dict["features.6.1.weight"]
            model_state_dict["10.block.0.block.0.bias"]   = state_dict["features.7.0.block.0.bias"]
            model_state_dict["10.block.0.block.0.weight"] = state_dict["features.7.0.block.0.weight"]
            model_state_dict["10.block.0.block.2.bias"]   = state_dict["features.7.0.block.2.bias"]
            model_state_dict["10.block.0.block.2.weight"] = state_dict["features.7.0.block.2.weight"]
            model_state_dict["10.block.0.block.3.bias"]   = state_dict["features.7.0.block.3.bias"]
            model_state_dict["10.block.0.block.3.weight"] = state_dict["features.7.0.block.3.weight"]
            model_state_dict["10.block.0.block.5.bias"]   = state_dict["features.7.0.block.5.bias"]
            model_state_dict["10.block.0.block.5.weight"] = state_dict["features.7.0.block.5.weight"]
            model_state_dict["10.block.0.layer_scale"]    = state_dict["features.7.0.layer_scale"]
            model_state_dict["10.block.1.block.0.bias"]   = state_dict["features.7.1.block.0.bias"]
            model_state_dict["10.block.1.block.0.weight"] = state_dict["features.7.1.block.0.weight"]
            model_state_dict["10.block.1.block.2.bias"]   = state_dict["features.7.1.block.2.bias"]
            model_state_dict["10.block.1.block.2.weight"] = state_dict["features.7.1.block.2.weight"]
            model_state_dict["10.block.1.block.3.bias"]   = state_dict["features.7.1.block.3.bias"]
            model_state_dict["10.block.1.block.3.weight"] = state_dict["features.7.1.block.3.weight"]
            model_state_dict["10.block.1.block.5.bias"]   = state_dict["features.7.1.block.5.bias"]
            model_state_dict["10.block.1.block.5.weight"] = state_dict["features.7.1.block.5.weight"]
            model_state_dict["10.block.1.layer_scale"]    = state_dict["features.7.1.layer_scale"]
            model_state_dict["10.block.2.block.0.bias"]   = state_dict["features.7.2.block.0.bias"]
            model_state_dict["10.block.2.block.0.weight"] = state_dict["features.7.2.block.0.weight"]
            model_state_dict["10.block.2.block.2.bias"]   = state_dict["features.7.2.block.2.bias"]
            model_state_dict["10.block.2.block.2.weight"] = state_dict["features.7.2.block.2.weight"]
            model_state_dict["10.block.2.block.3.bias"]   = state_dict["features.7.2.block.3.bias"]
            model_state_dict["10.block.2.block.3.weight"] = state_dict["features.7.2.block.3.weight"]
            model_state_dict["10.block.2.block.5.bias"]   = state_dict["features.7.2.block.5.bias"]
            model_state_dict["10.block.2.block.5.weight"] = state_dict["features.7.2.block.5.weight"]
            model_state_dict["10.block.2.layer_scale"]    = state_dict["features.7.2.layer_scale"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["12.norm.bias"]     = state_dict["classifier.0.bias"]
                model_state_dict["12.norm.weight"]   = state_dict["classifier.0.weight"]
                model_state_dict["12.linear.bias"]   = state_dict["classifier.2.bias"]
                model_state_dict["12.linear.weight"] = state_dict["classifier.2.weight"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()


@MODELS.register(name="convnext-small")
class ConvNeXtSmall(ConvNeXt):
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
            path        = "https://download.pytorch.org/models/convnext_small-0c510722.pth",
            filename    = "convnext-small-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "convnext-small",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "convnext",
        fullname   : str  | None         = "convnext-small",
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
        cfg = cfg or "convnext-small"
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
            pretrained  = ConvNeXtSmall.init_pretrained(pretrained),
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
            model_state_dict["0.0.bias"]                  = state_dict["features.0.0.bias"]
            model_state_dict["0.0.weight"]                = state_dict["features.0.0.weight"]
            model_state_dict["0.1.bias"]                  = state_dict["features.0.1.bias"]
            model_state_dict["0.1.weight"]                = state_dict["features.0.1.weight"]
            model_state_dict["1.block.0.block.0.bias"]    = state_dict["features.1.0.block.0.bias"]
            model_state_dict["1.block.0.block.0.weight"]  = state_dict["features.1.0.block.0.weight"]
            model_state_dict["1.block.0.block.2.bias"]    = state_dict["features.1.0.block.2.bias"]
            model_state_dict["1.block.0.block.2.weight"]  = state_dict["features.1.0.block.2.weight"]
            model_state_dict["1.block.0.block.3.bias"]    = state_dict["features.1.0.block.3.bias"]
            model_state_dict["1.block.0.block.3.weight"]  = state_dict["features.1.0.block.3.weight"]
            model_state_dict["1.block.0.block.5.bias"]    = state_dict["features.1.0.block.5.bias"]
            model_state_dict["1.block.0.block.5.weight"]  = state_dict["features.1.0.block.5.weight"]
            model_state_dict["1.block.0.layer_scale"]     = state_dict["features.1.0.layer_scale"]
            model_state_dict["1.block.1.block.0.bias"]    = state_dict["features.1.1.block.0.bias"]
            model_state_dict["1.block.1.block.0.weight"]  = state_dict["features.1.1.block.0.weight"]
            model_state_dict["1.block.1.block.2.bias"]    = state_dict["features.1.1.block.2.bias"]
            model_state_dict["1.block.1.block.2.weight"]  = state_dict["features.1.1.block.2.weight"]
            model_state_dict["1.block.1.block.3.bias"]    = state_dict["features.1.1.block.3.bias"]
            model_state_dict["1.block.1.block.3.weight"]  = state_dict["features.1.1.block.3.weight"]
            model_state_dict["1.block.1.block.5.bias"]    = state_dict["features.1.1.block.5.bias"]
            model_state_dict["1.block.1.block.5.weight"]  = state_dict["features.1.1.block.5.weight"]
            model_state_dict["1.block.1.layer_scale"]     = state_dict["features.1.1.layer_scale"]
            model_state_dict["1.block.2.block.0.bias"]    = state_dict["features.1.2.block.0.bias"]
            model_state_dict["1.block.2.block.0.weight"]  = state_dict["features.1.2.block.0.weight"]
            model_state_dict["1.block.2.block.2.bias"]    = state_dict["features.1.2.block.2.bias"]
            model_state_dict["1.block.2.block.2.weight"]  = state_dict["features.1.2.block.2.weight"]
            model_state_dict["1.block.2.block.3.bias"]    = state_dict["features.1.2.block.3.bias"]
            model_state_dict["1.block.2.block.3.weight"]  = state_dict["features.1.2.block.3.weight"]
            model_state_dict["1.block.2.block.5.bias"]    = state_dict["features.1.2.block.5.bias"]
            model_state_dict["1.block.2.block.5.weight"]  = state_dict["features.1.2.block.5.weight"]
            model_state_dict["1.block.2.layer_scale"]     = state_dict["features.1.2.layer_scale"]
            model_state_dict["2.bias"]                    = state_dict["features.2.0.bias"]
            model_state_dict["2.weight"]                  = state_dict["features.2.0.weight"]
            model_state_dict["3.bias"]                    = state_dict["features.2.1.bias"]
            model_state_dict["3.weight"]                  = state_dict["features.2.1.weight"]
            model_state_dict["4.block.0.block.0.bias"]    = state_dict["features.3.0.block.0.bias"]
            model_state_dict["4.block.0.block.0.weight"]  = state_dict["features.3.0.block.0.weight"]
            model_state_dict["4.block.0.block.2.bias"]    = state_dict["features.3.0.block.2.bias"]
            model_state_dict["4.block.0.block.2.weight"]  = state_dict["features.3.0.block.2.weight"]
            model_state_dict["4.block.0.block.3.bias"]    = state_dict["features.3.0.block.3.bias"]
            model_state_dict["4.block.0.block.3.weight"]  = state_dict["features.3.0.block.3.weight"]
            model_state_dict["4.block.0.block.5.bias"]    = state_dict["features.3.0.block.5.bias"]
            model_state_dict["4.block.0.block.5.weight"]  = state_dict["features.3.0.block.5.weight"]
            model_state_dict["4.block.0.layer_scale"]     = state_dict["features.3.0.layer_scale"]
            model_state_dict["4.block.1.block.0.bias"]    = state_dict["features.3.1.block.0.bias"]
            model_state_dict["4.block.1.block.0.weight"]  = state_dict["features.3.1.block.0.weight"]
            model_state_dict["4.block.1.block.2.bias"]    = state_dict["features.3.1.block.2.bias"]
            model_state_dict["4.block.1.block.2.weight"]  = state_dict["features.3.1.block.2.weight"]
            model_state_dict["4.block.1.block.3.bias"]    = state_dict["features.3.1.block.3.bias"]
            model_state_dict["4.block.1.block.3.weight"]  = state_dict["features.3.1.block.3.weight"]
            model_state_dict["4.block.1.block.5.bias"]    = state_dict["features.3.1.block.5.bias"]
            model_state_dict["4.block.1.block.5.weight"]  = state_dict["features.3.1.block.5.weight"]
            model_state_dict["4.block.1.layer_scale"]     = state_dict["features.3.1.layer_scale"]
            model_state_dict["4.block.2.block.0.bias"]    = state_dict["features.3.2.block.0.bias"]
            model_state_dict["4.block.2.block.0.weight"]  = state_dict["features.3.2.block.0.weight"]
            model_state_dict["4.block.2.block.2.bias"]    = state_dict["features.3.2.block.2.bias"]
            model_state_dict["4.block.2.block.2.weight"]  = state_dict["features.3.2.block.2.weight"]
            model_state_dict["4.block.2.block.3.bias"]    = state_dict["features.3.2.block.3.bias"]
            model_state_dict["4.block.2.block.3.weight"]  = state_dict["features.3.2.block.3.weight"]
            model_state_dict["4.block.2.block.5.bias"]    = state_dict["features.3.2.block.5.bias"]
            model_state_dict["4.block.2.block.5.weight"]  = state_dict["features.3.2.block.5.weight"]
            model_state_dict["4.block.2.layer_scale"]     = state_dict["features.3.2.layer_scale"]
            model_state_dict["5.bias"]                    = state_dict["features.4.0.bias"]
            model_state_dict["5.weight"]                  = state_dict["features.4.0.weight"]
            model_state_dict["6.bias"]                    = state_dict["features.4.1.bias"]
            model_state_dict["6.weight"]                  = state_dict["features.4.1.weight"]
            model_state_dict["7.block.0.block.0.bias"]    = state_dict["features.5.0.block.0.bias"]
            model_state_dict["7.block.0.block.0.weight"]  = state_dict["features.5.0.block.0.weight"]
            model_state_dict["7.block.0.block.2.bias"]    = state_dict["features.5.0.block.2.bias"]
            model_state_dict["7.block.0.block.2.weight"]  = state_dict["features.5.0.block.2.weight"]
            model_state_dict["7.block.0.block.3.bias"]    = state_dict["features.5.0.block.3.bias"]
            model_state_dict["7.block.0.block.3.weight"]  = state_dict["features.5.0.block.3.weight"]
            model_state_dict["7.block.0.block.5.bias"]    = state_dict["features.5.0.block.5.bias"]
            model_state_dict["7.block.0.block.5.weight"]  = state_dict["features.5.0.block.5.weight"]
            model_state_dict["7.block.0.layer_scale"]     = state_dict["features.5.0.layer_scale"]
            model_state_dict["7.block.1.block.0.bias"]    = state_dict["features.5.1.block.0.bias"]
            model_state_dict["7.block.1.block.0.weight"]  = state_dict["features.5.1.block.0.weight"]
            model_state_dict["7.block.1.block.2.bias"]    = state_dict["features.5.1.block.2.bias"]
            model_state_dict["7.block.1.block.2.weight"]  = state_dict["features.5.1.block.2.weight"]
            model_state_dict["7.block.1.block.3.bias"]    = state_dict["features.5.1.block.3.bias"]
            model_state_dict["7.block.1.block.3.weight"]  = state_dict["features.5.1.block.3.weight"]
            model_state_dict["7.block.1.block.5.bias"]    = state_dict["features.5.1.block.5.bias"]
            model_state_dict["7.block.1.block.5.weight"]  = state_dict["features.5.1.block.5.weight"]
            model_state_dict["7.block.1.layer_scale"]     = state_dict["features.5.1.layer_scale"]
            model_state_dict["7.block.10.block.0.bias"]   = state_dict["features.5.10.block.0.bias"]
            model_state_dict["7.block.10.block.0.weight"] = state_dict["features.5.10.block.0.weight"]
            model_state_dict["7.block.10.block.2.bias"]   = state_dict["features.5.10.block.2.bias"]
            model_state_dict["7.block.10.block.2.weight"] = state_dict["features.5.10.block.2.weight"]
            model_state_dict["7.block.10.block.3.bias"]   = state_dict["features.5.10.block.3.bias"]
            model_state_dict["7.block.10.block.3.weight"] = state_dict["features.5.10.block.3.weight"]
            model_state_dict["7.block.10.block.5.bias"]   = state_dict["features.5.10.block.5.bias"]
            model_state_dict["7.block.10.block.5.weight"] = state_dict["features.5.10.block.5.weight"]
            model_state_dict["7.block.10.layer_scale"]    = state_dict["features.5.10.layer_scale"]
            model_state_dict["7.block.11.block.0.bias"]   = state_dict["features.5.11.block.0.bias"]
            model_state_dict["7.block.11.block.0.weight"] = state_dict["features.5.11.block.0.weight"]
            model_state_dict["7.block.11.block.2.bias"]   = state_dict["features.5.11.block.2.bias"]
            model_state_dict["7.block.11.block.2.weight"] = state_dict["features.5.11.block.2.weight"]
            model_state_dict["7.block.11.block.3.bias"]   = state_dict["features.5.11.block.3.bias"]
            model_state_dict["7.block.11.block.3.weight"] = state_dict["features.5.11.block.3.weight"]
            model_state_dict["7.block.11.block.5.bias"]   = state_dict["features.5.11.block.5.bias"]
            model_state_dict["7.block.11.block.5.weight"] = state_dict["features.5.11.block.5.weight"]
            model_state_dict["7.block.11.layer_scale"]    = state_dict["features.5.11.layer_scale"]
            model_state_dict["7.block.12.block.0.bias"]   = state_dict["features.5.12.block.0.bias"]
            model_state_dict["7.block.12.block.0.weight"] = state_dict["features.5.12.block.0.weight"]
            model_state_dict["7.block.12.block.2.bias"]   = state_dict["features.5.12.block.2.bias"]
            model_state_dict["7.block.12.block.2.weight"] = state_dict["features.5.12.block.2.weight"]
            model_state_dict["7.block.12.block.3.bias"]   = state_dict["features.5.12.block.3.bias"]
            model_state_dict["7.block.12.block.3.weight"] = state_dict["features.5.12.block.3.weight"]
            model_state_dict["7.block.12.block.5.bias"]   = state_dict["features.5.12.block.5.bias"]
            model_state_dict["7.block.12.block.5.weight"] = state_dict["features.5.12.block.5.weight"]
            model_state_dict["7.block.12.layer_scale"]    = state_dict["features.5.12.layer_scale"]
            model_state_dict["7.block.13.block.0.bias"]   = state_dict["features.5.13.block.0.bias"]
            model_state_dict["7.block.13.block.0.weight"] = state_dict["features.5.13.block.0.weight"]
            model_state_dict["7.block.13.block.2.bias"]   = state_dict["features.5.13.block.2.bias"]
            model_state_dict["7.block.13.block.2.weight"] = state_dict["features.5.13.block.2.weight"]
            model_state_dict["7.block.13.block.3.bias"]   = state_dict["features.5.13.block.3.bias"]
            model_state_dict["7.block.13.block.3.weight"] = state_dict["features.5.13.block.3.weight"]
            model_state_dict["7.block.13.block.5.bias"]   = state_dict["features.5.13.block.5.bias"]
            model_state_dict["7.block.13.block.5.weight"] = state_dict["features.5.13.block.5.weight"]
            model_state_dict["7.block.13.layer_scale"]    = state_dict["features.5.13.layer_scale"]
            model_state_dict["7.block.14.block.0.bias"]   = state_dict["features.5.14.block.0.bias"]
            model_state_dict["7.block.14.block.0.weight"] = state_dict["features.5.14.block.0.weight"]
            model_state_dict["7.block.14.block.2.bias"]   = state_dict["features.5.14.block.2.bias"]
            model_state_dict["7.block.14.block.2.weight"] = state_dict["features.5.14.block.2.weight"]
            model_state_dict["7.block.14.block.3.bias"]   = state_dict["features.5.14.block.3.bias"]
            model_state_dict["7.block.14.block.3.weight"] = state_dict["features.5.14.block.3.weight"]
            model_state_dict["7.block.14.block.5.bias"]   = state_dict["features.5.14.block.5.bias"]
            model_state_dict["7.block.14.block.5.weight"] = state_dict["features.5.14.block.5.weight"]
            model_state_dict["7.block.14.layer_scale"]    = state_dict["features.5.14.layer_scale"]
            model_state_dict["7.block.15.block.0.bias"]   = state_dict["features.5.15.block.0.bias"]
            model_state_dict["7.block.15.block.0.weight"] = state_dict["features.5.15.block.0.weight"]
            model_state_dict["7.block.15.block.2.bias"]   = state_dict["features.5.15.block.2.bias"]
            model_state_dict["7.block.15.block.2.weight"] = state_dict["features.5.15.block.2.weight"]
            model_state_dict["7.block.15.block.3.bias"]   = state_dict["features.5.15.block.3.bias"]
            model_state_dict["7.block.15.block.3.weight"] = state_dict["features.5.15.block.3.weight"]
            model_state_dict["7.block.15.block.5.bias"]   = state_dict["features.5.15.block.5.bias"]
            model_state_dict["7.block.15.block.5.weight"] = state_dict["features.5.15.block.5.weight"]
            model_state_dict["7.block.15.layer_scale"]    = state_dict["features.5.15.layer_scale"]
            model_state_dict["7.block.16.block.0.bias"]   = state_dict["features.5.16.block.0.bias"]
            model_state_dict["7.block.16.block.0.weight"] = state_dict["features.5.16.block.0.weight"]
            model_state_dict["7.block.16.block.2.bias"]   = state_dict["features.5.16.block.2.bias"]
            model_state_dict["7.block.16.block.2.weight"] = state_dict["features.5.16.block.2.weight"]
            model_state_dict["7.block.16.block.3.bias"]   = state_dict["features.5.16.block.3.bias"]
            model_state_dict["7.block.16.block.3.weight"] = state_dict["features.5.16.block.3.weight"]
            model_state_dict["7.block.16.block.5.bias"]   = state_dict["features.5.16.block.5.bias"]
            model_state_dict["7.block.16.block.5.weight"] = state_dict["features.5.16.block.5.weight"]
            model_state_dict["7.block.16.layer_scale"]    = state_dict["features.5.16.layer_scale"]
            model_state_dict["7.block.17.block.0.bias"]   = state_dict["features.5.17.block.0.bias"]
            model_state_dict["7.block.17.block.0.weight"] = state_dict["features.5.17.block.0.weight"]
            model_state_dict["7.block.17.block.2.bias"]   = state_dict["features.5.17.block.2.bias"]
            model_state_dict["7.block.17.block.2.weight"] = state_dict["features.5.17.block.2.weight"]
            model_state_dict["7.block.17.block.3.bias"]   = state_dict["features.5.17.block.3.bias"]
            model_state_dict["7.block.17.block.3.weight"] = state_dict["features.5.17.block.3.weight"]
            model_state_dict["7.block.17.block.5.bias"]   = state_dict["features.5.17.block.5.bias"]
            model_state_dict["7.block.17.block.5.weight"] = state_dict["features.5.17.block.5.weight"]
            model_state_dict["7.block.17.layer_scale"]    = state_dict["features.5.17.layer_scale"]
            model_state_dict["7.block.18.block.0.bias"]   = state_dict["features.5.18.block.0.bias"]
            model_state_dict["7.block.18.block.0.weight"] = state_dict["features.5.18.block.0.weight"]
            model_state_dict["7.block.18.block.2.bias"]   = state_dict["features.5.18.block.2.bias"]
            model_state_dict["7.block.18.block.2.weight"] = state_dict["features.5.18.block.2.weight"]
            model_state_dict["7.block.18.block.3.bias"]   = state_dict["features.5.18.block.3.bias"]
            model_state_dict["7.block.18.block.3.weight"] = state_dict["features.5.18.block.3.weight"]
            model_state_dict["7.block.18.block.5.bias"]   = state_dict["features.5.18.block.5.bias"]
            model_state_dict["7.block.18.block.5.weight"] = state_dict["features.5.18.block.5.weight"]
            model_state_dict["7.block.18.layer_scale"]    = state_dict["features.5.18.layer_scale"]
            model_state_dict["7.block.19.block.0.bias"]   = state_dict["features.5.19.block.0.bias"]
            model_state_dict["7.block.19.block.0.weight"] = state_dict["features.5.19.block.0.weight"]
            model_state_dict["7.block.19.block.2.bias"]   = state_dict["features.5.19.block.2.bias"]
            model_state_dict["7.block.19.block.2.weight"] = state_dict["features.5.19.block.2.weight"]
            model_state_dict["7.block.19.block.3.bias"]   = state_dict["features.5.19.block.3.bias"]
            model_state_dict["7.block.19.block.3.weight"] = state_dict["features.5.19.block.3.weight"]
            model_state_dict["7.block.19.block.5.bias"]   = state_dict["features.5.19.block.5.bias"]
            model_state_dict["7.block.19.block.5.weight"] = state_dict["features.5.19.block.5.weight"]
            model_state_dict["7.block.19.layer_scale"]    = state_dict["features.5.19.layer_scale"]
            model_state_dict["7.block.2.block.0.bias"]    = state_dict["features.5.2.block.0.bias"]
            model_state_dict["7.block.2.block.0.weight"]  = state_dict["features.5.2.block.0.weight"]
            model_state_dict["7.block.2.block.2.bias"]    = state_dict["features.5.2.block.2.bias"]
            model_state_dict["7.block.2.block.2.weight"]  = state_dict["features.5.2.block.2.weight"]
            model_state_dict["7.block.2.block.3.bias"]    = state_dict["features.5.2.block.3.bias"]
            model_state_dict["7.block.2.block.3.weight"]  = state_dict["features.5.2.block.3.weight"]
            model_state_dict["7.block.2.block.5.bias"]    = state_dict["features.5.2.block.5.bias"]
            model_state_dict["7.block.2.block.5.weight"]  = state_dict["features.5.2.block.5.weight"]
            model_state_dict["7.block.2.layer_scale"]     = state_dict["features.5.2.layer_scale"]
            model_state_dict["7.block.20.block.0.bias"]   = state_dict["features.5.20.block.0.bias"]
            model_state_dict["7.block.20.block.0.weight"] = state_dict["features.5.20.block.0.weight"]
            model_state_dict["7.block.20.block.2.bias"]   = state_dict["features.5.20.block.2.bias"]
            model_state_dict["7.block.20.block.2.weight"] = state_dict["features.5.20.block.2.weight"]
            model_state_dict["7.block.20.block.3.bias"]   = state_dict["features.5.20.block.3.bias"]
            model_state_dict["7.block.20.block.3.weight"] = state_dict["features.5.20.block.3.weight"]
            model_state_dict["7.block.20.block.5.bias"]   = state_dict["features.5.20.block.5.bias"]
            model_state_dict["7.block.20.block.5.weight"] = state_dict["features.5.20.block.5.weight"]
            model_state_dict["7.block.20.layer_scale"]    = state_dict["features.5.20.layer_scale"]
            model_state_dict["7.block.21.block.0.bias"]   = state_dict["features.5.21.block.0.bias"]
            model_state_dict["7.block.21.block.0.weight"] = state_dict["features.5.21.block.0.weight"]
            model_state_dict["7.block.21.block.2.bias"]   = state_dict["features.5.21.block.2.bias"]
            model_state_dict["7.block.21.block.2.weight"] = state_dict["features.5.21.block.2.weight"]
            model_state_dict["7.block.21.block.3.bias"]   = state_dict["features.5.21.block.3.bias"]
            model_state_dict["7.block.21.block.3.weight"] = state_dict["features.5.21.block.3.weight"]
            model_state_dict["7.block.21.block.5.bias"]   = state_dict["features.5.21.block.5.bias"]
            model_state_dict["7.block.21.block.5.weight"] = state_dict["features.5.21.block.5.weight"]
            model_state_dict["7.block.21.layer_scale"]    = state_dict["features.5.21.layer_scale"]
            model_state_dict["7.block.22.block.0.bias"]   = state_dict["features.5.22.block.0.bias"]
            model_state_dict["7.block.22.block.0.weight"] = state_dict["features.5.22.block.0.weight"]
            model_state_dict["7.block.22.block.2.bias"]   = state_dict["features.5.22.block.2.bias"]
            model_state_dict["7.block.22.block.2.weight"] = state_dict["features.5.22.block.2.weight"]
            model_state_dict["7.block.22.block.3.bias"]   = state_dict["features.5.22.block.3.bias"]
            model_state_dict["7.block.22.block.3.weight"] = state_dict["features.5.22.block.3.weight"]
            model_state_dict["7.block.22.block.5.bias"]   = state_dict["features.5.22.block.5.bias"]
            model_state_dict["7.block.22.block.5.weight"] = state_dict["features.5.22.block.5.weight"]
            model_state_dict["7.block.22.layer_scale"]    = state_dict["features.5.22.layer_scale"]
            model_state_dict["7.block.23.block.0.bias"]   = state_dict["features.5.23.block.0.bias"]
            model_state_dict["7.block.23.block.0.weight"] = state_dict["features.5.23.block.0.weight"]
            model_state_dict["7.block.23.block.2.bias"]   = state_dict["features.5.23.block.2.bias"]
            model_state_dict["7.block.23.block.2.weight"] = state_dict["features.5.23.block.2.weight"]
            model_state_dict["7.block.23.block.3.bias"]   = state_dict["features.5.23.block.3.bias"]
            model_state_dict["7.block.23.block.3.weight"] = state_dict["features.5.23.block.3.weight"]
            model_state_dict["7.block.23.block.5.bias"]   = state_dict["features.5.23.block.5.bias"]
            model_state_dict["7.block.23.block.5.weight"] = state_dict["features.5.23.block.5.weight"]
            model_state_dict["7.block.23.layer_scale"]    = state_dict["features.5.23.layer_scale"]
            model_state_dict["7.block.24.block.0.bias"]   = state_dict["features.5.24.block.0.bias"]
            model_state_dict["7.block.24.block.0.weight"] = state_dict["features.5.24.block.0.weight"]
            model_state_dict["7.block.24.block.2.bias"]   = state_dict["features.5.24.block.2.bias"]
            model_state_dict["7.block.24.block.2.weight"] = state_dict["features.5.24.block.2.weight"]
            model_state_dict["7.block.24.block.3.bias"]   = state_dict["features.5.24.block.3.bias"]
            model_state_dict["7.block.24.block.3.weight"] = state_dict["features.5.24.block.3.weight"]
            model_state_dict["7.block.24.block.5.bias"]   = state_dict["features.5.24.block.5.bias"]
            model_state_dict["7.block.24.block.5.weight"] = state_dict["features.5.24.block.5.weight"]
            model_state_dict["7.block.24.layer_scale"]    = state_dict["features.5.24.layer_scale"]
            model_state_dict["7.block.25.block.0.bias"]   = state_dict["features.5.25.block.0.bias"]
            model_state_dict["7.block.25.block.0.weight"] = state_dict["features.5.25.block.0.weight"]
            model_state_dict["7.block.25.block.2.bias"]   = state_dict["features.5.25.block.2.bias"]
            model_state_dict["7.block.25.block.2.weight"] = state_dict["features.5.25.block.2.weight"]
            model_state_dict["7.block.25.block.3.bias"]   = state_dict["features.5.25.block.3.bias"]
            model_state_dict["7.block.25.block.3.weight"] = state_dict["features.5.25.block.3.weight"]
            model_state_dict["7.block.25.block.5.bias"]   = state_dict["features.5.25.block.5.bias"]
            model_state_dict["7.block.25.block.5.weight"] = state_dict["features.5.25.block.5.weight"]
            model_state_dict["7.block.25.layer_scale"]    = state_dict["features.5.25.layer_scale"]
            model_state_dict["7.block.26.block.0.bias"]   = state_dict["features.5.26.block.0.bias"]
            model_state_dict["7.block.26.block.0.weight"] = state_dict["features.5.26.block.0.weight"]
            model_state_dict["7.block.26.block.2.bias"]   = state_dict["features.5.26.block.2.bias"]
            model_state_dict["7.block.26.block.2.weight"] = state_dict["features.5.26.block.2.weight"]
            model_state_dict["7.block.26.block.3.bias"]   = state_dict["features.5.26.block.3.bias"]
            model_state_dict["7.block.26.block.3.weight"] = state_dict["features.5.26.block.3.weight"]
            model_state_dict["7.block.26.block.5.bias"]   = state_dict["features.5.26.block.5.bias"]
            model_state_dict["7.block.26.block.5.weight"] = state_dict["features.5.26.block.5.weight"]
            model_state_dict["7.block.26.layer_scale"]    = state_dict["features.5.26.layer_scale"]
            model_state_dict["7.block.3.block.0.bias"]    = state_dict["features.5.3.block.0.bias"]
            model_state_dict["7.block.3.block.0.weight"]  = state_dict["features.5.3.block.0.weight"]
            model_state_dict["7.block.3.block.2.bias"]    = state_dict["features.5.3.block.2.bias"]
            model_state_dict["7.block.3.block.2.weight"]  = state_dict["features.5.3.block.2.weight"]
            model_state_dict["7.block.3.block.3.bias"]    = state_dict["features.5.3.block.3.bias"]
            model_state_dict["7.block.3.block.3.weight"]  = state_dict["features.5.3.block.3.weight"]
            model_state_dict["7.block.3.block.5.bias"]    = state_dict["features.5.3.block.5.bias"]
            model_state_dict["7.block.3.block.5.weight"]  = state_dict["features.5.3.block.5.weight"]
            model_state_dict["7.block.3.layer_scale"]     = state_dict["features.5.3.layer_scale"]
            model_state_dict["7.block.4.block.0.bias"]    = state_dict["features.5.4.block.0.bias"]
            model_state_dict["7.block.4.block.0.weight"]  = state_dict["features.5.4.block.0.weight"]
            model_state_dict["7.block.4.block.2.bias"]    = state_dict["features.5.4.block.2.bias"]
            model_state_dict["7.block.4.block.2.weight"]  = state_dict["features.5.4.block.2.weight"]
            model_state_dict["7.block.4.block.3.bias"]    = state_dict["features.5.4.block.3.bias"]
            model_state_dict["7.block.4.block.3.weight"]  = state_dict["features.5.4.block.3.weight"]
            model_state_dict["7.block.4.block.5.bias"]    = state_dict["features.5.4.block.5.bias"]
            model_state_dict["7.block.4.block.5.weight"]  = state_dict["features.5.4.block.5.weight"]
            model_state_dict["7.block.4.layer_scale"]     = state_dict["features.5.4.layer_scale"]
            model_state_dict["7.block.5.block.0.bias"]    = state_dict["features.5.5.block.0.bias"]
            model_state_dict["7.block.5.block.0.weight"]  = state_dict["features.5.5.block.0.weight"]
            model_state_dict["7.block.5.block.2.bias"]    = state_dict["features.5.5.block.2.bias"]
            model_state_dict["7.block.5.block.2.weight"]  = state_dict["features.5.5.block.2.weight"]
            model_state_dict["7.block.5.block.3.bias"]    = state_dict["features.5.5.block.3.bias"]
            model_state_dict["7.block.5.block.3.weight"]  = state_dict["features.5.5.block.3.weight"]
            model_state_dict["7.block.5.block.5.bias"]    = state_dict["features.5.5.block.5.bias"]
            model_state_dict["7.block.5.block.5.weight"]  = state_dict["features.5.5.block.5.weight"]
            model_state_dict["7.block.5.layer_scale"]     = state_dict["features.5.5.layer_scale"]
            model_state_dict["7.block.6.block.0.bias"]    = state_dict["features.5.6.block.0.bias"]
            model_state_dict["7.block.6.block.0.weight"]  = state_dict["features.5.6.block.0.weight"]
            model_state_dict["7.block.6.block.2.bias"]    = state_dict["features.5.6.block.2.bias"]
            model_state_dict["7.block.6.block.2.weight"]  = state_dict["features.5.6.block.2.weight"]
            model_state_dict["7.block.6.block.3.bias"]    = state_dict["features.5.6.block.3.bias"]
            model_state_dict["7.block.6.block.3.weight"]  = state_dict["features.5.6.block.3.weight"]
            model_state_dict["7.block.6.block.5.bias"]    = state_dict["features.5.6.block.5.bias"]
            model_state_dict["7.block.6.block.5.weight"]  = state_dict["features.5.6.block.5.weight"]
            model_state_dict["7.block.6.layer_scale"]     = state_dict["features.5.6.layer_scale"]
            model_state_dict["7.block.7.block.0.bias"]    = state_dict["features.5.7.block.0.bias"]
            model_state_dict["7.block.7.block.0.weight"]  = state_dict["features.5.7.block.0.weight"]
            model_state_dict["7.block.7.block.2.bias"]    = state_dict["features.5.7.block.2.bias"]
            model_state_dict["7.block.7.block.2.weight"]  = state_dict["features.5.7.block.2.weight"]
            model_state_dict["7.block.7.block.3.bias"]    = state_dict["features.5.7.block.3.bias"]
            model_state_dict["7.block.7.block.3.weight"]  = state_dict["features.5.7.block.3.weight"]
            model_state_dict["7.block.7.block.5.bias"]    = state_dict["features.5.7.block.5.bias"]
            model_state_dict["7.block.7.block.5.weight"]  = state_dict["features.5.7.block.5.weight"]
            model_state_dict["7.block.7.layer_scale"]     = state_dict["features.5.7.layer_scale"]
            model_state_dict["7.block.8.block.0.bias"]    = state_dict["features.5.8.block.0.bias"]
            model_state_dict["7.block.8.block.0.weight"]  = state_dict["features.5.8.block.0.weight"]
            model_state_dict["7.block.8.block.2.bias"]    = state_dict["features.5.8.block.2.bias"]
            model_state_dict["7.block.8.block.2.weight"]  = state_dict["features.5.8.block.2.weight"]
            model_state_dict["7.block.8.block.3.bias"]    = state_dict["features.5.8.block.3.bias"]
            model_state_dict["7.block.8.block.3.weight"]  = state_dict["features.5.8.block.3.weight"]
            model_state_dict["7.block.8.block.5.bias"]    = state_dict["features.5.8.block.5.bias"]
            model_state_dict["7.block.8.block.5.weight"]  = state_dict["features.5.8.block.5.weight"]
            model_state_dict["7.block.8.layer_scale"]     = state_dict["features.5.8.layer_scale"]
            model_state_dict["7.block.9.block.0.bias"]    = state_dict["features.5.9.block.0.bias"]
            model_state_dict["7.block.9.block.0.weight"]  = state_dict["features.5.9.block.0.weight"]
            model_state_dict["7.block.9.block.2.bias"]    = state_dict["features.5.9.block.2.bias"]
            model_state_dict["7.block.9.block.2.weight"]  = state_dict["features.5.9.block.2.weight"]
            model_state_dict["7.block.9.block.3.bias"]    = state_dict["features.5.9.block.3.bias"]
            model_state_dict["7.block.9.block.3.weight"]  = state_dict["features.5.9.block.3.weight"]
            model_state_dict["7.block.9.block.5.bias"]    = state_dict["features.5.9.block.5.bias"]
            model_state_dict["7.block.9.block.5.weight"]  = state_dict["features.5.9.block.5.weight"]
            model_state_dict["7.block.9.layer_scale"]     = state_dict["features.5.9.layer_scale"]
            model_state_dict["8.bias"]                    = state_dict["features.6.0.bias"]
            model_state_dict["8.weight"]                  = state_dict["features.6.0.weight"]
            model_state_dict["9.bias"]                    = state_dict["features.6.1.bias"]
            model_state_dict["9.weight"]                  = state_dict["features.6.1.weight"]
            model_state_dict["10.block.0.block.0.bias"]   = state_dict["features.7.0.block.0.bias"]
            model_state_dict["10.block.0.block.0.weight"] = state_dict["features.7.0.block.0.weight"]
            model_state_dict["10.block.0.block.2.bias"]   = state_dict["features.7.0.block.2.bias"]
            model_state_dict["10.block.0.block.2.weight"] = state_dict["features.7.0.block.2.weight"]
            model_state_dict["10.block.0.block.3.bias"]   = state_dict["features.7.0.block.3.bias"]
            model_state_dict["10.block.0.block.3.weight"] = state_dict["features.7.0.block.3.weight"]
            model_state_dict["10.block.0.block.5.bias"]   = state_dict["features.7.0.block.5.bias"]
            model_state_dict["10.block.0.block.5.weight"] = state_dict["features.7.0.block.5.weight"]
            model_state_dict["10.block.0.layer_scale"]    = state_dict["features.7.0.layer_scale"]
            model_state_dict["10.block.1.block.0.bias"]   = state_dict["features.7.1.block.0.bias"]
            model_state_dict["10.block.1.block.0.weight"] = state_dict["features.7.1.block.0.weight"]
            model_state_dict["10.block.1.block.2.bias"]   = state_dict["features.7.1.block.2.bias"]
            model_state_dict["10.block.1.block.2.weight"] = state_dict["features.7.1.block.2.weight"]
            model_state_dict["10.block.1.block.3.bias"]   = state_dict["features.7.1.block.3.bias"]
            model_state_dict["10.block.1.block.3.weight"] = state_dict["features.7.1.block.3.weight"]
            model_state_dict["10.block.1.block.5.bias"]   = state_dict["features.7.1.block.5.bias"]
            model_state_dict["10.block.1.block.5.weight"] = state_dict["features.7.1.block.5.weight"]
            model_state_dict["10.block.1.layer_scale"]    = state_dict["features.7.1.layer_scale"]
            model_state_dict["10.block.2.block.0.bias"]   = state_dict["features.7.2.block.0.bias"]
            model_state_dict["10.block.2.block.0.weight"] = state_dict["features.7.2.block.0.weight"]
            model_state_dict["10.block.2.block.2.bias"]   = state_dict["features.7.2.block.2.bias"]
            model_state_dict["10.block.2.block.2.weight"] = state_dict["features.7.2.block.2.weight"]
            model_state_dict["10.block.2.block.3.bias"]   = state_dict["features.7.2.block.3.bias"]
            model_state_dict["10.block.2.block.3.weight"] = state_dict["features.7.2.block.3.weight"]
            model_state_dict["10.block.2.block.5.bias"]   = state_dict["features.7.2.block.5.bias"]
            model_state_dict["10.block.2.block.5.weight"] = state_dict["features.7.2.block.5.weight"]
            model_state_dict["10.block.2.layer_scale"]    = state_dict["features.7.2.layer_scale"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["12.norm.bias"]     = state_dict["classifier.0.bias"]
                model_state_dict["12.norm.weight"]   = state_dict["classifier.0.weight"]
                model_state_dict["12.linear.bias"]   = state_dict["classifier.2.bias"]
                model_state_dict["12.linear.weight"] = state_dict["classifier.2.weight"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()


@MODELS.register(name="convnext-large")
class ConvNeXtLarge(ConvNeXt):
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
            path        = "https://download.pytorch.org/models/convnext_large-ea097f82.pth",
            filename    = "convnext-large-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "convnext-large.yaml",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "convnext",
        fullname   : str  | None         = "convnext-large",
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
        cfg = cfg or "convnext-large"
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
            pretrained  = ConvNeXtLarge.init_pretrained(pretrained),
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
            model_state_dict["0.0.bias"]                  = state_dict["features.0.0.bias"]
            model_state_dict["0.0.weight"]                = state_dict["features.0.0.weight"]
            model_state_dict["0.1.bias"]                  = state_dict["features.0.1.bias"]
            model_state_dict["0.1.weight"]                = state_dict["features.0.1.weight"]
            model_state_dict["1.block.0.block.0.bias"]    = state_dict["features.1.0.block.0.bias"]
            model_state_dict["1.block.0.block.0.weight"]  = state_dict["features.1.0.block.0.weight"]
            model_state_dict["1.block.0.block.2.bias"]    = state_dict["features.1.0.block.2.bias"]
            model_state_dict["1.block.0.block.2.weight"]  = state_dict["features.1.0.block.2.weight"]
            model_state_dict["1.block.0.block.3.bias"]    = state_dict["features.1.0.block.3.bias"]
            model_state_dict["1.block.0.block.3.weight"]  = state_dict["features.1.0.block.3.weight"]
            model_state_dict["1.block.0.block.5.bias"]    = state_dict["features.1.0.block.5.bias"]
            model_state_dict["1.block.0.block.5.weight"]  = state_dict["features.1.0.block.5.weight"]
            model_state_dict["1.block.0.layer_scale"]     = state_dict["features.1.0.layer_scale"]
            model_state_dict["1.block.1.block.0.bias"]    = state_dict["features.1.1.block.0.bias"]
            model_state_dict["1.block.1.block.0.weight"]  = state_dict["features.1.1.block.0.weight"]
            model_state_dict["1.block.1.block.2.bias"]    = state_dict["features.1.1.block.2.bias"]
            model_state_dict["1.block.1.block.2.weight"]  = state_dict["features.1.1.block.2.weight"]
            model_state_dict["1.block.1.block.3.bias"]    = state_dict["features.1.1.block.3.bias"]
            model_state_dict["1.block.1.block.3.weight"]  = state_dict["features.1.1.block.3.weight"]
            model_state_dict["1.block.1.block.5.bias"]    = state_dict["features.1.1.block.5.bias"]
            model_state_dict["1.block.1.block.5.weight"]  = state_dict["features.1.1.block.5.weight"]
            model_state_dict["1.block.1.layer_scale"]     = state_dict["features.1.1.layer_scale"]
            model_state_dict["1.block.2.block.0.bias"]    = state_dict["features.1.2.block.0.bias"]
            model_state_dict["1.block.2.block.0.weight"]  = state_dict["features.1.2.block.0.weight"]
            model_state_dict["1.block.2.block.2.bias"]    = state_dict["features.1.2.block.2.bias"]
            model_state_dict["1.block.2.block.2.weight"]  = state_dict["features.1.2.block.2.weight"]
            model_state_dict["1.block.2.block.3.bias"]    = state_dict["features.1.2.block.3.bias"]
            model_state_dict["1.block.2.block.3.weight"]  = state_dict["features.1.2.block.3.weight"]
            model_state_dict["1.block.2.block.5.bias"]    = state_dict["features.1.2.block.5.bias"]
            model_state_dict["1.block.2.block.5.weight"]  = state_dict["features.1.2.block.5.weight"]
            model_state_dict["1.block.2.layer_scale"]     = state_dict["features.1.2.layer_scale"]
            model_state_dict["2.bias"]                    = state_dict["features.2.0.bias"]
            model_state_dict["2.weight"]                  = state_dict["features.2.0.weight"]
            model_state_dict["3.bias"]                    = state_dict["features.2.1.bias"]
            model_state_dict["3.weight"]                  = state_dict["features.2.1.weight"]
            model_state_dict["4.block.0.block.0.bias"]    = state_dict["features.3.0.block.0.bias"]
            model_state_dict["4.block.0.block.0.weight"]  = state_dict["features.3.0.block.0.weight"]
            model_state_dict["4.block.0.block.2.bias"]    = state_dict["features.3.0.block.2.bias"]
            model_state_dict["4.block.0.block.2.weight"]  = state_dict["features.3.0.block.2.weight"]
            model_state_dict["4.block.0.block.3.bias"]    = state_dict["features.3.0.block.3.bias"]
            model_state_dict["4.block.0.block.3.weight"]  = state_dict["features.3.0.block.3.weight"]
            model_state_dict["4.block.0.block.5.bias"]    = state_dict["features.3.0.block.5.bias"]
            model_state_dict["4.block.0.block.5.weight"]  = state_dict["features.3.0.block.5.weight"]
            model_state_dict["4.block.0.layer_scale"]     = state_dict["features.3.0.layer_scale"]
            model_state_dict["4.block.1.block.0.bias"]    = state_dict["features.3.1.block.0.bias"]
            model_state_dict["4.block.1.block.0.weight"]  = state_dict["features.3.1.block.0.weight"]
            model_state_dict["4.block.1.block.2.bias"]    = state_dict["features.3.1.block.2.bias"]
            model_state_dict["4.block.1.block.2.weight"]  = state_dict["features.3.1.block.2.weight"]
            model_state_dict["4.block.1.block.3.bias"]    = state_dict["features.3.1.block.3.bias"]
            model_state_dict["4.block.1.block.3.weight"]  = state_dict["features.3.1.block.3.weight"]
            model_state_dict["4.block.1.block.5.bias"]    = state_dict["features.3.1.block.5.bias"]
            model_state_dict["4.block.1.block.5.weight"]  = state_dict["features.3.1.block.5.weight"]
            model_state_dict["4.block.1.layer_scale"]     = state_dict["features.3.1.layer_scale"]
            model_state_dict["4.block.2.block.0.bias"]    = state_dict["features.3.2.block.0.bias"]
            model_state_dict["4.block.2.block.0.weight"]  = state_dict["features.3.2.block.0.weight"]
            model_state_dict["4.block.2.block.2.bias"]    = state_dict["features.3.2.block.2.bias"]
            model_state_dict["4.block.2.block.2.weight"]  = state_dict["features.3.2.block.2.weight"]
            model_state_dict["4.block.2.block.3.bias"]    = state_dict["features.3.2.block.3.bias"]
            model_state_dict["4.block.2.block.3.weight"]  = state_dict["features.3.2.block.3.weight"]
            model_state_dict["4.block.2.block.5.bias"]    = state_dict["features.3.2.block.5.bias"]
            model_state_dict["4.block.2.block.5.weight"]  = state_dict["features.3.2.block.5.weight"]
            model_state_dict["4.block.2.layer_scale"]     = state_dict["features.3.2.layer_scale"]
            model_state_dict["5.bias"]                    = state_dict["features.4.0.bias"]
            model_state_dict["5.weight"]                  = state_dict["features.4.0.weight"]
            model_state_dict["6.bias"]                    = state_dict["features.4.1.bias"]
            model_state_dict["6.weight"]                  = state_dict["features.4.1.weight"]
            model_state_dict["7.block.0.block.0.bias"]    = state_dict["features.5.0.block.0.bias"]
            model_state_dict["7.block.0.block.0.weight"]  = state_dict["features.5.0.block.0.weight"]
            model_state_dict["7.block.0.block.2.bias"]    = state_dict["features.5.0.block.2.bias"]
            model_state_dict["7.block.0.block.2.weight"]  = state_dict["features.5.0.block.2.weight"]
            model_state_dict["7.block.0.block.3.bias"]    = state_dict["features.5.0.block.3.bias"]
            model_state_dict["7.block.0.block.3.weight"]  = state_dict["features.5.0.block.3.weight"]
            model_state_dict["7.block.0.block.5.bias"]    = state_dict["features.5.0.block.5.bias"]
            model_state_dict["7.block.0.block.5.weight"]  = state_dict["features.5.0.block.5.weight"]
            model_state_dict["7.block.0.layer_scale"]     = state_dict["features.5.0.layer_scale"]
            model_state_dict["7.block.1.block.0.bias"]    = state_dict["features.5.1.block.0.bias"]
            model_state_dict["7.block.1.block.0.weight"]  = state_dict["features.5.1.block.0.weight"]
            model_state_dict["7.block.1.block.2.bias"]    = state_dict["features.5.1.block.2.bias"]
            model_state_dict["7.block.1.block.2.weight"]  = state_dict["features.5.1.block.2.weight"]
            model_state_dict["7.block.1.block.3.bias"]    = state_dict["features.5.1.block.3.bias"]
            model_state_dict["7.block.1.block.3.weight"]  = state_dict["features.5.1.block.3.weight"]
            model_state_dict["7.block.1.block.5.bias"]    = state_dict["features.5.1.block.5.bias"]
            model_state_dict["7.block.1.block.5.weight"]  = state_dict["features.5.1.block.5.weight"]
            model_state_dict["7.block.1.layer_scale"]     = state_dict["features.5.1.layer_scale"]
            model_state_dict["7.block.10.block.0.bias"]   = state_dict["features.5.10.block.0.bias"]
            model_state_dict["7.block.10.block.0.weight"] = state_dict["features.5.10.block.0.weight"]
            model_state_dict["7.block.10.block.2.bias"]   = state_dict["features.5.10.block.2.bias"]
            model_state_dict["7.block.10.block.2.weight"] = state_dict["features.5.10.block.2.weight"]
            model_state_dict["7.block.10.block.3.bias"]   = state_dict["features.5.10.block.3.bias"]
            model_state_dict["7.block.10.block.3.weight"] = state_dict["features.5.10.block.3.weight"]
            model_state_dict["7.block.10.block.5.bias"]   = state_dict["features.5.10.block.5.bias"]
            model_state_dict["7.block.10.block.5.weight"] = state_dict["features.5.10.block.5.weight"]
            model_state_dict["7.block.10.layer_scale"]    = state_dict["features.5.10.layer_scale"]
            model_state_dict["7.block.11.block.0.bias"]   = state_dict["features.5.11.block.0.bias"]
            model_state_dict["7.block.11.block.0.weight"] = state_dict["features.5.11.block.0.weight"]
            model_state_dict["7.block.11.block.2.bias"]   = state_dict["features.5.11.block.2.bias"]
            model_state_dict["7.block.11.block.2.weight"] = state_dict["features.5.11.block.2.weight"]
            model_state_dict["7.block.11.block.3.bias"]   = state_dict["features.5.11.block.3.bias"]
            model_state_dict["7.block.11.block.3.weight"] = state_dict["features.5.11.block.3.weight"]
            model_state_dict["7.block.11.block.5.bias"]   = state_dict["features.5.11.block.5.bias"]
            model_state_dict["7.block.11.block.5.weight"] = state_dict["features.5.11.block.5.weight"]
            model_state_dict["7.block.11.layer_scale"]    = state_dict["features.5.11.layer_scale"]
            model_state_dict["7.block.12.block.0.bias"]   = state_dict["features.5.12.block.0.bias"]
            model_state_dict["7.block.12.block.0.weight"] = state_dict["features.5.12.block.0.weight"]
            model_state_dict["7.block.12.block.2.bias"]   = state_dict["features.5.12.block.2.bias"]
            model_state_dict["7.block.12.block.2.weight"] = state_dict["features.5.12.block.2.weight"]
            model_state_dict["7.block.12.block.3.bias"]   = state_dict["features.5.12.block.3.bias"]
            model_state_dict["7.block.12.block.3.weight"] = state_dict["features.5.12.block.3.weight"]
            model_state_dict["7.block.12.block.5.bias"]   = state_dict["features.5.12.block.5.bias"]
            model_state_dict["7.block.12.block.5.weight"] = state_dict["features.5.12.block.5.weight"]
            model_state_dict["7.block.12.layer_scale"]    = state_dict["features.5.12.layer_scale"]
            model_state_dict["7.block.13.block.0.bias"]   = state_dict["features.5.13.block.0.bias"]
            model_state_dict["7.block.13.block.0.weight"] = state_dict["features.5.13.block.0.weight"]
            model_state_dict["7.block.13.block.2.bias"]   = state_dict["features.5.13.block.2.bias"]
            model_state_dict["7.block.13.block.2.weight"] = state_dict["features.5.13.block.2.weight"]
            model_state_dict["7.block.13.block.3.bias"]   = state_dict["features.5.13.block.3.bias"]
            model_state_dict["7.block.13.block.3.weight"] = state_dict["features.5.13.block.3.weight"]
            model_state_dict["7.block.13.block.5.bias"]   = state_dict["features.5.13.block.5.bias"]
            model_state_dict["7.block.13.block.5.weight"] = state_dict["features.5.13.block.5.weight"]
            model_state_dict["7.block.13.layer_scale"]    = state_dict["features.5.13.layer_scale"]
            model_state_dict["7.block.14.block.0.bias"]   = state_dict["features.5.14.block.0.bias"]
            model_state_dict["7.block.14.block.0.weight"] = state_dict["features.5.14.block.0.weight"]
            model_state_dict["7.block.14.block.2.bias"]   = state_dict["features.5.14.block.2.bias"]
            model_state_dict["7.block.14.block.2.weight"] = state_dict["features.5.14.block.2.weight"]
            model_state_dict["7.block.14.block.3.bias"]   = state_dict["features.5.14.block.3.bias"]
            model_state_dict["7.block.14.block.3.weight"] = state_dict["features.5.14.block.3.weight"]
            model_state_dict["7.block.14.block.5.bias"]   = state_dict["features.5.14.block.5.bias"]
            model_state_dict["7.block.14.block.5.weight"] = state_dict["features.5.14.block.5.weight"]
            model_state_dict["7.block.14.layer_scale"]    = state_dict["features.5.14.layer_scale"]
            model_state_dict["7.block.15.block.0.bias"]   = state_dict["features.5.15.block.0.bias"]
            model_state_dict["7.block.15.block.0.weight"] = state_dict["features.5.15.block.0.weight"]
            model_state_dict["7.block.15.block.2.bias"]   = state_dict["features.5.15.block.2.bias"]
            model_state_dict["7.block.15.block.2.weight"] = state_dict["features.5.15.block.2.weight"]
            model_state_dict["7.block.15.block.3.bias"]   = state_dict["features.5.15.block.3.bias"]
            model_state_dict["7.block.15.block.3.weight"] = state_dict["features.5.15.block.3.weight"]
            model_state_dict["7.block.15.block.5.bias"]   = state_dict["features.5.15.block.5.bias"]
            model_state_dict["7.block.15.block.5.weight"] = state_dict["features.5.15.block.5.weight"]
            model_state_dict["7.block.15.layer_scale"]    = state_dict["features.5.15.layer_scale"]
            model_state_dict["7.block.16.block.0.bias"]   = state_dict["features.5.16.block.0.bias"]
            model_state_dict["7.block.16.block.0.weight"] = state_dict["features.5.16.block.0.weight"]
            model_state_dict["7.block.16.block.2.bias"]   = state_dict["features.5.16.block.2.bias"]
            model_state_dict["7.block.16.block.2.weight"] = state_dict["features.5.16.block.2.weight"]
            model_state_dict["7.block.16.block.3.bias"]   = state_dict["features.5.16.block.3.bias"]
            model_state_dict["7.block.16.block.3.weight"] = state_dict["features.5.16.block.3.weight"]
            model_state_dict["7.block.16.block.5.bias"]   = state_dict["features.5.16.block.5.bias"]
            model_state_dict["7.block.16.block.5.weight"] = state_dict["features.5.16.block.5.weight"]
            model_state_dict["7.block.16.layer_scale"]    = state_dict["features.5.16.layer_scale"]
            model_state_dict["7.block.17.block.0.bias"]   = state_dict["features.5.17.block.0.bias"]
            model_state_dict["7.block.17.block.0.weight"] = state_dict["features.5.17.block.0.weight"]
            model_state_dict["7.block.17.block.2.bias"]   = state_dict["features.5.17.block.2.bias"]
            model_state_dict["7.block.17.block.2.weight"] = state_dict["features.5.17.block.2.weight"]
            model_state_dict["7.block.17.block.3.bias"]   = state_dict["features.5.17.block.3.bias"]
            model_state_dict["7.block.17.block.3.weight"] = state_dict["features.5.17.block.3.weight"]
            model_state_dict["7.block.17.block.5.bias"]   = state_dict["features.5.17.block.5.bias"]
            model_state_dict["7.block.17.block.5.weight"] = state_dict["features.5.17.block.5.weight"]
            model_state_dict["7.block.17.layer_scale"]    = state_dict["features.5.17.layer_scale"]
            model_state_dict["7.block.18.block.0.bias"]   = state_dict["features.5.18.block.0.bias"]
            model_state_dict["7.block.18.block.0.weight"] = state_dict["features.5.18.block.0.weight"]
            model_state_dict["7.block.18.block.2.bias"]   = state_dict["features.5.18.block.2.bias"]
            model_state_dict["7.block.18.block.2.weight"] = state_dict["features.5.18.block.2.weight"]
            model_state_dict["7.block.18.block.3.bias"]   = state_dict["features.5.18.block.3.bias"]
            model_state_dict["7.block.18.block.3.weight"] = state_dict["features.5.18.block.3.weight"]
            model_state_dict["7.block.18.block.5.bias"]   = state_dict["features.5.18.block.5.bias"]
            model_state_dict["7.block.18.block.5.weight"] = state_dict["features.5.18.block.5.weight"]
            model_state_dict["7.block.18.layer_scale"]    = state_dict["features.5.18.layer_scale"]
            model_state_dict["7.block.19.block.0.bias"]   = state_dict["features.5.19.block.0.bias"]
            model_state_dict["7.block.19.block.0.weight"] = state_dict["features.5.19.block.0.weight"]
            model_state_dict["7.block.19.block.2.bias"]   = state_dict["features.5.19.block.2.bias"]
            model_state_dict["7.block.19.block.2.weight"] = state_dict["features.5.19.block.2.weight"]
            model_state_dict["7.block.19.block.3.bias"]   = state_dict["features.5.19.block.3.bias"]
            model_state_dict["7.block.19.block.3.weight"] = state_dict["features.5.19.block.3.weight"]
            model_state_dict["7.block.19.block.5.bias"]   = state_dict["features.5.19.block.5.bias"]
            model_state_dict["7.block.19.block.5.weight"] = state_dict["features.5.19.block.5.weight"]
            model_state_dict["7.block.19.layer_scale"]    = state_dict["features.5.19.layer_scale"]
            model_state_dict["7.block.2.block.0.bias"]    = state_dict["features.5.2.block.0.bias"]
            model_state_dict["7.block.2.block.0.weight"]  = state_dict["features.5.2.block.0.weight"]
            model_state_dict["7.block.2.block.2.bias"]    = state_dict["features.5.2.block.2.bias"]
            model_state_dict["7.block.2.block.2.weight"]  = state_dict["features.5.2.block.2.weight"]
            model_state_dict["7.block.2.block.3.bias"]    = state_dict["features.5.2.block.3.bias"]
            model_state_dict["7.block.2.block.3.weight"]  = state_dict["features.5.2.block.3.weight"]
            model_state_dict["7.block.2.block.5.bias"]    = state_dict["features.5.2.block.5.bias"]
            model_state_dict["7.block.2.block.5.weight"]  = state_dict["features.5.2.block.5.weight"]
            model_state_dict["7.block.2.layer_scale"]     = state_dict["features.5.2.layer_scale"]
            model_state_dict["7.block.20.block.0.bias"]   = state_dict["features.5.20.block.0.bias"]
            model_state_dict["7.block.20.block.0.weight"] = state_dict["features.5.20.block.0.weight"]
            model_state_dict["7.block.20.block.2.bias"]   = state_dict["features.5.20.block.2.bias"]
            model_state_dict["7.block.20.block.2.weight"] = state_dict["features.5.20.block.2.weight"]
            model_state_dict["7.block.20.block.3.bias"]   = state_dict["features.5.20.block.3.bias"]
            model_state_dict["7.block.20.block.3.weight"] = state_dict["features.5.20.block.3.weight"]
            model_state_dict["7.block.20.block.5.bias"]   = state_dict["features.5.20.block.5.bias"]
            model_state_dict["7.block.20.block.5.weight"] = state_dict["features.5.20.block.5.weight"]
            model_state_dict["7.block.20.layer_scale"]    = state_dict["features.5.20.layer_scale"]
            model_state_dict["7.block.21.block.0.bias"]   = state_dict["features.5.21.block.0.bias"]
            model_state_dict["7.block.21.block.0.weight"] = state_dict["features.5.21.block.0.weight"]
            model_state_dict["7.block.21.block.2.bias"]   = state_dict["features.5.21.block.2.bias"]
            model_state_dict["7.block.21.block.2.weight"] = state_dict["features.5.21.block.2.weight"]
            model_state_dict["7.block.21.block.3.bias"]   = state_dict["features.5.21.block.3.bias"]
            model_state_dict["7.block.21.block.3.weight"] = state_dict["features.5.21.block.3.weight"]
            model_state_dict["7.block.21.block.5.bias"]   = state_dict["features.5.21.block.5.bias"]
            model_state_dict["7.block.21.block.5.weight"] = state_dict["features.5.21.block.5.weight"]
            model_state_dict["7.block.21.layer_scale"]    = state_dict["features.5.21.layer_scale"]
            model_state_dict["7.block.22.block.0.bias"]   = state_dict["features.5.22.block.0.bias"]
            model_state_dict["7.block.22.block.0.weight"] = state_dict["features.5.22.block.0.weight"]
            model_state_dict["7.block.22.block.2.bias"]   = state_dict["features.5.22.block.2.bias"]
            model_state_dict["7.block.22.block.2.weight"] = state_dict["features.5.22.block.2.weight"]
            model_state_dict["7.block.22.block.3.bias"]   = state_dict["features.5.22.block.3.bias"]
            model_state_dict["7.block.22.block.3.weight"] = state_dict["features.5.22.block.3.weight"]
            model_state_dict["7.block.22.block.5.bias"]   = state_dict["features.5.22.block.5.bias"]
            model_state_dict["7.block.22.block.5.weight"] = state_dict["features.5.22.block.5.weight"]
            model_state_dict["7.block.22.layer_scale"]    = state_dict["features.5.22.layer_scale"]
            model_state_dict["7.block.23.block.0.bias"]   = state_dict["features.5.23.block.0.bias"]
            model_state_dict["7.block.23.block.0.weight"] = state_dict["features.5.23.block.0.weight"]
            model_state_dict["7.block.23.block.2.bias"]   = state_dict["features.5.23.block.2.bias"]
            model_state_dict["7.block.23.block.2.weight"] = state_dict["features.5.23.block.2.weight"]
            model_state_dict["7.block.23.block.3.bias"]   = state_dict["features.5.23.block.3.bias"]
            model_state_dict["7.block.23.block.3.weight"] = state_dict["features.5.23.block.3.weight"]
            model_state_dict["7.block.23.block.5.bias"]   = state_dict["features.5.23.block.5.bias"]
            model_state_dict["7.block.23.block.5.weight"] = state_dict["features.5.23.block.5.weight"]
            model_state_dict["7.block.23.layer_scale"]    = state_dict["features.5.23.layer_scale"]
            model_state_dict["7.block.24.block.0.bias"]   = state_dict["features.5.24.block.0.bias"]
            model_state_dict["7.block.24.block.0.weight"] = state_dict["features.5.24.block.0.weight"]
            model_state_dict["7.block.24.block.2.bias"]   = state_dict["features.5.24.block.2.bias"]
            model_state_dict["7.block.24.block.2.weight"] = state_dict["features.5.24.block.2.weight"]
            model_state_dict["7.block.24.block.3.bias"]   = state_dict["features.5.24.block.3.bias"]
            model_state_dict["7.block.24.block.3.weight"] = state_dict["features.5.24.block.3.weight"]
            model_state_dict["7.block.24.block.5.bias"]   = state_dict["features.5.24.block.5.bias"]
            model_state_dict["7.block.24.block.5.weight"] = state_dict["features.5.24.block.5.weight"]
            model_state_dict["7.block.24.layer_scale"]    = state_dict["features.5.24.layer_scale"]
            model_state_dict["7.block.25.block.0.bias"]   = state_dict["features.5.25.block.0.bias"]
            model_state_dict["7.block.25.block.0.weight"] = state_dict["features.5.25.block.0.weight"]
            model_state_dict["7.block.25.block.2.bias"]   = state_dict["features.5.25.block.2.bias"]
            model_state_dict["7.block.25.block.2.weight"] = state_dict["features.5.25.block.2.weight"]
            model_state_dict["7.block.25.block.3.bias"]   = state_dict["features.5.25.block.3.bias"]
            model_state_dict["7.block.25.block.3.weight"] = state_dict["features.5.25.block.3.weight"]
            model_state_dict["7.block.25.block.5.bias"]   = state_dict["features.5.25.block.5.bias"]
            model_state_dict["7.block.25.block.5.weight"] = state_dict["features.5.25.block.5.weight"]
            model_state_dict["7.block.25.layer_scale"]    = state_dict["features.5.25.layer_scale"]
            model_state_dict["7.block.26.block.0.bias"]   = state_dict["features.5.26.block.0.bias"]
            model_state_dict["7.block.26.block.0.weight"] = state_dict["features.5.26.block.0.weight"]
            model_state_dict["7.block.26.block.2.bias"]   = state_dict["features.5.26.block.2.bias"]
            model_state_dict["7.block.26.block.2.weight"] = state_dict["features.5.26.block.2.weight"]
            model_state_dict["7.block.26.block.3.bias"]   = state_dict["features.5.26.block.3.bias"]
            model_state_dict["7.block.26.block.3.weight"] = state_dict["features.5.26.block.3.weight"]
            model_state_dict["7.block.26.block.5.bias"]   = state_dict["features.5.26.block.5.bias"]
            model_state_dict["7.block.26.block.5.weight"] = state_dict["features.5.26.block.5.weight"]
            model_state_dict["7.block.26.layer_scale"]    = state_dict["features.5.26.layer_scale"]
            model_state_dict["7.block.3.block.0.bias"]    = state_dict["features.5.3.block.0.bias"]
            model_state_dict["7.block.3.block.0.weight"]  = state_dict["features.5.3.block.0.weight"]
            model_state_dict["7.block.3.block.2.bias"]    = state_dict["features.5.3.block.2.bias"]
            model_state_dict["7.block.3.block.2.weight"]  = state_dict["features.5.3.block.2.weight"]
            model_state_dict["7.block.3.block.3.bias"]    = state_dict["features.5.3.block.3.bias"]
            model_state_dict["7.block.3.block.3.weight"]  = state_dict["features.5.3.block.3.weight"]
            model_state_dict["7.block.3.block.5.bias"]    = state_dict["features.5.3.block.5.bias"]
            model_state_dict["7.block.3.block.5.weight"]  = state_dict["features.5.3.block.5.weight"]
            model_state_dict["7.block.3.layer_scale"]     = state_dict["features.5.3.layer_scale"]
            model_state_dict["7.block.4.block.0.bias"]    = state_dict["features.5.4.block.0.bias"]
            model_state_dict["7.block.4.block.0.weight"]  = state_dict["features.5.4.block.0.weight"]
            model_state_dict["7.block.4.block.2.bias"]    = state_dict["features.5.4.block.2.bias"]
            model_state_dict["7.block.4.block.2.weight"]  = state_dict["features.5.4.block.2.weight"]
            model_state_dict["7.block.4.block.3.bias"]    = state_dict["features.5.4.block.3.bias"]
            model_state_dict["7.block.4.block.3.weight"]  = state_dict["features.5.4.block.3.weight"]
            model_state_dict["7.block.4.block.5.bias"]    = state_dict["features.5.4.block.5.bias"]
            model_state_dict["7.block.4.block.5.weight"]  = state_dict["features.5.4.block.5.weight"]
            model_state_dict["7.block.4.layer_scale"]     = state_dict["features.5.4.layer_scale"]
            model_state_dict["7.block.5.block.0.bias"]    = state_dict["features.5.5.block.0.bias"]
            model_state_dict["7.block.5.block.0.weight"]  = state_dict["features.5.5.block.0.weight"]
            model_state_dict["7.block.5.block.2.bias"]    = state_dict["features.5.5.block.2.bias"]
            model_state_dict["7.block.5.block.2.weight"]  = state_dict["features.5.5.block.2.weight"]
            model_state_dict["7.block.5.block.3.bias"]    = state_dict["features.5.5.block.3.bias"]
            model_state_dict["7.block.5.block.3.weight"]  = state_dict["features.5.5.block.3.weight"]
            model_state_dict["7.block.5.block.5.bias"]    = state_dict["features.5.5.block.5.bias"]
            model_state_dict["7.block.5.block.5.weight"]  = state_dict["features.5.5.block.5.weight"]
            model_state_dict["7.block.5.layer_scale"]     = state_dict["features.5.5.layer_scale"]
            model_state_dict["7.block.6.block.0.bias"]    = state_dict["features.5.6.block.0.bias"]
            model_state_dict["7.block.6.block.0.weight"]  = state_dict["features.5.6.block.0.weight"]
            model_state_dict["7.block.6.block.2.bias"]    = state_dict["features.5.6.block.2.bias"]
            model_state_dict["7.block.6.block.2.weight"]  = state_dict["features.5.6.block.2.weight"]
            model_state_dict["7.block.6.block.3.bias"]    = state_dict["features.5.6.block.3.bias"]
            model_state_dict["7.block.6.block.3.weight"]  = state_dict["features.5.6.block.3.weight"]
            model_state_dict["7.block.6.block.5.bias"]    = state_dict["features.5.6.block.5.bias"]
            model_state_dict["7.block.6.block.5.weight"]  = state_dict["features.5.6.block.5.weight"]
            model_state_dict["7.block.6.layer_scale"]     = state_dict["features.5.6.layer_scale"]
            model_state_dict["7.block.7.block.0.bias"]    = state_dict["features.5.7.block.0.bias"]
            model_state_dict["7.block.7.block.0.weight"]  = state_dict["features.5.7.block.0.weight"]
            model_state_dict["7.block.7.block.2.bias"]    = state_dict["features.5.7.block.2.bias"]
            model_state_dict["7.block.7.block.2.weight"]  = state_dict["features.5.7.block.2.weight"]
            model_state_dict["7.block.7.block.3.bias"]    = state_dict["features.5.7.block.3.bias"]
            model_state_dict["7.block.7.block.3.weight"]  = state_dict["features.5.7.block.3.weight"]
            model_state_dict["7.block.7.block.5.bias"]    = state_dict["features.5.7.block.5.bias"]
            model_state_dict["7.block.7.block.5.weight"]  = state_dict["features.5.7.block.5.weight"]
            model_state_dict["7.block.7.layer_scale"]     = state_dict["features.5.7.layer_scale"]
            model_state_dict["7.block.8.block.0.bias"]    = state_dict["features.5.8.block.0.bias"]
            model_state_dict["7.block.8.block.0.weight"]  = state_dict["features.5.8.block.0.weight"]
            model_state_dict["7.block.8.block.2.bias"]    = state_dict["features.5.8.block.2.bias"]
            model_state_dict["7.block.8.block.2.weight"]  = state_dict["features.5.8.block.2.weight"]
            model_state_dict["7.block.8.block.3.bias"]    = state_dict["features.5.8.block.3.bias"]
            model_state_dict["7.block.8.block.3.weight"]  = state_dict["features.5.8.block.3.weight"]
            model_state_dict["7.block.8.block.5.bias"]    = state_dict["features.5.8.block.5.bias"]
            model_state_dict["7.block.8.block.5.weight"]  = state_dict["features.5.8.block.5.weight"]
            model_state_dict["7.block.8.layer_scale"]     = state_dict["features.5.8.layer_scale"]
            model_state_dict["7.block.9.block.0.bias"]    = state_dict["features.5.9.block.0.bias"]
            model_state_dict["7.block.9.block.0.weight"]  = state_dict["features.5.9.block.0.weight"]
            model_state_dict["7.block.9.block.2.bias"]    = state_dict["features.5.9.block.2.bias"]
            model_state_dict["7.block.9.block.2.weight"]  = state_dict["features.5.9.block.2.weight"]
            model_state_dict["7.block.9.block.3.bias"]    = state_dict["features.5.9.block.3.bias"]
            model_state_dict["7.block.9.block.3.weight"]  = state_dict["features.5.9.block.3.weight"]
            model_state_dict["7.block.9.block.5.bias"]    = state_dict["features.5.9.block.5.bias"]
            model_state_dict["7.block.9.block.5.weight"]  = state_dict["features.5.9.block.5.weight"]
            model_state_dict["7.block.9.layer_scale"]     = state_dict["features.5.9.layer_scale"]
            model_state_dict["8.bias"]                    = state_dict["features.6.0.bias"]
            model_state_dict["8.weight"]                  = state_dict["features.6.0.weight"]
            model_state_dict["9.bias"]                    = state_dict["features.6.1.bias"]
            model_state_dict["9.weight"]                  = state_dict["features.6.1.weight"]
            model_state_dict["10.block.0.block.0.bias"]   = state_dict["features.7.0.block.0.bias"]
            model_state_dict["10.block.0.block.0.weight"] = state_dict["features.7.0.block.0.weight"]
            model_state_dict["10.block.0.block.2.bias"]   = state_dict["features.7.0.block.2.bias"]
            model_state_dict["10.block.0.block.2.weight"] = state_dict["features.7.0.block.2.weight"]
            model_state_dict["10.block.0.block.3.bias"]   = state_dict["features.7.0.block.3.bias"]
            model_state_dict["10.block.0.block.3.weight"] = state_dict["features.7.0.block.3.weight"]
            model_state_dict["10.block.0.block.5.bias"]   = state_dict["features.7.0.block.5.bias"]
            model_state_dict["10.block.0.block.5.weight"] = state_dict["features.7.0.block.5.weight"]
            model_state_dict["10.block.0.layer_scale"]    = state_dict["features.7.0.layer_scale"]
            model_state_dict["10.block.1.block.0.bias"]   = state_dict["features.7.1.block.0.bias"]
            model_state_dict["10.block.1.block.0.weight"] = state_dict["features.7.1.block.0.weight"]
            model_state_dict["10.block.1.block.2.bias"]   = state_dict["features.7.1.block.2.bias"]
            model_state_dict["10.block.1.block.2.weight"] = state_dict["features.7.1.block.2.weight"]
            model_state_dict["10.block.1.block.3.bias"]   = state_dict["features.7.1.block.3.bias"]
            model_state_dict["10.block.1.block.3.weight"] = state_dict["features.7.1.block.3.weight"]
            model_state_dict["10.block.1.block.5.bias"]   = state_dict["features.7.1.block.5.bias"]
            model_state_dict["10.block.1.block.5.weight"] = state_dict["features.7.1.block.5.weight"]
            model_state_dict["10.block.1.layer_scale"]    = state_dict["features.7.1.layer_scale"]
            model_state_dict["10.block.2.block.0.bias"]   = state_dict["features.7.2.block.0.bias"]
            model_state_dict["10.block.2.block.0.weight"] = state_dict["features.7.2.block.0.weight"]
            model_state_dict["10.block.2.block.2.bias"]   = state_dict["features.7.2.block.2.bias"]
            model_state_dict["10.block.2.block.2.weight"] = state_dict["features.7.2.block.2.weight"]
            model_state_dict["10.block.2.block.3.bias"]   = state_dict["features.7.2.block.3.bias"]
            model_state_dict["10.block.2.block.3.weight"] = state_dict["features.7.2.block.3.weight"]
            model_state_dict["10.block.2.block.5.bias"]   = state_dict["features.7.2.block.5.bias"]
            model_state_dict["10.block.2.block.5.weight"] = state_dict["features.7.2.block.5.weight"]
            model_state_dict["10.block.2.layer_scale"]    = state_dict["features.7.2.layer_scale"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["12.norm.bias"]     = state_dict["classifier.0.bias"]
                model_state_dict["12.norm.weight"]   = state_dict["classifier.0.weight"]
                model_state_dict["12.linear.bias"]   = state_dict["classifier.2.bias"]
                model_state_dict["12.linear.weight"] = state_dict["classifier.2.weight"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()
