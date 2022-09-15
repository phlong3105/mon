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
    "inception3": {
        "channels": 3,
        "backbone": [
            # [from,   number, module,               args(out_channels, ...)]
            [-1,       1,      InceptionBasicConv2d, [32, 3, 2   ]],  # 0
            [-1,       1,      InceptionBasicConv2d, [32, 3, 1   ]],  # 1
            [-1,       1,      InceptionBasicConv2d, [64, 3, 1, 0]],  # 2
            [-1,       1,      MaxPool2d,            [3, 2]],         # 3
            [-1,       1,      InceptionBasicConv2d, [80,  1]],       # 4
            [-1,       1,      InceptionBasicConv2d, [192, 3]],       # 5
            [-1,       1,      MaxPool2d,            [3, 2]],         # 6
            [-1,       1,      InceptionA,           [192, 32]],      # 7
            [-1,       1,      InceptionA,           [256, 64]],      # 8
            [-1,       1,      InceptionA,           [288, 64]],      # 8
            [-1,       1,      InceptionB,           [288]],          # 9
            [-1,       1,      InceptionC,           [768, 128]],     # 10
            [-1,       1,      InceptionC,           [768, 160]],     # 11
            [-1,       1,      InceptionC,           [768, 160]],     # 12
            [-1,       1,      InceptionC,           [768, 192]],     # 13
            [-1,       1,      InceptionAux1,        [768]],          # 14
            [-1,       1,      InceptionD,           [768]],          # 15
            [-1,       1,      InceptionE,           [1280]],         # 16
            [-1,       1,      InceptionE,           [2048]],         # 17
        ],             
        "head": [      
            [-1,       1,      InceptionClassifier,  [2048]],         # 18
            [[14, -1], 1,      Join,                 []],             # 19
        ]
    },
}


@MODELS.register(name="efficientnet")
class EfficientNet(ImageClassificationModel):
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
            path        = "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth",
            filename    = "inception3-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "inception3.yaml",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "inception",
        fullname   : str  | None         = "inception3",
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
        cfg = cfg or "inception3"
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
            pretrained  = EfficientNet.init_pretrained(pretrained),
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

        init_weights = self.cfg["init_weights"]
        if init_weights:
            if isinstance(m, Conv2d) or isinstance(m, nn.Linear):
                stddev = float(m.stddev) if hasattr(m, "stddev") else 0.1  # type: ignore
                torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=stddev, a=-2, b=2)
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
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
            model_state_dict["0.bn.bias"]                          = state_dict["Conv2d_1a_3x3.bn.bias"]
            model_state_dict["0.bn.running_mean"]                  = state_dict["Conv2d_1a_3x3.bn.running_mean"]
            model_state_dict["0.bn.running_var"]                   = state_dict["Conv2d_1a_3x3.bn.running_var"]
            model_state_dict["0.bn.weight"]                        = state_dict["Conv2d_1a_3x3.bn.weight"]
            model_state_dict["0.conv.weight"]                      = state_dict["Conv2d_1a_3x3.conv.weight"]
            model_state_dict["1.bn.bias"]                          = state_dict["Conv2d_2a_3x3.bn.bias"]
            model_state_dict["1.bn.running_mean"]                  = state_dict["Conv2d_2a_3x3.bn.running_mean"]
            model_state_dict["1.bn.running_var"]                   = state_dict["Conv2d_2a_3x3.bn.running_var"]
            model_state_dict["1.bn.weight"]                        = state_dict["Conv2d_2a_3x3.bn.weight"]
            model_state_dict["1.conv.weight"]                      = state_dict["Conv2d_2a_3x3.conv.weight"]
            model_state_dict["2.bn.bias"]                          = state_dict["Conv2d_2b_3x3.bn.bias"]
            model_state_dict["2.bn.running_mean"]                  = state_dict["Conv2d_2b_3x3.bn.running_mean"]
            model_state_dict["2.bn.running_var"]                   = state_dict["Conv2d_2b_3x3.bn.running_var"]
            model_state_dict["2.bn.weight"]                        = state_dict["Conv2d_2b_3x3.bn.weight"]
            model_state_dict["2.conv.weight"]                      = state_dict["Conv2d_2b_3x3.conv.weight"]
            model_state_dict["4.bn.bias"]                          = state_dict["Conv2d_3b_1x1.bn.bias"]
            model_state_dict["4.bn.running_mean"]                  = state_dict["Conv2d_3b_1x1.bn.running_mean"]
            model_state_dict["4.bn.running_var"]                   = state_dict["Conv2d_3b_1x1.bn.running_var"]
            model_state_dict["4.bn.weight"]                        = state_dict["Conv2d_3b_1x1.bn.weight"]
            model_state_dict["4.conv.weight"]                      = state_dict["Conv2d_3b_1x1.conv.weight"]
            model_state_dict["5.bn.bias"]                          = state_dict["Conv2d_4a_3x3.bn.bias"]
            model_state_dict["5.bn.running_mean"]                  = state_dict["Conv2d_4a_3x3.bn.running_mean"]
            model_state_dict["5.bn.running_var"]                   = state_dict["Conv2d_4a_3x3.bn.running_var"]
            model_state_dict["5.bn.weight"]                        = state_dict["Conv2d_4a_3x3.bn.weight"]
            model_state_dict["5.conv.weight"]                      = state_dict["Conv2d_4a_3x3.conv.weight"]
            model_state_dict["7.branch1x1.bn.bias"]                = state_dict["Mixed_5b.branch1x1.bn.bias"]
            model_state_dict["7.branch1x1.bn.running_mean"]        = state_dict["Mixed_5b.branch1x1.bn.running_mean"]
            model_state_dict["7.branch1x1.bn.running_var"]         = state_dict["Mixed_5b.branch1x1.bn.running_var"]
            model_state_dict["7.branch1x1.bn.weight"]              = state_dict["Mixed_5b.branch1x1.bn.weight"]
            model_state_dict["7.branch1x1.conv.weight"]            = state_dict["Mixed_5b.branch1x1.conv.weight"]
            model_state_dict["7.branch3x3dbl_1.bn.bias"]           = state_dict["Mixed_5b.branch3x3dbl_1.bn.bias"]
            model_state_dict["7.branch3x3dbl_1.bn.running_mean"]   = state_dict["Mixed_5b.branch3x3dbl_1.bn.running_mean"]
            model_state_dict["7.branch3x3dbl_1.bn.running_var"]    = state_dict["Mixed_5b.branch3x3dbl_1.bn.running_var"]
            model_state_dict["7.branch3x3dbl_1.bn.weight"]         = state_dict["Mixed_5b.branch3x3dbl_1.bn.weight"]
            model_state_dict["7.branch3x3dbl_1.conv.weight"]       = state_dict["Mixed_5b.branch3x3dbl_1.conv.weight"]
            model_state_dict["7.branch3x3dbl_2.bn.bias"]           = state_dict["Mixed_5b.branch3x3dbl_2.bn.bias"]
            model_state_dict["7.branch3x3dbl_2.bn.running_mean"]   = state_dict["Mixed_5b.branch3x3dbl_2.bn.running_mean"]
            model_state_dict["7.branch3x3dbl_2.bn.running_var"]    = state_dict["Mixed_5b.branch3x3dbl_2.bn.running_var"]
            model_state_dict["7.branch3x3dbl_2.bn.weight"]         = state_dict["Mixed_5b.branch3x3dbl_2.bn.weight"]
            model_state_dict["7.branch3x3dbl_2.conv.weight"]       = state_dict["Mixed_5b.branch3x3dbl_2.conv.weight"]
            model_state_dict["7.branch3x3dbl_3.bn.bias"]           = state_dict["Mixed_5b.branch3x3dbl_3.bn.bias"]
            model_state_dict["7.branch3x3dbl_3.bn.running_mean"]   = state_dict["Mixed_5b.branch3x3dbl_3.bn.running_mean"]
            model_state_dict["7.branch3x3dbl_3.bn.running_var"]    = state_dict["Mixed_5b.branch3x3dbl_3.bn.running_var"]
            model_state_dict["7.branch3x3dbl_3.bn.weight"]         = state_dict["Mixed_5b.branch3x3dbl_3.bn.weight"]
            model_state_dict["7.branch3x3dbl_3.conv.weight"]       = state_dict["Mixed_5b.branch3x3dbl_3.conv.weight"]
            model_state_dict["7.branch5x5_1.bn.bias"]              = state_dict["Mixed_5b.branch5x5_1.bn.bias"]
            model_state_dict["7.branch5x5_1.bn.running_mean"]      = state_dict["Mixed_5b.branch5x5_1.bn.running_mean"]
            model_state_dict["7.branch5x5_1.bn.running_var"]       = state_dict["Mixed_5b.branch5x5_1.bn.running_var"]
            model_state_dict["7.branch5x5_1.bn.weight"]            = state_dict["Mixed_5b.branch5x5_1.bn.weight"]
            model_state_dict["7.branch5x5_1.conv.weight"]          = state_dict["Mixed_5b.branch5x5_1.conv.weight"]
            model_state_dict["7.branch5x5_2.bn.bias"]              = state_dict["Mixed_5b.branch5x5_2.bn.bias"]
            model_state_dict["7.branch5x5_2.bn.running_mean"]      = state_dict["Mixed_5b.branch5x5_2.bn.running_mean"]
            model_state_dict["7.branch5x5_2.bn.running_var"]       = state_dict["Mixed_5b.branch5x5_2.bn.running_var"]
            model_state_dict["7.branch5x5_2.bn.weight"]            = state_dict["Mixed_5b.branch5x5_2.bn.weight"]
            model_state_dict["7.branch5x5_2.conv.weight"]          = state_dict["Mixed_5b.branch5x5_2.conv.weight"]
            model_state_dict["7.branch_pool.bn.bias"]              = state_dict["Mixed_5b.branch_pool.bn.bias"]
            model_state_dict["7.branch_pool.bn.running_mean"]      = state_dict["Mixed_5b.branch_pool.bn.running_mean"]
            model_state_dict["7.branch_pool.bn.running_var"]       = state_dict["Mixed_5b.branch_pool.bn.running_var"]
            model_state_dict["7.branch_pool.bn.weight"]            = state_dict["Mixed_5b.branch_pool.bn.weight"]
            model_state_dict["7.branch_pool.conv.weight"]          = state_dict["Mixed_5b.branch_pool.conv.weight"]
            model_state_dict["8.branch1x1.bn.bias"]                = state_dict["Mixed_5c.branch1x1.bn.bias"]
            model_state_dict["8.branch1x1.bn.running_mean"]        = state_dict["Mixed_5c.branch1x1.bn.running_mean"]
            model_state_dict["8.branch1x1.bn.running_var"]         = state_dict["Mixed_5c.branch1x1.bn.running_var"]
            model_state_dict["8.branch1x1.bn.weight"]              = state_dict["Mixed_5c.branch1x1.bn.weight"]
            model_state_dict["8.branch1x1.conv.weight"]            = state_dict["Mixed_5c.branch1x1.conv.weight"]
            model_state_dict["8.branch3x3dbl_1.bn.bias"]           = state_dict["Mixed_5c.branch3x3dbl_1.bn.bias"]
            model_state_dict["8.branch3x3dbl_1.bn.running_mean"]   = state_dict["Mixed_5c.branch3x3dbl_1.bn.running_mean"]
            model_state_dict["8.branch3x3dbl_1.bn.running_var"]    = state_dict["Mixed_5c.branch3x3dbl_1.bn.running_var"]
            model_state_dict["8.branch3x3dbl_1.bn.weight"]         = state_dict["Mixed_5c.branch3x3dbl_1.bn.weight"]
            model_state_dict["8.branch3x3dbl_1.conv.weight"]       = state_dict["Mixed_5c.branch3x3dbl_1.conv.weight"]
            model_state_dict["8.branch3x3dbl_2.bn.bias"]           = state_dict["Mixed_5c.branch3x3dbl_2.bn.bias"]
            model_state_dict["8.branch3x3dbl_2.bn.running_mean"]   = state_dict["Mixed_5c.branch3x3dbl_2.bn.running_mean"]
            model_state_dict["8.branch3x3dbl_2.bn.running_var"]    = state_dict["Mixed_5c.branch3x3dbl_2.bn.running_var"]
            model_state_dict["8.branch3x3dbl_2.bn.weight"]         = state_dict["Mixed_5c.branch3x3dbl_2.bn.weight"]
            model_state_dict["8.branch3x3dbl_2.conv.weight"]       = state_dict["Mixed_5c.branch3x3dbl_2.conv.weight"]
            model_state_dict["8.branch3x3dbl_3.bn.bias"]           = state_dict["Mixed_5c.branch3x3dbl_3.bn.bias"]
            model_state_dict["8.branch3x3dbl_3.bn.running_mean"]   = state_dict["Mixed_5c.branch3x3dbl_3.bn.running_mean"]
            model_state_dict["8.branch3x3dbl_3.bn.running_var"]    = state_dict["Mixed_5c.branch3x3dbl_3.bn.running_var"]
            model_state_dict["8.branch3x3dbl_3.bn.weight"]         = state_dict["Mixed_5c.branch3x3dbl_3.bn.weight"]
            model_state_dict["8.branch3x3dbl_3.conv.weight"]       = state_dict["Mixed_5c.branch3x3dbl_3.conv.weight"]
            model_state_dict["8.branch5x5_1.bn.bias"]              = state_dict["Mixed_5c.branch5x5_1.bn.bias"]
            model_state_dict["8.branch5x5_1.bn.running_mean"]      = state_dict["Mixed_5c.branch5x5_1.bn.running_mean"]
            model_state_dict["8.branch5x5_1.bn.running_var"]       = state_dict["Mixed_5c.branch5x5_1.bn.running_var"]
            model_state_dict["8.branch5x5_1.bn.weight"]            = state_dict["Mixed_5c.branch5x5_1.bn.weight"]
            model_state_dict["8.branch5x5_1.conv.weight"]          = state_dict["Mixed_5c.branch5x5_1.conv.weight"]
            model_state_dict["8.branch5x5_2.bn.bias"]              = state_dict["Mixed_5c.branch5x5_2.bn.bias"]
            model_state_dict["8.branch5x5_2.bn.running_mean"]      = state_dict["Mixed_5c.branch5x5_2.bn.running_mean"]
            model_state_dict["8.branch5x5_2.bn.running_var"]       = state_dict["Mixed_5c.branch5x5_2.bn.running_var"]
            model_state_dict["8.branch5x5_2.bn.weight"]            = state_dict["Mixed_5c.branch5x5_2.bn.weight"]
            model_state_dict["8.branch5x5_2.conv.weight"]          = state_dict["Mixed_5c.branch5x5_2.conv.weight"]
            model_state_dict["8.branch_pool.bn.bias"]              = state_dict["Mixed_5c.branch_pool.bn.bias"]
            model_state_dict["8.branch_pool.bn.running_mean"]      = state_dict["Mixed_5c.branch_pool.bn.running_mean"]
            model_state_dict["8.branch_pool.bn.running_var"]       = state_dict["Mixed_5c.branch_pool.bn.running_var"]
            model_state_dict["8.branch_pool.bn.weight"]            = state_dict["Mixed_5c.branch_pool.bn.weight"]
            model_state_dict["8.branch_pool.conv.weight"]          = state_dict["Mixed_5c.branch_pool.conv.weight"]
            model_state_dict["9.branch1x1.bn.bias"]                = state_dict["Mixed_5d.branch1x1.bn.bias"]
            model_state_dict["9.branch1x1.bn.running_mean"]        = state_dict["Mixed_5d.branch1x1.bn.running_mean"]
            model_state_dict["9.branch1x1.bn.running_var"]         = state_dict["Mixed_5d.branch1x1.bn.running_var"]
            model_state_dict["9.branch1x1.bn.weight"]              = state_dict["Mixed_5d.branch1x1.bn.weight"]
            model_state_dict["9.branch1x1.conv.weight"]            = state_dict["Mixed_5d.branch1x1.conv.weight"]
            model_state_dict["9.branch3x3dbl_1.bn.bias"]           = state_dict["Mixed_5d.branch3x3dbl_1.bn.bias"]
            model_state_dict["9.branch3x3dbl_1.bn.running_mean"]   = state_dict["Mixed_5d.branch3x3dbl_1.bn.running_mean"]
            model_state_dict["9.branch3x3dbl_1.bn.running_var"]    = state_dict["Mixed_5d.branch3x3dbl_1.bn.running_var"]
            model_state_dict["9.branch3x3dbl_1.bn.weight"]         = state_dict["Mixed_5d.branch3x3dbl_1.bn.weight"]
            model_state_dict["9.branch3x3dbl_1.conv.weight"]       = state_dict["Mixed_5d.branch3x3dbl_1.conv.weight"]
            model_state_dict["9.branch3x3dbl_2.bn.bias"]           = state_dict["Mixed_5d.branch3x3dbl_2.bn.bias"]
            model_state_dict["9.branch3x3dbl_2.bn.running_mean"]   = state_dict["Mixed_5d.branch3x3dbl_2.bn.running_mean"]
            model_state_dict["9.branch3x3dbl_2.bn.running_var"]    = state_dict["Mixed_5d.branch3x3dbl_2.bn.running_var"]
            model_state_dict["9.branch3x3dbl_2.bn.weight"]         = state_dict["Mixed_5d.branch3x3dbl_2.bn.weight"]
            model_state_dict["9.branch3x3dbl_2.conv.weight"]       = state_dict["Mixed_5d.branch3x3dbl_2.conv.weight"]
            model_state_dict["9.branch3x3dbl_3.bn.bias"]           = state_dict["Mixed_5d.branch3x3dbl_3.bn.bias"]
            model_state_dict["9.branch3x3dbl_3.bn.running_mean"]   = state_dict["Mixed_5d.branch3x3dbl_3.bn.running_mean"]
            model_state_dict["9.branch3x3dbl_3.bn.running_var"]    = state_dict["Mixed_5d.branch3x3dbl_3.bn.running_var"]
            model_state_dict["9.branch3x3dbl_3.bn.weight"]         = state_dict["Mixed_5d.branch3x3dbl_3.bn.weight"]
            model_state_dict["9.branch3x3dbl_3.conv.weight"]       = state_dict["Mixed_5d.branch3x3dbl_3.conv.weight"]
            model_state_dict["9.branch5x5_1.bn.bias"]              = state_dict["Mixed_5d.branch5x5_1.bn.bias"]
            model_state_dict["9.branch5x5_1.bn.running_mean"]      = state_dict["Mixed_5d.branch5x5_1.bn.running_mean"]
            model_state_dict["9.branch5x5_1.bn.running_var"]       = state_dict["Mixed_5d.branch5x5_1.bn.running_var"]
            model_state_dict["9.branch5x5_1.bn.weight"]            = state_dict["Mixed_5d.branch5x5_1.bn.weight"]
            model_state_dict["9.branch5x5_1.conv.weight"]          = state_dict["Mixed_5d.branch5x5_1.conv.weight"]
            model_state_dict["9.branch5x5_2.bn.bias"]              = state_dict["Mixed_5d.branch5x5_2.bn.bias"]
            model_state_dict["9.branch5x5_2.bn.running_mean"]      = state_dict["Mixed_5d.branch5x5_2.bn.running_mean"]
            model_state_dict["9.branch5x5_2.bn.running_var"]       = state_dict["Mixed_5d.branch5x5_2.bn.running_var"]
            model_state_dict["9.branch5x5_2.bn.weight"]            = state_dict["Mixed_5d.branch5x5_2.bn.weight"]
            model_state_dict["9.branch5x5_2.conv.weight"]          = state_dict["Mixed_5d.branch5x5_2.conv.weight"]
            model_state_dict["9.branch_pool.bn.bias"]              = state_dict["Mixed_5d.branch_pool.bn.bias"]
            model_state_dict["9.branch_pool.bn.running_mean"]      = state_dict["Mixed_5d.branch_pool.bn.running_mean"]
            model_state_dict["9.branch_pool.bn.running_var"]       = state_dict["Mixed_5d.branch_pool.bn.running_var"]
            model_state_dict["9.branch_pool.bn.weight"]            = state_dict["Mixed_5d.branch_pool.bn.weight"]
            model_state_dict["9.branch_pool.conv.weight"]          = state_dict["Mixed_5d.branch_pool.conv.weight"]
            model_state_dict["10.branch3x3.bn.bias"]               = state_dict["Mixed_6a.branch3x3.bn.bias"]
            model_state_dict["10.branch3x3.bn.running_mean"]       = state_dict["Mixed_6a.branch3x3.bn.running_mean"]
            model_state_dict["10.branch3x3.bn.running_var"]        = state_dict["Mixed_6a.branch3x3.bn.running_var"]
            model_state_dict["10.branch3x3.bn.weight"]             = state_dict["Mixed_6a.branch3x3.bn.weight"]
            model_state_dict["10.branch3x3.conv.weight"]           = state_dict["Mixed_6a.branch3x3.conv.weight"]
            model_state_dict["10.branch3x3dbl_1.bn.bias"]          = state_dict["Mixed_6a.branch3x3dbl_1.bn.bias"]
            model_state_dict["10.branch3x3dbl_1.bn.running_mean"]  = state_dict["Mixed_6a.branch3x3dbl_1.bn.running_mean"]
            model_state_dict["10.branch3x3dbl_1.bn.running_var"]   = state_dict["Mixed_6a.branch3x3dbl_1.bn.running_var"]
            model_state_dict["10.branch3x3dbl_1.bn.weight"]        = state_dict["Mixed_6a.branch3x3dbl_1.bn.weight"]
            model_state_dict["10.branch3x3dbl_1.conv.weight"]      = state_dict["Mixed_6a.branch3x3dbl_1.conv.weight"]
            model_state_dict["10.branch3x3dbl_2.bn.bias"]          = state_dict["Mixed_6a.branch3x3dbl_2.bn.bias"]
            model_state_dict["10.branch3x3dbl_2.bn.running_mean"]  = state_dict["Mixed_6a.branch3x3dbl_2.bn.running_mean"]
            model_state_dict["10.branch3x3dbl_2.bn.running_var"]   = state_dict["Mixed_6a.branch3x3dbl_2.bn.running_var"]
            model_state_dict["10.branch3x3dbl_2.bn.weight"]        = state_dict["Mixed_6a.branch3x3dbl_2.bn.weight"]
            model_state_dict["10.branch3x3dbl_2.conv.weight"]      = state_dict["Mixed_6a.branch3x3dbl_2.conv.weight"]
            model_state_dict["10.branch3x3dbl_3.bn.bias"]          = state_dict["Mixed_6a.branch3x3dbl_3.bn.bias"]
            model_state_dict["10.branch3x3dbl_3.bn.running_mean"]  = state_dict["Mixed_6a.branch3x3dbl_3.bn.running_mean"]
            model_state_dict["10.branch3x3dbl_3.bn.running_var"]   = state_dict["Mixed_6a.branch3x3dbl_3.bn.running_var"]
            model_state_dict["10.branch3x3dbl_3.bn.weight"]        = state_dict["Mixed_6a.branch3x3dbl_3.bn.weight"]
            model_state_dict["10.branch3x3dbl_3.conv.weight"]      = state_dict["Mixed_6a.branch3x3dbl_3.conv.weight"]
            model_state_dict["11.branch1x1.bn.bias"]               = state_dict["Mixed_6b.branch1x1.bn.bias"]
            model_state_dict["11.branch1x1.bn.running_mean"]       = state_dict["Mixed_6b.branch1x1.bn.running_mean"]
            model_state_dict["11.branch1x1.bn.running_var"]        = state_dict["Mixed_6b.branch1x1.bn.running_var"]
            model_state_dict["11.branch1x1.bn.weight"]             = state_dict["Mixed_6b.branch1x1.bn.weight"]
            model_state_dict["11.branch1x1.conv.weight"]           = state_dict["Mixed_6b.branch1x1.conv.weight"]
            model_state_dict["11.branch7x7_1.bn.bias"]             = state_dict["Mixed_6b.branch7x7_1.bn.bias"]
            model_state_dict["11.branch7x7_1.bn.running_mean"]     = state_dict["Mixed_6b.branch7x7_1.bn.running_mean"]
            model_state_dict["11.branch7x7_1.bn.running_var"]      = state_dict["Mixed_6b.branch7x7_1.bn.running_var"]
            model_state_dict["11.branch7x7_1.bn.weight"]           = state_dict["Mixed_6b.branch7x7_1.bn.weight"]
            model_state_dict["11.branch7x7_1.conv.weight"]         = state_dict["Mixed_6b.branch7x7_1.conv.weight"]
            model_state_dict["11.branch7x7_2.bn.bias"]             = state_dict["Mixed_6b.branch7x7_2.bn.bias"]
            model_state_dict["11.branch7x7_2.bn.running_mean"]     = state_dict["Mixed_6b.branch7x7_2.bn.running_mean"]
            model_state_dict["11.branch7x7_2.bn.running_var"]      = state_dict["Mixed_6b.branch7x7_2.bn.running_var"]
            model_state_dict["11.branch7x7_2.bn.weight"]           = state_dict["Mixed_6b.branch7x7_2.bn.weight"]
            model_state_dict["11.branch7x7_2.conv.weight"]         = state_dict["Mixed_6b.branch7x7_2.conv.weight"]
            model_state_dict["11.branch7x7_3.bn.bias"]             = state_dict["Mixed_6b.branch7x7_3.bn.bias"]
            model_state_dict["11.branch7x7_3.bn.running_mean"]     = state_dict["Mixed_6b.branch7x7_3.bn.running_mean"]
            model_state_dict["11.branch7x7_3.bn.running_var"]      = state_dict["Mixed_6b.branch7x7_3.bn.running_var"]
            model_state_dict["11.branch7x7_3.bn.weight"]           = state_dict["Mixed_6b.branch7x7_3.bn.weight"]
            model_state_dict["11.branch7x7_3.conv.weight"]         = state_dict["Mixed_6b.branch7x7_3.conv.weight"]
            model_state_dict["11.branch7x7dbl_1.bn.bias"]          = state_dict["Mixed_6b.branch7x7dbl_1.bn.bias"]
            model_state_dict["11.branch7x7dbl_1.bn.running_mean"]  = state_dict["Mixed_6b.branch7x7dbl_1.bn.running_mean"]
            model_state_dict["11.branch7x7dbl_1.bn.running_var"]   = state_dict["Mixed_6b.branch7x7dbl_1.bn.running_var"]
            model_state_dict["11.branch7x7dbl_1.bn.weight"]        = state_dict["Mixed_6b.branch7x7dbl_1.bn.weight"]
            model_state_dict["11.branch7x7dbl_1.conv.weight"]      = state_dict["Mixed_6b.branch7x7dbl_1.conv.weight"]
            model_state_dict["11.branch7x7dbl_2.bn.bias"]          = state_dict["Mixed_6b.branch7x7dbl_2.bn.bias"]
            model_state_dict["11.branch7x7dbl_2.bn.running_mean"]  = state_dict["Mixed_6b.branch7x7dbl_2.bn.running_mean"]
            model_state_dict["11.branch7x7dbl_2.bn.running_var"]   = state_dict["Mixed_6b.branch7x7dbl_2.bn.running_var"]
            model_state_dict["11.branch7x7dbl_2.bn.weight"]        = state_dict["Mixed_6b.branch7x7dbl_2.bn.weight"]
            model_state_dict["11.branch7x7dbl_2.conv.weight"]      = state_dict["Mixed_6b.branch7x7dbl_2.conv.weight"]
            model_state_dict["11.branch7x7dbl_3.bn.bias"]          = state_dict["Mixed_6b.branch7x7dbl_3.bn.bias"]
            model_state_dict["11.branch7x7dbl_3.bn.running_mean"]  = state_dict["Mixed_6b.branch7x7dbl_3.bn.running_mean"]
            model_state_dict["11.branch7x7dbl_3.bn.running_var"]   = state_dict["Mixed_6b.branch7x7dbl_3.bn.running_var"]
            model_state_dict["11.branch7x7dbl_3.bn.weight"]        = state_dict["Mixed_6b.branch7x7dbl_3.bn.weight"]
            model_state_dict["11.branch7x7dbl_3.conv.weight"]      = state_dict["Mixed_6b.branch7x7dbl_3.conv.weight"]
            model_state_dict["11.branch7x7dbl_4.bn.bias"]          = state_dict["Mixed_6b.branch7x7dbl_4.bn.bias"]
            model_state_dict["11.branch7x7dbl_4.bn.running_mean"]  = state_dict["Mixed_6b.branch7x7dbl_4.bn.running_mean"]
            model_state_dict["11.branch7x7dbl_4.bn.running_var"]   = state_dict["Mixed_6b.branch7x7dbl_4.bn.running_var"]
            model_state_dict["11.branch7x7dbl_4.bn.weight"]        = state_dict["Mixed_6b.branch7x7dbl_4.bn.weight"]
            model_state_dict["11.branch7x7dbl_4.conv.weight"]      = state_dict["Mixed_6b.branch7x7dbl_4.conv.weight"]
            model_state_dict["11.branch7x7dbl_5.bn.bias"]          = state_dict["Mixed_6b.branch7x7dbl_5.bn.bias"]
            model_state_dict["11.branch7x7dbl_5.bn.running_mean"]  = state_dict["Mixed_6b.branch7x7dbl_5.bn.running_mean"]
            model_state_dict["11.branch7x7dbl_5.bn.running_var"]   = state_dict["Mixed_6b.branch7x7dbl_5.bn.running_var"]
            model_state_dict["11.branch7x7dbl_5.bn.weight"]        = state_dict["Mixed_6b.branch7x7dbl_5.bn.weight"]
            model_state_dict["11.branch7x7dbl_5.conv.weight"]      = state_dict["Mixed_6b.branch7x7dbl_5.conv.weight"]
            model_state_dict["11.branch_pool.bn.bias"]             = state_dict["Mixed_6b.branch_pool.bn.bias"]
            model_state_dict["11.branch_pool.bn.running_mean"]     = state_dict["Mixed_6b.branch_pool.bn.running_mean"]
            model_state_dict["11.branch_pool.bn.running_var"]      = state_dict["Mixed_6b.branch_pool.bn.running_var"]
            model_state_dict["11.branch_pool.bn.weight"]           = state_dict["Mixed_6b.branch_pool.bn.weight"]
            model_state_dict["11.branch_pool.conv.weight"]         = state_dict["Mixed_6b.branch_pool.conv.weight"]
            model_state_dict["12.branch1x1.bn.bias"]               = state_dict["Mixed_6c.branch1x1.bn.bias"]
            model_state_dict["12.branch1x1.bn.running_mean"]       = state_dict["Mixed_6c.branch1x1.bn.running_mean"]
            model_state_dict["12.branch1x1.bn.running_var"]        = state_dict["Mixed_6c.branch1x1.bn.running_var"]
            model_state_dict["12.branch1x1.bn.weight"]             = state_dict["Mixed_6c.branch1x1.bn.weight"]
            model_state_dict["12.branch1x1.conv.weight"]           = state_dict["Mixed_6c.branch1x1.conv.weight"]
            model_state_dict["12.branch7x7_1.bn.bias"]             = state_dict["Mixed_6c.branch7x7_1.bn.bias"]
            model_state_dict["12.branch7x7_1.bn.running_mean"]     = state_dict["Mixed_6c.branch7x7_1.bn.running_mean"]
            model_state_dict["12.branch7x7_1.bn.running_var"]      = state_dict["Mixed_6c.branch7x7_1.bn.running_var"]
            model_state_dict["12.branch7x7_1.bn.weight"]           = state_dict["Mixed_6c.branch7x7_1.bn.weight"]
            model_state_dict["12.branch7x7_1.conv.weight"]         = state_dict["Mixed_6c.branch7x7_1.conv.weight"]
            model_state_dict["12.branch7x7_2.bn.bias"]             = state_dict["Mixed_6c.branch7x7_2.bn.bias"]
            model_state_dict["12.branch7x7_2.bn.running_mean"]     = state_dict["Mixed_6c.branch7x7_2.bn.running_mean"]
            model_state_dict["12.branch7x7_2.bn.running_var"]      = state_dict["Mixed_6c.branch7x7_2.bn.running_var"]
            model_state_dict["12.branch7x7_2.bn.weight"]           = state_dict["Mixed_6c.branch7x7_2.bn.weight"]
            model_state_dict["12.branch7x7_2.conv.weight"]         = state_dict["Mixed_6c.branch7x7_2.conv.weight"]
            model_state_dict["12.branch7x7_3.bn.bias"]             = state_dict["Mixed_6c.branch7x7_3.bn.bias"]
            model_state_dict["12.branch7x7_3.bn.running_mean"]     = state_dict["Mixed_6c.branch7x7_3.bn.running_mean"]
            model_state_dict["12.branch7x7_3.bn.running_var"]      = state_dict["Mixed_6c.branch7x7_3.bn.running_var"]
            model_state_dict["12.branch7x7_3.bn.weight"]           = state_dict["Mixed_6c.branch7x7_3.bn.weight"]
            model_state_dict["12.branch7x7_3.conv.weight"]         = state_dict["Mixed_6c.branch7x7_3.conv.weight"]
            model_state_dict["12.branch7x7dbl_1.bn.bias"]          = state_dict["Mixed_6c.branch7x7dbl_1.bn.bias"]
            model_state_dict["12.branch7x7dbl_1.bn.running_mean"]  = state_dict["Mixed_6c.branch7x7dbl_1.bn.running_mean"]
            model_state_dict["12.branch7x7dbl_1.bn.running_var"]   = state_dict["Mixed_6c.branch7x7dbl_1.bn.running_var"]
            model_state_dict["12.branch7x7dbl_1.bn.weight"]        = state_dict["Mixed_6c.branch7x7dbl_1.bn.weight"]
            model_state_dict["12.branch7x7dbl_1.conv.weight"]      = state_dict["Mixed_6c.branch7x7dbl_1.conv.weight"]
            model_state_dict["12.branch7x7dbl_2.bn.bias"]          = state_dict["Mixed_6c.branch7x7dbl_2.bn.bias"]
            model_state_dict["12.branch7x7dbl_2.bn.running_mean"]  = state_dict["Mixed_6c.branch7x7dbl_2.bn.running_mean"]
            model_state_dict["12.branch7x7dbl_2.bn.running_var"]   = state_dict["Mixed_6c.branch7x7dbl_2.bn.running_var"]
            model_state_dict["12.branch7x7dbl_2.bn.weight"]        = state_dict["Mixed_6c.branch7x7dbl_2.bn.weight"]
            model_state_dict["12.branch7x7dbl_2.conv.weight"]      = state_dict["Mixed_6c.branch7x7dbl_2.conv.weight"]
            model_state_dict["12.branch7x7dbl_3.bn.bias"]          = state_dict["Mixed_6c.branch7x7dbl_3.bn.bias"]
            model_state_dict["12.branch7x7dbl_3.bn.running_mean"]  = state_dict["Mixed_6c.branch7x7dbl_3.bn.running_mean"]
            model_state_dict["12.branch7x7dbl_3.bn.running_var"]   = state_dict["Mixed_6c.branch7x7dbl_3.bn.running_var"]
            model_state_dict["12.branch7x7dbl_3.bn.weight"]        = state_dict["Mixed_6c.branch7x7dbl_3.bn.weight"]
            model_state_dict["12.branch7x7dbl_3.conv.weight"]      = state_dict["Mixed_6c.branch7x7dbl_3.conv.weight"]
            model_state_dict["12.branch7x7dbl_4.bn.bias"]          = state_dict["Mixed_6c.branch7x7dbl_4.bn.bias"]
            model_state_dict["12.branch7x7dbl_4.bn.running_mean"]  = state_dict["Mixed_6c.branch7x7dbl_4.bn.running_mean"]
            model_state_dict["12.branch7x7dbl_4.bn.running_var"]   = state_dict["Mixed_6c.branch7x7dbl_4.bn.running_var"]
            model_state_dict["12.branch7x7dbl_4.bn.weight"]        = state_dict["Mixed_6c.branch7x7dbl_4.bn.weight"]
            model_state_dict["12.branch7x7dbl_4.conv.weight"]      = state_dict["Mixed_6c.branch7x7dbl_4.conv.weight"]
            model_state_dict["12.branch7x7dbl_5.bn.bias"]          = state_dict["Mixed_6c.branch7x7dbl_5.bn.bias"]
            model_state_dict["12.branch7x7dbl_5.bn.running_mean"]  = state_dict["Mixed_6c.branch7x7dbl_5.bn.running_mean"]
            model_state_dict["12.branch7x7dbl_5.bn.running_var"]   = state_dict["Mixed_6c.branch7x7dbl_5.bn.running_var"]
            model_state_dict["12.branch7x7dbl_5.bn.weight"]        = state_dict["Mixed_6c.branch7x7dbl_5.bn.weight"]
            model_state_dict["12.branch7x7dbl_5.conv.weight"]      = state_dict["Mixed_6c.branch7x7dbl_5.conv.weight"]
            model_state_dict["12.branch_pool.bn.bias"]             = state_dict["Mixed_6c.branch_pool.bn.bias"]
            model_state_dict["12.branch_pool.bn.running_mean"]     = state_dict["Mixed_6c.branch_pool.bn.running_mean"]
            model_state_dict["12.branch_pool.bn.running_var"]      = state_dict["Mixed_6c.branch_pool.bn.running_var"]
            model_state_dict["12.branch_pool.bn.weight"]           = state_dict["Mixed_6c.branch_pool.bn.weight"]
            model_state_dict["12.branch_pool.conv.weight"]         = state_dict["Mixed_6c.branch_pool.conv.weight"]
            model_state_dict["13.branch1x1.bn.bias"]               = state_dict["Mixed_6d.branch1x1.bn.bias"]
            model_state_dict["13.branch1x1.bn.running_mean"]       = state_dict["Mixed_6d.branch1x1.bn.running_mean"]
            model_state_dict["13.branch1x1.bn.running_var"]        = state_dict["Mixed_6d.branch1x1.bn.running_var"]
            model_state_dict["13.branch1x1.bn.weight"]             = state_dict["Mixed_6d.branch1x1.bn.weight"]
            model_state_dict["13.branch1x1.conv.weight"]           = state_dict["Mixed_6d.branch1x1.conv.weight"]
            model_state_dict["13.branch7x7_1.bn.bias"]             = state_dict["Mixed_6d.branch7x7_1.bn.bias"]
            model_state_dict["13.branch7x7_1.bn.running_mean"]     = state_dict["Mixed_6d.branch7x7_1.bn.running_mean"]
            model_state_dict["13.branch7x7_1.bn.running_var"]      = state_dict["Mixed_6d.branch7x7_1.bn.running_var"]
            model_state_dict["13.branch7x7_1.bn.weight"]           = state_dict["Mixed_6d.branch7x7_1.bn.weight"]
            model_state_dict["13.branch7x7_1.conv.weight"]         = state_dict["Mixed_6d.branch7x7_1.conv.weight"]
            model_state_dict["13.branch7x7_2.bn.bias"]             = state_dict["Mixed_6d.branch7x7_2.bn.bias"]
            model_state_dict["13.branch7x7_2.bn.running_mean"]     = state_dict["Mixed_6d.branch7x7_2.bn.running_mean"]
            model_state_dict["13.branch7x7_2.bn.running_var"]      = state_dict["Mixed_6d.branch7x7_2.bn.running_var"]
            model_state_dict["13.branch7x7_2.bn.weight"]           = state_dict["Mixed_6d.branch7x7_2.bn.weight"]
            model_state_dict["13.branch7x7_2.conv.weight"]         = state_dict["Mixed_6d.branch7x7_2.conv.weight"]
            model_state_dict["13.branch7x7_3.bn.bias"]             = state_dict["Mixed_6d.branch7x7_3.bn.bias"]
            model_state_dict["13.branch7x7_3.bn.running_mean"]     = state_dict["Mixed_6d.branch7x7_3.bn.running_mean"]
            model_state_dict["13.branch7x7_3.bn.running_var"]      = state_dict["Mixed_6d.branch7x7_3.bn.running_var"]
            model_state_dict["13.branch7x7_3.bn.weight"]           = state_dict["Mixed_6d.branch7x7_3.bn.weight"]
            model_state_dict["13.branch7x7_3.conv.weight"]         = state_dict["Mixed_6d.branch7x7_3.conv.weight"]
            model_state_dict["13.branch7x7dbl_1.bn.bias"]          = state_dict["Mixed_6d.branch7x7dbl_1.bn.bias"]
            model_state_dict["13.branch7x7dbl_1.bn.running_mean"]  = state_dict["Mixed_6d.branch7x7dbl_1.bn.running_mean"]
            model_state_dict["13.branch7x7dbl_1.bn.running_var"]   = state_dict["Mixed_6d.branch7x7dbl_1.bn.running_var"]
            model_state_dict["13.branch7x7dbl_1.bn.weight"]        = state_dict["Mixed_6d.branch7x7dbl_1.bn.weight"]
            model_state_dict["13.branch7x7dbl_1.conv.weight"]      = state_dict["Mixed_6d.branch7x7dbl_1.conv.weight"]
            model_state_dict["13.branch7x7dbl_2.bn.bias"]          = state_dict["Mixed_6d.branch7x7dbl_2.bn.bias"]
            model_state_dict["13.branch7x7dbl_2.bn.running_mean"]  = state_dict["Mixed_6d.branch7x7dbl_2.bn.running_mean"]
            model_state_dict["13.branch7x7dbl_2.bn.running_var"]   = state_dict["Mixed_6d.branch7x7dbl_2.bn.running_var"]
            model_state_dict["13.branch7x7dbl_2.bn.weight"]        = state_dict["Mixed_6d.branch7x7dbl_2.bn.weight"]
            model_state_dict["13.branch7x7dbl_2.conv.weight"]      = state_dict["Mixed_6d.branch7x7dbl_2.conv.weight"]
            model_state_dict["13.branch7x7dbl_3.bn.bias"]          = state_dict["Mixed_6d.branch7x7dbl_3.bn.bias"]
            model_state_dict["13.branch7x7dbl_3.bn.running_mean"]  = state_dict["Mixed_6d.branch7x7dbl_3.bn.running_mean"]
            model_state_dict["13.branch7x7dbl_3.bn.running_var"]   = state_dict["Mixed_6d.branch7x7dbl_3.bn.running_var"]
            model_state_dict["13.branch7x7dbl_3.bn.weight"]        = state_dict["Mixed_6d.branch7x7dbl_3.bn.weight"]
            model_state_dict["13.branch7x7dbl_3.conv.weight"]      = state_dict["Mixed_6d.branch7x7dbl_3.conv.weight"]
            model_state_dict["13.branch7x7dbl_4.bn.bias"]          = state_dict["Mixed_6d.branch7x7dbl_4.bn.bias"]
            model_state_dict["13.branch7x7dbl_4.bn.running_mean"]  = state_dict["Mixed_6d.branch7x7dbl_4.bn.running_mean"]
            model_state_dict["13.branch7x7dbl_4.bn.running_var"]   = state_dict["Mixed_6d.branch7x7dbl_4.bn.running_var"]
            model_state_dict["13.branch7x7dbl_4.bn.weight"]        = state_dict["Mixed_6d.branch7x7dbl_4.bn.weight"]
            model_state_dict["13.branch7x7dbl_4.conv.weight"]      = state_dict["Mixed_6d.branch7x7dbl_4.conv.weight"]
            model_state_dict["13.branch7x7dbl_5.bn.bias"]          = state_dict["Mixed_6d.branch7x7dbl_5.bn.bias"]
            model_state_dict["13.branch7x7dbl_5.bn.running_mean"]  = state_dict["Mixed_6d.branch7x7dbl_5.bn.running_mean"]
            model_state_dict["13.branch7x7dbl_5.bn.running_var"]   = state_dict["Mixed_6d.branch7x7dbl_5.bn.running_var"]
            model_state_dict["13.branch7x7dbl_5.bn.weight"]        = state_dict["Mixed_6d.branch7x7dbl_5.bn.weight"]
            model_state_dict["13.branch7x7dbl_5.conv.weight"]      = state_dict["Mixed_6d.branch7x7dbl_5.conv.weight"]
            model_state_dict["13.branch_pool.bn.bias"]             = state_dict["Mixed_6d.branch_pool.bn.bias"]
            model_state_dict["13.branch_pool.bn.running_mean"]     = state_dict["Mixed_6d.branch_pool.bn.running_mean"]
            model_state_dict["13.branch_pool.bn.running_var"]      = state_dict["Mixed_6d.branch_pool.bn.running_var"]
            model_state_dict["13.branch_pool.bn.weight"]           = state_dict["Mixed_6d.branch_pool.bn.weight"]
            model_state_dict["13.branch_pool.conv.weight"]         = state_dict["Mixed_6d.branch_pool.conv.weight"]
            model_state_dict["14.branch1x1.bn.bias"]               = state_dict["Mixed_6e.branch1x1.bn.bias"]
            model_state_dict["14.branch1x1.bn.running_mean"]       = state_dict["Mixed_6e.branch1x1.bn.running_mean"]
            model_state_dict["14.branch1x1.bn.running_var"]        = state_dict["Mixed_6e.branch1x1.bn.running_var"]
            model_state_dict["14.branch1x1.bn.weight"]             = state_dict["Mixed_6e.branch1x1.bn.weight"]
            model_state_dict["14.branch1x1.conv.weight"]           = state_dict["Mixed_6e.branch1x1.conv.weight"]
            model_state_dict["14.branch7x7_1.bn.bias"]             = state_dict["Mixed_6e.branch7x7_1.bn.bias"]
            model_state_dict["14.branch7x7_1.bn.running_mean"]     = state_dict["Mixed_6e.branch7x7_1.bn.running_mean"]
            model_state_dict["14.branch7x7_1.bn.running_var"]      = state_dict["Mixed_6e.branch7x7_1.bn.running_var"]
            model_state_dict["14.branch7x7_1.bn.weight"]           = state_dict["Mixed_6e.branch7x7_1.bn.weight"]
            model_state_dict["14.branch7x7_1.conv.weight"]         = state_dict["Mixed_6e.branch7x7_1.conv.weight"]
            model_state_dict["14.branch7x7_2.bn.bias"]             = state_dict["Mixed_6e.branch7x7_2.bn.bias"]
            model_state_dict["14.branch7x7_2.bn.running_mean"]     = state_dict["Mixed_6e.branch7x7_2.bn.running_mean"]
            model_state_dict["14.branch7x7_2.bn.running_var"]      = state_dict["Mixed_6e.branch7x7_2.bn.running_var"]
            model_state_dict["14.branch7x7_2.bn.weight"]           = state_dict["Mixed_6e.branch7x7_2.bn.weight"]
            model_state_dict["14.branch7x7_2.conv.weight"]         = state_dict["Mixed_6e.branch7x7_2.conv.weight"]
            model_state_dict["14.branch7x7_3.bn.bias"]             = state_dict["Mixed_6e.branch7x7_3.bn.bias"]
            model_state_dict["14.branch7x7_3.bn.running_mean"]     = state_dict["Mixed_6e.branch7x7_3.bn.running_mean"]
            model_state_dict["14.branch7x7_3.bn.running_var"]      = state_dict["Mixed_6e.branch7x7_3.bn.running_var"]
            model_state_dict["14.branch7x7_3.bn.weight"]           = state_dict["Mixed_6e.branch7x7_3.bn.weight"]
            model_state_dict["14.branch7x7_3.conv.weight"]         = state_dict["Mixed_6e.branch7x7_3.conv.weight"]
            model_state_dict["14.branch7x7dbl_1.bn.bias"]          = state_dict["Mixed_6e.branch7x7dbl_1.bn.bias"]
            model_state_dict["14.branch7x7dbl_1.bn.running_mean"]  = state_dict["Mixed_6e.branch7x7dbl_1.bn.running_mean"]
            model_state_dict["14.branch7x7dbl_1.bn.running_var"]   = state_dict["Mixed_6e.branch7x7dbl_1.bn.running_var"]
            model_state_dict["14.branch7x7dbl_1.bn.weight"]        = state_dict["Mixed_6e.branch7x7dbl_1.bn.weight"]
            model_state_dict["14.branch7x7dbl_1.conv.weight"]      = state_dict["Mixed_6e.branch7x7dbl_1.conv.weight"]
            model_state_dict["14.branch7x7dbl_2.bn.bias"]          = state_dict["Mixed_6e.branch7x7dbl_2.bn.bias"]
            model_state_dict["14.branch7x7dbl_2.bn.running_mean"]  = state_dict["Mixed_6e.branch7x7dbl_2.bn.running_mean"]
            model_state_dict["14.branch7x7dbl_2.bn.running_var"]   = state_dict["Mixed_6e.branch7x7dbl_2.bn.running_var"]
            model_state_dict["14.branch7x7dbl_2.bn.weight"]        = state_dict["Mixed_6e.branch7x7dbl_2.bn.weight"]
            model_state_dict["14.branch7x7dbl_2.conv.weight"]      = state_dict["Mixed_6e.branch7x7dbl_2.conv.weight"]
            model_state_dict["14.branch7x7dbl_3.bn.bias"]          = state_dict["Mixed_6e.branch7x7dbl_3.bn.bias"]
            model_state_dict["14.branch7x7dbl_3.bn.running_mean"]  = state_dict["Mixed_6e.branch7x7dbl_3.bn.running_mean"]
            model_state_dict["14.branch7x7dbl_3.bn.running_var"]   = state_dict["Mixed_6e.branch7x7dbl_3.bn.running_var"]
            model_state_dict["14.branch7x7dbl_3.bn.weight"]        = state_dict["Mixed_6e.branch7x7dbl_3.bn.weight"]
            model_state_dict["14.branch7x7dbl_3.conv.weight"]      = state_dict["Mixed_6e.branch7x7dbl_3.conv.weight"]
            model_state_dict["14.branch7x7dbl_4.bn.bias"]          = state_dict["Mixed_6e.branch7x7dbl_4.bn.bias"]
            model_state_dict["14.branch7x7dbl_4.bn.running_mean"]  = state_dict["Mixed_6e.branch7x7dbl_4.bn.running_mean"]
            model_state_dict["14.branch7x7dbl_4.bn.running_var"]   = state_dict["Mixed_6e.branch7x7dbl_4.bn.running_var"]
            model_state_dict["14.branch7x7dbl_4.bn.weight"]        = state_dict["Mixed_6e.branch7x7dbl_4.bn.weight"]
            model_state_dict["14.branch7x7dbl_4.conv.weight"]      = state_dict["Mixed_6e.branch7x7dbl_4.conv.weight"]
            model_state_dict["14.branch7x7dbl_5.bn.bias"]          = state_dict["Mixed_6e.branch7x7dbl_5.bn.bias"]
            model_state_dict["14.branch7x7dbl_5.bn.running_mean"]  = state_dict["Mixed_6e.branch7x7dbl_5.bn.running_mean"]
            model_state_dict["14.branch7x7dbl_5.bn.running_var"]   = state_dict["Mixed_6e.branch7x7dbl_5.bn.running_var"]
            model_state_dict["14.branch7x7dbl_5.bn.weight"]        = state_dict["Mixed_6e.branch7x7dbl_5.bn.weight"]
            model_state_dict["14.branch7x7dbl_5.conv.weight"]      = state_dict["Mixed_6e.branch7x7dbl_5.conv.weight"]
            model_state_dict["14.branch_pool.bn.bias"]             = state_dict["Mixed_6e.branch_pool.bn.bias"]
            model_state_dict["14.branch_pool.bn.running_mean"]     = state_dict["Mixed_6e.branch_pool.bn.running_mean"]
            model_state_dict["14.branch_pool.bn.running_var"]      = state_dict["Mixed_6e.branch_pool.bn.running_var"]
            model_state_dict["14.branch_pool.bn.weight"]           = state_dict["Mixed_6e.branch_pool.bn.weight"]
            model_state_dict["14.branch_pool.conv.weight"]         = state_dict["Mixed_6e.branch_pool.conv.weight"]
            model_state_dict["15.conv0.bn.bias"]                   = state_dict["AuxLogits.conv0.bn.bias"]
            model_state_dict["15.conv0.bn.running_mean"]           = state_dict["AuxLogits.conv0.bn.running_mean"]
            model_state_dict["15.conv0.bn.running_var"]            = state_dict["AuxLogits.conv0.bn.running_var"]
            model_state_dict["15.conv0.bn.weight"]                 = state_dict["AuxLogits.conv0.bn.weight"]
            model_state_dict["15.conv0.conv.weight"]               = state_dict["AuxLogits.conv0.conv.weight"]
            model_state_dict["15.conv1.bn.bias"]                   = state_dict["AuxLogits.conv1.bn.bias"]
            model_state_dict["15.conv1.bn.running_mean"]           = state_dict["AuxLogits.conv1.bn.running_mean"]
            model_state_dict["15.conv1.bn.running_var"]            = state_dict["AuxLogits.conv1.bn.running_var"]
            model_state_dict["15.conv1.bn.weight"]                 = state_dict["AuxLogits.conv1.bn.weight"]
            model_state_dict["15.conv1.conv.weight"]               = state_dict["AuxLogits.conv1.conv.weight"]
            model_state_dict["15.fc.bias"]                         = state_dict["AuxLogits.fc.bias"]
            model_state_dict["15.fc.weight"]                       = state_dict["AuxLogits.fc.weight"]
            model_state_dict["16.branch3x3_1.bn.bias"]             = state_dict["Mixed_7a.branch3x3_1.bn.bias"]
            model_state_dict["16.branch3x3_1.bn.running_mean"]     = state_dict["Mixed_7a.branch3x3_1.bn.running_mean"]
            model_state_dict["16.branch3x3_1.bn.running_var"]      = state_dict["Mixed_7a.branch3x3_1.bn.running_var"]
            model_state_dict["16.branch3x3_1.bn.weight"]           = state_dict["Mixed_7a.branch3x3_1.bn.weight"]
            model_state_dict["16.branch3x3_1.conv.weight"]         = state_dict["Mixed_7a.branch3x3_1.conv.weight"]
            model_state_dict["16.branch3x3_2.bn.bias"]             = state_dict["Mixed_7a.branch3x3_2.bn.bias"]
            model_state_dict["16.branch3x3_2.bn.running_mean"]     = state_dict["Mixed_7a.branch3x3_2.bn.running_mean"]
            model_state_dict["16.branch3x3_2.bn.running_var"]      = state_dict["Mixed_7a.branch3x3_2.bn.running_var"]
            model_state_dict["16.branch3x3_2.bn.weight"]           = state_dict["Mixed_7a.branch3x3_2.bn.weight"]
            model_state_dict["16.branch3x3_2.conv.weight"]         = state_dict["Mixed_7a.branch3x3_2.conv.weight"]
            model_state_dict["16.branch7x7x3_1.bn.bias"]           = state_dict["Mixed_7a.branch7x7x3_1.bn.bias"]
            model_state_dict["16.branch7x7x3_1.bn.running_mean"]   = state_dict["Mixed_7a.branch7x7x3_1.bn.running_mean"]
            model_state_dict["16.branch7x7x3_1.bn.running_var"]    = state_dict["Mixed_7a.branch7x7x3_1.bn.running_var"]
            model_state_dict["16.branch7x7x3_1.bn.weight"]         = state_dict["Mixed_7a.branch7x7x3_1.bn.weight"]
            model_state_dict["16.branch7x7x3_1.conv.weight"]       = state_dict["Mixed_7a.branch7x7x3_1.conv.weight"]
            model_state_dict["16.branch7x7x3_2.bn.bias"]           = state_dict["Mixed_7a.branch7x7x3_2.bn.bias"]
            model_state_dict["16.branch7x7x3_2.bn.running_mean"]   = state_dict["Mixed_7a.branch7x7x3_2.bn.running_mean"]
            model_state_dict["16.branch7x7x3_2.bn.running_var"]    = state_dict["Mixed_7a.branch7x7x3_2.bn.running_var"]
            model_state_dict["16.branch7x7x3_2.bn.weight"]         = state_dict["Mixed_7a.branch7x7x3_2.bn.weight"]
            model_state_dict["16.branch7x7x3_2.conv.weight"]       = state_dict["Mixed_7a.branch7x7x3_2.conv.weight"]
            model_state_dict["16.branch7x7x3_3.bn.bias"]           = state_dict["Mixed_7a.branch7x7x3_3.bn.bias"]
            model_state_dict["16.branch7x7x3_3.bn.running_mean"]   = state_dict["Mixed_7a.branch7x7x3_3.bn.running_mean"]
            model_state_dict["16.branch7x7x3_3.bn.running_var"]    = state_dict["Mixed_7a.branch7x7x3_3.bn.running_var"]
            model_state_dict["16.branch7x7x3_3.bn.weight"]         = state_dict["Mixed_7a.branch7x7x3_3.bn.weight"]
            model_state_dict["16.branch7x7x3_3.conv.weight"]       = state_dict["Mixed_7a.branch7x7x3_3.conv.weight"]
            model_state_dict["16.branch7x7x3_4.bn.bias"]           = state_dict["Mixed_7a.branch7x7x3_4.bn.bias"]
            model_state_dict["16.branch7x7x3_4.bn.running_mean"]   = state_dict["Mixed_7a.branch7x7x3_4.bn.running_mean"]
            model_state_dict["16.branch7x7x3_4.bn.running_var"]    = state_dict["Mixed_7a.branch7x7x3_4.bn.running_var"]
            model_state_dict["16.branch7x7x3_4.bn.weight"]         = state_dict["Mixed_7a.branch7x7x3_4.bn.weight"]
            model_state_dict["16.branch7x7x3_4.conv.weight"]       = state_dict["Mixed_7a.branch7x7x3_4.conv.weight"]
            model_state_dict["17.branch1x1.bn.bias"]               = state_dict["Mixed_7b.branch1x1.bn.bias"]
            model_state_dict["17.branch1x1.bn.running_mean"]       = state_dict["Mixed_7b.branch1x1.bn.running_mean"]
            model_state_dict["17.branch1x1.bn.running_var"]        = state_dict["Mixed_7b.branch1x1.bn.running_var"]
            model_state_dict["17.branch1x1.bn.weight"]             = state_dict["Mixed_7b.branch1x1.bn.weight"]
            model_state_dict["17.branch1x1.conv.weight"]           = state_dict["Mixed_7b.branch1x1.conv.weight"]
            model_state_dict["17.branch3x3_1.bn.bias"]             = state_dict["Mixed_7b.branch3x3_1.bn.bias"]
            model_state_dict["17.branch3x3_1.bn.running_mean"]     = state_dict["Mixed_7b.branch3x3_1.bn.running_mean"]
            model_state_dict["17.branch3x3_1.bn.running_var"]      = state_dict["Mixed_7b.branch3x3_1.bn.running_var"]
            model_state_dict["17.branch3x3_1.bn.weight"]           = state_dict["Mixed_7b.branch3x3_1.bn.weight"]
            model_state_dict["17.branch3x3_1.conv.weight"]         = state_dict["Mixed_7b.branch3x3_1.conv.weight"]
            model_state_dict["17.branch3x3_2a.bn.bias"]            = state_dict["Mixed_7b.branch3x3_2a.bn.bias"]
            model_state_dict["17.branch3x3_2a.bn.running_mean"]    = state_dict["Mixed_7b.branch3x3_2a.bn.running_mean"]
            model_state_dict["17.branch3x3_2a.bn.running_var"]     = state_dict["Mixed_7b.branch3x3_2a.bn.running_var"]
            model_state_dict["17.branch3x3_2a.bn.weight"]          = state_dict["Mixed_7b.branch3x3_2a.bn.weight"]
            model_state_dict["17.branch3x3_2a.conv.weight"]        = state_dict["Mixed_7b.branch3x3_2a.conv.weight"]
            model_state_dict["17.branch3x3_2b.bn.bias"]            = state_dict["Mixed_7b.branch3x3_2b.bn.bias"]
            model_state_dict["17.branch3x3_2b.bn.running_mean"]    = state_dict["Mixed_7b.branch3x3_2b.bn.running_mean"]
            model_state_dict["17.branch3x3_2b.bn.running_var"]     = state_dict["Mixed_7b.branch3x3_2b.bn.running_var"]
            model_state_dict["17.branch3x3_2b.bn.weight"]          = state_dict["Mixed_7b.branch3x3_2b.bn.weight"]
            model_state_dict["17.branch3x3_2b.conv.weight"]        = state_dict["Mixed_7b.branch3x3_2b.conv.weight"]
            model_state_dict["17.branch3x3dbl_1.bn.bias"]          = state_dict["Mixed_7b.branch3x3dbl_1.bn.bias"]
            model_state_dict["17.branch3x3dbl_1.bn.running_mean"]  = state_dict["Mixed_7b.branch3x3dbl_1.bn.running_mean"]
            model_state_dict["17.branch3x3dbl_1.bn.running_var"]   = state_dict["Mixed_7b.branch3x3dbl_1.bn.running_var"]
            model_state_dict["17.branch3x3dbl_1.bn.weight"]        = state_dict["Mixed_7b.branch3x3dbl_1.bn.weight"]
            model_state_dict["17.branch3x3dbl_1.conv.weight"]      = state_dict["Mixed_7b.branch3x3dbl_1.conv.weight"]
            model_state_dict["17.branch3x3dbl_2.bn.bias"]          = state_dict["Mixed_7b.branch3x3dbl_2.bn.bias"]
            model_state_dict["17.branch3x3dbl_2.bn.running_mean"]  = state_dict["Mixed_7b.branch3x3dbl_2.bn.running_mean"]
            model_state_dict["17.branch3x3dbl_2.bn.running_var"]   = state_dict["Mixed_7b.branch3x3dbl_2.bn.running_var"]
            model_state_dict["17.branch3x3dbl_2.bn.weight"]        = state_dict["Mixed_7b.branch3x3dbl_2.bn.weight"]
            model_state_dict["17.branch3x3dbl_2.conv.weight"]      = state_dict["Mixed_7b.branch3x3dbl_2.conv.weight"]
            model_state_dict["17.branch3x3dbl_3a.bn.bias"]         = state_dict["Mixed_7b.branch3x3dbl_3a.bn.bias"]
            model_state_dict["17.branch3x3dbl_3a.bn.running_mean"] = state_dict["Mixed_7b.branch3x3dbl_3a.bn.running_mean"]
            model_state_dict["17.branch3x3dbl_3a.bn.running_var"]  = state_dict["Mixed_7b.branch3x3dbl_3a.bn.running_var"]
            model_state_dict["17.branch3x3dbl_3a.bn.weight"]       = state_dict["Mixed_7b.branch3x3dbl_3a.bn.weight"]
            model_state_dict["17.branch3x3dbl_3a.conv.weight"]     = state_dict["Mixed_7b.branch3x3dbl_3a.conv.weight"]
            model_state_dict["17.branch3x3dbl_3b.bn.bias"]         = state_dict["Mixed_7b.branch3x3dbl_3b.bn.bias"]
            model_state_dict["17.branch3x3dbl_3b.bn.running_mean"] = state_dict["Mixed_7b.branch3x3dbl_3b.bn.running_mean"]
            model_state_dict["17.branch3x3dbl_3b.bn.running_var"]  = state_dict["Mixed_7b.branch3x3dbl_3b.bn.running_var"]
            model_state_dict["17.branch3x3dbl_3b.bn.weight"]       = state_dict["Mixed_7b.branch3x3dbl_3b.bn.weight"]
            model_state_dict["17.branch3x3dbl_3b.conv.weight"]     = state_dict["Mixed_7b.branch3x3dbl_3b.conv.weight"]
            model_state_dict["17.branch_pool.bn.bias"]             = state_dict["Mixed_7b.branch_pool.bn.bias"]
            model_state_dict["17.branch_pool.bn.running_mean"]     = state_dict["Mixed_7b.branch_pool.bn.running_mean"]
            model_state_dict["17.branch_pool.bn.running_var"]      = state_dict["Mixed_7b.branch_pool.bn.running_var"]
            model_state_dict["17.branch_pool.bn.weight"]           = state_dict["Mixed_7b.branch_pool.bn.weight"]
            model_state_dict["17.branch_pool.conv.weight"]         = state_dict["Mixed_7b.branch_pool.conv.weight"]
            model_state_dict["18.branch1x1.bn.bias"]               = state_dict["Mixed_7c.branch1x1.bn.bias"]
            model_state_dict["18.branch1x1.bn.running_mean"]       = state_dict["Mixed_7c.branch1x1.bn.running_mean"]
            model_state_dict["18.branch1x1.bn.running_var"]        = state_dict["Mixed_7c.branch1x1.bn.running_var"]
            model_state_dict["18.branch1x1.bn.weight"]             = state_dict["Mixed_7c.branch1x1.bn.weight"]
            model_state_dict["18.branch1x1.conv.weight"]           = state_dict["Mixed_7c.branch1x1.conv.weight"]
            model_state_dict["18.branch3x3_1.bn.bias"]             = state_dict["Mixed_7c.branch3x3_1.bn.bias"]
            model_state_dict["18.branch3x3_1.bn.running_mean"]     = state_dict["Mixed_7c.branch3x3_1.bn.running_mean"]
            model_state_dict["18.branch3x3_1.bn.running_var"]      = state_dict["Mixed_7c.branch3x3_1.bn.running_var"]
            model_state_dict["18.branch3x3_1.bn.weight"]           = state_dict["Mixed_7c.branch3x3_1.bn.weight"]
            model_state_dict["18.branch3x3_1.conv.weight"]         = state_dict["Mixed_7c.branch3x3_1.conv.weight"]
            model_state_dict["18.branch3x3_2a.bn.bias"]            = state_dict["Mixed_7c.branch3x3_2a.bn.bias"]
            model_state_dict["18.branch3x3_2a.bn.running_mean"]    = state_dict["Mixed_7c.branch3x3_2a.bn.running_mean"]
            model_state_dict["18.branch3x3_2a.bn.running_var"]     = state_dict["Mixed_7c.branch3x3_2a.bn.running_var"]
            model_state_dict["18.branch3x3_2a.bn.weight"]          = state_dict["Mixed_7c.branch3x3_2a.bn.weight"]
            model_state_dict["18.branch3x3_2a.conv.weight"]        = state_dict["Mixed_7c.branch3x3_2a.conv.weight"]
            model_state_dict["18.branch3x3_2b.bn.bias"]            = state_dict["Mixed_7c.branch3x3_2b.bn.bias"]
            model_state_dict["18.branch3x3_2b.bn.running_mean"]    = state_dict["Mixed_7c.branch3x3_2b.bn.running_mean"]
            model_state_dict["18.branch3x3_2b.bn.running_var"]     = state_dict["Mixed_7c.branch3x3_2b.bn.running_var"]
            model_state_dict["18.branch3x3_2b.bn.weight"]          = state_dict["Mixed_7c.branch3x3_2b.bn.weight"]
            model_state_dict["18.branch3x3_2b.conv.weight"]        = state_dict["Mixed_7c.branch3x3_2b.conv.weight"]
            model_state_dict["18.branch3x3dbl_1.bn.bias"]          = state_dict["Mixed_7c.branch3x3dbl_1.bn.bias"]
            model_state_dict["18.branch3x3dbl_1.bn.running_mean"]  = state_dict["Mixed_7c.branch3x3dbl_1.bn.running_mean"]
            model_state_dict["18.branch3x3dbl_1.bn.running_var"]   = state_dict["Mixed_7c.branch3x3dbl_1.bn.running_var"]
            model_state_dict["18.branch3x3dbl_1.bn.weight"]        = state_dict["Mixed_7c.branch3x3dbl_1.bn.weight"]
            model_state_dict["18.branch3x3dbl_1.conv.weight"]      = state_dict["Mixed_7c.branch3x3dbl_1.conv.weight"]
            model_state_dict["18.branch3x3dbl_2.bn.bias"]          = state_dict["Mixed_7c.branch3x3dbl_2.bn.bias"]
            model_state_dict["18.branch3x3dbl_2.bn.running_mean"]  = state_dict["Mixed_7c.branch3x3dbl_2.bn.running_mean"]
            model_state_dict["18.branch3x3dbl_2.bn.running_var"]   = state_dict["Mixed_7c.branch3x3dbl_2.bn.running_var"]
            model_state_dict["18.branch3x3dbl_2.bn.weight"]        = state_dict["Mixed_7c.branch3x3dbl_2.bn.weight"]
            model_state_dict["18.branch3x3dbl_2.conv.weight"]      = state_dict["Mixed_7c.branch3x3dbl_2.conv.weight"]
            model_state_dict["18.branch3x3dbl_3a.bn.bias"]         = state_dict["Mixed_7c.branch3x3dbl_3a.bn.bias"]
            model_state_dict["18.branch3x3dbl_3a.bn.running_mean"] = state_dict["Mixed_7c.branch3x3dbl_3a.bn.running_mean"]
            model_state_dict["18.branch3x3dbl_3a.bn.running_var"]  = state_dict["Mixed_7c.branch3x3dbl_3a.bn.running_var"]
            model_state_dict["18.branch3x3dbl_3a.bn.weight"]       = state_dict["Mixed_7c.branch3x3dbl_3a.bn.weight"]
            model_state_dict["18.branch3x3dbl_3a.conv.weight"]     = state_dict["Mixed_7c.branch3x3dbl_3a.conv.weight"]
            model_state_dict["18.branch3x3dbl_3b.bn.bias"]         = state_dict["Mixed_7c.branch3x3dbl_3b.bn.bias"]
            model_state_dict["18.branch3x3dbl_3b.bn.running_mean"] = state_dict["Mixed_7c.branch3x3dbl_3b.bn.running_mean"]
            model_state_dict["18.branch3x3dbl_3b.bn.running_var"]  = state_dict["Mixed_7c.branch3x3dbl_3b.bn.running_var"]
            model_state_dict["18.branch3x3dbl_3b.bn.weight"]       = state_dict["Mixed_7c.branch3x3dbl_3b.bn.weight"]
            model_state_dict["18.branch3x3dbl_3b.conv.weight"]     = state_dict["Mixed_7c.branch3x3dbl_3b.conv.weight"]
            model_state_dict["18.branch_pool.bn.bias"]             = state_dict["Mixed_7c.branch_pool.bn.bias"]
            model_state_dict["18.branch_pool.bn.running_mean"]     = state_dict["Mixed_7c.branch_pool.bn.running_mean"]
            model_state_dict["18.branch_pool.bn.running_var"]      = state_dict["Mixed_7c.branch_pool.bn.running_var"]
            model_state_dict["18.branch_pool.bn.weight"]           = state_dict["Mixed_7c.branch_pool.bn.weight"]
            model_state_dict["18.branch_pool.conv.weight"]         = state_dict["Mixed_7c.branch_pool.conv.weight"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["19.fc.bias"]   = state_dict["fc.bias"]
                model_state_dict["19.fc.weight"] = state_dict["fc.weight"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()
