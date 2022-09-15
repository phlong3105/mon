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
    "googlenet": {
        "init_weights": False,
        "channels": 3,
        "backbone": [
            # [from,       number, module,               args(out_channels, ...)]
            [-1,           1,      InceptionBasicConv2d, [64,  7, 2, 3]],                      # 0
            [-1,           1,      MaxPool2d,            [3, 2, 0, 1, False, True]],           # 1
            [-1,           1,      InceptionBasicConv2d, [64, 1]],                             # 2
            [-1,           1,      InceptionBasicConv2d, [192, 3, 1, 1]],                      # 3
            [-1,           1,      MaxPool2d,            [3, 2, 0, 1, False, True]],           # 4
            [-1,           1,      Inception,            [192, 64,  96,  128, 16, 32,  32]],   # 5
            [-1,           1,      Inception,            [256, 128, 128, 192, 32, 96,  64]],   # 6
            [-1,           1,      MaxPool2d,            [3, 2, 0, 1, False, True]],           # 7
            [-1,           1,      Inception,            [480, 192, 96,  208, 16, 48,  64]],   # 8
            [-1,           1,      Inception,            [512, 160, 112, 224, 24, 64,  64]],   # 9
            [-1,           1,      Inception,            [512, 128, 128, 256, 24, 64,  64]],   # 10
            [-1,           1,      Inception,            [512, 112, 144, 288, 32, 64,  64]],   # 11
            [-1,           1,      Inception,            [528, 256, 160, 320, 32, 128, 128]],  # 12
            [-1,           1,      MaxPool2d,            [2, 2, 0, 1, False, True]],           # 13
            [-1,           1,      Inception,            [832, 256, 160, 320, 32, 128, 128]],  # 14
            [-1,           1,      Inception,            [832, 384, 192, 384, 48, 128, 128]],  # 15
            [ 8,           1,      InceptionAux2,        [512, 0.7]],                          # 16
            [11,           1,      InceptionAux2,        [528, 0.7]],                          # 17
        ],
        "head": [
            [-1,           1,      GoogleNetClassifier,  [1024, 0.2]],                         # 18
            [[16, 17, -1], 1,      Join,                 []],                                  # 19
        ]
    },
}


@MODELS.register(name="googlenet")
class GoogleNet(ImageClassificationModel):
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
            path        = "https://download.pytorch.org/models/googlenet-1378be20.pth",
            filename    = "googlenet-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "googlenet.yaml",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "googlenet",
        fullname   : str  | None         = "googlenet",
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
        cfg = cfg or "googlenet"
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
            pretrained  = GoogleNet.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def init_weights(self, m: Module):
        classname    = m.__class__.__name__
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
            model_state_dict["0.bn.bias"]                           = state_dict["conv1.bn.bias"]
            model_state_dict["0.bn.num_batches_tracked"]            = state_dict["conv1.bn.num_batches_tracked"]
            model_state_dict["0.bn.running_mean"]                   = state_dict["conv1.bn.running_mean"]
            model_state_dict["0.bn.running_var"]                    = state_dict["conv1.bn.running_var"]
            model_state_dict["0.bn.weight"]                         = state_dict["conv1.bn.weight"]
            model_state_dict["0.conv.weight"]                       = state_dict["conv1.conv.weight"]
            model_state_dict["2.bn.bias"]                           = state_dict["conv2.bn.bias"]
            model_state_dict["2.bn.num_batches_tracked"]            = state_dict["conv2.bn.num_batches_tracked"]
            model_state_dict["2.bn.running_mean"]                   = state_dict["conv2.bn.running_mean"]
            model_state_dict["2.bn.running_var"]                    = state_dict["conv2.bn.running_var"]
            model_state_dict["2.bn.weight"]                         = state_dict["conv2.bn.weight"]
            model_state_dict["2.conv.weight"]                       = state_dict["conv2.conv.weight"]
            model_state_dict["3.bn.bias"]                           = state_dict["conv3.bn.bias"]
            model_state_dict["3.bn.num_batches_tracked"]            = state_dict["conv3.bn.num_batches_tracked"]
            model_state_dict["3.bn.running_mean"]                   = state_dict["conv3.bn.running_mean"]
            model_state_dict["3.bn.running_var"]                    = state_dict["conv3.bn.running_var"]
            model_state_dict["3.bn.weight"]                         = state_dict["conv3.bn.weight"]
            model_state_dict["3.conv.weight"]                       = state_dict["conv3.conv.weight"]
            model_state_dict["5.branch1.bn.bias"]                   = state_dict["inception3a.branch1.bn.bias"]
            model_state_dict["5.branch1.bn.num_batches_tracked"]    = state_dict["inception3a.branch1.bn.num_batches_tracked"]
            model_state_dict["5.branch1.bn.running_mean"]           = state_dict["inception3a.branch1.bn.running_mean"]
            model_state_dict["5.branch1.bn.running_var"]            = state_dict["inception3a.branch1.bn.running_var"]
            model_state_dict["5.branch1.bn.weight"]                 = state_dict["inception3a.branch1.bn.weight"]
            model_state_dict["5.branch1.conv.weight"]               = state_dict["inception3a.branch1.conv.weight"]
            model_state_dict["5.branch2.0.bn.bias"]                 = state_dict["inception3a.branch2.0.bn.bias"]
            model_state_dict["5.branch2.0.bn.num_batches_tracked"]  = state_dict["inception3a.branch2.0.bn.num_batches_tracked"]
            model_state_dict["5.branch2.0.bn.running_mean"]         = state_dict["inception3a.branch2.0.bn.running_mean"]
            model_state_dict["5.branch2.0.bn.running_var"]          = state_dict["inception3a.branch2.0.bn.running_var"]
            model_state_dict["5.branch2.0.bn.weight"]               = state_dict["inception3a.branch2.0.bn.weight"]
            model_state_dict["5.branch2.0.conv.weight"]             = state_dict["inception3a.branch2.0.conv.weight"]
            model_state_dict["5.branch2.1.bn.bias"]                 = state_dict["inception3a.branch2.1.bn.bias"]
            model_state_dict["5.branch2.1.bn.num_batches_tracked"]  = state_dict["inception3a.branch2.1.bn.num_batches_tracked"]
            model_state_dict["5.branch2.1.bn.running_mean"]         = state_dict["inception3a.branch2.1.bn.running_mean"]
            model_state_dict["5.branch2.1.bn.running_var"]          = state_dict["inception3a.branch2.1.bn.running_var"]
            model_state_dict["5.branch2.1.bn.weight"]               = state_dict["inception3a.branch2.1.bn.weight"]
            model_state_dict["5.branch2.1.conv.weight"]             = state_dict["inception3a.branch2.1.conv.weight"]
            model_state_dict["5.branch3.0.bn.bias"]                 = state_dict["inception3a.branch3.0.bn.bias"]
            model_state_dict["5.branch3.0.bn.num_batches_tracked"]  = state_dict["inception3a.branch3.0.bn.num_batches_tracked"]
            model_state_dict["5.branch3.0.bn.running_mean"]         = state_dict["inception3a.branch3.0.bn.running_mean"]
            model_state_dict["5.branch3.0.bn.running_var"]          = state_dict["inception3a.branch3.0.bn.running_var"]
            model_state_dict["5.branch3.0.bn.weight"]               = state_dict["inception3a.branch3.0.bn.weight"]
            model_state_dict["5.branch3.0.conv.weight"]             = state_dict["inception3a.branch3.0.conv.weight"]
            model_state_dict["5.branch3.1.bn.bias"]                 = state_dict["inception3a.branch3.1.bn.bias"]
            model_state_dict["5.branch3.1.bn.num_batches_tracked"]  = state_dict["inception3a.branch3.1.bn.num_batches_tracked"]
            model_state_dict["5.branch3.1.bn.running_mean"]         = state_dict["inception3a.branch3.1.bn.running_mean"]
            model_state_dict["5.branch3.1.bn.running_var"]          = state_dict["inception3a.branch3.1.bn.running_var"]
            model_state_dict["5.branch3.1.bn.weight"]               = state_dict["inception3a.branch3.1.bn.weight"]
            model_state_dict["5.branch3.1.conv.weight"]             = state_dict["inception3a.branch3.1.conv.weight"]
            model_state_dict["5.branch4.1.bn.bias"]                 = state_dict["inception3a.branch4.1.bn.bias"]
            model_state_dict["5.branch4.1.bn.num_batches_tracked"]  = state_dict["inception3a.branch4.1.bn.num_batches_tracked"]
            model_state_dict["5.branch4.1.bn.running_mean"]         = state_dict["inception3a.branch4.1.bn.running_mean"]
            model_state_dict["5.branch4.1.bn.running_var"]          = state_dict["inception3a.branch4.1.bn.running_var"]
            model_state_dict["5.branch4.1.bn.weight"]               = state_dict["inception3a.branch4.1.bn.weight"]
            model_state_dict["5.branch4.1.conv.weight"]             = state_dict["inception3a.branch4.1.conv.weight"]
            model_state_dict["6.branch1.bn.bias"]                   = state_dict["inception3b.branch1.bn.bias"]
            model_state_dict["6.branch1.bn.num_batches_tracked"]    = state_dict["inception3b.branch1.bn.num_batches_tracked"]
            model_state_dict["6.branch1.bn.running_mean"]           = state_dict["inception3b.branch1.bn.running_mean"]
            model_state_dict["6.branch1.bn.running_var"]            = state_dict["inception3b.branch1.bn.running_var"]
            model_state_dict["6.branch1.bn.weight"]                 = state_dict["inception3b.branch1.bn.weight"]
            model_state_dict["6.branch1.conv.weight"]               = state_dict["inception3b.branch1.conv.weight"]
            model_state_dict["6.branch2.0.bn.bias"]                 = state_dict["inception3b.branch2.0.bn.bias"]
            model_state_dict["6.branch2.0.bn.num_batches_tracked"]  = state_dict["inception3b.branch2.0.bn.num_batches_tracked"]
            model_state_dict["6.branch2.0.bn.running_mean"]         = state_dict["inception3b.branch2.0.bn.running_mean"]
            model_state_dict["6.branch2.0.bn.running_var"]          = state_dict["inception3b.branch2.0.bn.running_var"]
            model_state_dict["6.branch2.0.bn.weight"]               = state_dict["inception3b.branch2.0.bn.weight"]
            model_state_dict["6.branch2.0.conv.weight"]             = state_dict["inception3b.branch2.0.conv.weight"]
            model_state_dict["6.branch2.1.bn.bias"]                 = state_dict["inception3b.branch2.1.bn.bias"]
            model_state_dict["6.branch2.1.bn.num_batches_tracked"]  = state_dict["inception3b.branch2.1.bn.num_batches_tracked"]
            model_state_dict["6.branch2.1.bn.running_mean"]         = state_dict["inception3b.branch2.1.bn.running_mean"]
            model_state_dict["6.branch2.1.bn.running_var"]          = state_dict["inception3b.branch2.1.bn.running_var"]
            model_state_dict["6.branch2.1.bn.weight"]               = state_dict["inception3b.branch2.1.bn.weight"]
            model_state_dict["6.branch2.1.conv.weight"]             = state_dict["inception3b.branch2.1.conv.weight"]
            model_state_dict["6.branch3.0.bn.bias"]                 = state_dict["inception3b.branch3.0.bn.bias"]
            model_state_dict["6.branch3.0.bn.num_batches_tracked"]  = state_dict["inception3b.branch3.0.bn.num_batches_tracked"]
            model_state_dict["6.branch3.0.bn.running_mean"]         = state_dict["inception3b.branch3.0.bn.running_mean"]
            model_state_dict["6.branch3.0.bn.running_var"]          = state_dict["inception3b.branch3.0.bn.running_var"]
            model_state_dict["6.branch3.0.bn.weight"]               = state_dict["inception3b.branch3.0.bn.weight"]
            model_state_dict["6.branch3.0.conv.weight"]             = state_dict["inception3b.branch3.0.conv.weight"]
            model_state_dict["6.branch3.1.bn.bias"]                 = state_dict["inception3b.branch3.1.bn.bias"]
            model_state_dict["6.branch3.1.bn.num_batches_tracked"]  = state_dict["inception3b.branch3.1.bn.num_batches_tracked"]
            model_state_dict["6.branch3.1.bn.running_mean"]         = state_dict["inception3b.branch3.1.bn.running_mean"]
            model_state_dict["6.branch3.1.bn.running_var"]          = state_dict["inception3b.branch3.1.bn.running_var"]
            model_state_dict["6.branch3.1.bn.weight"]               = state_dict["inception3b.branch3.1.bn.weight"]
            model_state_dict["6.branch3.1.conv.weight"]             = state_dict["inception3b.branch3.1.conv.weight"]
            model_state_dict["6.branch4.1.bn.bias"]                 = state_dict["inception3b.branch4.1.bn.bias"]
            model_state_dict["6.branch4.1.bn.num_batches_tracked"]  = state_dict["inception3b.branch4.1.bn.num_batches_tracked"]
            model_state_dict["6.branch4.1.bn.running_mean"]         = state_dict["inception3b.branch4.1.bn.running_mean"]
            model_state_dict["6.branch4.1.bn.running_var"]          = state_dict["inception3b.branch4.1.bn.running_var"]
            model_state_dict["6.branch4.1.bn.weight"]               = state_dict["inception3b.branch4.1.bn.weight"]
            model_state_dict["6.branch4.1.conv.weight"]             = state_dict["inception3b.branch4.1.conv.weight"]
            model_state_dict["8.branch1.bn.bias"]                   = state_dict["inception4a.branch1.bn.bias"]
            model_state_dict["8.branch1.bn.num_batches_tracked"]    = state_dict["inception4a.branch1.bn.num_batches_tracked"]
            model_state_dict["8.branch1.bn.running_mean"]           = state_dict["inception4a.branch1.bn.running_mean"]
            model_state_dict["8.branch1.bn.running_var"]            = state_dict["inception4a.branch1.bn.running_var"]
            model_state_dict["8.branch1.bn.weight"]                 = state_dict["inception4a.branch1.bn.weight"]
            model_state_dict["8.branch1.conv.weight"]               = state_dict["inception4a.branch1.conv.weight"]
            model_state_dict["8.branch2.0.bn.bias"]                 = state_dict["inception4a.branch2.0.bn.bias"]
            model_state_dict["8.branch2.0.bn.num_batches_tracked"]  = state_dict["inception4a.branch2.0.bn.num_batches_tracked"]
            model_state_dict["8.branch2.0.bn.running_mean"]         = state_dict["inception4a.branch2.0.bn.running_mean"]
            model_state_dict["8.branch2.0.bn.running_var"]          = state_dict["inception4a.branch2.0.bn.running_var"]
            model_state_dict["8.branch2.0.bn.weight"]               = state_dict["inception4a.branch2.0.bn.weight"]
            model_state_dict["8.branch2.0.conv.weight"]             = state_dict["inception4a.branch2.0.conv.weight"]
            model_state_dict["8.branch2.1.bn.bias"]                 = state_dict["inception4a.branch2.1.bn.bias"]
            model_state_dict["8.branch2.1.bn.num_batches_tracked"]  = state_dict["inception4a.branch2.1.bn.num_batches_tracked"]
            model_state_dict["8.branch2.1.bn.running_mean"]         = state_dict["inception4a.branch2.1.bn.running_mean"]
            model_state_dict["8.branch2.1.bn.running_var"]          = state_dict["inception4a.branch2.1.bn.running_var"]
            model_state_dict["8.branch2.1.bn.weight"]               = state_dict["inception4a.branch2.1.bn.weight"]
            model_state_dict["8.branch2.1.conv.weight"]             = state_dict["inception4a.branch2.1.conv.weight"]
            model_state_dict["8.branch3.0.bn.bias"]                 = state_dict["inception4a.branch3.0.bn.bias"]
            model_state_dict["8.branch3.0.bn.num_batches_tracked"]  = state_dict["inception4a.branch3.0.bn.num_batches_tracked"]
            model_state_dict["8.branch3.0.bn.running_mean"]         = state_dict["inception4a.branch3.0.bn.running_mean"]
            model_state_dict["8.branch3.0.bn.running_var"]          = state_dict["inception4a.branch3.0.bn.running_var"]
            model_state_dict["8.branch3.0.bn.weight"]               = state_dict["inception4a.branch3.0.bn.weight"]
            model_state_dict["8.branch3.0.conv.weight"]             = state_dict["inception4a.branch3.0.conv.weight"]
            model_state_dict["8.branch3.1.bn.bias"]                 = state_dict["inception4a.branch3.1.bn.bias"]
            model_state_dict["8.branch3.1.bn.num_batches_tracked"]  = state_dict["inception4a.branch3.1.bn.num_batches_tracked"]
            model_state_dict["8.branch3.1.bn.running_mean"]         = state_dict["inception4a.branch3.1.bn.running_mean"]
            model_state_dict["8.branch3.1.bn.running_var"]          = state_dict["inception4a.branch3.1.bn.running_var"]
            model_state_dict["8.branch3.1.bn.weight"]               = state_dict["inception4a.branch3.1.bn.weight"]
            model_state_dict["8.branch3.1.conv.weight"]             = state_dict["inception4a.branch3.1.conv.weight"]
            model_state_dict["8.branch4.1.bn.bias"]                 = state_dict["inception4a.branch4.1.bn.bias"]
            model_state_dict["8.branch4.1.bn.num_batches_tracked"]  = state_dict["inception4a.branch4.1.bn.num_batches_tracked"]
            model_state_dict["8.branch4.1.bn.running_mean"]         = state_dict["inception4a.branch4.1.bn.running_mean"]
            model_state_dict["8.branch4.1.bn.running_var"]          = state_dict["inception4a.branch4.1.bn.running_var"]
            model_state_dict["8.branch4.1.bn.weight"]               = state_dict["inception4a.branch4.1.bn.weight"]
            model_state_dict["8.branch4.1.conv.weight"]             = state_dict["inception4a.branch4.1.conv.weight"]
            model_state_dict["9.branch1.bn.bias"]                   = state_dict["inception4b.branch1.bn.bias"]
            model_state_dict["9.branch1.bn.num_batches_tracked"]    = state_dict["inception4b.branch1.bn.num_batches_tracked"]
            model_state_dict["9.branch1.bn.running_mean"]           = state_dict["inception4b.branch1.bn.running_mean"]
            model_state_dict["9.branch1.bn.running_var"]            = state_dict["inception4b.branch1.bn.running_var"]
            model_state_dict["9.branch1.bn.weight"]                 = state_dict["inception4b.branch1.bn.weight"]
            model_state_dict["9.branch1.conv.weight"]               = state_dict["inception4b.branch1.conv.weight"]
            model_state_dict["9.branch2.0.bn.bias"]                 = state_dict["inception4b.branch2.0.bn.bias"]
            model_state_dict["9.branch2.0.bn.num_batches_tracked"]  = state_dict["inception4b.branch2.0.bn.num_batches_tracked"]
            model_state_dict["9.branch2.0.bn.running_mean"]         = state_dict["inception4b.branch2.0.bn.running_mean"]
            model_state_dict["9.branch2.0.bn.running_var"]          = state_dict["inception4b.branch2.0.bn.running_var"]
            model_state_dict["9.branch2.0.bn.weight"]               = state_dict["inception4b.branch2.0.bn.weight"]
            model_state_dict["9.branch2.0.conv.weight"]             = state_dict["inception4b.branch2.0.conv.weight"]
            model_state_dict["9.branch2.1.bn.bias"]                 = state_dict["inception4b.branch2.1.bn.bias"]
            model_state_dict["9.branch2.1.bn.num_batches_tracked"]  = state_dict["inception4b.branch2.1.bn.num_batches_tracked"]
            model_state_dict["9.branch2.1.bn.running_mean"]         = state_dict["inception4b.branch2.1.bn.running_mean"]
            model_state_dict["9.branch2.1.bn.running_var"]          = state_dict["inception4b.branch2.1.bn.running_var"]
            model_state_dict["9.branch2.1.bn.weight"]               = state_dict["inception4b.branch2.1.bn.weight"]
            model_state_dict["9.branch2.1.conv.weight"]             = state_dict["inception4b.branch2.1.conv.weight"]
            model_state_dict["9.branch3.0.bn.bias"]                 = state_dict["inception4b.branch3.0.bn.bias"]
            model_state_dict["9.branch3.0.bn.num_batches_tracked"]  = state_dict["inception4b.branch3.0.bn.num_batches_tracked"]
            model_state_dict["9.branch3.0.bn.running_mean"]         = state_dict["inception4b.branch3.0.bn.running_mean"]
            model_state_dict["9.branch3.0.bn.running_var"]          = state_dict["inception4b.branch3.0.bn.running_var"]
            model_state_dict["9.branch3.0.bn.weight"]               = state_dict["inception4b.branch3.0.bn.weight"]
            model_state_dict["9.branch3.0.conv.weight"]             = state_dict["inception4b.branch3.0.conv.weight"]
            model_state_dict["9.branch3.1.bn.bias"]                 = state_dict["inception4b.branch3.1.bn.bias"]
            model_state_dict["9.branch3.1.bn.num_batches_tracked"]  = state_dict["inception4b.branch3.1.bn.num_batches_tracked"]
            model_state_dict["9.branch3.1.bn.running_mean"]         = state_dict["inception4b.branch3.1.bn.running_mean"]
            model_state_dict["9.branch3.1.bn.running_var"]          = state_dict["inception4b.branch3.1.bn.running_var"]
            model_state_dict["9.branch3.1.bn.weight"]               = state_dict["inception4b.branch3.1.bn.weight"]
            model_state_dict["9.branch3.1.conv.weight"]             = state_dict["inception4b.branch3.1.conv.weight"]
            model_state_dict["9.branch4.1.bn.bias"]                 = state_dict["inception4b.branch4.1.bn.bias"]
            model_state_dict["9.branch4.1.bn.num_batches_tracked"]  = state_dict["inception4b.branch4.1.bn.num_batches_tracked"]
            model_state_dict["9.branch4.1.bn.running_mean"]         = state_dict["inception4b.branch4.1.bn.running_mean"]
            model_state_dict["9.branch4.1.bn.running_var"]          = state_dict["inception4b.branch4.1.bn.running_var"]
            model_state_dict["9.branch4.1.bn.weight"]               = state_dict["inception4b.branch4.1.bn.weight"]
            model_state_dict["9.branch4.1.conv.weight"]             = state_dict["inception4b.branch4.1.conv.weight"]
            model_state_dict["10.branch1.bn.bias"]                  = state_dict["inception4c.branch1.bn.bias"]
            model_state_dict["10.branch1.bn.num_batches_tracked"]   = state_dict["inception4c.branch1.bn.num_batches_tracked"]
            model_state_dict["10.branch1.bn.running_mean"]          = state_dict["inception4c.branch1.bn.running_mean"]
            model_state_dict["10.branch1.bn.running_var"]           = state_dict["inception4c.branch1.bn.running_var"]
            model_state_dict["10.branch1.bn.weight"]                = state_dict["inception4c.branch1.bn.weight"]
            model_state_dict["10.branch1.conv.weight"]              = state_dict["inception4c.branch1.conv.weight"]
            model_state_dict["10.branch2.0.bn.bias"]                = state_dict["inception4c.branch2.0.bn.bias"]
            model_state_dict["10.branch2.0.bn.num_batches_tracked"] = state_dict["inception4c.branch2.0.bn.num_batches_tracked"]
            model_state_dict["10.branch2.0.bn.running_mean"]        = state_dict["inception4c.branch2.0.bn.running_mean"]
            model_state_dict["10.branch2.0.bn.running_var"]         = state_dict["inception4c.branch2.0.bn.running_var"]
            model_state_dict["10.branch2.0.bn.weight"]              = state_dict["inception4c.branch2.0.bn.weight"]
            model_state_dict["10.branch2.0.conv.weight"]            = state_dict["inception4c.branch2.0.conv.weight"]
            model_state_dict["10.branch2.1.bn.bias"]                = state_dict["inception4c.branch2.1.bn.bias"]
            model_state_dict["10.branch2.1.bn.num_batches_tracked"] = state_dict["inception4c.branch2.1.bn.num_batches_tracked"]
            model_state_dict["10.branch2.1.bn.running_mean"]        = state_dict["inception4c.branch2.1.bn.running_mean"]
            model_state_dict["10.branch2.1.bn.running_var"]         = state_dict["inception4c.branch2.1.bn.running_var"]
            model_state_dict["10.branch2.1.bn.weight"]              = state_dict["inception4c.branch2.1.bn.weight"]
            model_state_dict["10.branch2.1.conv.weight"]            = state_dict["inception4c.branch2.1.conv.weight"]
            model_state_dict["10.branch3.0.bn.bias"]                = state_dict["inception4c.branch3.0.bn.bias"]
            model_state_dict["10.branch3.0.bn.num_batches_tracked"] = state_dict["inception4c.branch3.0.bn.num_batches_tracked"]
            model_state_dict["10.branch3.0.bn.running_mean"]        = state_dict["inception4c.branch3.0.bn.running_mean"]
            model_state_dict["10.branch3.0.bn.running_var"]         = state_dict["inception4c.branch3.0.bn.running_var"]
            model_state_dict["10.branch3.0.bn.weight"]              = state_dict["inception4c.branch3.0.bn.weight"]
            model_state_dict["10.branch3.0.conv.weight"]            = state_dict["inception4c.branch3.0.conv.weight"]
            model_state_dict["10.branch3.1.bn.bias"]                = state_dict["inception4c.branch3.1.bn.bias"]
            model_state_dict["10.branch3.1.bn.num_batches_tracked"] = state_dict["inception4c.branch3.1.bn.num_batches_tracked"]
            model_state_dict["10.branch3.1.bn.running_mean"]        = state_dict["inception4c.branch3.1.bn.running_mean"]
            model_state_dict["10.branch3.1.bn.running_var"]         = state_dict["inception4c.branch3.1.bn.running_var"]
            model_state_dict["10.branch3.1.bn.weight"]              = state_dict["inception4c.branch3.1.bn.weight"]
            model_state_dict["10.branch3.1.conv.weight"]            = state_dict["inception4c.branch3.1.conv.weight"]
            model_state_dict["10.branch4.1.bn.bias"]                = state_dict["inception4c.branch4.1.bn.bias"]
            model_state_dict["10.branch4.1.bn.num_batches_tracked"] = state_dict["inception4c.branch4.1.bn.num_batches_tracked"]
            model_state_dict["10.branch4.1.bn.running_mean"]        = state_dict["inception4c.branch4.1.bn.running_mean"]
            model_state_dict["10.branch4.1.bn.running_var"]         = state_dict["inception4c.branch4.1.bn.running_var"]
            model_state_dict["10.branch4.1.bn.weight"]              = state_dict["inception4c.branch4.1.bn.weight"]
            model_state_dict["10.branch4.1.conv.weight"]            = state_dict["inception4c.branch4.1.conv.weight"]
            model_state_dict["11.branch1.bn.bias"]                  = state_dict["inception4d.branch1.bn.bias"]
            model_state_dict["11.branch1.bn.num_batches_tracked"]   = state_dict["inception4d.branch1.bn.num_batches_tracked"]
            model_state_dict["11.branch1.bn.running_mean"]          = state_dict["inception4d.branch1.bn.running_mean"]
            model_state_dict["11.branch1.bn.running_var"]           = state_dict["inception4d.branch1.bn.running_var"]
            model_state_dict["11.branch1.bn.weight"]                = state_dict["inception4d.branch1.bn.weight"]
            model_state_dict["11.branch1.conv.weight"]              = state_dict["inception4d.branch1.conv.weight"]
            model_state_dict["11.branch2.0.bn.bias"]                = state_dict["inception4d.branch2.0.bn.bias"]
            model_state_dict["11.branch2.0.bn.num_batches_tracked"] = state_dict["inception4d.branch2.0.bn.num_batches_tracked"]
            model_state_dict["11.branch2.0.bn.running_mean"]        = state_dict["inception4d.branch2.0.bn.running_mean"]
            model_state_dict["11.branch2.0.bn.running_var"]         = state_dict["inception4d.branch2.0.bn.running_var"]
            model_state_dict["11.branch2.0.bn.weight"]              = state_dict["inception4d.branch2.0.bn.weight"]
            model_state_dict["11.branch2.0.conv.weight"]            = state_dict["inception4d.branch2.0.conv.weight"]
            model_state_dict["11.branch2.1.bn.bias"]                = state_dict["inception4d.branch2.1.bn.bias"]
            model_state_dict["11.branch2.1.bn.num_batches_tracked"] = state_dict["inception4d.branch2.1.bn.num_batches_tracked"]
            model_state_dict["11.branch2.1.bn.running_mean"]        = state_dict["inception4d.branch2.1.bn.running_mean"]
            model_state_dict["11.branch2.1.bn.running_var"]         = state_dict["inception4d.branch2.1.bn.running_var"]
            model_state_dict["11.branch2.1.bn.weight"]              = state_dict["inception4d.branch2.1.bn.weight"]
            model_state_dict["11.branch2.1.conv.weight"]            = state_dict["inception4d.branch2.1.conv.weight"]
            model_state_dict["11.branch3.0.bn.bias"]                = state_dict["inception4d.branch3.0.bn.bias"]
            model_state_dict["11.branch3.0.bn.num_batches_tracked"] = state_dict["inception4d.branch3.0.bn.num_batches_tracked"]
            model_state_dict["11.branch3.0.bn.running_mean"]        = state_dict["inception4d.branch3.0.bn.running_mean"]
            model_state_dict["11.branch3.0.bn.running_var"]         = state_dict["inception4d.branch3.0.bn.running_var"]
            model_state_dict["11.branch3.0.bn.weight"]              = state_dict["inception4d.branch3.0.bn.weight"]
            model_state_dict["11.branch3.0.conv.weight"]            = state_dict["inception4d.branch3.0.conv.weight"]
            model_state_dict["11.branch3.1.bn.bias"]                = state_dict["inception4d.branch3.1.bn.bias"]
            model_state_dict["11.branch3.1.bn.num_batches_tracked"] = state_dict["inception4d.branch3.1.bn.num_batches_tracked"]
            model_state_dict["11.branch3.1.bn.running_mean"]        = state_dict["inception4d.branch3.1.bn.running_mean"]
            model_state_dict["11.branch3.1.bn.running_var"]         = state_dict["inception4d.branch3.1.bn.running_var"]
            model_state_dict["11.branch3.1.bn.weight"]              = state_dict["inception4d.branch3.1.bn.weight"]
            model_state_dict["11.branch3.1.conv.weight"]            = state_dict["inception4d.branch3.1.conv.weight"]
            model_state_dict["11.branch4.1.bn.bias"]                = state_dict["inception4d.branch4.1.bn.bias"]
            model_state_dict["11.branch4.1.bn.num_batches_tracked"] = state_dict["inception4d.branch4.1.bn.num_batches_tracked"]
            model_state_dict["11.branch4.1.bn.running_mean"]        = state_dict["inception4d.branch4.1.bn.running_mean"]
            model_state_dict["11.branch4.1.bn.running_var"]         = state_dict["inception4d.branch4.1.bn.running_var"]
            model_state_dict["11.branch4.1.bn.weight"]              = state_dict["inception4d.branch4.1.bn.weight"]
            model_state_dict["11.branch4.1.conv.weight"]            = state_dict["inception4d.branch4.1.conv.weight"]
            model_state_dict["12.branch1.bn.bias"]                  = state_dict["inception4e.branch1.bn.bias"]
            model_state_dict["12.branch1.bn.num_batches_tracked"]   = state_dict["inception4e.branch1.bn.num_batches_tracked"]
            model_state_dict["12.branch1.bn.running_mean"]          = state_dict["inception4e.branch1.bn.running_mean"]
            model_state_dict["12.branch1.bn.running_var"]           = state_dict["inception4e.branch1.bn.running_var"]
            model_state_dict["12.branch1.bn.weight"]                = state_dict["inception4e.branch1.bn.weight"]
            model_state_dict["12.branch1.conv.weight"]              = state_dict["inception4e.branch1.conv.weight"]
            model_state_dict["12.branch2.0.bn.bias"]                = state_dict["inception4e.branch2.0.bn.bias"]
            model_state_dict["12.branch2.0.bn.num_batches_tracked"] = state_dict["inception4e.branch2.0.bn.num_batches_tracked"]
            model_state_dict["12.branch2.0.bn.running_mean"]        = state_dict["inception4e.branch2.0.bn.running_mean"]
            model_state_dict["12.branch2.0.bn.running_var"]         = state_dict["inception4e.branch2.0.bn.running_var"]
            model_state_dict["12.branch2.0.bn.weight"]              = state_dict["inception4e.branch2.0.bn.weight"]
            model_state_dict["12.branch2.0.conv.weight"]            = state_dict["inception4e.branch2.0.conv.weight"]
            model_state_dict["12.branch2.1.bn.bias"]                = state_dict["inception4e.branch2.1.bn.bias"]
            model_state_dict["12.branch2.1.bn.num_batches_tracked"] = state_dict["inception4e.branch2.1.bn.num_batches_tracked"]
            model_state_dict["12.branch2.1.bn.running_mean"]        = state_dict["inception4e.branch2.1.bn.running_mean"]
            model_state_dict["12.branch2.1.bn.running_var"]         = state_dict["inception4e.branch2.1.bn.running_var"]
            model_state_dict["12.branch2.1.bn.weight"]              = state_dict["inception4e.branch2.1.bn.weight"]
            model_state_dict["12.branch2.1.conv.weight"]            = state_dict["inception4e.branch2.1.conv.weight"]
            model_state_dict["12.branch3.0.bn.bias"]                = state_dict["inception4e.branch3.0.bn.bias"]
            model_state_dict["12.branch3.0.bn.num_batches_tracked"] = state_dict["inception4e.branch3.0.bn.num_batches_tracked"]
            model_state_dict["12.branch3.0.bn.running_mean"]        = state_dict["inception4e.branch3.0.bn.running_mean"]
            model_state_dict["12.branch3.0.bn.running_var"]         = state_dict["inception4e.branch3.0.bn.running_var"]
            model_state_dict["12.branch3.0.bn.weight"]              = state_dict["inception4e.branch3.0.bn.weight"]
            model_state_dict["12.branch3.0.conv.weight"]            = state_dict["inception4e.branch3.0.conv.weight"]
            model_state_dict["12.branch3.1.bn.bias"]                = state_dict["inception4e.branch3.1.bn.bias"]
            model_state_dict["12.branch3.1.bn.num_batches_tracked"] = state_dict["inception4e.branch3.1.bn.num_batches_tracked"]
            model_state_dict["12.branch3.1.bn.running_mean"]        = state_dict["inception4e.branch3.1.bn.running_mean"]
            model_state_dict["12.branch3.1.bn.running_var"]         = state_dict["inception4e.branch3.1.bn.running_var"]
            model_state_dict["12.branch3.1.bn.weight"]              = state_dict["inception4e.branch3.1.bn.weight"]
            model_state_dict["12.branch3.1.conv.weight"]            = state_dict["inception4e.branch3.1.conv.weight"]
            model_state_dict["12.branch4.1.bn.bias"]                = state_dict["inception4e.branch4.1.bn.bias"]
            model_state_dict["12.branch4.1.bn.num_batches_tracked"] = state_dict["inception4e.branch4.1.bn.num_batches_tracked"]
            model_state_dict["12.branch4.1.bn.running_mean"]        = state_dict["inception4e.branch4.1.bn.running_mean"]
            model_state_dict["12.branch4.1.bn.running_var"]         = state_dict["inception4e.branch4.1.bn.running_var"]
            model_state_dict["12.branch4.1.bn.weight"]              = state_dict["inception4e.branch4.1.bn.weight"]
            model_state_dict["12.branch4.1.conv.weight"]            = state_dict["inception4e.branch4.1.conv.weight"]
            model_state_dict["14.branch1.bn.bias"]                  = state_dict["inception5a.branch1.bn.bias"]
            model_state_dict["14.branch1.bn.num_batches_tracked"]   = state_dict["inception5a.branch1.bn.num_batches_tracked"]
            model_state_dict["14.branch1.bn.running_mean"]          = state_dict["inception5a.branch1.bn.running_mean"]
            model_state_dict["14.branch1.bn.running_var"]           = state_dict["inception5a.branch1.bn.running_var"]
            model_state_dict["14.branch1.bn.weight"]                = state_dict["inception5a.branch1.bn.weight"]
            model_state_dict["14.branch1.conv.weight"]              = state_dict["inception5a.branch1.conv.weight"]
            model_state_dict["14.branch2.0.bn.bias"]                = state_dict["inception5a.branch2.0.bn.bias"]
            model_state_dict["14.branch2.0.bn.num_batches_tracked"] = state_dict["inception5a.branch2.0.bn.num_batches_tracked"]
            model_state_dict["14.branch2.0.bn.running_mean"]        = state_dict["inception5a.branch2.0.bn.running_mean"]
            model_state_dict["14.branch2.0.bn.running_var"]         = state_dict["inception5a.branch2.0.bn.running_var"]
            model_state_dict["14.branch2.0.bn.weight"]              = state_dict["inception5a.branch2.0.bn.weight"]
            model_state_dict["14.branch2.0.conv.weight"]            = state_dict["inception5a.branch2.0.conv.weight"]
            model_state_dict["14.branch2.1.bn.bias"]                = state_dict["inception5a.branch2.1.bn.bias"]
            model_state_dict["14.branch2.1.bn.num_batches_tracked"] = state_dict["inception5a.branch2.1.bn.num_batches_tracked"]
            model_state_dict["14.branch2.1.bn.running_mean"]        = state_dict["inception5a.branch2.1.bn.running_mean"]
            model_state_dict["14.branch2.1.bn.running_var"]         = state_dict["inception5a.branch2.1.bn.running_var"]
            model_state_dict["14.branch2.1.bn.weight"]              = state_dict["inception5a.branch2.1.bn.weight"]
            model_state_dict["14.branch2.1.conv.weight"]            = state_dict["inception5a.branch2.1.conv.weight"]
            model_state_dict["14.branch3.0.bn.bias"]                = state_dict["inception5a.branch3.0.bn.bias"]
            model_state_dict["14.branch3.0.bn.num_batches_tracked"] = state_dict["inception5a.branch3.0.bn.num_batches_tracked"]
            model_state_dict["14.branch3.0.bn.running_mean"]        = state_dict["inception5a.branch3.0.bn.running_mean"]
            model_state_dict["14.branch3.0.bn.running_var"]         = state_dict["inception5a.branch3.0.bn.running_var"]
            model_state_dict["14.branch3.0.bn.weight"]              = state_dict["inception5a.branch3.0.bn.weight"]
            model_state_dict["14.branch3.0.conv.weight"]            = state_dict["inception5a.branch3.0.conv.weight"]
            model_state_dict["14.branch3.1.bn.bias"]                = state_dict["inception5a.branch3.1.bn.bias"]
            model_state_dict["14.branch3.1.bn.num_batches_tracked"] = state_dict["inception5a.branch3.1.bn.num_batches_tracked"]
            model_state_dict["14.branch3.1.bn.running_mean"]        = state_dict["inception5a.branch3.1.bn.running_mean"]
            model_state_dict["14.branch3.1.bn.running_var"]         = state_dict["inception5a.branch3.1.bn.running_var"]
            model_state_dict["14.branch3.1.bn.weight"]              = state_dict["inception5a.branch3.1.bn.weight"]
            model_state_dict["14.branch3.1.conv.weight"]            = state_dict["inception5a.branch3.1.conv.weight"]
            model_state_dict["14.branch4.1.bn.bias"]                = state_dict["inception5a.branch4.1.bn.bias"]
            model_state_dict["14.branch4.1.bn.num_batches_tracked"] = state_dict["inception5a.branch4.1.bn.num_batches_tracked"]
            model_state_dict["14.branch4.1.bn.running_mean"]        = state_dict["inception5a.branch4.1.bn.running_mean"]
            model_state_dict["14.branch4.1.bn.running_var"]         = state_dict["inception5a.branch4.1.bn.running_var"]
            model_state_dict["14.branch4.1.bn.weight"]              = state_dict["inception5a.branch4.1.bn.weight"]
            model_state_dict["14.branch4.1.conv.weight"]            = state_dict["inception5a.branch4.1.conv.weight"]
            model_state_dict["15.branch1.bn.bias"]                  = state_dict["inception5b.branch1.bn.bias"]
            model_state_dict["15.branch1.bn.num_batches_tracked"]   = state_dict["inception5b.branch1.bn.num_batches_tracked"]
            model_state_dict["15.branch1.bn.running_mean"]          = state_dict["inception5b.branch1.bn.running_mean"]
            model_state_dict["15.branch1.bn.running_var"]           = state_dict["inception5b.branch1.bn.running_var"]
            model_state_dict["15.branch1.bn.weight"]                = state_dict["inception5b.branch1.bn.weight"]
            model_state_dict["15.branch1.conv.weight"]              = state_dict["inception5b.branch1.conv.weight"]
            model_state_dict["15.branch2.0.bn.bias"]                = state_dict["inception5b.branch2.0.bn.bias"]
            model_state_dict["15.branch2.0.bn.num_batches_tracked"] = state_dict["inception5b.branch2.0.bn.num_batches_tracked"]
            model_state_dict["15.branch2.0.bn.running_mean"]        = state_dict["inception5b.branch2.0.bn.running_mean"]
            model_state_dict["15.branch2.0.bn.running_var"]         = state_dict["inception5b.branch2.0.bn.running_var"]
            model_state_dict["15.branch2.0.bn.weight"]              = state_dict["inception5b.branch2.0.bn.weight"]
            model_state_dict["15.branch2.0.conv.weight"]            = state_dict["inception5b.branch2.0.conv.weight"]
            model_state_dict["15.branch2.1.bn.bias"]                = state_dict["inception5b.branch2.1.bn.bias"]
            model_state_dict["15.branch2.1.bn.num_batches_tracked"] = state_dict["inception5b.branch2.1.bn.num_batches_tracked"]
            model_state_dict["15.branch2.1.bn.running_mean"]        = state_dict["inception5b.branch2.1.bn.running_mean"]
            model_state_dict["15.branch2.1.bn.running_var"]         = state_dict["inception5b.branch2.1.bn.running_var"]
            model_state_dict["15.branch2.1.bn.weight"]              = state_dict["inception5b.branch2.1.bn.weight"]
            model_state_dict["15.branch2.1.conv.weight"]            = state_dict["inception5b.branch2.1.conv.weight"]
            model_state_dict["15.branch3.0.bn.bias"]                = state_dict["inception5b.branch3.0.bn.bias"]
            model_state_dict["15.branch3.0.bn.num_batches_tracked"] = state_dict["inception5b.branch3.0.bn.num_batches_tracked"]
            model_state_dict["15.branch3.0.bn.running_mean"]        = state_dict["inception5b.branch3.0.bn.running_mean"]
            model_state_dict["15.branch3.0.bn.running_var"]         = state_dict["inception5b.branch3.0.bn.running_var"]
            model_state_dict["15.branch3.0.bn.weight"]              = state_dict["inception5b.branch3.0.bn.weight"]
            model_state_dict["15.branch3.0.conv.weight"]            = state_dict["inception5b.branch3.0.conv.weight"]
            model_state_dict["15.branch3.1.bn.bias"]                = state_dict["inception5b.branch3.1.bn.bias"]
            model_state_dict["15.branch3.1.bn.num_batches_tracked"] = state_dict["inception5b.branch3.1.bn.num_batches_tracked"]
            model_state_dict["15.branch3.1.bn.running_mean"]        = state_dict["inception5b.branch3.1.bn.running_mean"]
            model_state_dict["15.branch3.1.bn.running_var"]         = state_dict["inception5b.branch3.1.bn.running_var"]
            model_state_dict["15.branch3.1.bn.weight"]              = state_dict["inception5b.branch3.1.bn.weight"]
            model_state_dict["15.branch3.1.conv.weight"]            = state_dict["inception5b.branch3.1.conv.weight"]
            model_state_dict["15.branch4.1.bn.bias"]                = state_dict["inception5b.branch4.1.bn.bias"]
            model_state_dict["15.branch4.1.bn.num_batches_tracked"] = state_dict["inception5b.branch4.1.bn.num_batches_tracked"]
            model_state_dict["15.branch4.1.bn.running_mean"]        = state_dict["inception5b.branch4.1.bn.running_mean"]
            model_state_dict["15.branch4.1.bn.running_var"]         = state_dict["inception5b.branch4.1.bn.running_var"]
            model_state_dict["15.branch4.1.bn.weight"]              = state_dict["inception5b.branch4.1.bn.weight"]
            model_state_dict["15.branch4.1.conv.weight"]            = state_dict["inception5b.branch4.1.conv.weight"]
            model_state_dict["16.conv.bn.bias"]                     = state_dict["aux1.conv.bn.bias"]
            model_state_dict["16.conv.bn.num_batches_tracked"]      = state_dict["aux1.conv.bn.num_batches_tracked"]
            model_state_dict["16.conv.bn.running_mean"]             = state_dict["aux1.conv.bn.running_mean"]
            model_state_dict["16.conv.bn.running_var"]              = state_dict["aux1.conv.bn.running_var"]
            model_state_dict["16.conv.bn.weight"]                   = state_dict["aux1.conv.bn.weight"]
            model_state_dict["16.conv.conv.weight"]                 = state_dict["aux1.conv.conv.weight"]
            model_state_dict["16.fc1.bias"]                         = state_dict["aux1.fc1.bias"]
            model_state_dict["16.fc1.weight"]                       = state_dict["aux1.fc1.weight"]
            model_state_dict["16.fc2.bias"]                         = state_dict["aux1.fc2.bias"]
            model_state_dict["16.fc2.weight"]                       = state_dict["aux1.fc2.weight"]
            model_state_dict["17.conv.bn.bias"]                     = state_dict["aux2.conv.bn.bias"]
            model_state_dict["17.conv.bn.num_batches_tracked"]      = state_dict["aux2.conv.bn.num_batches_tracked"]
            model_state_dict["17.conv.bn.running_mean"]             = state_dict["aux2.conv.bn.running_mean"]
            model_state_dict["17.conv.bn.running_var"]              = state_dict["aux2.conv.bn.running_var"]
            model_state_dict["17.conv.bn.weight"]                   = state_dict["aux2.conv.bn.weight"]
            model_state_dict["17.conv.conv.weight"]                 = state_dict["aux2.conv.conv.weight"]
            model_state_dict["17.fc1.bias"]                         = state_dict["aux2.fc1.bias"]
            model_state_dict["17.fc1.weight"]                       = state_dict["aux2.fc1.weight"]
            model_state_dict["17.fc2.bias"]                         = state_dict["aux2.fc2.bias"]
            model_state_dict["17.fc2.weight"]                       = state_dict["aux2.fc2.weight"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["18.fc.bias"]   = state_dict["fc.bias"]
                model_state_dict["18.fc.weight"] = state_dict["fc.weight"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()
