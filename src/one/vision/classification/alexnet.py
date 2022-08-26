#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from torchvision.models import vgg19
from torchvision.models import VGG19_Weights

from one.nn import *

CURRENT_DIR = Path(__file__).resolve().parent
CFG_DIR     = CURRENT_DIR / "cfg"


# H1: - Model ------------------------------------------------------------------

cfgs = {
    "alexnet": {
        "channels": 3,
        "backbone": [
            # [from, number, module,            args(out_channels, ...)]
            [-1,     1,      Conv2d,            [64, 11, 4, 2]],       # 0
            [-1,     1,      ReLU,              [True]],               # 1
            [-1,     1,      MaxPool2d,         [3, 2]],               # 2
            [-1,     1,      Conv2d,            [192, 5, 1, 2]],       # 3
            [-1,     1,      ReLU,              [True]],               # 4
            [-1,     1,      MaxPool2d,         [3, 2]],               # 5
            [-1,     1,      Conv2d,            [384, 3, 1, 1]],       # 6
            [-1,     1,      ReLU,              [True]],               # 7
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 8
            [-1,     1,      ReLU,              [True]],               # 9
            [-1,     1,      Conv2d,            [256, 3, 1, 1]],       # 10
            [-1,     1,      ReLU,              [True]],               # 11
            [-1,     1,      MaxPool2d,         [3, 2]],               # 12
            [-1,     1,      AdaptiveAvgPool2d, [6]],                  # 13
        ],
        "head": [
            [-1,     1,      AlexNetClassifier, []],                   # 14
        ]
    },
}


@MODELS.register(name="alexnet")
class AlexNet(ImageClassificationModel):
    
    model_zoo = {
        "imagenet": dict(
            name        = "imagenet",
            path        = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
            filename    = "alexnet-owt-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "alexnet",
        fullname   : str  | None         = "alexnet",
        cfg        : dict | Path_ | None = "alexnet.yaml",
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
        cfg = cfg or "alexnet"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg
        
        super().__init__(
            root        = root,
            name        = name,
            fullname    = fullname,
            cfg         = cfg,
            channels    = channels,
            num_classes = num_classes,
            pretrained  = AlexNet.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def init_weights(self, m: Module):
        pass
    
    def load_pretrained(self):
        """
        Load pretrained weights. It only loads the intersection layers of
        matching keys and shapes between current model and pretrained.
        """
        if is_dict(self.pretrained) and self.pretrained["name"] == "imagenet":
            state_dict = load_state_dict_from_path(
                model_dir=self.pretrained_dir, **self.pretrained
            )
            # print(self.model.state_dict().keys())
            # print(state_dict.keys())
            model_state_dict = self.model.state_dict()
            model_state_dict["0.weight"]          = state_dict["features.0.weight"]
            model_state_dict["0.bias"]            = state_dict["features.0.bias"]
            model_state_dict["3.weight"]          = state_dict["features.3.weight"]
            model_state_dict["3.bias"]            = state_dict["features.3.bias"]
            model_state_dict["6.weight"]          = state_dict["features.6.weight"]
            model_state_dict["6.bias"]            = state_dict["features.6.bias"]
            model_state_dict["8.weight"]          = state_dict["features.8.weight"]
            model_state_dict["8.bias"]            = state_dict["features.8.bias"]
            model_state_dict["10.weight"]         = state_dict["features.10.weight"]
            model_state_dict["10.bias"]           = state_dict["features.10.bias"]
            model_state_dict["14.linear1.weight"] = state_dict["classifier.1.weight"]
            model_state_dict["14.linear1.bias"]   = state_dict["classifier.1.bias"]
            model_state_dict["14.linear2.weight"] = state_dict["classifier.4.weight"]
            model_state_dict["14.linear2.bias"]   = state_dict["classifier.4.bias"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["14.linear3.weight"] = state_dict["classifier.6.weight"]
                model_state_dict["14.linear3.bias"]   = state_dict["classifier.6.bias"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()
