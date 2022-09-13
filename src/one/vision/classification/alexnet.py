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
            path        = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
            filename    = "alexnet-owt-imagenet.pth",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "alexnet.yaml",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "alexnet",
        fullname   : str  | None         = "alexnet",
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
            cfg         = cfg,
            root        = root,
            name        = name,
            fullname    = fullname,
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
            for k, v in state_dict.items():
                k = k.replace("features.", "")
                if k in model_state_dict:
                    model_state_dict[k] = v
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
