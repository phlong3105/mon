#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DualCNN model.
"""

from __future__ import annotations

from one.nn import *

CURRENT_DIR = Path(__file__).resolve().parent
CFG_DIR     = CURRENT_DIR / "cfg"


# H1: - Model ------------------------------------------------------------------

cfgs = {
    "dual-cnn": {
        "channels": 3,
        "backbone": [
            # [from,   number, module,   args(out_channels, ...)]
            [-1,       1,      Identity, []],                          # 0  (x)
            # Branch 1: Details (Edge Preserving)
            [0,        1,      Conv2d,   [64, 3, 1, 1, 1, 1, False]],  # 1
            [-1,       1,      ReLU,     [True]],                      # 2  (input)
            [-1,       1,      Conv2d,   [64, 3, 1, 1, 1, 1, False]],  # 3
            [-1,       1,      ReLU,     [True]],                      # 4  (1)
            [-1,       1,      Conv2d,   [64, 3, 1, 1, 1, 1, False]],  # 5
            [-1,       1,      ReLU,     [True]],                      # 6  (2)
            [-1,       1,      Conv2d,   [64, 3, 1, 1, 1, 1, False]],  # 7
            [-1,       1,      ReLU,     [True]],                      # 8  (3)
            [-1,       1,      Conv2d,   [64, 3, 1, 1, 1, 1, False]],  # 9
            [-1,       1,      ReLU,     [True]],                      # 10 (4)
            [-1,       1,      Conv2d,   [64, 3, 1, 1, 1, 1, False]],  # 11
            [-1,       1,      ReLU,     [True]],                      # 12 (5)
            [-1,       1,      Conv2d,   [64, 3, 1, 1, 1, 1, False]],  # 13
            [-1,       1,      ReLU,     [True]],                      # 14 (6)
            [-1,       1,      Conv2d,   [64, 3, 1, 1, 1, 1, False]],  # 15
            [-1,       1,      ReLU,     [True]],                      # 16 (7)
            [-1,       1,      Conv2d,   [64, 3, 1, 1, 1, 1, False]],  # 17
            [-1,       1,      ReLU,     [True]],                      # 18 (8)
            [-1,       1,      Conv2d,   [64, 3, 1, 1, 1, 1, False]],  # 19
            [-1,       1,      ReLU,     [True]],                      # 20 (9)
            [-1,       1,      Conv2d,   [64, 3, 1, 1, 1, 1, False]],  # 21
            [-1,       1,      ReLU,     [True]],                      # 22 (10)
            [-1,       1,      Conv2d,   [64, 3, 1, 1, 1, 1, False]],  # 23
            [-1,       1,      ReLU,     [True]],                      # 24 (11)
            [-1,       1,      Conv2d,   [64, 3, 1, 1, 1, 1, False]],  # 25
            [-1,       1,      ReLU,     [True]],                      # 26 (12)
            [-1,       1,      Conv2d,   [64, 3, 1, 1, 1, 1, False]],  # 27
            [-1,       1,      ReLU,     [True]],                      # 28 (13)
            [-1,       1,      Conv2d,   [64, 3, 1, 1, 1, 1, False]],  # 29
            [-1,       1,      ReLU,     [True]],                      # 30 (14)
            [-1,       1,      Conv2d,   [64, 3, 1, 1, 1, 1, False]],  # 31
            [-1,       1,      ReLU,     [True]],                      # 32 (15)
            [-1,       1,      Conv2d,   [64, 3, 1, 1, 1, 1, False]],  # 33
            [-1,       1,      ReLU,     [True]],                      # 34 (16)
            [-1,       1,      Conv2d,   [64, 3, 1, 1, 1, 1, False]],  # 35
            [-1,       1,      ReLU,     [True]],                      # 36 (17)
            [-1,       1,      Conv2d,   [64, 3, 1, 1, 1, 1, False]],  # 37
            [-1,       1,      ReLU,     [True]],                      # 38 (18)
            [-1,       1,      Conv2d,   [3, 3, 1, 1, 1, 1, False]],   # 39
            # Branch 2: Structures
            [0,        1,      Conv2d,   [64, 9, 1, 0]],               # 40
            [-1,       1,      ReLU,     [True]],                      # 41
            [-1,       1,      Conv2d,   [32, 1, 1, 0]],               # 42
            [-1,       1,      ReLU,     [True]],                      # 43
            [-1,       1,      Conv2d,   [3,  5, 1, 0]],               # 44
        ],
        "head": [
            [39,       1,      CropTBLR, [6, -6, 6, -6]],              # 45
            [[-1, 44], 1,      Sum,      []],                          # 46
        ]
    },
}


@MODELS.register(name="dual-cnn")
class DualCNN(ImageEnhancementModel):
    """
    
    References:
        https://github.com/jspan/dualcnn
        
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
    
    model_zoo = {}
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "dual-cnn",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "dual-cnn",
        fullname   : str  | None         = "dual-cnn",
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
        cfg = cfg or "dual-cnn"
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
            classlabels = classlabels,
            pretrained  = DualCNN.init_pretrained(pretrained),
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
                m.conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "dw_conv"):
                m.dw_conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "pw_conv"):
                m.pw_conv.weight.data.normal_(0.0, 0.02)
            else:
                m.weight.data.normal_(0.0, 0.02)
    
    def load_pretrained(self):
        """
        Load pretrained weights. It only loads the intersection layers of
        matching keys and shapes between current model and pretrained.
        """
        if is_dict(self.pretrained) \
            and self.pretrained["name"] in ["sice"]:
            state_dict = load_state_dict_from_path(
                model_dir=self.pretrained_dir, **self.pretrained
            )
            
            for k in sorted(self.model.state_dict().keys()):
                print(f"model_state_dict[\"{k}\"] = ")
            for k in sorted(state_dict.keys()):
                print(f"state_dict[\"{k}\"]")
            
            model_state_dict = self.model.state_dict()
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()
    
    def forward_loss(
        self,
        input : Tensor,
        target: Tensor,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        """
        Forward pass with loss value. Loss function may require more arguments
        beside the ground-truth and prediction values. For calculating the
        metrics, we only need the final predictions and ground-truth.

        Args:
            input (Tensor): Input of shape [B, C, H, W].
            target (Tensor): Ground-truth of shape [B, C, H, W].
            
        Returns:
            Predictions and loss value.
        """
        pred  = self.forward(input=input, *args, **kwargs)
        loss  = self.loss(input, pred) if self.loss else None
        loss += self.regularization_loss(alpha=0.1)
        return pred[-1], loss
