#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from one.nn import *
from one.vision.enhancement.zerodce import CombinedLoss

CURRENT_DIR = Path(__file__).resolve().parent
CFG_DIR     = CURRENT_DIR / "cfg"


# H1: - Model ------------------------------------------------------------------

cfgs = {
    "zerodce++": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,                      args(out_channels, ...)]
            [-1,      1,      Identity,                    []],                      # 0  (x)
            [-1,      1,      Downsample,                  [None, 1, "bilinear"]],   # 1  (x_down)
            [-1,      1,      DepthwiseSeparableConv2d,    [32, 3, 1, 1, 1, 1, 0]],  # 2
            [-1,      1,      ReLU,                        [True]],                  # 3  (x1)
            [-1,      1,      DepthwiseSeparableConv2d,    [32, 3, 1, 1, 1, 1, 0]],  # 4
            [-1,      1,      ReLU,                        [True]],                  # 5  (x2)
            [-1,      1,      DepthwiseSeparableConv2d,    [32, 3, 1, 1, 1, 1, 0]],  # 6
            [-1,      1,      ReLU,                        [True]],                  # 7  (x3)
            [-1,      1,      DepthwiseSeparableConv2d,    [32, 3, 1, 1, 1, 1, 0]],  # 8
            [-1,      1,      ReLU,                        [True]],                  # 9  (x4)
            [[7, 9],  1,      Concat,                      []],                      # 10
            [-1,      1,      DepthwiseSeparableConv2d,    [32, 3, 1, 1, 1, 1, 0]],  # 11
            [-1,      1,      ReLU,                        [True]],                  # 12 (x5)
            [[5, 12], 1,      Concat,                      []],                      # 13
            [-1,      1,      DepthwiseSeparableConv2d,    [32, 3, 1, 1, 1, 1, 0]],  # 14
            [-1,      1,      ReLU,                        [True]],                  # 15 (x6)
            [[3, 15], 1,      Concat,                      []],                      # 16
            [-1,      1,      DepthwiseSeparableConv2d,    [3,  3, 1, 1, 1, 1, 0]],  # 17 (x_r)
            [-1,      1,      Tanh,                        []],                      # 18
            [-1,      1,      UpsamplingBilinear2d,        [None, 1]],               # 19
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                     # 20
        ]
    }
}


@MODELS.register(name="zerodce++")
class ZeroDCEPP(ImageEnhancementModel):
    """
    
    References:
        https://github.com/Li-Chongyi/Zero-DCE_extension
    
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
        "sice": dict(
            name        = "sice",
            path        = "",
            filename    = "zerodce++-sice.pth",
            num_classes = None,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "zerodce++.yaml",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "zerodce++",
        fullname   : str  | None         = "zerodce++",
        channels   : int                 = 3,
        num_classes: int  | None 		 = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = CombinedLoss(tv_weight=1600.0),
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg = cfg or "zerodce++"
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
            pretrained  = ZeroDCEPP.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss or CombinedLoss(tv_weight=1600.0),
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
            """
            for k in sorted(self.model.state_dict().keys()):
                print(f"model_state_dict[\"{k}\"] = ")
            for k in sorted(state_dict.keys()):
                print(f"state_dict[\"{k}\"]")
            """
            model_state_dict = self.model.state_dict()
            model_state_dict["2.dw_conv.weight"]  = state_dict["e_conv1.depth_conv.weight"]
            model_state_dict["2.dw_conv.bias"]    = state_dict["e_conv1.depth_conv.bias"]
            model_state_dict["2.pw_conv.weight"]  = state_dict["e_conv1.point_conv.weight"]
            model_state_dict["2.pw_conv.bias"]    = state_dict["e_conv1.point_conv.bias"]
            model_state_dict["4.dw_conv.weight"]  = state_dict["e_conv2.depth_conv.weight"]
            model_state_dict["4.dw_conv.bias"]    = state_dict["e_conv2.depth_conv.bias"]
            model_state_dict["4.pw_conv.weight"]  = state_dict["e_conv2.point_conv.weight"]
            model_state_dict["4.pw_conv.bias"]    = state_dict["e_conv2.point_conv.bias"]
            model_state_dict["6.dw_conv.weight"]  = state_dict["e_conv3.depth_conv.weight"]
            model_state_dict["6.dw_conv.bias"]    = state_dict["e_conv3.depth_conv.bias"]
            model_state_dict["6.pw_conv.weight"]  = state_dict["e_conv3.point_conv.weight"]
            model_state_dict["6.pw_conv.bias"]    = state_dict["e_conv3.point_conv.bias"]
            model_state_dict["8.dw_conv.weight"]  = state_dict["e_conv4.depth_conv.weight"]
            model_state_dict["8.dw_conv.bias"]    = state_dict["e_conv4.depth_conv.bias"]
            model_state_dict["8.pw_conv.weight"]  = state_dict["e_conv4.point_conv.weight"]
            model_state_dict["8.pw_conv.bias"]    = state_dict["e_conv4.point_conv.bias"]
            model_state_dict["11.dw_conv.weight"] = state_dict["e_conv5.depth_conv.weight"]
            model_state_dict["11.dw_conv.bias"]   = state_dict["e_conv5.depth_conv.bias"]
            model_state_dict["11.pw_conv.weight"] = state_dict["e_conv5.point_conv.weight"]
            model_state_dict["11.pw_conv.bias"]   = state_dict["e_conv5.point_conv.bias"]
            model_state_dict["14.dw_conv.weight"] = state_dict["e_conv6.depth_conv.weight"]
            model_state_dict["14.dw_conv.bias"]   = state_dict["e_conv6.depth_conv.bias"]
            model_state_dict["14.pw_conv.weight"] = state_dict["e_conv6.point_conv.weight"]
            model_state_dict["14.pw_conv.bias"]   = state_dict["e_conv6.point_conv.bias"]
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
        pred = self.forward(input=input, *args, **kwargs)
        loss = self.loss(input, pred) if self.loss else None
        return pred[-1], loss
