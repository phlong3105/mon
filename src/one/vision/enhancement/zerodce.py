#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from one.nn import *

CURRENT_DIR = Path(__file__).resolve().parent
CFG_DIR     = CURRENT_DIR / "cfg"


# H1: - Loss -------------------------------------------------------------------

class CombinedLoss(BaseLoss):
    """
    Loss = loss_spa + loss_exp + loss_col + loss_tv.
    """
    
    def __init__(
        self,
        spa_weight    : Floats = 1.0,
        exp_patch_size: int    = 16,
        exp_mean_val  : float  = 0.6,
        exp_weight    : Floats = 10.0,
        col_weight    : Floats = 5.0,
        tv_weight     : Floats = 200.0,
        reduction     : str    = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.name     = "combined_loss"
        self.loss_spa = SpatialConsistencyLoss(
            weight    = spa_weight,
            reduction = reduction,
        )
        self.loss_exp = ExposureControlLoss(
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
            weight     = exp_weight,
            reduction  = reduction,
        )
        self.loss_col = ColorConstancyLoss(
            weight    = col_weight,
            reduction = reduction,
        )
        self.loss_tv  = IlluminationSmoothnessLoss(
            weight    = tv_weight,
            reduction = reduction,
        )
     
    def forward(self, input: Tensors, target: Sequence[Tensor], **_) -> Tensor:
        if isinstance(target, Sequence):
            a       = target[0]
            enhance = target[-1]
            # print(enhance)
        else:
            raise TypeError()
        loss_spa = self.loss_spa(input=enhance, target=input)
        loss_exp = self.loss_exp(input=enhance)
        loss_col = self.loss_col(input=enhance)
        loss_tv  = self.loss_tv(input=a)
        # print(float(loss_spa), float(loss_exp), float(loss_col), float(loss_tv))
        return loss_spa + loss_exp + loss_col + loss_tv


# H1: - Model ------------------------------------------------------------------

cfgs = {
    "zerodce": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],                    # 0  (x)
            [-1,      1,      Conv2d,     [32, 3, 1, 1]],         # 1
            [-1,      1,      ReLU,       [True]],                # 2  (x1)
            [-1,      1,      Conv2d,     [32, 3, 1, 1]],         # 3
            [-1,      1,      ReLU,       [True]],                # 4  (x2)
            [-1,      1,      Conv2d,     [32, 3, 1, 1]],         # 5
            [-1,      1,      ReLU,       [True]],                # 6  (x3)
            [-1,      1,      Conv2d,     [32, 3, 1, 1]],         # 7
            [-1,      1,      ReLU,       [True]],                # 8  (x4)
            [[6, 8],  1,      Concat,     []],                    # 9
            [-1,      1,      Conv2d,     [32, 3, 1, 1]],         # 10
            [-1,      1,      ReLU,       [True]],                # 11 (x5)
            [[4, 11], 1,      Concat,     []],                    # 12
            [-1,      1,      Conv2d,     [32, 3, 1, 1]],         # 13
            [-1,      1,      ReLU,       [True]],                # 14 (x6)
            [[2, 14], 1,      Concat,     []],                    # 15
            [-1,      1,      Conv2d,     [24, 3, 1, 1]],         # 16 (x_r)
            [-1,      1,      Tanh,       []],                    # 17
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],  # 18
        ]
    },
}


@MODELS.register(name="zerodce")
class ZeroDCE(ImageEnhancementModel):
    """
    
    References:
        https://github.com/Li-Chongyi/Zero-DCE
        
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
            filename    = "zerodce-sice.pth",
            num_classes = None,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "zerodce.yaml",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "zerodce",
        fullname   : str  | None         = "zerodce",
        channels   : int                 = 3,
        num_classes: int  | None 		 = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = CombinedLoss(tv_weight=200.0),
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg = cfg or "zerodce"
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
            pretrained  = ZeroDCE.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss or CombinedLoss(),
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
            else:
                m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    
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
            # print(self.model.state_dict().keys())
            # print(state_dict.keys())
            model_state_dict = self.model.state_dict()
            model_state_dict["1.weight"]  = state_dict["e_conv1.weight"]
            model_state_dict["1.bias"]    = state_dict["e_conv1.bias"]
            model_state_dict["3.weight"]  = state_dict["e_conv2.weight"]
            model_state_dict["3.bias"]    = state_dict["e_conv2.bias"]
            model_state_dict["5.weight"]  = state_dict["e_conv3.weight"]
            model_state_dict["5.bias"]    = state_dict["e_conv3.bias"]
            model_state_dict["7.weight"]  = state_dict["e_conv4.weight"]
            model_state_dict["7.bias"]    = state_dict["e_conv4.bias"]
            model_state_dict["10.weight"] = state_dict["e_conv5.weight"]
            model_state_dict["10.bias"]   = state_dict["e_conv5.bias"]
            model_state_dict["13.weight"] = state_dict["e_conv6.weight"]
            model_state_dict["13.bias"]   = state_dict["e_conv6.bias"]
            model_state_dict["16.weight"] = state_dict["e_conv7.weight"]
            model_state_dict["16.bias"]   = state_dict["e_conv7.bias"]
            self.model.load_state_dict(model_state_dict)
            assert_same_state_dicts(self.model.state_dict(), state_dict)
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
