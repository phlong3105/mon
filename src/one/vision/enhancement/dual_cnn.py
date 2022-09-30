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
            # [from,     number, module,   args(out_channels, ...)]
            [-1,         1,      Identity, []],                              # 0  (x)
            # Branch 1: Details (Edge Preserving)                          
            [0,          1,      VDSR,     [3]],                             # 1
            # Branch 2: Structures
            [0,          1,      SRCNN,    [3, 9, 1, 0, 1, 1, 0, 5, 1, 0]],  # 2
        ],
        "head": [
            [1,          1,      CropTBLR, [6, -6, 6, -6]],                  # 3
            [[2, -1],    1,      Sum,      []],                              # 4
            [[1, 2, -1], 1,      Join,     []],                              # 5 (detail, structure, output)
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
    
    model_zoo = {}
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "dual-cnn",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "dual-cnn",
        fullname   : str          | None = "dual-cnn",
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
        cfg = cfg or "dual-cnn"
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
        pass
    
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
        import one.vision.transformation as t
        pred        = self.forward(input=input, *args, **kwargs)  # (detail, structure, output)
        crop_target = t.crop_tblr(target, 6, -6, 6, -6)
        loss = 0.01 * self.loss(pred[1], crop_target) \
               + self.loss(pred[-1], crop_target) if self.loss else None
        return pred[-1], loss

    def on_fit_start(self):
        """
        Called at the very beginning of fit.
        """
        self.optims = Adam(
            params       = [
                {"params": self.model[1].parameters(), "lr": 2e-5},
                {"params": self.model[2].parameters(), "lr": 2e-4}
            ],
            weight_decay = 1e-4,
        )
        super().on_fit_start()
        
    def on_train_epoch_start(self):
        """
        Called in the training loop at the very beginning of the epoch.
        """
        vdsr_lr  = 2e-5 * (0.2 ** (self.current_epoch // 50))
        srcnn_lr = 2e-4 * (0.2 ** (self.current_epoch // 50))
        self.optims.param_groups[0]["lr"] = vdsr_lr
        self.optims.param_groups[1]["lr"] = srcnn_lr
        super().on_train_epoch_start()
