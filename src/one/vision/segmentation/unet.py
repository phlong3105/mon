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
    "unet-32": {
        "channels": 3,
        "backbone": [
            # [from,   number, module,          args(out_channels, ...)]
            [-1,       1,      UNetBlock,       [32]],          # 0  enc1
            [-1,       1,      MaxPool2d,       [2, 2]],        # 1
            [-1,       1,      UNetBlock,       [64]],          # 2  enc2
            [-1,       1,      MaxPool2d,       [2, 2]],        # 3
            [-1,       1,      UNetBlock,       [128]],         # 4  enc3
            [-1,       1,      MaxPool2d,       [2, 2]],        # 5
            [-1,       1,      UNetBlock,       [256]],         # 6  enc4
            [-1,       1,      MaxPool2d,       [2, 2]],        # 7
            [-1,       1,      UNetBlock,       [512]],         # 8  bottleneck
            [-1,       1,      ConvTranspose2d, [256, 2, 2]],   # 9  dec4
            [[-1, 6],  1,      Concat,          []],            # 10 dec4 = dec4 + enc4
            [-1,       1,      UNetBlock,       [256]],         # 11 dec4
            [-1,       1,      ConvTranspose2d, [128, 2, 2]],   # 12 dec3
            [[-1, 4],  1,      Concat,          []],            # 13 dec3 = dec3 + enc3
            [-1,       1,      UNetBlock,       [128]],         # 14 dec3
            [-1,       1,      ConvTranspose2d, [64, 2, 2]],    # 15 dec2
            [[-1, 2],  1,      Concat,          []],            # 16 dec2 = dec2 + enc2
            [-1,       1,      UNetBlock,       [64]],          # 17 dec2
            [-1,       1,      ConvTranspose2d, [32, 2, 2]],    # 18 dec1
            [[-1, 0],  1,      Concat,          []],            # 19 dec1 = dec1 + enc1
            [-1,       1,      UNetBlock,       [32]],          # 20 dec1
            [-1,       1,      Conv2d,          [1, 1]],        # 21 conv
        ],
        "head": [
            [-1,       1,      Sigmoid,         []],            # 22
        ]
    },
}


@MODELS.register(name="unet")
class UNet(ImageClassificationModel):
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
        "lgg": dict(
            name        = "lgg",
            path        = "https://download.pytorch.org/models/googlenet-1378be20.pth",
            filename    = "unet-32-lgg.pt",
            num_classes = 1000,
        ),
    }
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "unet-32.yaml",
        root       : Path_               = RUNS_DIR,
        name       : str  | None         = "unet",
        fullname   : str  | None         = "unet-32",
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
        cfg = cfg or "unet-32"
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
            pretrained  = UNet.init_pretrained(pretrained),
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
        pass
    
    def load_pretrained(self):
        """
        Load pretrained weights. It only loads the intersection layers of
        matching keys and shapes between current model and pretrained.
        """
        if is_dict(self.pretrained) and self.pretrained["name"] == "lgg":
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
            model_state_dict["0.conv1.weight"]               = state_dict["encoder1.enc1conv1.weight"]
            model_state_dict["0.conv2.weight"]               = state_dict["encoder1.enc1conv2.weight"]
            model_state_dict["0.norm1.bias"]                 = state_dict["encoder1.enc1norm1.bias"]
            model_state_dict["0.norm1.num_batches_tracked"]  = state_dict["encoder1.enc1norm1.num_batches_tracked"]
            model_state_dict["0.norm1.running_mean"]         = state_dict["encoder1.enc1norm1.running_mean"]
            model_state_dict["0.norm1.running_var"]          = state_dict["encoder1.enc1norm1.running_var"]
            model_state_dict["0.norm1.weight"]               = state_dict["encoder1.enc1norm1.weight"]
            model_state_dict["0.norm2.bias"]                 = state_dict["encoder1.enc1norm2.bias"]
            model_state_dict["0.norm2.num_batches_tracked"]  = state_dict["encoder1.enc1norm2.num_batches_tracked"]
            model_state_dict["0.norm2.running_mean"]         = state_dict["encoder1.enc1norm2.running_mean"]
            model_state_dict["0.norm2.running_var"]          = state_dict["encoder1.enc1norm2.running_var"]
            model_state_dict["0.norm2.weight"]               = state_dict["encoder1.enc1norm2.weight"]
            model_state_dict["2.conv1.weight"]               = state_dict["encoder2.enc2conv1.weight"]
            model_state_dict["2.conv2.weight"]               = state_dict["encoder2.enc2conv2.weight"]
            model_state_dict["2.norm1.bias"]                 = state_dict["encoder2.enc2norm1.bias"]
            model_state_dict["2.norm1.num_batches_tracked"]  = state_dict["encoder2.enc2norm1.num_batches_tracked"]
            model_state_dict["2.norm1.running_mean"]         = state_dict["encoder2.enc2norm1.running_mean"]
            model_state_dict["2.norm1.running_var"]          = state_dict["encoder2.enc2norm1.running_var"]
            model_state_dict["2.norm1.weight"]               = state_dict["encoder2.enc2norm1.weight"]
            model_state_dict["2.norm2.bias"]                 = state_dict["encoder2.enc2norm2.bias"]
            model_state_dict["2.norm2.num_batches_tracked"]  = state_dict["encoder2.enc2norm2.num_batches_tracked"]
            model_state_dict["2.norm2.running_mean"]         = state_dict["encoder2.enc2norm2.running_mean"]
            model_state_dict["2.norm2.running_var"]          = state_dict["encoder2.enc2norm2.running_var"]
            model_state_dict["2.norm2.weight"]               = state_dict["encoder2.enc2norm2.weight"]
            model_state_dict["4.conv1.weight"]               = state_dict["encoder3.enc3conv1.weight"]
            model_state_dict["4.conv2.weight"]               = state_dict["encoder3.enc3conv2.weight"]
            model_state_dict["4.norm1.bias"]                 = state_dict["encoder3.enc3norm1.bias"]
            model_state_dict["4.norm1.num_batches_tracked"]  = state_dict["encoder3.enc3norm1.num_batches_tracked"]
            model_state_dict["4.norm1.running_mean"]         = state_dict["encoder3.enc3norm1.running_mean"]
            model_state_dict["4.norm1.running_var"]          = state_dict["encoder3.enc3norm1.running_var"]
            model_state_dict["4.norm1.weight"]               = state_dict["encoder3.enc3norm1.weight"]
            model_state_dict["4.norm2.bias"]                 = state_dict["encoder3.enc3norm2.bias"]
            model_state_dict["4.norm2.num_batches_tracked"]  = state_dict["encoder3.enc3norm2.num_batches_tracked"]
            model_state_dict["4.norm2.running_mean"]         = state_dict["encoder3.enc3norm2.running_mean"]
            model_state_dict["4.norm2.running_var"]          = state_dict["encoder3.enc3norm2.running_var"]
            model_state_dict["4.norm2.weight"]               = state_dict["encoder3.enc3norm2.weight"]
            model_state_dict["6.conv1.weight"]               = state_dict["encoder4.enc4conv1.weight"]
            model_state_dict["6.conv2.weight"]               = state_dict["encoder4.enc4conv2.weight"]
            model_state_dict["6.norm1.bias"]                 = state_dict["encoder4.enc4norm1.bias"]
            model_state_dict["6.norm1.num_batches_tracked"]  = state_dict["encoder4.enc4norm1.num_batches_tracked"]
            model_state_dict["6.norm1.running_mean"]         = state_dict["encoder4.enc4norm1.running_mean"]
            model_state_dict["6.norm1.running_var"]          = state_dict["encoder4.enc4norm1.running_var"]
            model_state_dict["6.norm1.weight"]               = state_dict["encoder4.enc4norm1.weight"]
            model_state_dict["6.norm2.bias"]                 = state_dict["encoder4.enc4norm2.bias"]
            model_state_dict["6.norm2.num_batches_tracked"]  = state_dict["encoder4.enc4norm2.num_batches_tracked"]
            model_state_dict["6.norm2.running_mean"]         = state_dict["encoder4.enc4norm2.running_mean"]
            model_state_dict["6.norm2.running_var"]          = state_dict["encoder4.enc4norm2.running_var"]
            model_state_dict["6.norm2.weight"]               = state_dict["encoder4.enc4norm2.weight"]
            model_state_dict["8.conv1.weight"]               = state_dict["bottleneck.bottleneckconv1.weight"]
            model_state_dict["8.conv2.weight"]               = state_dict["bottleneck.bottleneckconv2.weight"]
            model_state_dict["8.norm1.bias"]                 = state_dict["bottleneck.bottlenecknorm1.bias"]
            model_state_dict["8.norm1.num_batches_tracked"]  = state_dict["bottleneck.bottlenecknorm1.num_batches_tracked"]
            model_state_dict["8.norm1.running_mean"]         = state_dict["bottleneck.bottlenecknorm1.running_mean"]
            model_state_dict["8.norm1.running_var"]          = state_dict["bottleneck.bottlenecknorm1.running_var"]
            model_state_dict["8.norm1.weight"]               = state_dict["bottleneck.bottlenecknorm1.weight"]
            model_state_dict["8.norm2.bias"]                 = state_dict["bottleneck.bottlenecknorm2.bias"]
            model_state_dict["8.norm2.num_batches_tracked"]  = state_dict["bottleneck.bottlenecknorm2.num_batches_tracked"]
            model_state_dict["8.norm2.running_mean"]         = state_dict["bottleneck.bottlenecknorm2.running_mean"]
            model_state_dict["8.norm2.running_var"]          = state_dict["bottleneck.bottlenecknorm2.running_var"]
            model_state_dict["8.norm2.weight"]               = state_dict["bottleneck.bottlenecknorm2.weight"]
            model_state_dict["9.bias"]                       = state_dict["upconv4.bias"]
            model_state_dict["9.weight"]                     = state_dict["upconv4.weight"]
            model_state_dict["11.conv1.weight"]              = state_dict["decoder4.dec4conv1.weight"]
            model_state_dict["11.conv2.weight"]              = state_dict["decoder4.dec4conv2.weight"]
            model_state_dict["11.norm1.bias"]                = state_dict["decoder4.dec4norm1.bias"]
            model_state_dict["11.norm1.num_batches_tracked"] = state_dict["decoder4.dec4norm1.num_batches_tracked"]
            model_state_dict["11.norm1.running_mean"]        = state_dict["decoder4.dec4norm1.running_mean"]
            model_state_dict["11.norm1.running_var"]         = state_dict["decoder4.dec4norm1.running_var"]
            model_state_dict["11.norm1.weight"]              = state_dict["decoder4.dec4norm1.weight"]
            model_state_dict["11.norm2.bias"]                = state_dict["decoder4.dec4norm2.bias"]
            model_state_dict["11.norm2.num_batches_tracked"] = state_dict["decoder4.dec4norm2.num_batches_tracked"]
            model_state_dict["11.norm2.running_mean"]        = state_dict["decoder4.dec4norm2.running_mean"]
            model_state_dict["11.norm2.running_var"]         = state_dict["decoder4.dec4norm2.running_var"]
            model_state_dict["11.norm2.weight"]              = state_dict["decoder4.dec4norm2.weight"]
            model_state_dict["12.bias"]                      = state_dict["upconv3.bias"]
            model_state_dict["12.weight"]                    = state_dict["upconv3.weight"]
            model_state_dict["14.conv1.weight"]              = state_dict["decoder3.dec3conv1.weight"]
            model_state_dict["14.conv2.weight"]              = state_dict["decoder3.dec3conv2.weight"]
            model_state_dict["14.norm1.bias"]                = state_dict["decoder3.dec3norm1.bias"]
            model_state_dict["14.norm1.num_batches_tracked"] = state_dict["decoder3.dec3norm1.num_batches_tracked"]
            model_state_dict["14.norm1.running_mean"]        = state_dict["decoder3.dec3norm1.running_mean"]
            model_state_dict["14.norm1.running_var"]         = state_dict["decoder3.dec3norm1.running_var"]
            model_state_dict["14.norm1.weight"]              = state_dict["decoder3.dec3norm1.weight"]
            model_state_dict["14.norm2.bias"]                = state_dict["decoder3.dec3norm2.bias"]
            model_state_dict["14.norm2.num_batches_tracked"] = state_dict["decoder3.dec3norm2.num_batches_tracked"]
            model_state_dict["14.norm2.running_mean"]        = state_dict["decoder3.dec3norm2.running_mean"]
            model_state_dict["14.norm2.running_var"]         = state_dict["decoder3.dec3norm2.running_var"]
            model_state_dict["14.norm2.weight"]              = state_dict["decoder3.dec3norm2.weight"]
            model_state_dict["15.bias"]                      = state_dict["upconv2.bias"]
            model_state_dict["15.weight"]                    = state_dict["upconv2.weight"]
            model_state_dict["17.conv1.weight"]              = state_dict["decoder2.dec2conv1.weight"]
            model_state_dict["17.conv2.weight"]              = state_dict["decoder2.dec2conv2.weight"]
            model_state_dict["17.norm1.bias"]                = state_dict["decoder2.dec2norm1.bias"]
            model_state_dict["17.norm1.num_batches_tracked"] = state_dict["decoder2.dec2norm1.num_batches_tracked"]
            model_state_dict["17.norm1.running_mean"]        = state_dict["decoder2.dec2norm1.running_mean"]
            model_state_dict["17.norm1.running_var"]         = state_dict["decoder2.dec2norm1.running_var"]
            model_state_dict["17.norm1.weight"]              = state_dict["decoder2.dec2norm1.weight"]
            model_state_dict["17.norm2.bias"]                = state_dict["decoder2.dec2norm2.bias"]
            model_state_dict["17.norm2.num_batches_tracked"] = state_dict["decoder2.dec2norm2.num_batches_tracked"]
            model_state_dict["17.norm2.running_mean"]        = state_dict["decoder2.dec2norm2.running_mean"]
            model_state_dict["17.norm2.running_var"]         = state_dict["decoder2.dec2norm2.running_var"]
            model_state_dict["17.norm2.weight"]              = state_dict["decoder2.dec2norm2.weight"]
            model_state_dict["18.bias"]                      = state_dict["upconv1.bias"]
            model_state_dict["18.weight"]                    = state_dict["upconv1.weight"]
            model_state_dict["20.conv1.weight"]              = state_dict["decoder1.dec1conv1.weight"]
            model_state_dict["20.conv2.weight"]              = state_dict["decoder1.dec1conv2.weight"]
            model_state_dict["20.norm1.bias"]                = state_dict["decoder1.dec1norm1.bias"]
            model_state_dict["20.norm1.num_batches_tracked"] = state_dict["decoder1.dec1norm1.num_batches_tracked"]
            model_state_dict["20.norm1.running_mean"]        = state_dict["decoder1.dec1norm1.running_mean"]
            model_state_dict["20.norm1.running_var"]         = state_dict["decoder1.dec1norm1.running_var"]
            model_state_dict["20.norm1.weight"]              = state_dict["decoder1.dec1norm1.weight"]
            model_state_dict["20.norm2.bias"]                = state_dict["decoder1.dec1norm2.bias"]
            model_state_dict["20.norm2.num_batches_tracked"] = state_dict["decoder1.dec1norm2.num_batches_tracked"]
            model_state_dict["20.norm2.running_mean"]        = state_dict["decoder1.dec1norm2.running_mean"]
            model_state_dict["20.norm2.running_var"]         = state_dict["decoder1.dec1norm2.running_var"]
            model_state_dict["20.norm2.weight"]              = state_dict["decoder1.dec1norm2.weight"]
            if self.pretrained["num_classes"] == self.num_classes:
                model_state_dict["21.bias"]   = state_dict["conv.bias"]
                model_state_dict["21.weight"] = state_dict["conv.weight"]
            self.model.load_state_dict(model_state_dict)
            # assert_same_state_dicts(self.model.state_dict(), state_dict)
        else:
            super().load_pretrained()
