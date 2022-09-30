#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from one.nn import *

CURRENT_DIR = Path(__file__).resolve().parent
CFG_DIR     = CURRENT_DIR / "cfg"


# H1: - Loss -------------------------------------------------------------------

def gradient(image: Tensor, direction: str) -> Tensor:
    """
    Calculate the gradient in the image with the desired direction.

    Args:
        image (Tensor): Input image.
        direction (str): Direction to calculate the gradient. Can be ["x", "y"].

    Returns:
        grad (Tensor): Gradient.
    """
    if direction not in ["x", "y"]:
        raise ValueError
    
    smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2))
    smooth_kernel_y = torch.transpose(smooth_kernel_x, 2, 3)
    
    if direction == "x":
        kernel = smooth_kernel_x
    else:
        kernel = smooth_kernel_y
    
    kernel = kernel.cuda()
    grad   = torch.abs(F.conv2d(input=image, weight=kernel, stride=1, padding=1))
    return grad


def avg_gradient(image: Tensor, direction: str) -> Tensor:
    """
    Calculate the average gradient in the image with the desired direction.

    Args:
        image (Tensor): Input image.
        direction (str): Direction to calculate the gradient. Can be ["x", "y"].

    Returns:
        avg_gradient (Tensor): Average gradient.
    """
    return F.avg_pool2d(
        gradient(image=image, direction=direction),
        kernel_size = 3,
        stride      = 1,
        padding     = 1
    )


def smooth(r: Tensor, i: Tensor) -> Tensor:
    """
    Get the smooth reconstructed image from the given illumination map and
    reflectance map.
    
    Args:
        r (Tensor): Reflectance map.
        i (Tensor): Illumination map.
        
    Returns:
        grad (Tensor): Smoothed reconstructed image.
    """
    r    = ((0.299 * r[:, 0, :, :])
            + (0.587 * r[:, 1, :, :])
            + (0.114 * r[:, 2, :, :]))
    r    = torch.unsqueeze(input=r, dim=1)
    grad = gradient(image=i, direction="x") \
           * torch.exp(-10 * avg_gradient(image=r, direction="x")) + \
           gradient(image=i, direction="y") \
           * torch.exp(-10 * avg_gradient(image=r, direction="y"))
    return torch.mean(input=grad)


class DecomLoss(BaseLoss):
    """
    Calculate the decomposition loss according to RetinexNet paper
    (https://arxiv.org/abs/1808.04560).
    """
    
    def __init__(
        self,
        weight   : Floats = 1.0,
        reduction: str    = "mean",
        *args, **kwargs
    ):
        super().__init__(
            weight    = weight,
            reduction = reduction,
            *args, **kwargs
        )
        self.name = "decom_loss"
        
    def forward(self, input : Tensor, target: Tensor, **_) -> Tensor:
        input, r_low, i_low, r_high, i_high = \
            input[0], input[1], input[2], input[3], input[4]
        i_low_3                = torch.cat(tensors=(i_low,  i_low,  i_low),  dim=1)
        i_high_3               = torch.cat(tensors=(i_high, i_high, i_high), dim=1)
        recon_loss_low         = F.l1_loss(r_low  * i_low_3,  input)
        recon_loss_high        = F.l1_loss(r_high * i_high_3, target)
        recon_loss_mutual_low  = F.l1_loss(r_high * i_low_3,  input)
        recon_loss_mutual_high = F.l1_loss(r_low  * i_high_3, target)
        equal_r_loss           = F.l1_loss(r_low, r_high)
        i_smooth_loss_low      = smooth(r_low,  i_low)
        i_smooth_loss_high     = smooth(r_high, i_high)
        loss = (recon_loss_low
                + recon_loss_high
                + 0.001 * recon_loss_mutual_low
                + 0.001 * recon_loss_mutual_high
                + 0.1   * i_smooth_loss_low
                + 0.1   * i_smooth_loss_high
                + 0.01  * equal_r_loss)
        return loss


class EnhanceLoss(BaseLoss):
    """
    Calculate the enhancement loss according to RetinexNet paper
    (https://arxiv.org/abs/1808.04560).
    """
    
    def __init__(
        self,
        weight   : Floats = 1.0,
        reduction: str    = "mean",
        *args, **kwargs
    ):
        super().__init__(
            weight    = weight,
            reduction = reduction,
            *args, **kwargs
        )
        self.name = "enhance_loss"
        
    def forward(self, input : Tensor, target: Tensor, **_) -> Tensor:
        r_low, i_low, i_delta, i_delta_3, enhance = \
            input[0], input[1], input[2], input[3], input[4]
        
        i_delta_3 = (torch.cat(tensors=(i_delta, i_delta, i_delta), dim=1)
                     if i_delta_3 is None else i_delta_3)
        enhance   = (r_low * i_delta_3) if enhance is None else enhance
        
        relight_loss 		= F.l1_loss(enhance, target)
        i_smooth_loss_delta = smooth(i_delta, r_low)
        loss 				= relight_loss + 3 * i_smooth_loss_delta
        return loss


class RetinexLoss(BaseLoss):
    """
    Calculate the retinex loss according to RetinexNet paper
    (https://arxiv.org/abs/1808.04560).
    """
    
    def __init__(
        self,
        weight   : Floats = 1.0,
        reduction: str    = "mean",
        *args, **kwargs
    ):
        super().__init__(
            weight    = weight,
            reduction = reduction,
            *args, **kwargs
        )
        self.name         = "retinex_loss"
        self.decom_loss   = DecomLoss()
        self.enhance_loss = EnhanceLoss()
        
    def forward(self, input : Tensor, target: Tensor, **_) -> Tensor:
        input, r_low, i_low, r_high, i_high, i_delta, i_delta_3, enhance = \
            input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7]
        decom_loss = self.decom_loss(
            input  = (input, r_low, i_low, r_high, i_high),
            target = target,
        )
        enhance_loss = self.enhance_loss(
            input  = (r_low, i_low, i_delta, i_delta_3, enhance),
            target = target,
        )
        loss = decom_loss + enhance_loss
        return loss


# H1: - Model ------------------------------------------------------------------

cfgs = {
    "decomnet": {
        "channels": 3,
        "backbone": [
            # [from,   number, module,          args(out_channels, ...)]
            [-1,       1,      Identity,        []],                                      # 0  (x)
            [-1,       1,      Max,             [1, True]],                               # 1  (x_max)
            [[0, 1],   1,      Concat,          []],                                      # 2  (x_concat)
            [-1,       1,      Conv2d,          [64, 3, 1, 4, 1, 1, True, "replicate"]],  # 3
            [-1,       1,      Conv2d,          [64, 3, 1, 1, 1, 1, True, "replicate"]],  # 4
            [-1,       1,      ReLU,            [True]],                                  # 5
            [-1,       1,      Conv2d,          [64, 3, 1, 1, 1, 1, True, "replicate"]],  # 6
            [-1,       1,      ReLU,            [True]],                                  # 7
            [-1,       1,      Conv2d,          [64, 3, 1, 1, 1, 1, True, "replicate"]],  # 8
            [-1,       1,      ReLU,            [True]],                                  # 9
            [-1,       1,      Conv2d,          [64, 3, 1, 1, 1, 1, True, "replicate"]],  # 10
            [-1,       1,      ReLU,            [True]],                                  # 11
            [-1,       1,      Conv2d,          [64, 3, 1, 1, 1, 1, True, "replicate"]],  # 12
            [-1,       1,      ReLU,            [True]],                                  # 13
            [-1,       1,      Conv2d,          [4,  3, 1, 1, 1, 1, True, "replicate"]],  # 14
            [-1,       1,      BatchNorm2d,     [4]],                                     # 15
        ],                                              
        "head": [                                       
            [-1,       1,      ExtractFeatures, [0, 3]],                                  # 16
            [-2,       1,      ExtractFeatures, [3, 4]],                                  # 17
            [-2,       1,      Sigmoid,         []],                                      # 18  (r)
            [-2,       1,      Sigmoid,         []],                                      # 19  (i)
            [[-2, -1], 1,      Join,            []],                                      # 20  (r, i)
        ]
    },
    "enhancenet": {
        "channels": 4,
        "backbone": [
            # [from,        number, module,            args(out_channels, ...)]
            [-1,            1,      Identity,          []],                                      # 0  (r, i)
            [0,             1,      ExtractItem,       [0]],                                     # 1  (r)
            [0,             1,      ExtractItem,       [1]],                                     # 2  (i)
            [[1, 2],        1,      Concat,            []],                                      # 3  (x)
            [-1,            1,      Conv2d,            [64, 3, 1, 1, 1, 1, True, "replicate"]],  # 4  (conv_0)
            [-1,            1,      Conv2d,            [64, 3, 2, 1, 1, 1, True, "replicate"]],  # 5
            [-1,            1,      ReLU,              [True]],                                  # 6  (conv_1)
            [-1,            1,      Conv2d,            [64, 3, 2, 1, 1, 1, True, "replicate"]],  # 7
            [-1,            1,      ReLU,              [True]],                                  # 8  (conv_2)
            [-1,            1,      Conv2d,            [64, 3, 2, 1, 1, 1, True, "replicate"]],  # 9
            [-1,            1,      ReLU,              [True]],                                  # 10 (conv_3)
            [[-1, 8],       1,      InterpolateConcat, [1]],                                     # 11 (conv_3_up, conv_2)
            [-1,            1,      Conv2d,            [64, 3, 1, 1, 1, 1, True, "replicate"]],  # 12
            [-1,            1,      ReLU,              [True]],                                  # 13 (deconv_1)
            [[-1, 6],       1,      InterpolateConcat, [1]],                                     # 14 (deconv_1_up, conv_1)
            [-1,            1,      Conv2d,            [64, 3, 1, 1, 1, 1, True, "replicate"]],  # 15
            [-1,            1,      ReLU,              [True]],                                  # 16 (deconv_2)
            [[-1, 4],       1,      InterpolateConcat, [1]],                                     # 17 (deconv_2_up, conv_0)
            [-1,            1,      Conv2d,            [64, 3, 1, 1, 1, 1, True, "replicate"]],  # 18
            [-1,            1,      ReLU,              [True]],                                  # 19 (deconv_3)
            [[-1, 13, 16],  1,      InterpolateConcat, [1]],                                     # 20 (deconv_1, deconv_2, deconv_3)
            [-1,            1,      Conv2d,            [64, 3, 1, 1, 1, 1, True, "replicate"]],  # 21
            [-1,            1,      BatchNorm2d,       [64]],                                    # 22
        ],
        "head": [
            [-1,            1,      Conv2d,            [1,  3]],                                 # 23
        ]
    },
}


@MODELS.register(name="decomnet")
class DecomNet(ImageEnhancementModel):
    """
    
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
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "decomnet.yaml",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "decomnet",
        fullname   : str          | None = "decomnet",
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
        cfg = cfg or "decomnet"
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
        pass


@MODELS.register(name="enhancenet")
class EnhanceNet(ImageEnhancementModel):
    """
    
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
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "enhancenet.yaml",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "enhancenet",
        fullname   : str          | None = "enhancenet",
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
        cfg = cfg or "enhancenet"
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
        pass


class ModelPhase(Enum):
    DECOMNET   = "decomnet"
    # Train the EnhanceNet ONLY. Produce predictions, calculate losses and
    # metrics, update weights at the end of each epoch/step.
    ENHANCENET = "enhancenet"
    # Train the whole network. Produce predictions, calculate losses and
    # metrics, update weights at the end of each epoch/step.
    RETINEXNET = "retinexnet"
    TRAINING   = "training"
    # Produce predictions, calculate losses and metrics, update weights at
    # the end of each epoch/step.
    TESTING    = "testing"
    # Produce predictions, calculate losses and metrics,
    # DO NOT update weights at the end of each epoch/step.
    INFERENCE  = "inference"
    # Produce predictions ONLY.
    
    @classmethod
    def str_mapping(cls) -> dict:
        """
        It returns a dictionary that maps strings to the corresponding enum.
        
        Returns:
            A dictionary with the keys being the string representation of the
                enum and the values being the enum itself.
        """
        return {
            "decomnet"  : cls.DECOMNET,
            "enhancenet": cls.ENHANCENET,
            "retinexnet": cls.RETINEXNET,
            "training"  : cls.TRAINING,
            "testing"   : cls.TESTING,
            "inference" : cls.INFERENCE,
        }

    @classmethod
    def int_mapping(cls) -> dict:
        """
        It returns a dictionary that maps integers to the enum.
        
        Returns:
            A dictionary with the keys being the integer values and the values
                being the enum itself.
        """
        return {
            0: cls.DECOMNET,
            1: cls.ENHANCENET,
            2: cls.RETINEXNET,
            3: cls.TRAINING,
            4: cls.TESTING,
            5: cls.INFERENCE,
        }

    @classmethod
    def from_str(cls, value: str) -> ModelPhase:
        """
        It takes a string and returns an enum.
        
        Args:
            value (str): The string to convert to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.str_mapping(), value.lower())
        return cls.str_mapping()[value]
    
    @classmethod
    def from_int(cls, value: int) -> ModelPhase:
        """
        It takes an integer and returns an enum.
        
        Args:
            value (int): The value to be converted to an enum.
        
        Returns:
            The enum.
        """
        assert_dict_contain_key(cls.int_mapping(), value)
        return cls.int_mapping()[value]

    @classmethod
    def from_value(cls, value: Any) -> ModelPhase | None:
        """
        It converts an arbitrary value to an enum.
        
        Args:
            value (Any): The value to be converted.
        
        Returns:
            The enum.
        """
        if isinstance(value, ModelPhase):
            return value
        if isinstance(value, str):
            return cls.from_str(value)
        if isinstance(value, int):
            return cls.from_int(value)
        error_console.log(
            f"`value` must be `ModelPhase`, `dict`, or `str`. "
            f"But got: {type(value)}."
        )
        return None

    @classmethod
    def keys(cls) -> list:
        """
        Return a list of all the keys of the enumeration.
        
        Returns:
            A list of the keys of the enumeration.
        """
        return [e for e in cls]

    @classmethod
    def values(cls) -> list:
        """
        Return a list of all the values of the enumeration.
        
        Returns:
            A list of the values of the enumeration.
        """
        return [e.value for e in cls]


@MODELS.register(name="retinexnet")
class RetinexNet(ImageEnhancementModel):
    """
    
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
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "retinexnet.yaml",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "retinexnet",
        fullname   : str          | None = "retinexnet",
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
        cfg = cfg or "retinexnet.yaml"
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
            pretrained  = pretrained,
            phase       = phase,
            loss        = loss,
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
        self.decom_loss	  = DecomLoss()
        self.enhance_loss = EnhanceLoss()
        self.retinex_loss = RetinexLoss()
    
    @property
    def phase(self) -> ModelPhase:
        return self._phase
    
    @phase.setter
    def phase(self, phase: ModelPhase_ = "training"):
        """
        Assign the model's running phase.
        """
        self._phase = ModelPhase.from_value(phase)
        if self._phase is ModelPhase.TRAINING:
            self.unfreeze()
            freeze = self.cfg.get("freeze", None)
            if is_list(freeze):
                for k, v in self.model.named_parameters():
                    if any(x in k for x in freeze):
                        v.requires_grad = False
        elif self._phase is ModelPhase.DECOMNET:
            self.model[0].unfreeze()
            self.model[1].freeze()
        elif self._phase is ModelPhase.ENHANCENET:
            self.model[0].freeze()
            self.model[1].unfreeze()
        elif self._phase is ModelPhase.RETINEXNET:
            self.model[0].unfreeze()
            self.model[1].unfreeze()
        else:
            self.freeze()
    
    def parse_model(
        self,
        d : dict      | None = None,
        ch: list[int] | None = None
    ) -> tuple[Sequential, list[int], list[dict]]:
        anchors = d.get("anchors",        None)
        nc      = d.get("num_classes",    None)
        gd      = d.get("depth_multiple", 1)
        gw      = d.get("width_multiple", 1)
        
        layers = []      # layers
        save   = []      # savelist
        ch     = ch or [3]
        c2     = ch[-1]  # out_channels
        info   = []      # print data as table
        for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
            # Convert string class name into class
            m = eval(m) if isinstance(m, str) else m  # eval strings
            for j, a in enumerate(args):
                try:
                    args[j] = eval(a) if isinstance(a, str) else a  # eval strings
                except:
                    pass
            # print(f, n, m, args)
            
            m_    = Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
            m_.i  = i
            m_.f  = f
            m_.t  = t  = str(m)[8:-2].replace("__main__.", "")      # module type
            m_.np = np = sum([x.numel() for x in m_.parameters()])  # number params
            sa    = [x % i for x in ([f] if isinstance(f, int) else f) if x != -1]
            save.extend(sa)  # append to savelist
            layers.append(m_)
            info.append({
                "index"    : i,
                "from"     : f,
                "n"        : n,
                "params"   : np,
                "module"   : t,
                "arguments": args
            })
            
        return Sequential(*layers), sorted(save), info
    
    def init_weights(self, m: Module):
        pass
    
    def forward_loss(
        self,
        input : Tensor,
        target: Tensor,
        *args, **kwargs
    ) -> tuple[Any, Tensor | None]:
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
        if self.phase is ModelPhase.DECOMNET:
            r_low,  i_low  = self.model[0](input=input)
            r_high, i_high = self.model[0](input=target)
            pred           = (input, r_low, i_low, r_high, i_high)
            loss           = self.decom_loss(input=pred, target=target)
            return pred, loss
        
        elif self.phase is ModelPhase.ENHANCENET:
            r_low, i_low = self.model[0](input=input)
            i_delta      = self.model[1](input=(r_low, i_low))
            i_delta_3 	 = torch.cat((i_delta, i_delta, i_delta), dim=1)
            enhance  	 = r_low * i_delta_3
            pred         = (r_low, i_low, i_delta, i_delta_3, enhance)
            loss         = self.enhance_loss(input=pred, target=target)
            return pred, loss

        else:
            r_low, i_low   = self.model[0](input=input)
            r_high, i_high = self.model[0](input=target)
            i_delta        = self.model[1](input=(r_low, i_low))
            i_delta_3 	   = torch.cat((i_delta, i_delta, i_delta), dim=1)
            enhance  	   = r_low * i_delta_3
            pred           = (input, r_low, i_low, r_high, i_high, i_delta, i_delta_3, enhance)
            loss           = self.retinex_loss(input=pred, target=target)
            return pred, loss
        
    def forward_once(
        self,
        input    : Tensor,
        profile  : bool = False,
        out_index: int = -1,
        *args, **kwargs
    ) -> Tensor:
        """
        Forward pass once. Implement the logic for a single forward pass.

        Args:
            input (Tensor): Input of shape [B, C, H, W].
            profile (bool): Measure processing time. Defaults to False.
            out_index (int): Return specific layer's output from `out_index`.
                Defaults to -1 means the last layer.
                
        Returns:
            Predictions.
        """
        r_low, i_low = self.model[0](input=input)
        i_delta      = self.model[1](input=(r_low, i_low))
        i_delta_3 	 = torch.cat((i_delta, i_delta, i_delta), dim=1)
        output       = r_low * i_delta_3
        return output
    
    def show_results(
        self,
        input        : Tensor | None = None,
        target	     : Tensor | None = None,
        pred		 : Tensor | None = None,
        filepath     : Path_  | None = None,
        image_quality: int           = 95,
        max_n        : int | None    = 8,
        nrow         : int | None    = 8,
        wait_time    : float         = 0.01,
        verbose      : bool          = False,
        *args, **kwargs
    ):
        """
        Show results.

        Args:
            input (Tensor | None): Input.
            target (Tensor | None): Ground-truth.
            pred (Tensor | None): Predictions.
            filepath (Path_ | None): File path to save the debug result.
            image_quality (int): Image quality to be saved. Defaults to 95.
            max_n (int | None): Show max n images if `image` has a batch size
                of more than `max_n` images. Defaults to None means show all.
            nrow (int | None): The maximum number of items to display in a row.
                The final grid size is (n / nrow, nrow). If None, then the
                number of items in a row will be the same as the number of
                items in the list. Defaults to 8.
            wait_time (float): Wait some time (in seconds) to display the
                figure then reset. Defaults to 0.01.
            verbose (bool): If True shows the results on the screen.
                Defaults to False.
        """
        from one.plot import imshow_enhancement

        result = {}
        if input is not None:
            result["input"]  = input
        if target is not None:
            result["target"] = target
        if pred is not None:
            if self.phase is ModelPhase.DECOMNET:
                input, r_low, i_low, r_high, i_high = pred
                result["r_low"]  = r_low
                result["i_low"]  = torch.cat(tensors=(i_low, i_low, i_low), dim=1)
                result["r_high"] = r_high
                result["i_high"] = torch.cat(tensors=(i_high, i_high, i_high), dim=1)
            elif self.phase is ModelPhase.RETINEXNET:
                r_low, i_low, i_delta, i_delta_3, enhance = pred
                result["r_low"]   = r_low
                result["i_low"]   = torch.cat(tensors=(i_low, i_low, i_low), dim=1)
                result["i_delta"] = i_delta_3
                result["enhance"] = enhance
            else:
                input, r_low, i_low, r_high, i_high, i_delta, i_delta_3, enhance = pred
                result["r_low"]   = r_low
                result["i_low"]   = torch.cat(tensors=(i_low, i_low, i_low), dim=1)
                result["i_delta"] = i_delta_3
                result["enhance"] = enhance
                
        save_cfg = {
            "filepath"  : filepath or self.debug_image_filepath ,
            "pil_kwargs": dict(quality=image_quality)
        }
        imshow_enhancement(
            winname   = self.phase.value,
            image     = result,
            scale     = 2,
            save_cfg  = save_cfg,
            max_n     = self.debug.max_n,
            nrow      = self.debug.nrow,
            wait_time = self.debug.wait_time,
        )
