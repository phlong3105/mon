"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import PIL.Image
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision import transforms as _transforms

from .._misc import (
    BoundingBoxes,
    Image,
    Mask,
    SanitizeBoundingBoxes,
    Video,
    _boxes_keys,
    convert_to_tv_tensor,
)
from ...core import register

cv2.setNumThreads(0)
torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut            = register()(T.RandomZoomOut)
RandomHorizontalFlip     = register()(T.RandomHorizontalFlip)
Resize                   = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes    = register(name="SanitizeBoundingBoxes")(SanitizeBoundingBoxes)
RandomCrop               = register()(T.RandomCrop)
Normalize                = register()(T.Normalize)


@register()
class ResizeCV(T.Transform):
    """Resize the input to the given size using cv2."""

    _v1_transform_cls = _transforms.Resize

    def __init__(
        self,
        size         : Union[int, Sequence[int], None],
        interpolation: int            = cv2.INTER_AREA,
        max_size     : Optional[int]  = None,
        antialias    : Optional[bool] = True,
    ) -> None:
        super().__init__()

        if isinstance(size, int):
            size = [size]
        elif isinstance(size, Sequence) and len(size) in {1, 2}:
            size = list(size)
        elif size is None:
            if not isinstance(max_size, int):
                raise ValueError(f"max_size must be an integer when size is None, but got {max_size} instead.")
        else:
            raise ValueError(f"size can be an integer, a sequence of one or two integers, or None, but got {size} instead.")
        self.size          = size
        self.interpolation = interpolation
        self.max_size      = max_size
        self.antialias     = antialias
        cv2.setNumThreads(0)  # Disable multithreading in OpenCV to avoid potential issues with parallel processing

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, PIL.Image.Image):
            inpt = np.array(inpt)
            inpt = cv2.resize(inpt, self.size, interpolation=self.interpolation)
            # print(inpt.shape)
            return inpt
        else:
            return inpt


@register()
class EmptyTransform(T.Transform):

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):

    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode="constant") -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params["padding"]
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]["padding"] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):

    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
        p: float = 1.0,
    ):
        super().__init__(
            min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials
        )
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):

    _transformed_types = (BoundingBoxes,)

    def __init__(self, fmt="", normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(
                inpt, key="boxes", box_format=self.fmt.upper(), spatial_size=spatial_size
            )

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):

    _transformed_types = (PIL.Image.Image,)

    def __init__(self, dtype="float32", scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == "float32":
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.0

        inpt = Image(inpt)

        return inpt


@register()
class ConvertNumpyImage(T.Transform):

    _transformed_types = (
        np.ndarray,
    )

    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = torch.from_numpy(inpt).contiguous()
        inpt = inpt.permute(2, 0, 1)

        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt
