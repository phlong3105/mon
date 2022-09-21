#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import multiprocessing
import subprocess
from abc import ABCMeta

import ffmpeg
import numpy as np
import PIL
import torchvision
import torchvision.transforms.functional as F
from joblib import delayed
from joblib import Parallel
from PIL import ExifTags
from PIL import Image
from torch import Tensor

from one.constants import *
from one.core import *
from one.math import make_divisible

# Get orientation exif tag

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


# H1: - Assertion ------------------------------------------------------------

def check_image_size(size: Ints, stride: int = 32) -> int:
    """
    If the input image size is not a multiple of the stride, then the image
    size is updated to the next multiple of the stride.
    
    Args:
        size (Ints): The size of the image.
        stride (int): The stride of the network. Defaults to 32.
    
    Returns:
        The new size of the image.
    """
    if isinstance(size, (list, tuple)):
        if len(size) == 3:    # [C, H, W]
            size = size[1]
        elif len(size) == 2:  # [H, W]
            size = size[0]
        
    new_size = make_divisible(size, int(stride))  # ceil gs-multiple
    if new_size != size:
        error_console.log(
            "WARNING: image_size %g must be multiple of max stride %g, "
            "updating to %g" % (size, stride, new_size)
        )
    return new_size


def is_channel_first(image: Tensor) -> bool:
    """
    If the first dimension is the smallest, then it's channel first.
    
    Args:
        image (Tensor): The image to be checked.
    
    Returns:
        A boolean value.
    """
    assert_tensor_of_ndim_in_range(image, 3, 5)
    if image.ndim == 5:
        _, _, s2, s3, s4 = list(image.shape)
        if (s2 < s3) and (s2 < s4):
            return True
        elif (s4 < s2) and (s4 < s3):
            return False
    elif image.ndim == 4:
        _, s1, s2, s3 = list(image.shape)
        if (s1 < s2) and (s1 < s3):
            return True
        elif (s3 < s1) and (s3 < s2):
            return False
    elif image.ndim == 3:
        s0, s1, s2 = list(image.shape)
        if (s0 < s1) and (s0 < s2):
            return True
        elif (s2 < s0) and (s2 < s1):
            return False
    return False


def is_channel_last(image: Tensor) -> bool:
    """
    If the image is not channel first, then it is channel last.
    
    Args:
        image (Tensor): The image tensor.
    
    Returns:
        A boolean value.
    """
    return not is_channel_first(image)


# H1: - Conversion -----------------------------------------------------------

def to_channel_first(
    image: Tensor, keepdim: bool = True, inplace: bool = False
) -> Tensor:
    """
    It takes a tensor of any shape and returns a tensor of the same shape,
    but with the channel first.
    
    Args:
        image (Tensor): The image to convert.
        keepdim (bool): If True, the dimension will be kept. Defaults to True.
        inplace (bool): If True, the operation will be done in-place.
            Defaults to False.
    
    Returns:
        A tensor with the channel first.
    """
    assert_tensor_of_ndim_in_range(image, 2, 5)
    if not inplace:
        image = image.clone()
    if is_channel_first(image):
        pass
    elif image.ndim == 2:
        image     = image.unsqueeze(0)
    elif image.ndim == 3:
        image     = image.permute(2, 0, 1)
    elif image.ndim == 4:
        image     = image.permute(0, 3, 1, 2)
        keepdim = True
    elif image.ndim == 5:
        image     = image.permute(0, 1, 4, 2, 3)
        keepdim = True
    return image.unsqueeze(0) if not keepdim else image


def to_channel_last(
    image: Tensor, keepdim: bool = True, inplace: bool = False
) -> Tensor:
    """
    It takes a tensor of any shape and returns a tensor of the same shape,
    but with the channel last.
    
    Args:
        image (Tensor): The image to convert.
        keepdim (bool): If True, the dimension will be kept.
            Defaults to True
        inplace (bool): If True, the operation will be done in-place.
            Defaults to False
    
    Returns:
        A tensor with the channel last.
    """
    assert_tensor_of_ndim_in_range(image, 2, 5)
    if not inplace:
        image = image.clone()

    input_shape = image.shape
    if is_channel_last(image):
        pass
    elif image.ndim == 2:
        pass
    elif image.ndim == 3:
        if input_shape[0] == 1:
            # Grayscale for proper plt.imshow needs to be [H, W]
            image = image.squeeze()
        else:
            image = image.permute(1, 2, 0)
    elif image.ndim == 4:  # [..., C, H, W] -> [..., H, W, C]
        image = image.permute(0, 2, 3, 1)
        if input_shape[0] == 1 and not keepdim:
            image = image.squeeze(0)
        if input_shape[1] == 1:
            image = image.squeeze(-1)
    elif image.ndim == 5:
        image = image.permute(0, 1, 3, 4, 2)
        if input_shape[0] == 1 and not keepdim:
            image = image.squeeze(0)
        if input_shape[2] == 1:
            image = image.squeeze(-1)
    return image


def to_image(
    image      : Tensor,
    keepdim    : bool = True,
    denormalize: bool = False,
    inplace    : bool = False,
) -> np.ndarray:
    """
    It converts a tensor to a numpy array.
    
    Args:
        image (Tensor): Tensor
        keepdim (bool): If True, the function will keep the dimensions of
            the input tensor. Defaults to True.
        denormalize (bool): If True, the image will be denormalized to [0, 255].
            Defaults to False.
        inplace (bool): If True, the input tensor will be modified inplace.
            Defaults to False.
    
    Returns:
        A numpy array of the image.
    """
    import one.vision.transformation as t
    
    if not inplace:
        image = image.clone()
    image = image.detach()
    image = to_3d_tensor(image)
    image = t.denormalize_simple(image) if denormalize else image
    image = to_channel_last(image, keepdim=keepdim)
    image = image.cpu().numpy()
    return image.astype(np.uint8)


def to_pil_image(image: Tensor | np.ndarray) -> PIL.Image:
    """
    It converts a tensor or numpy array to a PIL image.
    
    Args:
        image (Tensor | np.ndarray): The image to be transformed.
    
    Returns:
        A PIL image.
    """
    if torch.is_tensor(image):
        # Equivalent to: `np_image = image.numpy()` but more efficient
        return F.pil_to_tensor(image)
    elif isinstance(image, np.ndarray):
        return PIL.Image.fromarray(image.astype(np.uint8), "RGB")
    raise TypeError(f"Do not support {type(image)}.")


def to_tensor(
    image    : Tensor | np.ndarray | PIL.Image,
    keepdim  : bool = True,
    normalize: bool = False,
    inplace  : bool = False,
) -> Tensor:
    """
    Convert a Tensor, np.ndarray, or PIL.Image to a Tensor with channel
    first format.
    
    Args:
        image (Tensor | np.ndarray | PIL.Image): The image to be converted.
        keepdim (bool): If True, the channel dimension will be kept. If False
            unsqueeze the image to match the shape [..., C, H, W].
            Defaults to True
        normalize (bool): If True, normalize the image to [0, 1].
            Defaults to False
        inplace (bool): If True, the input image will be modified inplace.
            Defaults to False
    
    Returns:
        A tensor.
    """
    import one.vision.transformation as t

    if not (F._is_numpy(image) or torch.is_tensor(image)
            or F._is_pil_image(image)):
        raise TypeError(
            f"`image` must be a `Tensor`, `np.ndarray`, or `PIL.Image. "
            f"But got: {type(image)}."
        )
    if ((F._is_numpy(image) or torch.is_tensor(image))
        and not (2 <= image.ndim <= 4)):
        raise ValueError(
            f"Expect 2 <= `image.ndim` <= 4. But got: {image.ndim}."
        )

    # Handle PIL Image
    if F._is_pil_image(image):
        mode = image.mode
        if not inplace:
            image = deepcopy(image)
        mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
        image = np.array(image, mode_to_nptype.get(image.mode, np.uint8), copy=True)
        if mode == "1":
            image = 255 * image
    
    # Handle numpy array
    if F._is_numpy(image):
        if not inplace:
            image = deepcopy(image)
        image = torch.from_numpy(image).contiguous()
    
    # Channel first format
    image = to_channel_first(image, keepdim=keepdim)
   
    # Normalize
    if normalize:
        image = t.normalize_simple(image)
    
    # Convert type
    if isinstance(image, torch.ByteTensor):
        return image.to(dtype=torch.get_default_dtype())
    
    # Place in memory
    image = image.contiguous()
    return image


@TRANSFORMS.register(name="to_image")
class ToImage(Transform):
    """
    Converts a Tensor to a numpy image. In case the image is in the GPU,
    it will be copied back to CPU.

    Args:
        keepdim (bool): If False squeeze the input image to match the shape
            [H, W, C] or [H, W]. Else, keep the original dimension.
            Defaults to True.
        denormalize (bool): If True, converts the image in the range [0.0, 1.0]
            to the range [0, 255]. Defaults to False.
    """
    
    def __init__(
        self,
        keepdim    : bool = True,
        denormalize: bool = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.keepdim     = keepdim
        self.denormalize = denormalize
        
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[np.ndrray, np.ndrray | None]:
        return \
            to_image(
                image       = input,
                keepdim= self.keepdim,
                denormalize = self.denormalize
            ), \
            to_image(
                image       = target,
                keepdim= self.keepdim,
                denormalize = self.denormalize
            ) if target is not None else None


@TRANSFORMS.register(name="to_tensor")
class ToTensor(Transform):
    """
    Convert a `PIL Image` or `np.ndarray` image to a 4D tensor.
    
    Args:
        keepdim (bool): If False unsqueeze the image to match the shape
            [..., C, H, W]. Else, keep the original dimension. Defaults to True.
        normalize (bool): If True, converts the tensor in the range [0, 255]
            to the range [0.0, 1.0]. Defaults to False.
    """
    
    def __init__(
        self,
        keepdim  : bool = False,
        normalize: bool = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.keepdim   = keepdim
        self.normalize = normalize
        
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            to_tensor(
                image     = input,
                keepdim   = self.keepdim,
                normalize = self.normalize
            ), \
            to_tensor(
                image     = input,
                keepdim   = self.keepdim,
                normalize = self.normalize
            ) if target is not None else None
    

# H1: - Image Property -------------------------------------------------------

def get_exif_size(image: PIL.Image) -> Ints:
    """
    If the image has an EXIF orientation tag, and the orientation is 6 or 8,
    then the image is rotated 90 or 270 degrees, so we swap the width and height.
    
    Args:
        image (PIL.Image): PIL.Image.
    
    Returns:
        The height and width of the image.
    """
    size = image.size  # (width, height)
    try:
        rotation = dict(image._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            size = [size[1], size[0]]
        elif rotation == 8:  # rotation 90
            size = [size[1], size[0]]
    except:
        pass
    return [size[1], size[0]]


def get_image_center(image: Tensor) -> Tensor:
    """
    It takes an image and returns the center of the image as (x=h/2, y=w/2).
    
    Args:
        image (Tensor): The image to get the center of.
    
    Returns:
        The center of the image.
    """
    assert_tensor(image)
    h, w = get_image_hw(image)
    return torch.Tensor([h / 2, w / 2])


def get_image_center4(image: Tensor) -> Tensor:
    """
    It takes an image and returns the coordinates of the center of the image
    as (x=h/2, y=w/2, x=h/2, y=w/2).
    
    Args:
        image (Tensor): The image tensor.
    
    Returns:
        The center of the image.
    """
    assert_tensor(image)
    h, w = get_image_hw(image)
    return torch.Tensor([h / 2, w / 2, h / 2, w / 2])
    

def get_image_hw(image: Tensor) -> Ints:
    """
    Given an image tensor, return its height and width
    
    Args:
        image (Tensor[..., C, H, W])): The image tensor.
    
    Returns:
        The height and width of the image.
    """
    assert_tensor_of_atleast_ndim(image, 3)
    if is_channel_first(image):  # [.., C, H, W]
        return [image.shape[-2], image.shape[-1]]
    else:  # [.., H, W, C]
        return [image.shape[-3], image.shape[-2]]
    
    
get_image_size = get_image_hw


def get_image_shape(image: Tensor) -> Ints:
    """
    It returns the shape of the image as a list of integers.
    
    Args:
        image (Tensor): The image tensor.
    
    Returns:
        The shape of the image as [C, H, W].
    """
    assert_tensor_of_atleast_ndim(image, 3)
    if is_channel_first(image):  # [.., C, H, W]
        return [image.shape[-3], image.shape[-2], image.shape[-1]]
    else:  # [.., H, W, C]
        return [image.shape[-1], image.shape[-3], image.shape[-2]]


def get_num_channels(image: Tensor) -> int:
    """
    It returns the number of channels in an image.
    
    Args:
        image (Tensor): The image to get the number of channels from.
    
    Returns:
        The number of channels in the image.
    """
    assert_tensor_of_ndim_in_range(image, 3, 4)
    if image.ndim == 4:
        if is_channel_first(image):
            _, c, h, w = list(image.shape)
        else:
            _, h, w, c = list(image.shape)
        return c
    elif image.ndim == 3:
        if is_channel_first(image):
            c, h, w = list(image.shape)
        else:
            h, w, c = list(image.shape)
        return c
    return 0


# H1: - IO -------------------------------------------------------------------

# H2: - Functional -------------------------------------------------------------

def read_image_cv(path: Path_) -> Tensor:
    """
    It reads an image from a path, converts it to RGB, and converts it to a
    PyTorch tensor.
    
    Args:
        path (Path_): The path to the image file.
    
    Returns:
        A tensor of shape [1, C, H, W].
    """
    image = cv2.imread(str(path))  # BGR
    image = image[:, :, ::-1]      # BGR -> RGB
    image = to_tensor(image=image, keepdim=False, normalize=True)
    return image

'''
def read_image_libvips(path: str) -> np.ndarray:
    """Read image using libvips."""
    image   = pyvips.Image.new_from_file(path, access="sequential")
    mem_img = image.write_to_memory()
    image   = np.frombuffer(mem_img, dtype=np.uint8).reshape(image.height, image.width, 3)
    return image
'''


def read_image_pil(path: Path_) -> Tensor:
    """
    It reads an image from a file, converts it to a tensor, and returns the
    tensor.
    
    Args:
        path (Path_): The path to the image file.
    
    Returns:
        A tensor of shape [1, C, H, W].
    """
    image = Image.open(path)                                       # PIL Image
    image = to_tensor(image=image, keepdim=False, normalize=True)  # Tensor[C, H, W]
    return image


def read_image(path: Path_, backend: VisionBackend_ = VisionBackend.CV) -> Tensor:
    """
    It reads an image from a file path and returns a tensor.
    
    Args:
        path (Path_): The path to the image file.
        backend (VisionBackend_): Vision backend to use.
    
    Returns:
        A tensor of shape [1, C, H, W].
    """
    backend = VisionBackend.from_value(backend)
    if backend == VisionBackend.CV:
        return read_image_cv(path)
    elif backend == VisionBackend.LIBVIPS:
        # return read_image_libvips(path)
        pass
    elif backend == VisionBackend.PIL:
        return read_image_pil(path)
    raise ValueError(f"Do not supported {backend}.")


def read_video_ffmpeg(process, height: int, width: int) -> Tensor:
    """
    Read raw bytes from video stream using ffmpeg and return a Tensor.
    
    Args:
        process: Subprocess that manages ffmpeg.
        height (int): The height of the video frame.
        width (int): The width of the video.
    
    Returns:
        A Tensor of shape [1, C, H, W].
    """
    # RGB24 == 3 bytes per pixel.
    image_size = height * width * 3
    in_bytes   = process.stdout.read(image_size)
    if len(in_bytes) == 0:
        image = None
    else:
        if len(in_bytes) != image_size:
            raise ValueError()
        image = (
            np
                .frombuffer(in_bytes, np.uint8)
                .reshape([height, width, 3])
        )  # Numpy
        image = to_tensor(image=image, keepdim=False, normalize=True)
    return image
  
        
def write_image_pil(
    image      : Tensor,
    dir        : str,
    name       : str,
    prefix     : str  = "",
    extension  : str  = ".png",
    denormalize: bool = True
):
    """
    It takes an image, a directory, a name, a prefix, and an extension, and
    writes the image to the directory with the name and prefix and extension.
    
    Args:
        image (Tensor): The image to write.
        dir (str): The directory to write the image to.
        name (str): The name of the image.
        prefix (str): The prefix to add to the name of the image.
        extension (str): The extension of the image file. Defaults to .png
        denormalize (bool): If True, the image will be denormalized to
                [0, 255]. Defaults to True.
    """

    # Convert image
    image = to_image(image, denormalize=denormalize)
    image = Image.fromarray(image.astype(np.uint8))
    
    # Write image
    dir  = Path(dir)
    create_dirs(paths=[dir])
    name = Path(name)
    stem = name.stem
    ext  = name.suffix
    ext  = f".{extension}"    if ext == ""      else ext
    ext  = f".{ext}"          if "." not in ext else ext
    stem = f"{prefix}_{stem}" if prefix != ""   else stem
    name = f"{stem}{extension}"
    filepath = dir / name
    image.save(filepath)
    
    
def write_image_torch(
    image      : Tensor,
    dir        : Path_,
    name       : str,
    prefix     : str  = "",
    extension  : str  = ".png",
    denormalize: bool = True
):
    """
    It takes an image, a directory, a name, a prefix, and an extension, and
    writes the image to the directory with the name and prefix and extension.
    
    Args:
        image (Tensor): The image to write.
        dir (Path_): The directory to write the image to.
        name (str): The name of the image.
        prefix (str): The prefix to add to the name.
        extension (str): The extension of the image file. Defaults to .png
        denormalize (bool): If True, the image will be denormalized to
                [0, 255]. Defaults to True.
    """
    from one.vision.transformation.intensity import denormalize
    assert_tensor_of_ndim_in_range(image, 2, 3)
    
    # Convert image
    image = denormalize(image) if denormalize else image
    image = to_channel_last(image)
    
    # Write image
    dir  = Path(dir)
    create_dirs(paths=[dir])
    name = Path(name)
    stem = name.stem
    ext  = name.suffix
    ext  = f".{extension}"    if ext == ""      else ext
    ext  = f".{ext}"          if "." not in ext else ext
    stem = f"{prefix}_{stem}" if prefix != ""   else stem
    name = f"{stem}{extension}"
    filepath = dir / name
    
    if ext in [".jpg", ".jpeg"]:
        torchvision.io.image.write_jpeg(input=image, filename=str(filepath))
    elif ext in [".png"]:
        torchvision.io.image.write_png(input=image, filename=str(filepath))


def write_images_pil(
    images     : np.ndarray,
    dir        : Path_,
    names      : Strs,
    prefixes   : Strs = "",
    extension  : str  = ".png",
    denormalize: bool = True
):
    """
    It writes a batch of images to disk.
    
    Args:
        images (np.ndarray): a 4D numpy array of shape [B, H, W, C].
        dir (str): The directory to write the images to.
        names (Strs): Names of the images.
        prefixes (Strs): The prefixes to add to the names.
        extension (str): The extension of the images file. Defaults to .png
        denormalize (bool): If True, the image will be denormalized to
                [0, 255]. Defaults to True.
    """
    images = to_3d_array_list(images)
    if isinstance(names, str):
        names    = [names for _ in range(len(images))]
    if isinstance(prefixes, str):
        prefixes = [prefixes for _ in range(len(prefixes))]
    assert_same_length(images, names)
    assert_same_length(images, prefixes)
    
    num_jobs = multiprocessing.cpu_count()
    Parallel(n_jobs=num_jobs)(
        delayed(write_image_pil)(
            image, dir, names[i], prefixes[i], extension, denormalize
        )
        for i, image in enumerate(images)
    )


def write_images_torch(
    images     : Tensor,
    dir        : Path_,
    names      : Strs,
    prefixes   : Strs = "",
    extension  : str  = ".png",
    denormalize: bool = True
):
    """
    It writes a batch of images to disk.
    
    Args:
        images (Tensor): A 4D tensor of shape [B, C, H, W]
        dir (str): The directory to write the images to.
        names (Strs): Names of the images.
        prefixes (Strs): The prefixes to add to the names.
        extension (str): The extension of the images file. Defaults to .png
        denormalize (bool): If True, the image will be denormalized to
                [0, 255]. Defaults to True.
    """
    images = to_3d_tensor_list(images)
    if isinstance(names, str):
        names    = [names for _ in range(len(images))]
    if isinstance(prefixes, str):
        prefixes = [prefixes for _ in range(len(prefixes))]
    assert_same_length(images, names)
    assert_same_length(images, prefixes)
    
    num_jobs = multiprocessing.cpu_count()
    Parallel(n_jobs=num_jobs)(
        delayed(write_image_torch)(
            image, dir, names[i], prefixes[i], extension, denormalize
        )
        for i, image in enumerate(images)
    )


def write_video_ffmpeg(process, image: Tensor | None, denormalize: bool = True):
    """
    Write a tensor image to video file using ffmpeg.

    Args:
        process: Subprocess that manages ffmpeg.
        image (Tensor | None): frame/image of shape [1, C, H, W].
            Defaults to None.
        denormalize (bool): If True, the image will be denormalized to
            [0, 255]. Defaults to True.
    """
    if isinstance(image, Tensor):
        image = to_image(image=image, keepdim=False, denormalize=denormalize)
        process.stdin.write(
            image
                .astype(np.uint8)
                .tobytes()
        )
    else:
        error_console(f"`image` must be `Tensor`. But got: {type(image)}.")
      

# H2: - Base Module ------------------------------------------------------------

class BaseLoader(metaclass=ABCMeta):
    """
    A baseclass/interface for all VideoLoader classes.
    
    Args:
        source (Path_): Data source. Can be a path to an image file, a
            directory, a video, or a stream. It can also be a pathname pattern
            to images.
        batch_size (int): Number of samples in one forward & backward pass.
            Defaults to 1.
        verbose (bool): Verbosity mode of video loader backend. Defaults to False.
    """
    
    def __init__(
        self,
        source    : Path_,
        batch_size: int  = 1,
        verbose   : bool = False,
        *args, **kwargs
    ):
        super().__init__()
        self.source     = Path(source)
        self.batch_size = batch_size
        self.verbose    = verbose
        self.index      =  0
        self.num_images = -1
        self.init()
    
    def __len__(self) -> int:
        """
        The function returns the number of images in the dataset.
        
        Returns:
            The number of images in the dataset.
        """
        return self.num_images
    
    def __iter__(self):
        """
        The __iter__ function returns an iterator object starting at index 0.
        
        Returns:
            The iterator object itself.
        """
        self.reset()
        return self
    
    @abstractmethod
    def __next__(self):
        pass
    
    def __del__(self):
        """
        Close.
        """
        self.close()
    
    def batch_len(self) -> int:
        """
        Return the total batches calculated from `batch_size`.
        """
        return int(self.__len__() / self.batch_size)
    
    @abstractmethod
    def init(self):
        """
        Initialize data source.
        """
        pass
    
    @abstractmethod
    def reset(self):
        """
        Reset and start over.
        """
        pass
    
    @abstractmethod
    def close(self):
        """
        Stop and release.
        """
        pass
    

class BaseWriter(metaclass=ABCMeta):
    """
    Video Writer saves images to a destination directory.

    Args:
        dst (Path_): The destination folder where the images will be saved.
        shape (Ints): Output size as [C, H, W]. This is also used to reshape
            the input. Defaults to (3, 480, 640).
        verbose (bool): Verbosity mode of video writer backend. Defaults to False.
    """
    
    def __init__(
        self,
        dst    : Path_,
        shape  : Ints  = (3, 480, 640),
        verbose: bool  = False,
        *args, **kwargs
    ):
        super().__init__()
        self.dst	    = Path(dst)
        self.shape      = shape
        self.image_size = to_size(shape)
        self.verbose    = verbose
        self.index      = 0
        self.init()
    
    def __len__(self) -> int:
        """
        Return the number of already written frames
        
        Returns:
            The number of frames that have been written to the file.
        """
        return self.index
    
    def __del__(self):
        """
        Close the `video_writer`.
        """
        self.close()
    
    @abstractmethod
    def init(self):
        """
        Initialize output.
        """
        pass
    
    @abstractmethod
    def close(self):
        """
        Close the `video_writer`.
        """
        pass
        
    @abstractmethod
    def write(
        self,
        image      : Tensor,
        image_file : str | None = None,
        denormalize: bool       = True
    ):
        """
        Add a frame to writing video.

        Args:
            image (Tensor): Image.
            image_file (str | None): Image file. Defaults to None.
            denormalize (bool): If True, the image will be denormalized to
                [0, 255]. Defaults to True.
        """
        pass
    
    @abstractmethod
    def write_batch(
        self,
        images     : Tensors,
        image_files: list[str] | None = None,
        denormalize: bool             = True
    ):
        """Add batch of frames to video.

        Args:
            images (Tensors): List of images.
            image_files (list[str] | None): Image files. Defaults to None.
            denormalize (bool): If True, the image will be denormalized to
                [0, 255]. Defaults to True.
        """
        pass
    
    
# H2: - Image Loader -----------------------------------------------------------

class ImageLoader(BaseLoader):
    """
    Image Loader retrieves and loads image(s) from a filepath, a pathname
    pattern, or directory.
    
    Notes:
        We don't need to define the image shape since images in directory can
        have different shapes.
    
    Args:
        source (Path_): The path to the image file. Can be a path to an image
            file or a directory, or a pathname pattern to images.
        batch_size (int): The number of images to be processed at once.
            Defaults to 1.
        backend (VisionBackend_): The backend to use for image processing.
            Default to VisionBackend.CV.
        verbose (bool): Verbosity mode of video loader backend. Defaults to False.
    """

    def __init__(
        self,
        source    : Path_,
        batch_size: int            = 1,
        backend   : VisionBackend_ = VisionBackend.CV,
        verbose   : bool           = False,
    ):
        self.images  = []
        self.backend = VisionBackend.from_value(backend)
        super().__init__(
            source     = source,
            batch_size = batch_size,
            verbose    = verbose
        )

    def __next__(self) -> tuple[Tensor, list, list, list]:
        """
        It reads a batch of images from the disk, converts them to RGB, and
        returns them as a tensor.
        
        Examples:
            >>> images = ImageLoader("cam_1.mp4")
            >>> for index, image in enumerate(images):
        
        Returns:
            Images tensor of shape [B, C, H, W].
            List of image indexes
            List of image files.
            List of images' relative paths corresponding to data.
        """
        if self.index >= self.num_images:
            raise StopIteration
        else:
            images    = []
            indexes   = []
            files     = []
            rel_paths = []
            
            for i in range(self.batch_size):
                if self.index >= self.num_images:
                    break
                
                file     = self.images[self.index]
                rel_path = str(file).replace(str(self.source), "")
                image    = read_image(
                    path    = self.images[self.index],
                    backend = self.backend,
                )
                # image  = image[:, :, ::-1]  # BGR to RGB
                
                images.append(image)
                indexes.append(self.index)
                files.append(file)
                rel_paths.append(rel_path)
                
                self.index += 1
            
            images = torch.stack(images)
            images = torch.squeeze(images, dim=0)
            return images, indexes, files, rel_paths
    
    def init(self):
        self.images = []
        if is_image_file(self.source):
            self.images = [self.source]
        elif is_dir(self.source):
            self.images = [i for i in Path(self.source).rglob("*")
                           if is_image_file(i)]
        elif isinstance(self.source, str):
            self.images = [Path(i) for i in glob.glob(self.source)
                           if is_image_file(i)]
        else:
            raise IOError("Error when listing image files.")
        self.num_images = len(self.images)

    def reset(self):
        self.index = 0

    def close(self):
        pass


# H2: - Image Writer -----------------------------------------------------------

class ImageWriter(BaseWriter):
    """
    Image Writer saves images to a destination directory.

    Args:
        dst (Path_): The destination folder where the images will be saved.
        shape (Ints): Output size as [C, H, W]. This is also used to reshape
            the input. Defaults to (3, 480, 640).
        extension (str): The extension of the file to be saved.
            Defaults to .jpg
        verbose (bool): Verbosity mode of video writer backend. Defaults to False.
    """

    def __init__(
        self,
        dst      : Path_,
        shape    : Ints = (3, 480, 640),
        extension: str  = ".jpg",
        verbose  : bool = False,
        *args, **kwargs
    ):
        super().__init__(
            dst     = dst,
            shape   = shape,
            verbose = verbose
        )
        self.extension = extension

    def __len__(self) -> int:
        """
        The function returns the number of items in the stack.
        
        Returns:
            The index of the last item in the list.
        """
        return self.index
    
    def init(self):
        pass

    def close(self):
        pass

    def write(
        self,
        image      : Tensor,
        image_file : Path_ | None = None,
        denormalize: bool         = True
    ):
        """
        It takes an image, converts it to a numpy array, and saves it to disk.
        
        Args:
            image (Tensor): The image to write.
            image_file (Path_ | None): The name of the image file.
            denormalize (bool): If True, the image will be denormalized to
                [0, 255]. Defaults to True.
        """
        """
        image = to_image(image=image, keepdim=False, denormalize=denormalize)
        if image_file is not None:
            image_file = (image_file[1:] if image_file.startswith("\\")
                          else image_file)
            image_file = (image_file[1:] if image_file.startswith("/")
                          else image_file)
            image_name = os.path.splitext(image_file)[0]
        else:
            image_name = f"{self.index}"
        
        stem        = Path(image_file).stem
        output_file = self.dst / stem / self.extension
        parent_dir  = str(Path(output_file).parent)
        create_dirs(paths=[parent_dir])
        cv2.imwrite(output_file, image)
        """
        if isinstance(image_file, (Path, str)):
            image_file = self.dst / f"{Path(image_file).stem}{self.extension}"
        else:
            raise ValueError(f"`image_file` must be given.")
        image_file = Path(image_file)
        write_image_torch(
            image       = image,
            dir         = image_file.parent,
            name        = image_file.name,
            extension   = self.extension,
            denormalize = denormalize
        )
        self.index += 1

    def write_batch(
        self,
        images     : Tensors,
        image_files: Paths_ | None = None,
        denormalize: bool          = True,
    ):
        """
        Write a list of images to disk.
        
        Args:
            images (Tensors): A list of tensors, each of which is a single
                image.
            image_files (Paths_ | None): Paths to save images. Defaults to None.
            denormalize (bool): If True, the image will be denormalized to
                [0, 255]. Defaults to True.
        """
        images = to_3d_tensor_list(images)
        
        if image_files is None:
            image_files = [None for _ in range(len(images))]
        
        for image, image_file in zip(images, image_files):
            self.write(
                image=image, image_file=image_file, denormalize=denormalize
            )


# H2: - Video Loader -----------------------------------------------------------

class VideoLoader(BaseLoader, metaclass=ABCMeta):
    """
    A baseclass/interface for all VideoLoader classes.
    """
    
    def __init__(
        self,
        data      : Path_,
        batch_size: int  = 1,
        verbose   : bool = False,
        *args, **kwargs
    ):
        self.frame_rate = 0
        super().__init__(
            data       = data,
            batch_size = batch_size,
            verbose    = verbose,
            *args, **kwargs
        )
    
    @property
    @abstractmethod
    def fourcc(self) -> str:
        """
        Return 4-character code of codec.
        """
        pass
    
    @property
    @abstractmethod
    def fps(self) -> int:
        """
        Return frame rate.
        """
        pass
   
    @property
    @abstractmethod
    def frame_height(self) -> int:
        """
        Return height of the frames in the video stream.
        """
        pass
    
    @property
    @abstractmethod
    def frame_width(self) -> int:
        """
        Return width of the frames in the video stream.
        """
        pass
    
    @property
    def is_stream(self) -> bool:
        """
        Return True if it is a video stream, i.e, unknown `frame_count`.
        """
        return self.num_images == -1
    
    @property
    def shape(self) -> Ints:
        """
        It returns the shape of the frames in the video stream in [C, H, W]
        format.
        
        Returns:
            The height and width of the frame.
        """
        return 3, self.frame_height, self.frame_width


class VideoLoaderCV(VideoLoader):
    """
    Loads frame(s) from a video or a stream using OpenCV.

    Args:
        data (Path_): Data source. Can be a path to an image file, a directory,
            a video, or a stream. It can also be a pathname pattern to images.
        batch_size (int): Number of samples in one forward & backward pass.
            Defaults to 1.
        api_preference (int): Preferred Capture API backends to use. Can be
            used to enforce a specific reader implementation. Two most used
            options are: [cv2.CAP_ANY=0, cv2.CAP_FFMPEG=1900].
            See more: https://docs.opencv.org/4.5.5/d4/d15/group__videoio__flags__base.html#ggaeb8dd9c89c10a5c63c139bf7c4f5704da7b235a04f50a444bc2dc72f5ae394aaf
            Defaults to cv2.CAP_FFMPEG.
        verbose (bool): Verbosity mode of video loader backend. Defaults to False.
    """

    def __init__(
        self,
        data          : str,
        batch_size    : int  = 1,
        api_preference: int  = cv2.CAP_FFMPEG,
        verbose       : bool = False,
        *args, **kwargs
    ):
        self.api_preference = api_preference
        self.video_capture  = None
        super().__init__(
            data       = data,
            batch_size = batch_size,
            verbose    = verbose,
            *args, **kwargs
        )
   
    def __next__(self) -> tuple[Tensor, list, list, list]:
        """
        Load next batch of images.
        
        Returns:
            Image Tensor of shape [B, C, H, W].
            List of image indexes.
            List of image files.
            List of images' relative paths corresponding to data.
        """
        if not self.is_stream and self.index >= self.frame_count:
            self.close()
            raise StopIteration
        else:
            images    = []
            indexes   = []
            files     = []
            rel_paths = []
            
            for i in range(self.batch_size):
                if not self.is_stream and self.index >= self.frame_count:
                    break
                
                if isinstance(self.video_capture, cv2.VideoCapture):
                    ret_val, image = self.video_capture.read()
                    rel_path       = self.data.name
                else:
                    raise RuntimeError(
                        f"`video_capture` has not been initialized."
                    )
                
                if image is None:
                    continue
                image = image[:, :, ::-1]  # BGR to RGB
                image = to_tensor(image=image, keepdim=False)
                
                images.append(image)
                indexes.append(self.index)
                files.append(self.data)
                rel_paths.append(rel_path)
                
                self.index += 1

            images = torch.stack(images)
            images = torch.squeeze(images, dim=0)
            return images, indexes, files, rel_paths
    
    @property
    def format(self):  # Flag=8
        """
        Return format of the Mat objects (see Mat::type()) returned by
        VideoCapture::retrieve(). Set value -1 to fetch undecoded RAW video
        streams (as Mat 8UC1).
        """
        return self.video_capture.get(cv2.CAP_PROP_FORMAT)
    
    @property
    def fourcc(self) -> str:  # Flag=6
        """
        Return 4-character code of codec.
        """
        return str(self.video_capture.get(cv2.CAP_PROP_FOURCC))
    
    @property
    def fps(self) -> int:  # Flag=5
        """Return frame rate."""
        return int(self.video_capture.get(cv2.CAP_PROP_FPS))
    
    @property
    def frame_count(self) -> int:  # Flag=7
        """
        Return number of frames in the video file.
        """
        if isinstance(self.video_capture, cv2.VideoCapture):
            return int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        elif isinstance(self.video_capture, list):
            return len(self.video_capture)
        else:
            return -1
    
    @property
    def frame_height(self) -> int:  # Flag=4
        """
        Return height of the frames in the video stream.
        """
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    @property
    def frame_width(self) -> int:  # Flag=3
        """Return width of the frames in the video stream."""
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    @property
    def mode(self):  # Flag=10
        """
        Return backend-specific value indicating the current capture mode.
        """
        return self.video_capture.get(cv2.CAP_PROP_MODE)
    
    @property
    def pos_avi_ratio(self) -> int:  # Flag=2
        """
        Return relative position of the video file:
        0=start of the film, 1=end of the film.
        """
        return int(self.video_capture.get(cv2.CAP_PROP_POS_AVI_RATIO))
    
    @property
    def pos_msec(self) -> int:  # Flag=0
        """
        Return current position of the video file in milliseconds.
        """
        return int(self.video_capture.get(cv2.CAP_PROP_POS_MSEC))
    
    @property
    def pos_frames(self) -> int:  # Flag=1
        """
        Return 0-based index of the frame to be decoded/captured next.
        """
        return int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
    
    def init(self):
        if is_video_file(self.data):
            self.video_capture = cv2.VideoCapture(self.data, self.api_preference)
            self.num_images    = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_rate    = int(self.video_capture.get(cv2.CAP_PROP_FPS))
        elif is_video_stream(self.data):
            self.video_capture = cv2.VideoCapture(self.data, self.api_preference)  # stream
            self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, self.batch_size)
            self.num_images    = -1
        else:
            raise IOError("Error when reading input stream or video file!")
        
    def reset(self):
        """
        Reset and start over.
        """
        self.index = 0
        if isinstance(self.video_capture, cv2.VideoCapture):
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.index)
    
    def close(self):
        """
        Stop and release the current `video_capture` object.
        """
        if isinstance(self.video_capture, cv2.VideoCapture) \
            and self.video_capture:
            self.video_capture.release()
   

class VideoLoaderFFmpeg(VideoLoader):
    """
    Loads frame(s) from a video or a stream using FFmpeg.
    
    Args:
        data (str): Data source. Can be a path to an image file, pathname
            pattern to  image files, a directory, a video, or a stream.
        batch_size (int): Number of samples in one forward & backward pass.
            Defaults to 1.
        verbose (bool): Verbosity mode of video loader backend. Defaults to False.
        kwargs: Any supplied kwargs are passed to ffmpeg verbatim.
        
    References:
        https://github.com/kkroening/ffmpeg-python/tree/master/examples
    """
    
    def __init__(
        self,
        data      : str,
        batch_size: int  = 1,
        verbose   : bool = False,
        *args, **kwargs
    ):
        self.ffmpeg_cmd     = None
        self.ffmpeg_process = None
        self.ffmpeg_kwargs  = kwargs
        self.video_info     = None
        super().__init__(
            data       = data,
            batch_size = batch_size,
            verbose    = verbose,
            *args, **kwargs
        )
        
    def __next__(self) -> tuple[Tensor, list, list, list]:
        """
        Load next batch of images.
    
        Returns:
            Image Tensor of shape [B, C, H, W].
            List of image indexes.
            List of image files.
            List of images' relative paths corresponding to data.
        """
        if not self.is_stream and self.index >= self.frame_count:
            self.close()
            raise StopIteration
        else:
            images    = []
            indexes   = []
            files     = []
            rel_paths = []
            
            for i in range(self.batch_size):
                if not self.is_stream and self.index >= self.frame_count:
                    break
                
                if self.ffmpeg_process:
                    image = read_video_ffmpeg(
                        process = self.ffmpeg_process,
                        width   = self.frame_width,
                        height  = self.frame_height
                    )  # Already in RGB
                    rel_path = self.data.name
                else:
                    raise RuntimeError(
                        f"`video_capture` has not been initialized."
                    )
                
                images.append(image)
                indexes.append(self.index)
                files.append(self.data)
                rel_paths.append(rel_path)
                
                self.index += 1

            images = torch.stack(images)
            images = torch.squeeze(images, dim=0)
            return images, indexes, files, rel_paths
        
    @property
    def fourcc(self) -> str:
        """
        Return 4-character code of codec.
        """
        return self.video_info["codec_name"]
    
    @property
    def fps(self) -> int:
        """
        Return frame rate.
        """
        return int(self.video_info["avg_frame_rate"].split("/")[0])
    
    @property
    def frame_count(self) -> int:
        """
        Return number of frames in the video file.
        """
        if is_video_file(self.data):
            return int(self.video_info["nb_frames"])
        else:
            return -1
    
    @property
    def frame_width(self) -> int:
        """
        Return width of the frames in the video stream.
        """
        return int(self.video_info["width"])
    
    @property
    def frame_height(self) -> int:
        """
        Return height of the frames in the video stream.
        """
        return int(self.video_info["height"])
        
    def init(self):
        """
        Initialize ffmpeg cmd.
        """
        probe           = ffmpeg.probe(self.data, **self.ffmpeg_kwargs)
        self.video_info = next(
            s for s in probe["streams"] if s["codec_type"] == "video"
        )
        if self.verbose:
            self.ffmpeg_cmd = (
                ffmpeg
                    .input(self.data, **self.ffmpeg_kwargs)
                    .output("pipe:", format="rawvideo", pix_fmt="rgb24")
                    .compile()
            )
        else:
            self.ffmpeg_cmd = (
                ffmpeg
                    .input(self.data, **self.ffmpeg_kwargs)
                    .output("pipe:", format="rawvideo", pix_fmt="rgb24")
                    .global_args("-loglevel", "quiet")
                    .compile()
            )
    
    def reset(self):
        """
        Reset and start over.
        """
        self.close()
        self.index = 0
        if self.ffmpeg_cmd:
            self.ffmpeg_process = subprocess.Popen(
                self.ffmpeg_cmd,
                stdout  = subprocess.PIPE,
                bufsize = 10**8
            )
    
    def close(self):
        """
        Stop and release the current `ffmpeg_process`.
        """
        if self.ffmpeg_process and self.ffmpeg_process.poll() is not None:
            # os.killpg(os.getpgid(self.ffmpeg_process.pid), signal.SIGTERM)
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()
            self.ffmpeg_process = None
            # raise StopIteration


# H2: - Video Writer -----------------------------------------------------------

class VideoWriter(BaseWriter, metaclass=ABCMeta):
    """
    A baseclass/interface for all VideoWriter classes.

    Args:
        dst (Path_): The destination folder where the images will be saved.
        shape (Ints): Output size as [C, H, W]. This is also used to reshape
            the input. Defaults to (3, 480, 640).
        frame_rate (float): Frame rate of the video. Defaults to 10.
        save_image (bool): Should write individual image? Defaults to False.
        verbose (bool): Verbosity mode of video writer backend. Defaults to False.
    """
    
    def __init__(
        self,
        dst       : Path_,
        shape     : Ints  = (3, 480, 640),
        frame_rate: float = 10,
        save_image: bool  = False,
        verbose   : bool  = False,
        *args, **kwargs
    ):
        self.frame_rate = frame_rate
        self.save_image = save_image
        super().__init__(
            dst     = dst,
            shape   = shape,
            verbose = verbose,
            *args, **kwargs
        )


class VideoWriterCV(VideoWriter):
    """
    Saves frames to individual image files or appends all to a video file.

    Args:
        dst (str): Output video file or a directory.
        shape (Ints): Output size as [C, H, W]. This is also used to reshape the
            input. Defaults to (3, 480, 640).
        frame_rate (int): Frame rate of the video. Defaults to 10.
        fourcc (str): Video codec. One of: ["mp4v", "xvid", "mjpg", "wmv"].
            Defaults to .mp4v.
        save_image (bool): Should write individual image? Defaults to False.
        save_video (bool): Should write video? Defaults to True.
        verbose (bool): Verbosity mode of video writer backend. Defaults to False.
    """
    
    def __init__(
        self,
        dst		  : str,
        shape     : Ints  = (3, 480, 640),
        frame_rate: float = 10,
        fourcc    : str   = ".mp4v",
        save_image: bool  = False,
        verbose   : bool  = False,
        *args, **kwargs
    ):
        self.fourcc       = fourcc
        self.video_writer = None
        super().__init__(
            dst        = dst,
            shape      = shape,
            frame_rate = frame_rate,
            save_image = save_image,
            verbose    = verbose,
            *args, **kwargs
        )
        
    def init(self):
        """
        Initialize output.
        """
        if self.dst.is_dir():
            video_file = self.dst / f"result.mp4"
        else:
            video_file = self.dst.parent / f"{self.dst.stem}.mp4"
        create_dirs(paths=[video_file.parent])

        fourcc			  = cv2.VideoWriter_fourcc(*self.fourcc)
        self.video_writer = cv2.VideoWriter(
            video_file, fourcc, self.frame_rate, self.image_size[::-1]  # Must be in [W, H]
        )

        if self.video_writer is None:
            raise FileNotFoundError(f"Cannot create video file at: {video_file}.")

    def close(self):
        """
        Close the `video_writer`.
        """
        if self.video_writer:
            self.video_writer.release()

    def write(
        self,
        image      : Tensor,
        image_file : str  | None = None,
        denormalize: bool        = True
    ):
        """
        Add a frame to writing video.

        Args:
            image (Tensor): Image.
            image_file (str | None): Image file. Defaults to None.
            denormalize (bool): If True, the image will be denormalized to
                [0, 255]. Defaults to True.
        """
        image = to_image(image=image, keepdim=False, denormalize=denormalize)
        
        if self.save_image:
            if isinstance(image_file, (Path, str)):
                image_file = self.dst / f"{Path(image_file).stem}.png"
            else:
                raise ValueError(f"`image_file` must be given.")
            create_dirs(paths=[image_file.parent])
            cv2.imwrite(str(image_file), image)
        
        self.video_writer.write(image)
        self.index += 1

    def write_batch(
        self,
        images     : Tensors,
        image_files: list[str] | None = None,
        denormalize: bool             = True
    ):
        """
        Add batch of frames to video.

        Args:
            images (Tensors): Images.
            image_files (list[str] | None): Image files. Defaults to None.
            denormalize (bool): If True, the image will be denormalized to
                [0, 255]. Defaults to True.
        """
        images = to_3d_tensor_list(images)
        
        if image_files is None:
            image_files = [None for _ in range(len(images))]

        for image, image_file in zip(images, image_files):
            self.write(
                image=image, image_file=image_file, denormalize=denormalize
            )


class VideoWriterFFmpeg(VideoWriter):
    """
    Saves frames to individual image files or appends all to a video file.

    Args:
        dst (Path_): Output video file or a directory.
        shape (Int3T): Output size as [C, H, W]. This is also used to reshape
            the input. Defaults to (3, 480, 640).
        frame_rate (int): Frame rate of the video. Defaults to 10.
        pix_fmt (str): Video codec. Defaults to yuv420p.
        save_image (bool): Should write individual image? Defaults to False.
        save_video (bool): Should write video? Defaults to True.
        verbose (bool): Verbosity mode of video loader backend. Defaults to False.
        ffmpeg_process (subprocess.Popen): Subprocess that manages ffmpeg.
        index (int): Current index.
        kwargs: Any supplied kwargs are passed to ffmpeg verbatim.
    """
    
    def __init__(
        self,
        dst		  : str,
        shape     : Ints  = (3, 480, 640),
        frame_rate: float = 10,
        pix_fmt   : str   = "yuv420p",
        save_image: bool  = False,
        save_video: bool  = True,
        verbose   : bool  = False,
        *args, **kwargs
    ):
        self.pix_fmt        = pix_fmt
        self.ffmpeg_process = None
        self.ffmpeg_kwargs  = kwargs
        super().__init__(
            dst        = dst,
            shape      = shape,
            frame_rate = frame_rate,
            save_image = save_image,
            save_video = save_video,
            verbose    = verbose,
            *args, **kwargs
        )

    def init(self):
        """
        Initialize output.
        """
        if self.dst.is_dir():
            video_file = self.dst / f"result.mp4"
        else:
            video_file = self.dst.parent / f"{self.dst.stem}.mp4"
        create_dirs(paths=[video_file.parent])
        
        if self.verbose:
            self.ffmpeg_process = (
                ffmpeg
                    .input("pipe:", format="rawvideo", pix_fmt="rgb24", s="{}x{}".format(self.image_size[1], self.image_size[0]))
                    .output(filename=str(video_file), pix_fmt=self.pix_fmt, **self.ffmpeg_kwargs)
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
            )
        else:
            self.ffmpeg_process = (
                ffmpeg
                    .input("pipe:", format="rawvideo", pix_fmt="rgb24", s="{}x{}".format(self.image_size[1], self.image_size[0]))
                    .output(filename=str(video_file), pix_fmt=self.pix_fmt, **self.ffmpeg_kwargs)
                    .global_args("-loglevel", "quiet")
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
            )
        
    def close(self):
        """
        Stop and release the current `ffmpeg_process`.
        """
        if self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()
            self.ffmpeg_process = None
    
    def write(
        self,
        image      : Tensor,
        image_file : str | None = None,
        denormalize: bool       = True
    ):
        """
        Add a frame to writing video.

        Args:
            image (Tensor): Image of shape [C, H, W].
            image_file (str | None): Image file. Defaults to None.
            denormalize (bool): If True, the image will be denormalized to
                [0, 255]. Defaults to True.
        """
        if self.save_image:
            if isinstance(image_file, (Path, str)):
                image_file = self.dst / f"{Path(image_file).stem}.png"
            else:
                raise ValueError(f"`image_file` must be given.")
            create_dirs(paths=[image_file.parent])
            image = to_image(image=image, keepdim=False, denormalize=denormalize)
            cv2.imwrite(str(image_file), image)
        
        write_video_ffmpeg(
            process=self.ffmpeg_process, image=image, denormalize=denormalize
        )
        self.index += 1

    def write_batch(
        self,
        images     : Tensors,
        image_files: list[str] | None = None,
        denormalize: bool             = True
    ):
        """
        Add batch of frames to video.

        Args:
            images (Tensors): Images.
            image_files (list[str] | None): Image files. Defaults to None.
            denormalize (bool): If True, the image will be denormalized to
                [0, 255]. Defaults to True.
        """
        images = to_3d_tensor_list(images)
        
        if image_files is None:
            image_files = [None for _ in range(len(images))]

        for image, image_file in zip(images, image_files):
            self.write(image=image, image_file=image_file)
