#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements image and video I/O functions."""

from __future__ import annotations

__all__ = [
    "ImageLoader", "ImageWriter", "Loader", "VideoLoader", "VideoLoaderCV",
    "VideoLoaderFFmpeg", "VideoWriter", "VideoWriterCV", "VideoWriterFFmpeg",
    "Writer", "read_image", "read_video_ffmpeg", "write_image_cv",
    "write_image_torch", "write_images_cv", "write_images_torch",
    "write_video_ffmpeg",
]

import glob
import multiprocessing
import subprocess
from abc import ABC, abstractmethod
from typing import Sequence

import cv2
import ffmpeg
import joblib
import numpy as np
import torch
import torchvision

from mon.foundation import pathlib
from mon.vision.image import base


# region Read

def read_image(
    path     : pathlib.Path,
    to_rgb   : bool = True,
    to_tensor: bool = False,
    normalize: bool = False,
) -> torch.Tensor | np.ndarray:
    """Read an image from a filepath using :mod:`cv2`. Optionally, convert it to
    RGB format, and :class:`torch.Tensor` type of shape 1CHW.

    Args:
        path: An image filepath.
        to_rgb: If True, convert the image from BGR to RGB. Defaults to True.
        to_tensor: If True, convert the image from :class:`numpy.ndarray` to
            :class:`torch.Tensor`. Defaults to False.
        normalize: If True, normalize the image to [0.0, 1.0]. Defaults to
            False.
        
    Return:
        A :class:`numpy.ndarray` image of shape HWC with value in range [0, 255]
        or a :class:`torch.Tensor` image of shape 1CHW with value in range [0.0,
        1.0].
    """
    image = cv2.imread(str(path))  # BGR
    if to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if to_tensor:
        image = base.to_tensor(image=image, keepdim=False, normalize=normalize)
    return image


def read_video_ffmpeg(
    process,
    height   : int,
    width    : int,
    to_tensor: bool = False,
    normalize: bool = False,
) -> torch.Tensor | np.ndarray:
    """Read raw bytes from a video stream using :mod`ffmpeg`. Optionally,
    convert it to :class:`torch.Tensor` type of shape 1CHW.
    
    Args:
        process: The subprocess that manages :mod:`ffmpeg` instance.
        height: The height of the video frame.
        width: The width of the video.
        to_tensor: If True convert the image from :class:`numpy.ndarray` to
            :class:`torch.Tensor`. Defaults to False.
        normalize: If True, normalize the image to [0.0, 1.0]. Defaults to
            False.
    
    Return:
        A :class:`numpy.ndarray` image of shape HCW with value in range [0, 255]
        or a :class:`torch.Tensor` image of shape 1CHW with value in range
        [0,1].
    """
    # RGB24 == 3 bytes per pixel.
    img_size = height * width * 3
    in_bytes = process.stdout.read(img_size)
    if len(in_bytes) == 0:
        image = None
    else:
        if len(in_bytes) != img_size:
            raise ValueError()
        image = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )  # Numpy
        if to_tensor:
            image = base.to_tensor(
                image     = image,
                keepdim   = False,
                normalize = normalize
            )
    return image


class Loader(ABC):
    """The base class for all image and video loaders.
    
    Args:
        source: A data source. It can be a path to a single image file, a
            directory, a video file, or a stream. It can also be a path pattern.
        max_samples: The maximum number of datapoints from the given
            :param:`source` to process. Defaults to None.
        batch_size: The number of samples in a single forward pass. Defaults to
            1.
        to_rgb: If True, convert the image from BGR to RGB. Defaults to False.
        to_tensor: If True, convert the image from :class:`numpy.ndarray` to
            :class:`torch.Tensor`. Defaults to False.
        normalize: If True, normalize the image to [0.0, 1.0]. Defaults to True.
        verbose: Verbosity. Defaults to False.
    """
    
    def __init__(
        self,
        source     : pathlib.Path,
        max_samples: int | None = None,
        batch_size : int        = 1,
        to_rgb     : bool       = True,
        to_tensor  : bool       = False,
        normalize  : bool       = False,
        verbose    : bool       = False,
        *args, **kwargs
    ):
        super().__init__()
        self.source      = pathlib.Path(source)
        self.batch_size  = batch_size
        self.to_rgb      = to_rgb
        self.to_tensor   = to_tensor
        self.normalize   = normalize
        self.verbose     = verbose
        self.index       = 0
        self.max_samples = max_samples
        self.num_images  = 0
        self.init()
    
    def __iter__(self):
        """Return an iterator object starting at index 0."""
        self.reset()
        return self
    
    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return self.num_images
    
    @abstractmethod
    def __next__(self):
        pass
    
    def __del__(self):
        """Close."""
        self.close()
    
    def batch_len(self) -> int:
        """Return the number of batches."""
        return int(self.__len__() / self.batch_size)
    
    @abstractmethod
    def init(self):
        """Initialize the data source."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset and start over."""
        pass
    
    @abstractmethod
    def close(self):
        """Stop and release."""
        pass


class ImageLoader(Loader):
    """An image loader that retrieves and loads image(s) from a filepath,
    filepath pattern, or directory.
    
    Notes:
        We don't need to define the image shape since images in a directory can
        have different shapes.
    
    Args:
        source: A data source. It can be a filepath, a filepath pattern, or a
            directory.
        max_samples: The maximum number of datapoints from the given
            :param:`source` to process. Defaults to None.
        batch_size: The number of samples in a single forward pass. Defaults to
            1.
        to_rgb: If True, convert the image from BGR to RGB. Defaults to False.
        to_tensor: If True, convert the image from :class:`numpy.ndarray` to
            :class:`torch.Tensor`. Defaults to False.
        normalize: If True, normalize the image to [0.0, 1.0]. Defaults to True.
        verbose: Verbosity mode of video loader backend. Defaults to False.
    """
    
    def __init__(
        self,
        source     : pathlib.Path,
        max_samples: int | None = None,
        batch_size : int        = 1,
        to_rgb     : bool       = True,
        to_tensor  : bool       = False,
        normalize  : bool       = False,
        verbose    : bool       = False,
    ):
        self.images = []
        super().__init__(
            source      = source,
            max_samples = max_samples,
            batch_size  = batch_size,
            to_rgb      = to_rgb,
            to_tensor   = to_tensor,
            normalize   = normalize,
            verbose     = verbose
        )
    
    def __next__(self) -> tuple[torch.Tensor | np.ndarray, list, list, list]:
        """Load the next batch of images from the disk.
        
        Examples:
            >>> images = ImageLoader("cam_1.mp4")
            >>> for index, image in enumerate(images):
        
        Return:
            Images of shape HWC or NCHW.
            A list of image indexes
            A list of image files.
            A list of images' relative paths corresponding to data.
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
                rel_path = str(file).replace(str(self.source) + "/", "")
                image    = read_image(
                    path      = self.images[self.index],
                    to_rgb    = self.to_rgb,
                    to_tensor = self.to_tensor,
                    normalize = self.normalize,
                )
                images.append(image)
                indexes.append(self.index)
                files.append(file)
                rel_paths.append(rel_path)
                self.index += 1
            
            if self.to_tensor:
                images = torch.stack(images, dim=1)
                images = torch.squeeze(images, dim=0)
            else:
                images = np.hstack(images)
            return images, indexes, files, rel_paths
    
    def init(self):
        """Initialize the data source."""
        self.images = []
        if self.source.is_image_file():
            self.images = [self.source]
        elif self.source.is_dir():
            self.images = [i for i in self.source.rglob("*") if i.is_image_file()]
        elif isinstance(self.source, str):
            images = [pathlib.Path(i) for i in glob.glob(self.source)]
            images = [i for i in images if i.is_image_file()]
            self.images = images
        else:
            raise IOError(f"Error when listing image files.")
        
        if self.num_images == 0:
            self.num_images = len(self.images)
        if self.max_samples is not None and self.num_images > self.max_samples:
            self.num_images = self.max_samples
    
    def reset(self):
        """Reset and start over."""
        self.index = 0
    
    def close(self):
        """Stop and release."""
        pass


class VideoLoader(Loader, ABC):
    """The base class for all video loaders.
    
    Args:
        source: A data source. It can be a path to a single video file, a
            directory, or a stream. It can also be a path pattern.
        max_samples: The maximum number of datapoints from the given
            :param:`source` to process. Defaults to None.
        batch_size: The number of samples in a single forward pass. Defaults to
            1.
        to_rgb: If True, convert the image from BGR to RGB. Defaults to True.
        to_tensor: If True, convert the image from :class:`numpy.ndarray` to
            :class:`torch.Tensor`. Defaults to False.
        normalize: If True, normalize the image to [0.0, 1.0]. Defaults to
            False.
        verbose: Verbosity mode of video loader backend. Defaults to False.
    """
    
    def __init__(
        self,
        source     : pathlib.Path,
        max_samples: int | None = None,
        batch_size : int        = 1,
        to_rgb     : bool       = True,
        to_tensor  : bool       = False,
        normalize  : bool       = False,
        verbose    : bool       = False,
        *args, **kwargs
    ):
        self.frame_rate = 0
        super().__init__(
            source      = source,
            max_samples = max_samples,
            batch_size  = batch_size,
            to_rgb      = to_rgb,
            to_tensor   = to_tensor,
            normalize   = normalize,
            verbose     = verbose,
            *args, **kwargs
        )
    
    @property
    @abstractmethod
    def fourcc(self) -> str:
        """Return the 4-character code of codec."""
        pass
    
    @property
    @abstractmethod
    def fps(self) -> int:
        """Return the frame rate."""
        pass
    
    @property
    @abstractmethod
    def frame_height(self) -> int:
        """Return the height of the frames in the video stream."""
        pass
    
    @property
    @abstractmethod
    def frame_width(self) -> int:
        """Return the width of the frames in the video stream."""
        pass
    
    @property
    def is_stream(self) -> bool:
        """Return True if it is a video stream, i.e, unknown
        :attr:`frame_count`.
        """
        return self.num_images == -1
    
    @property
    def shape(self) -> list[int]:
        """Return the shape of the frames in the video stream in HWC format."""
        return [self.frame_height, self.frame_width, 3]


class VideoLoaderCV(VideoLoader):
    """A video loader that retrieves and loads frame(s) from a video or a stream
    using :mod:`cv2`.

    Args:
        source: A data source. It can be a filepath, a filepath pattern, or a
            directory.
        max_samples: The maximum number of datapoints from the given
            :param:`source` to process. Defaults to None.
        batch_size: The number of samples in a single forward pass. Defaults to
            1.
        to_rgb: If True, convert the image from BGR to RGB. Defaults to True.
        to_tensor: If True, convert the image from :class:`numpy.ndarray` to
            :class:`torch.Tensor`. Defaults to False.
        normalize: If True, normalize the image to [0.0, 1.0]. Defaults to
            False.
        api_preference: Preferred Capture API backends to use. It can be used to
            enforce a specific reader implementation. Two most used options are:
            [cv2.CAP_ANY=0, cv2.CAP_FFMPEG=1900]. See more:
            https://docs.opencv.org/4.5.5/d4/d15/group__videoio__flags__base
            .htmlggaeb8dd9c89c10a5c63c139bf7c4f5704da7b235a04f50a444bc2dc72f5ae394aaf
            Defaults to cv2.CAP_FFMPEG.
        verbose: Verbosity mode of video loader backend. Defaults to False.
    """
    
    def __init__(
        self,
        source        : pathlib.Path,
        max_samples   : int | None = None,
        batch_size    : int        = 1,
        to_rgb        : bool       = True,
        to_tensor     : bool       = False,
        normalize     : bool       = False,
        api_preference: int        = cv2.CAP_FFMPEG,
        verbose       : bool       = False,
        *args, **kwargs
    ):
        self.api_preference = api_preference
        self.video_capture  = None
        super().__init__(
            source      = source,
            max_samples = max_samples,
            batch_size  = batch_size,
            to_rgb      = to_rgb,
            to_tensor   = to_tensor,
            normalize   = normalize,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def __next__(self) -> tuple[torch.Tensor | np.ndarray, list, list, list]:
        """Load the next batch of frames in the video.
        
        Return:
            Image of shape HWC or NCHW.
            A list of frames indexes.
            A list of frames files.
            A list of frames' relative paths corresponding to data.
        """
        if not self.is_stream and self.index >= self.num_images:
            self.close()
            raise StopIteration
        else:
            images    = []
            indexes   = []
            files     = []
            rel_paths = []
            
            for i in range(self.batch_size):
                if not self.is_stream and self.index >= self.num_images:
                    break
                
                if isinstance(self.video_capture, cv2.VideoCapture):
                    ret_val, image = self.video_capture.read()
                    rel_path       = self.source.name
                else:
                    raise RuntimeError(
                        f"'video_capture' has not been initialized."
                    )
                
                if image is None:
                    continue
                if self.to_rgb:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if self.to_tensor:
                    image = base.to_tensor(
                        image     = image,
                        keepdim   = False,
                        normalize = self.normalize
                    )
                images.append(image)
                indexes.append(self.index)
                files.append(self.source)
                rel_paths.append(rel_path)
                self.index += 1
            
            if self.to_tensor:
                images = torch.stack(images, dim=1)
                images = torch.squeeze(images, dim=0)
            else:
                images = np.hstack(images)
            return images, indexes, files, rel_paths
    
    @property
    def format(self):  # Flag=8
        """Return the format of the Mat objects (see Mat::type()) returned by
        VideoCapture::retrieve(). Set value -1 to fetch undecoded RAW video
        streams (as Mat 8UC1).
        """
        return self.video_capture.get(cv2.CAP_PROP_FORMAT)
    
    @property
    def fourcc(self) -> str:  # Flag=6
        """Return the 4-character code of codec."""
        return str(self.video_capture.get(cv2.CAP_PROP_FOURCC))
    
    @property
    def fps(self) -> int:  # Flag=5
        """Return the frame rate."""
        return int(self.video_capture.get(cv2.CAP_PROP_FPS))
    
    @property
    def frame_count(self) -> int:  # Flag=7
        """Return the number of frames in the video file."""
        if isinstance(self.video_capture, cv2.VideoCapture):
            return int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        elif isinstance(self.video_capture, list):
            return len(self.video_capture)
        else:
            return -1
    
    @property
    def frame_height(self) -> int:  # Flag=4
        """Return the height of the frames in the video stream."""
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    @property
    def frame_width(self) -> int:  # Flag=3
        """Return the width of the frames in the video stream."""
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    @property
    def mode(self):  # Flag=10
        """Return the backend-specific value indicating the current capture
        mode.
        """
        return self.video_capture.get(cv2.CAP_PROP_MODE)
    
    @property
    def pos_avi_ratio(self) -> int:  # Flag=2
        """Return the relative position of the video file: 0=start of the film,
        1=end of the film.
        """
        return int(self.video_capture.get(cv2.CAP_PROP_POS_AVI_RATIO))
    
    @property
    def pos_msec(self) -> int:  # Flag=0
        """Return the current position of the video file in milliseconds."""
        return int(self.video_capture.get(cv2.CAP_PROP_POS_MSEC))
    
    @property
    def pos_frames(self) -> int:  # Flag=1
        """Return the 0-based index of the frame to be decoded/captured next."""
        return int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
    
    def init(self):
        """Initialize the data source."""
        source = pathlib.Path(self.source)
        if source.is_video_file():
            self.video_capture = cv2.VideoCapture(
                str(source),
                self.api_preference
            )
            num_images = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_rate = int(self.video_capture.get(cv2.CAP_PROP_FPS))
        elif source.is_video_stream():
            self.video_capture = cv2.VideoCapture(
                str(source),
                self.api_preference
            )  # stream
            num_images = -1
            self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, self.batch_size)
        else:
            raise IOError(f"Error when reading input stream or video file!")
        
        if self.num_images == 0:
            self.num_images = num_images
        if self.max_samples is not None and self.num_images > self.max_samples:
            self.num_images = self.max_samples
    
    def reset(self):
        """Reset and start over."""
        self.index = 0
        if isinstance(self.video_capture, cv2.VideoCapture):
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.index)
    
    def close(self):
        """Stop and release the current attr:`video_capture` object."""
        if isinstance(self.video_capture, cv2.VideoCapture) \
            and self.video_capture:
            self.video_capture.release()


class VideoLoaderFFmpeg(VideoLoader):
    """A video loader that retrieves and loads frame(s) from a video or a stream
    using :mod:`ffmpeg`.
    
    Args:
        source: A data source. It can be a filepath, a filepath pattern, or a
            directory.
        max_samples: The maximum number of datapoints from the given
            :param:`source` to process. Defaults to None.
        batch_size: The number of samples in a single forward pass. Defaults to
            1.
        to_rgb: If True, convert the image from BGR to RGB. Defaults to True.
        to_tensor: If True, convert the image from :class:`numpy.ndarray` to
            :class:`torch.Tensor`. Defaults to False.
        normalize: If True, normalize the image to [0.0, 1.0]. Defaults to
            False.
        verbose: Verbosity mode of video loader backend. Defaults to False.
        kwargs: Any supplied kwargs are passed to :mod:`ffmpeg` verbatim.
        
    References:
        https://github.com/kkroening/ffmpeg-python/tree/master/examples
    """
    
    def __init__(
        self,
        source     : pathlib.Path,
        max_samples: int  = 0,
        batch_size : int  = 1,
        to_rgb     : bool = True,
        to_tensor  : bool = False,
        normalize  : bool = False,
        verbose    : bool = False,
        *args, **kwargs
    ):
        self.ffmpeg_cmd     = None
        self.ffmpeg_process = None
        self.ffmpeg_kwargs  = kwargs
        self.video_info     = None
        super().__init__(
            source      = source,
            max_samples = max_samples,
            batch_size  = batch_size,
            to_rgb      = to_rgb,
            to_tensor   = to_tensor,
            normalize   = normalize,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def __next__(self) -> tuple[torch.Tensor | np.ndarray, list, list, list]:
        """Load the next batch of frames in the video.
        
        Return:
            Image of shape HWC or NCHW.
            A list of frames indexes.
            A list of frames files.
            A list of frames' relative paths corresponding to data.
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
                    rel_path = self.source.name
                else:
                    raise RuntimeError(f"video_capture has not been initialized.")
                
                if image is None:
                    continue
                if self.to_rgb:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if self.to_tensor:
                    image = base.to_tensor(
                        image     = image,
                        keepdim   = False,
                        normalize = self.normalize
                    )
                images.append(image)
                indexes.append(self.index)
                files.append(self.source)
                rel_paths.append(rel_path)
                self.index += 1
            
            if self.to_tensor:
                images = torch.stack(images, dim=1)
                images = torch.squeeze(images, dim=0)
            else:
                images = np.hstack(images)
            return images, indexes, files, rel_paths
    
    @property
    def fourcc(self) -> str:
        """Return the 4-character code of codec."""
        return self.video_info["codec_name"]
    
    @property
    def fps(self) -> int:
        """Return the frame rate."""
        return int(self.video_info["avg_frame_rate"].split("/")[0])
    
    @property
    def frame_count(self) -> int:
        """Return the number of frames in the video file."""
        if self.source.is_video_file():
            return int(self.video_info["nb_frames"])
        else:
            return -1
    
    @property
    def frame_width(self) -> int:
        """Return the width of the frames in the video stream."""
        return int(self.video_info["width"])
    
    @property
    def frame_height(self) -> int:
        """Return the height of the frames in the video stream."""
        return int(self.video_info["height"])
    
    def init(self):
        """Initialize ffmpeg cmd."""
        source = str(self.source)
        probe  = ffmpeg.probe(source, **self.ffmpeg_kwargs)
        self.video_info = next(
            s for s in probe["streams"] if s["codec_type"] == "video"
        )
        if self.verbose:
            self.ffmpeg_cmd = (
                ffmpeg
                .input(source, **self.ffmpeg_kwargs)
                .output("pipe:", format="rawvideo", pix_fmt="rgb24")
                .compile()
            )
        else:
            self.ffmpeg_cmd = (
                ffmpeg
                .input(source, **self.ffmpeg_kwargs)
                .output("pipe:", format="rawvideo", pix_fmt="rgb24")
                .global_args("-loglevel", "quiet")
                .compile()
            )
    
    def reset(self):
        """Reset and start over."""
        self.close()
        self.index = 0
        if self.ffmpeg_cmd:
            self.ffmpeg_process = subprocess.Popen(
                self.ffmpeg_cmd,
                stdout  = subprocess.PIPE,
                bufsize = 10 ** 8
            )
    
    def close(self):
        """Stop and release the current :attr:`ffmpeg_process`."""
        if self.ffmpeg_process and self.ffmpeg_process.poll() is not None:
            # os.killpg(os.getpgid(self.ffmpeg_process.pid), signal.SIGTERM)
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()
            self.ffmpeg_process = None
            # raise StopIteration


# endregion


# region Write

def write_image_cv(
    image      : torch.Tensor | np.ndarray,
    dirpath    : pathlib.Path,
    name       : str,
    prefix     : str  = "",
    extension  : str  = ".png",
    denormalize: bool = True
):
    """Write an image to a directory using :mod:`cv2`.
    
    Args:
        image: An image to write.
        dirpath: A directory to write the image to.
        name: An image's name.
        prefix: A prefix to add to the :param:`name`.
        extension: An extension of the image file. Defaults to '.png'.
        denormalize: If True, convert the image to [0, 255]. Defaults to True.
    """
    # Convert image
    if isinstance(image, torch.Tensor):
        image = base.to_nparray(image=image, keepdim=True, denormalize=denormalize)
    image = base.to_channel_last(image=image)
    if 2 <= image.ndim <= 3:
        raise ValueError(
            f"img's number of dimensions must be between 2 and 3, but got "
            f"{image.ndim}."
        )
    # Write image
    dirpath   = pathlib.Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    name      = pathlib.Path(name)
    stem      = name.stem
    extension = extension  # name.suffix
    extension = f"{name.suffix}" if extension == "" else extension
    extension = f".{extension}" if "." not in extension else extension
    stem      = f"{prefix}_{stem}" if prefix != "" else stem
    name      = f"{stem}{extension}"
    filepath  = dirpath / name
    cv2.imwrite(image, str(filepath))


def write_image_torch(
    image      : torch.Tensor | np.ndarray,
    dirpath    : pathlib.Path,
    name       : str,
    prefix     : str  = "",
    extension  : str  = ".png",
    denormalize: bool = True
):
    """Write an image to a directory.
    
    Args:
        image: An image to write.
        dirpath: A directory to write the image to.
        name: An image's name.
        prefix: A prefix to add to the :param:`name`.
        extension: An extension of the image file. Defaults to '.png'.
        denormalize: If True, convert the image to [0, 255]. Defaults to True.
    """
    # Convert image
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
        image = base.to_channel_first(image=image)
    image = base.denormalize_image(image=image) if denormalize else image
    image = image.to(torch.uint8)
    image = image.cpu()
    if 2 <= image.ndim <= 3:
        raise ValueError(
            f"img's number of dimensions must be between 2 and 3, but got "
            f"{image.ndim}."
        )
    # Write image
    dirpath   = pathlib.Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    name      = pathlib.Path(name)
    stem      = name.stem
    extension = extension  # name.suffix
    extension = f"{name.suffix}" if extension == "" else extension
    extension = f".{extension}" if "." not in extension else extension
    stem      = f"{prefix}_{stem}" if prefix != "" else stem
    name      = f"{stem}{extension}"
    filepath  = dirpath / name
    if extension in [".jpg", ".jpeg"]:
        torchvision.io.image.write_jpeg(input=image, filename=str(filepath))
    elif extension in [".png"]:
        torchvision.io.image.write_png(input=image, filename=str(filepath))


def write_images_cv(
    images     : list[torch.Tensor | np.ndarray],
    dirpath    : pathlib.Path,
    names      : list[str],
    prefixes   : list[str] = "",
    extension  : str       = ".png",
    denormalize: bool      = True
):
    """Write a list of images to a directory using :mod:`cv2`.
   
    Args:
        images: A list of 3-D images.
        dirpath: A directory to write the images to.
        names: A list of images' names.
        prefixes: A prefix to add to the :param:`names`.
        extension: An extension of image files. Defaults to ".png".
        denormalize: If True, convert image to [0, 255]. Defaults to True.
    """
    if isinstance(names, str):
        names = [names for _ in range(len(images))]
    if isinstance(prefixes, str):
        prefixes = [prefixes for _ in range(len(prefixes))]
    if not len(images) == len(names):
        raise ValueError(
            f"The length of images and names must be the same, but got "
            f"{len(images)} and {len(names)}."
        )
    if not len(images) == len(prefixes):
        raise ValueError(
            f"The length of images and prefixes must be the same, but got "
            f"{len(images)} and {len(prefixes)}."
        )
    num_jobs = multiprocessing.cpu_count()
    joblib.Parallel(n_jobs=num_jobs)(
        joblib.delayed(write_image_cv)(
            image, dirpath, names[i], prefixes[i], extension, denormalize
        )
        for i, image in enumerate(images)
    )


def write_images_torch(
    images     : Sequence[torch.Tensor | np.ndarray],
    dirpath    : pathlib.Path,
    names      : list[str],
    prefixes   : list[str] = "",
    extension  : str       = ".png",
    denormalize: bool      = True
):
    """Write a list of images to a directory using :mod:`torchvision`.
   
    Args:
        images: A list of 3-D images.
        dirpath: A directory to write the images to.
        names: A list of images' names.
        prefixes: A prefix to add to the :param:`names`.
        extension: An extension of image files. Defaults to ".png".
        denormalize: If True, convert image to [0, 255]. Defaults to True.
    """
    if isinstance(names, str):
        names = [names for _ in range(len(images))]
    if isinstance(prefixes, str):
        prefixes = [prefixes for _ in range(len(prefixes))]
    if not len(images) == len(names):
        raise ValueError(
            f"The length of images and names must be the same, but got "
            f"{len(images)} and {len(names)}."
        )
    if not len(images) == len(prefixes):
        raise ValueError(
            f"The length of images and prefixes must be the same, but got "
            f"{len(images)} and {len(prefixes)}."
        )
    num_jobs = multiprocessing.cpu_count()
    joblib.Parallel(n_jobs=num_jobs)(
        joblib.delayed(write_image_torch)(
            image, dirpath, names[i], prefixes[i], extension, denormalize
        )
        for i, image in enumerate(images)
    )


def write_video_ffmpeg(
    process,
    image      : torch.Tensor | np.ndarray,
    denormalize: bool = True
):
    """Write an image to a video file using :mod:`ffmpeg`.

    Args:
        process: A subprocess that manages ffmpeg.
        image: A frame/image of shape 1CHW.
        denormalize: If True, convert image to [0, 255]. Defaults to True.
    """
    if isinstance(image, np.ndarray):
        if base.is_normalized_image(image=image):
            image = base.denormalize_image(image=image)
        if base.is_channel_first(image=image):
            image = base.to_channel_last(image=image)
    elif isinstance(image, torch.Tensor):
        image = base.to_nparray(
            image= image,
            keepdim     = False,
            denormalize = denormalize
        )
    else:
        raise ValueError(
            f"image must be a torch.Tensor or numpy.ndarray, but got "
            f"{type(image)}."
        )
    process.stdin.write(
        image
        .astype(np.uint8)
        .tobytes()
    )


class Writer(ABC):
    """The base class for all image and video writers that save images to a
    destination directory.

    Args:
        destination: A directory to save images.
        image_size: An output image size of shape HW. Defaults to [480, 640].
        denormalize: If True, convert image to [0, 255]. Defaults to True.
        verbose: Verbosity. Defaults to False.
    """
    
    def __init__(
        self,
        destination: pathlib.Path,
        image_size   : int | list[int] = [480, 640],
        denormalize: bool = True,
        verbose    : bool = False,
        *args, **kwargs
    ):
        super().__init__()
        self.dst         = pathlib.Path(destination)
        self.img_size    = base.get_hw(size=image_size)
        self.denormalize = denormalize
        self.verbose     = verbose
        self.index       = 0
        self.init()
    
    def __len__(self) -> int:
        """Return the number frames of already written frames."""
        return self.index
    
    def __del__(self):
        """Close."""
        self.close()
    
    @abstractmethod
    def init(self):
        """Initialize the output handler."""
        pass
    
    @abstractmethod
    def close(self):
        """Close."""
        pass
    
    @abstractmethod
    def write(
        self,
        image      : torch.Tensor | np.ndarray,
        path       : pathlib.Path | None = None,
        denormalize: bool = True
    ):
        """Write an image to :attr:`dst`.

        Args:
            image: An image.
            path: An image filepath with an extension. Defaults to None.
            denormalize: If True, convert image to [0, 255]. Defaults to True.
        """
        pass
    
    @abstractmethod
    def write_batch(
        self,
        images     : list[torch.Tensor  | np.ndarray],
        paths      : list[pathlib.Path] | None = None,
        denormalize: bool = True
    ):
        """Write a batch of images to :attr:`dst`.

        Args:
            images: A list of images.
            paths: A list of image filepaths with extensions. Defaults to None.
            denormalize: If True, convert image to [0, 255]. Defaults to True.
        """
        pass


class ImageWriter(Writer):
    """An image writer that saves images to a destination directory.

    Args:
        destination: A directory to save images.
        image_size: A desired output size of shape HW. This is used to
        reshape the
            input. Defaults to [480, 640].
        denormalize: If True, convert image to [0, 255]. Defaults to True.
        extension: The extension of the file to be saved. Defaults to '.png'.
        verbose: Verbosity. Defaults to False.
    """
    
    def __init__(
        self,
        destination: pathlib.Path,
        image_size : int | list[int] = [480, 640],
        extension  : str  = ".png",
        denormalize: bool = True,
        verbose    : bool = False,
        *args, **kwargs
    ):
        super().__init__(
            destination = destination,
            image_size  = image_size,
            denormalize = denormalize,
            verbose     = verbose,
            *args, **kwargs
        )
        self.extension = extension
    
    def __len__(self) -> int:
        """Return the number frames of already written frames (in other words,
        the index of the last item in the list).
        """
        return self.index
    
    def init(self):
        """Initialize the output destination."""
        pass
    
    def close(self):
        """Close."""
        pass
    
    def write(
        self,
        image      : torch.Tensor | np.ndarray,
        path       : pathlib.Path | None = None,
        denormalize: bool = True
    ):
        """Write an image to :attr:`dst`.

        Args:
            image: An image.
            path: An image filepath with an extension. Defaults to None.
            denormalize: If True, convert image to [0, 255]. Defaults to True.
        """
        if isinstance(path, pathlib.Path):
            path = self.dst / f"{path.stem}{self.extension}"
        elif isinstance(path, str):
            path = self.dst / path
        else:
            raise ValueError(f"'file' must be given.")
        path = pathlib.Path(path)
        write_image_cv(
            image       = image,
            dirpath= path.parent,
            name        = path.name,
            extension   = self.extension,
            denormalize = denormalize or self.denormalize
        )
        self.index += 1
    
    def write_batch(
        self,
        images     : list[torch.Tensor | np.ndarray],
        paths      : list[pathlib.Path] | None = None,
        denormalize: bool = True,
    ):
        """Write a batch of images to :attr:`dst`.

        Args:
            images: A list of 3-D images.
            paths: A list of image filepaths with extensions. Defaults to None.
            denormalize: If True, convert image to [0, 255]. Defaults to True.
        """
        if paths is None:
            paths = [None for _ in range(len(images))]
        for image, file in zip(images, paths):
            self.write(
                image       = image,
                path= file,
                denormalize = denormalize or self.denormalize,
            )


class VideoWriter(Writer, ABC):
    """The base class for all video writers.

    Args:
        destination: A directory to save images.
        image_size: A desired output size of shape HW. This is used to
        reshape the
            input. Defaults to [480, 640].
        frame_rate: A frame rate of the output video. Defaults to 10.
        save_image: If True save each image separately. Defaults to False.
        denormalize: If True, convert image to [0, 255]. Defaults to True.
        verbose: Verbosity. Defaults to False.
    """
    
    def __init__(
        self,
        destination: pathlib.Path,
        image_size : int | list[int] = [480, 640],
        frame_rate : float = 10,
        save_image : bool  = False,
        denormalize: bool  = True,
        verbose    : bool  = False,
        *args, **kwargs
    ):
        self.frame_rate = frame_rate
        self.save_image = save_image
        super().__init__(
            destination = destination,
            image_size  = image_size,
            denormalize = denormalize,
            verbose     = verbose,
            *args, **kwargs
        )


class VideoWriterCV(VideoWriter):
    """A video writer that writes images to a video file using :mod:`cv2`.

    Args:
        destination: A destination directory to save images.
        image_size: A desired output size of shape HW. This is used to
        reshape the
            input. Defaults to [480, 640].
        frame_rate: A frame rate of the output video. Defaults to 10.
        fourcc: Video codec. One of: ["mp4v", "xvid", "mjpg", "wmv"]. Defaults
            to ".mp4v".
        save_image: If True save each image separately. Defaults to False.
        denormalize: If True, convert image to [0, 255]. Defaults to True.
        verbose: Verbosity. Defaults to False.
    """
    
    def __init__(
        self,
        destination: pathlib.Path,
        image_size : list[int] = [480, 640],
        frame_rate : float = 30,
        fourcc     : str   = "mp4v",
        save_image : bool  = False,
        denormalize: bool  = True,
        verbose    : bool  = False,
        *args, **kwargs
    ):
        self.fourcc       = fourcc
        self.video_writer = None
        super().__init__(
            destination = destination,
            image_size  = image_size,
            frame_rate  = frame_rate,
            save_image  = save_image,
            denormalize = denormalize,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def init(self):
        """Initialize the output destination."""
        if self.dst.is_dir():
            video_file = self.dst / f"result.mp4"
        else:
            video_file = self.dst.parent / f"{self.dst.stem}.mp4"
        video_file.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        self.video_writer = cv2.VideoWriter(
            filename  = str(video_file),
            fourcc    = fourcc,
            fps       = float(self.frame_rate),
            frameSize =self.img_size[::-1],  # Must be in [W, H]
            isColor   = True
        )
        
        if self.video_writer is None:
            raise FileNotFoundError(
                f"Cannot create video file at {video_file}."
            )
    
    def close(self):
        """Close the :attr:`video_writer`."""
        if self.video_writer:
            self.video_writer.release()
    
    def write(
        self,
        image      : torch.Tensor | np.ndarray,
        path       : pathlib.Path | None = None,
        denormalize: bool = True
    ):
        """Write an image to :attr:`dst`.

        Args:
            image: An image.
            path: An image filepath with an extension. Defaults to None.
            denormalize: If True, convert image to [0, 255]. Defaults to True.
        """
        if self.save_image:
            write_image_cv(
                image       = image,
                dirpath= self.dst,
                name        = f"{pathlib.Path(path).stem}.png",
                prefix      = "",
                extension   =".png",
                denormalize = denormalize or self.denormalize
            )
        
        image = base.to_nparray(
            image       = image,
            keepdim     = False,
            denormalize = denormalize or self.denormalize,
        )
        # IMPORTANT: Image must be in a BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        self.video_writer.write(image)
        self.index += 1
    
    def write_batch(
        self,
        images     : list[torch.Tensor | np.ndarray],
        paths      : list[pathlib.Path] | None = None,
        denormalize: bool  = True
    ):
        """Write a batch of images to :attr:`dst`.

        Args:
            images: A list of images.
            paths: A list of image filepaths with extensions. Defaults to None.
            denormalize: If True, convert image to [0, 255]. Defaults to True.
        """
        if paths is None:
            paths = [None for _ in range(len(images))]
        for image, file in zip(images, paths):
            self.write(image=image, path=file, denormalize=denormalize)


class VideoWriterFFmpeg(VideoWriter):
    """A video writer that writes images to a video file using :mod:`ffmpeg`.

    Args:
        destination: A destination directory to save images.
        image_size: A desired output size of shape HW. This is used to reshape
            the input. Defaults to [480, 640].
        frame_rate: A frame rate of the output video. Defaults to 10.
        pix_fmt: A video codec. Defaults to 'yuv420p'.
        save_image: If True save each image separately. Defaults to False.
        denormalize: If True, convert image to [0, 255]. Defaults to True.
        verbose: Verbosity. Defaults to False.
        kwargs: Any supplied kwargs are passed to :mod:`ffmpeg` verbatim.
    """
    
    def __init__(
        self,
        destination: pathlib.Path,
        image_size : int | list[int] = [480, 640],
        frame_rate : float = 10,
        pix_fmt    : str   = "yuv420p",
        save_image : bool  = False,
        denormalize: bool  = True,
        verbose    : bool  = False,
        *args, **kwargs
    ):
        self.pix_fmt        = pix_fmt
        self.ffmpeg_process = None
        self.ffmpeg_kwargs  = kwargs
        super().__init__(
            destination = destination,
            image_size  = image_size,
            frame_rate  = frame_rate,
            save_image  = save_image,
            denormalize = denormalize,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def init(self):
        """Initialize the output destination."""
        if self.dst.is_dir():
            video_file = self.dst / f"result.mp4"
        else:
            video_file = self.dst.parent / f"{self.dst.stem}.mp4"
        video_file.parent.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            self.ffmpeg_process = (
                ffmpeg
                .input(
                    "pipe:",
                    format  = "rawvideo",
                    pix_fmt = "rgb24",
                    s       = "{}x{}".format(self.img_size[1], self.img_size[0])
                )
                .output(
                    filename = str(video_file),
                    pix_fmt  = self.pix_fmt,
                    **self.ffmpeg_kwargs
                )
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
        else:
            self.ffmpeg_process = (
                ffmpeg
                .input(
                    "pipe:",
                    format  = "rawvideo",
                    pix_fmt = "rgb24",
                    s       = "{}x{}".format(self.img_size[1], self.img_size[0])
                )
                .output(
                    filename = str(video_file),
                    pix_fmt  = self.pix_fmt,
                    **self.ffmpeg_kwargs
                )
                .global_args("-loglevel", "quiet")
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
    
    def close(self):
        """Stop and release the current :attr:`ffmpeg_process`."""
        if self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()
            self.ffmpeg_process = None
    
    def write(
        self,
        image      : torch.Tensor | np.ndarray,
        path       : pathlib.Path | None = None,
        denormalize: bool = True
    ):
        """Write an image to :attr:`dst`.

        Args:
            image: An image.
            path: An image filepath with an extension. Defaults to None.
            denormalize: If True, convert image to [0, 255]. Defaults to True.
        """
        if self.save_image:
            assert isinstance(path, pathlib.Path)
            write_image_cv(
                image       = image,
                dirpath= self.dst,
                name        = f"{pathlib.Path(path).stem}.png",
                prefix      = "",
                extension   =".png",
                denormalize = denormalize or self.denormalize
            )
        
        write_video_ffmpeg(
            process     = self.ffmpeg_process,
            image       = image,
            denormalize = denormalize or self.denormalize
        )
        self.index += 1
    
    def write_batch(
        self,
        images     : list[torch.Tensor | np.ndarray],
        paths      : list[pathlib.Path] | None = None,
        denormalize: bool = True,
    ):
        """Write a batch of images to :attr:`dst`.

        Args:
            images: A list of images.
            paths: A list of image filepaths with extensions. Defaults to None.
            denormalize: If True, convert image to [0, 255]. Defaults to True.
        """
        if paths is None:
            paths = [None for _ in range(len(images))]
        for image, file in zip(images, paths):
            self.write(image=image, path=file, denormalize=denormalize)

# endregion
