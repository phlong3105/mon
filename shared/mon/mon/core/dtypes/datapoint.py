#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines base classes and data types for handling tensor-like objects
with device management capabilities.
"""

__all__ = [
    "BaseTensorOrArray",
    "Probs",
]

import abc
from typing import Union

import numpy as np
import torch


# ----- Base -----
class BaseTensorOrArray(abc.ABC):
    """Base tensor class with additional methods for easy manipulation and
    device handling.

    This class provides a foundation for tensor-like objects with device
    management capabilities, supporting both PyTorch tensors and NumPy arrays.
    It includes methods for moving data between devices and converting between
    tensor types.

    Args:
        data: Input data as a ``torch.Tensor`` or ``numpy.ndarray``.
            Can be ``None`` if not loaded yet (lazy loading).
        orig_shape: Original shape of the image as a tuple of :math:`(H, W)`.
            Can be ``None`` if not known (lazy loading).

    Examples:
        >>> import torch
        >>> data        = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> orig_shape  = (720, 1280)
        >>> base_tensor = BaseTensorOrArray(data, orig_shape)
        >>> cpu_tensor  = base_tensor.cpu()
        >>> numpy_array = base_tensor.numpy()
        >>> gpu_tensor  = base_tensor.cuda()
    """

    def __init__(self, data: Union[torch.Tensor, np.ndarray], orig_shape: tuple[int, int]):
        self._data       = data
        self._orig_shape = orig_shape

    def __len__(self) -> int:
        """Return the number of elements in the first dimension of the data tensor.

        Examples:
            >>> data        = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensorOrArray(data, orig_shape=(720, 1280))
            >>> len(base_tensor)
            2
        """
        return len(self.data)

    def __getitem__(self, idx: int | list[int] | torch.Tensor):
        """Returns a new ``BaseTensor`` instance containing the specified indexed
        elements of the data tensor.

        Args:
            idx: Index or indices to select from the data tensor.

        Returns:
            A new ``BaseTensor`` instance containing the indexed data.

        Examples:
            >>> data        = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensorOrArray(data, orig_shape=(720, 1280))
            >>> result      = base_tensor[0]  # Select the first row
            >>> print(result.data)
            tensor([1, 2, 3])
        """
        return self.__class__(self.data[idx], self.orig_shape)

    @property
    def data(self):
        """Returns the data if already loaded, otherwise calls the ``load()``
        method.

        This property allows for lazy loading of the data to avoid unnecessary
        memory usage (e.g., when the data is large and not immediately needed).
        """
        return self._data if self._data is not None else self.load()

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the current shape of the underlying data tensor.

        Examples:
            >>> data = torch.rand(100, 4)
            >>> base_tensor = BaseTensorOrArray(data, orig_shape=(720, 1280))
            >>> print(base_tensor.shape)
            (100, 4)
        """
        return self.data.shape

    @property
    def orig_shape(self) -> tuple[int, ...]:
        """Returns the original shape of the data tensor.

        Examples:
            >>> data        = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape  = (720, 1280)
            >>> base_tensor = BaseTensorOrArray(data, orig_shape)
            >>> print(base_tensor.orig_shape)
            (720, 1280)
        """
        return self._orig_shape

    def cpu(self):
        """Return a new ``BaseTensor`` object with the data tensor moved to CPU
        memory.

        Examples:
            >>> data        = torch.tensor([[1, 2, 3], [4, 5, 6]]).cuda()
            >>> base_tensor = BaseTensorOrArray(data, orig_shape=(720, 1280))
            >>> cpu_tensor  = base_tensor.cpu()
            >>> isinstance(cpu_tensor, BaseTensorOrArray)
            True
            >>> cpu_tensor.data.device
            device(type='cpu')
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def cuda(self):
        """Returns a new ``BaseTensor`` instance with the data moved to GPU
        memory if it is not already an ``numpy.ndarray`` array, otherwise returns
        self.

        Examples:
            >>> import torch
            >>> from ultralytics.engine.results import BaseTensor
            >>> data        = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> base_tensor = BaseTensor(data, orig_shape=(720, 1280))
            >>> gpu_tensor  = base_tensor.cuda()
            >>> print(gpu_tensor.data.device)
            cuda:0
        """
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def numpy(self):
        """Returns an ``numpy.ndarray`` array containing the same data as the
        original tensor.

        Examples:
            >>> data        = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> orig_shape  = (720, 1280)
            >>> base_tensor = BaseTensorOrArray(data, orig_shape)
            >>> numpy_array = base_tensor.numpy()
            >>> print(type(numpy_array))
            <class 'numpy.ndarray'>
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def to(self, *args, **kwargs):
        """Returns a copy of the tensor with the specified device and dtype.

        Args:
            *args: Variable length argument list to be passed to ``torch.Tensor.to()``.
            **kwargs: Arbitrary keyword arguments to be passed to ``torch.Tensor.to()``.

        Returns:
            A new ``BaseTensor`` instance with the data moved to the specified
            device and/or dtype.

        Examples:
            >>> base_tensor    = BaseTensorOrArray(torch.randn(3, 4), orig_shape=(480, 640))
            >>> cuda_tensor    = base_tensor.to("cuda")
            >>> float16_tensor = base_tensor.to(dtype=torch.float16)
        """
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def load(self):
        """Loads the data into memory.

        This method should be overridden in subclasses to implement specific
        loading logic.

        Returns:
            The loaded data as a ``torch.Tensor`` or ``numpy.ndarray``.

        Raises:
            NotImplementedError: If not overridden in a subclass.
        """
        raise NotImplementedError("Subclasses must implement the load method.")


# ----- Basic Datapoints -----
class Probs(BaseTensorOrArray):
    """A class for storing and manipulating classification probabilities.

    This class extends ``BaseTensor`` and provides methods for accessing and
    manipulating classification probabilities, including top-1 and top-5 predictions.

    Args:
        data: Input data as a ``torch.Tensor`` or ``numpy.ndarray`` of shape
            :math:`(C)` where ``C`` is the number of classes. If given an ``int``
            representing the class ID, it will convert it to a probability tensor
            in :math:`[classes]` format with the class ID set to ``1.0`` and
            others set to ``0.0``.
        num_classes: Total number of classes. Default: ``None``.
        orig_shape: Original shape of the image a tuple of :math:`(H, W)`.
            Default: ``None``.
    """

    def __init__(
        self,
        data       : Union[torch.Tensor, np.ndarray] | int,
        num_classes: int = None,
        orig_shape : tuple[int, int] = None,
    ):
        if isinstance(data, int):
            if num_classes is None:
                raise ValueError("If ``data`` is an int, ``num_classes`` must be provided.")
            else:
                data = Probs.to_probs(data, num_classes)
        if not isinstance(data, Union[torch.Tensor, np.ndarray]):
            raise TypeError(f"``data`` must be a torch.Tensor or numpy.ndarray, got {type(data)}.")

        super().__init__(data, orig_shape)

    @property
    def top1(self) -> int:
        """Returns the index of the class with the highest probability."""
        if isinstance(self.data, torch.Tensor):
            return int(self.data.argmax())
        else:
            return int(np.argmax(self.data))

    @property
    def top5(self) -> list[int]:
        """Returns the indices of the top 5 classes with the highest probabilities."""
        if isinstance(self.data, torch.Tensor):
            return list(self.data.topk(5).indices.cpu().numpy())
        else:
            return list(np.argsort(self.data)[-5:][::-1])

    @property
    def top1_conf(self) -> Union[torch.Tensor, np.ndarray]:
        """Returns the confidence score of the top-1 class."""
        return self.data[self.top1]

    @property
    def top5_conf(self) -> Union[torch.Tensor, np.ndarray]:
        """Returns the confidence scores of the top-5 classes."""
        return self.data[self.top5]

    @staticmethod
    def to_probs(class_id: int, num_classes: int) -> np.ndarray:
        """Converts a class ID to a probability tensor.

        Args:
            class_id: The class ID.
            num_classes: Total number of classes.

        Returns:
            A ``torch.Tensor`` or ``numpy.ndarray`` with the class ID set to
            ``1.0`` and others set to ``0.0``.
        """
        if not isinstance(class_id, int):
            raise TypeError(f"``class_id`` must be an int, got {type(class_id)}.")
        if not isinstance(num_classes, int):
            raise TypeError(f"``num_classes`` must be an int, got {type(num_classes)}.")

        probs           = np.zeros(num_classes, dtype=np.float32)
        probs[class_id] = 1.0
        return probs
