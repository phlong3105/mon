#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for abstract complex data types and base tensor/array handling.

This module provides abstract classes and base implementations for handling
tensor/array-like objects with device management capabilities.
"""

__all__ = [
    "Data",
    "BaseTensorOrArray",
]

import abc
from typing import Any

import numpy as np
import torch


# ----- Abstract -----
class Data(abc.ABC):
    """An abstract class for all complex data types.
    
    This class defines a common interface for complex data types that encapsulate
    tensor/array-like objects with device management capabilities. It includes
    methods for loading data, accessing properties, and moving data between
    devices.
    """
    
    # ----- Magic Methods -----
    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the length of the object.
        
        Returns:
            int: The number of elements, i.e., the size of the first dimension
                of the underlying data.
        """
        pass
    
    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """Gets the element(s) at the specified index from the underlying data.

        Args:
            idx (int): The index or indices to select from the data.
        
        Returns:
            Any: The selected element(s) from the underlying data.
        """
        pass
    
    # ----- Properties -----
    @property
    @abc.abstractmethod
    def data(self) -> Any:
        """Getter for the underlying data.
        
        This property should return the underlying data, loading the actual data
        from disk or other sources if necessary.
        
        Returns:
             Any: The underlying data.
        """
        pass
    
    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Getter for the shape of the data.
        
        This property should return the shape of the underlying data.
        
        Returns:
            tuple[int, ...]: The shape of the data.
        """
        pass
    
    @property
    @abc.abstractmethod
    def meta(self) -> dict:
        """Getter for metadata about the data.
        
        This property should return a dictionary containing metadata about the
        data, such as shape, dtype, and other relevant information.
        
        Returns:
            dict: Metadata about the data.
        """
        pass
    
    # ----- Initialize -----
    @abc.abstractmethod
    def load(self, reload: bool = False) -> Any:
        """Loads the data from disk into memory.

        For complex data type, this method can be overridden to implement
        lazy loading functionality from disk or other sources. This method can
        be called internally or externally to reload the data if needed.
        
        Args:
            reload (bool): If True, forces reloading the data from disk.
                Defaults to False.
                
        Returns:
            Any: The loaded data.

        Raises:
            NotImplementedError: If not overridden in a subclass.
        """
        pass
    
    # ----- Device Management Methods -----
    @abc.abstractmethod
    def cpu(self) -> Any:
        """Moves the data to CPU memory and returns a new instance.
        
        Returns:
            Any: A new instance with the data moved to CPU memory.
        """
        pass
    
    @abc.abstractmethod
    def cuda(self) -> Any:
        """Moves the data to GPU memory and returns a new instance.
        
        Returns:
            Any: A new instance with the data moved to GPU memory.
        """
        pass
    
    @abc.abstractmethod
    def numpy(self) -> Any:
        """Converts the data to a numpy.ndarray and returns a new instance.
        
        Returns:
            Any: A new instance with the data as a numpy.ndarray.
        """
        pass
    
    @abc.abstractmethod
    def to(self, *args, **kwargs) -> Any:
        """Moves the data to the specified device and/or dtype, returning a new
        instance. The underlying data must support the .to() method.

        Args:
            *args: Variable length argument list to be passed to torch.Tensor.to().
            **kwargs: Arbitrary keyword arguments to be passed to torch.Tensor.to().

        Returns:
            Any: A new instance with the data moved to the specified device
                and/or dtype.
        """
        pass


# ----- Base Class -----
class BaseTensorOrArray(Data):
    """A base class where the underlying data is a torch.Tensor or numpy.ndarray.
    
    Attributes:
        data (torch.Tensor or np.ndarray): The underlying data tensor or array.
    """
    
    def __init__(self, data: torch.Tensor | np.ndarray):
        """Initializes the instance.
        
        Args:
            data(torch.Tensor or np.ndarray): The underlying data tensor or array.
        """
        super().__init__()
        self.data = data
    
    # ----- Magic Methods -----
    def __len__(self) -> int:
        """Returns the length of the object.
        
        Returns:
            int: The number of elements, i.e., the size of the first dimension
                of the underlying data.
        """
        return len(self.data)
    
    def __getitem__(self, idx: int | list[int] | torch.Tensor) -> "BaseTensorOrArray":
        """Get the element(s) at the specified index from the underlying data.

        Args:
            idx (int, list[int], or torch.Tensor): The index or indices to select
                from the data.
            
        Returns:
            BaseTensorOrArray: The selected element(s) wrapped in a new instance.
        """
        return self.__class__(self.data[idx])
    
    # ----- Properties -----
    @property
    def data(self) -> torch.Tensor | np.ndarray:
        """Getter for the underlying data.
        
        Returns:
            torch.Tensor or np.ndarray: The underlying data tensor or array.
        """
        return self._data
    
    @data.setter
    def data(self, data: torch.Tensor | np.ndarray):
        """Setter for the underlying data.
        
        Args:
            data (torch.Tensor or np.ndarray): The new underlying data tensor or
                array.
            
        Raises:
            TypeError: If the provided data is not a torch.Tensor or numpy.ndarray.
        """
        if not isinstance(data, (torch.Tensor, np.ndarray)):
            raise TypeError(f"``data`` must be a torch.Tensor or numpy.ndarray, got {type(data)}.")
        self._data = data
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Getter for the shape of the data.
        
        Returns:
            tuple[int, ...]: The shape of the data.
        """
        return self.data.shape
    
    @property
    def meta(self) -> dict:
        """Getter for metadata about the data.
        
        Returns:
            dict: Metadata about the data.
        """
        return {
            "shape": self.shape,
            "dtype": self.data.dtype,
            "type" : type(self.data),
        }
    
    # ----- Initialize -----
    def load(self, reload: bool = False) -> torch.Tensor | np.ndarray:
        """No need to load data from disk, just return the underlying data.
        
        Args:
            reload (bool): Ignored for this base class. Defaults to False.
            
        Returns:
            torch.Tensor or np.ndarray: The underlying data tensor or array.
        """
        return self.data
    
    # ----- Device Management Methods -----
    def cpu(self) -> "BaseTensorOrArray":
        """Returns a new BaseTensorOrArray instance with the tensor data moved
        to CPU memory. If the data is a numpy.ndarray, returns self.
        
        Returns:
            BaseTensorOrArray: A new instance with the data moved to CPU memory.
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu())
    
    def cuda(self) -> "BaseTensorOrArray":
        """Returns a new BaseTensorOrArray instance with the tensor data moved
        to GPU memory. If the data is a numpy.ndarray, converts it to a
        torch.Tensor first.
        
        Returns:
            BaseTensorOrArray: A new instance with the data moved to GPU memory.
        """
        if isinstance(self.data, np.ndarray):
            return self.__class__(torch.as_tensor(self.data).cuda())
        else:
            return self.__class__(self.data.cuda())
    
    def numpy(self) -> np.ndarray:
        """Returns a numpy.ndarray array containing the same data as the
        original tensor. If the data is already a numpy.ndarray, returns self.
        
        Returns:
            np.ndarray: A numpy.ndarray containing the data.
        """
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy())
    
    def to(self, *args, **kwargs) -> "BaseTensorOrArray":
        """Returns a new BaseTensorOrArray instance with the data moved to the
        specified device and/or dtype. The underlying data must support the .to() method.
        
        Args:
            *args: Variable length argument list to be passed to torch.Tensor.to().
            **kwargs: Arbitrary keyword arguments to be passed to torch.Tensor.to().
        
        Returns:
            BaseTensorOrArray: A new instance with the data moved to the specified
                device and/or dtype.
        """
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs))
