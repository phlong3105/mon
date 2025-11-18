#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base classes for various types of datasets."""

__all__ = [
    "BaseDataset",
    "DualDomainDataset",
    "EvalDataset",
    "Modalities",
    "Modality",
]

import abc
import os
from collections import namedtuple
from typing import Any, Dict, TypeAlias

import numpy as np
import torch
from torch.utils.data import dataset

from mon.core import create_progress_bar, log, Path, Split, Task
from ..classes import Classes

Modality = namedtuple("Modality", [
    "name",     # The containing directory name in file system.
    "type",     # Albumentations target type, e.g. "image", "mask", etc.
    "module",   # Dataclass module that performs I/O operations.
    "train",    # If ``True``, this modality is included in train/val set.
    "test",     # If ``True``, this modality is included in test set.
    "primary"   # If ``True``, this is the primary modality.
], defaults=[None, None, True, False, False])
Modalities: TypeAlias = Dict[str, Modality]


# ----- Base Dataset -----
class BaseDataset(dataset.Dataset, abc.ABC):
    """Base class for all datasets.

    Attributes:
        root_name: Dataset's root directory name.
        tasks: List of supported tasks.
        splits: List of supported splits.
        modalities: Dictionary of datapoint modalities.
        classes: List of class-labels. Default: ``None``.
    
    Args:
        root: Absolute path to the dataset root directory.
        split: Data split subset to use. One of: ``Split.TRAIN``, ``Split.VAL``,
            ``Split.TEST``, or ``Split.PREDICT``. Default: ``Split.TRAIN``.
        transform: Transformations for input/target. Default: ``None``.
        verbose: If ``True``, enables verbose output. Default: ``False``.
    """
    
    root_name : str         = None
    tasks     : list[Task]  = []
    splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST, Split.PREDICT]
    modalities: Modalities  = {}
    classes   : Classes     = None
    
    def __init__(
        self,
        root     : Path,
        split    : Split = Split.TRAIN,
        transform: Any   = None,
        verbose  : bool  = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not self.modalities:
            raise ValueError("``modalities`` has no defined attributes.")
        
        # Set attributes
        self.root       = root
        self.split      = split
        self.transform  = None
        self.verbose    = verbose
        self.index      = 0  # Used with `__iter__` and `__next__`
        self.datapoints = {}
        # Order-specific, DO NOT CHANGE
        self.init_transform(transform)
        self.init_data()
        
    # ----- Magic Methods -----
    def __del__(self):
        """Closes the dataset."""
        self.close()
    
    @abc.abstractmethod
    def __getitem__(self, index: int) -> dict:
        """Retrieves a datapoint and metadata at given ``index`` as a ``dict``."""
        pass
    
    def __iter__(self):
        """Initializes the dataset iterator."""
        self.reset()
        return self
    
    @abc.abstractmethod
    def __len__(self) -> int:
        """Retrieves the total number of datapoints."""
        pass
    
    def __next__(self) -> dict:
        """Retrieves the next datapoint and metadata as a ``dict``.

        Raises:
            StopIteration: If index exceeds the dataset length.
        """
        if self.index >= self.__len__():
            raise StopIteration
        result = self.__getitem__(self.index)
        self.index += 1
        return result
    
    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root:
            body.append(f"Root location: {self.root}")
        if hasattr(self, "transform") and self.transform:
            body += [repr(self.transform)]
        lines = [head]
        return "\n".join(lines)
    
    # ----- Properties -----
    @property
    def root(self) -> Path:
        """Returns the dataset root directory."""
        return self._root
    
    @root.setter
    def root(self, root: Path):
        root = Path(root)
        if self.root_name not in [None, ""] and root.name != self.root_name:
            root = root / self.root_name
        if not root.is_dir():
            raise FileNotFoundError(f"``root`` directory not found: {root}.")
        self._root = root
    
    @property
    def split(self) -> Split:
        """Return the current dataset ``Split``."""
        return self._split
    
    @split.setter
    def split(self, split: Split):
        split = Split.from_str(split) if isinstance(split, str) else split
        if split in self.splits:
            self._split = split
        else:
            raise ValueError(f"``split`` must be one of {self.splits}, got {split}.")
    
    @property
    def split_str(self) -> str:
        """Returns the ``str`` representation of the current dataset ``Split``."""
        return self.split.value
    
    @property
    def primary_modality(self) -> tuple[str, Modality]:
        """Returns the primary modality of the dataset, which is the first key
        in ``modalities`` that is marked as ``"primary"``.
        """
        for k, v in self.modalities.items():
            if v.primary:
                return k, v
        raise ValueError(f"``modalities`` has no primary modality. "
                         f"Please set `primary=True` for one of the modalities.")
    
    @property
    def disable_pbar(self) -> bool:
        """Returns ``True`` if progress bar disabled, ``False`` otherwise."""
        return not self.verbose
    
    # ----- Initialize -----
    @abc.abstractmethod
    def init_transform(self, transform: Any = None):
        """Initializes transformation operations.

        Args:
            transform: Transformations to apply. Default: ``None``.
        """
        pass
    
    def init_data(self):
        """Initializes all datapoints in the dataset.
        
        Raises:
            ValueError: If ``modalities`` has no attributes.
        """
        # Initialize empty datapoints dictionary with modalities
        datapoints = {}
        for k, v in self.modalities.items():
            if ((v.type  is None  or  v.module is None) or
                (v.train is False and self.split in [Split.TRAIN, Split.VAL]) or
                (v.test  is False and self.split in [Split.TEST,  Split.PREDICT])):
                continue
            datapoints[k] = []
        self.datapoints = datapoints
        
        # List data
        pk, _ = self.primary_modality
        self.datapoints[pk] = self.list_primary_data()  # List primary modality
        for k, v in self.datapoints.items():            # List other modalities
            if k != pk:
                self.datapoints[k] = self.list_modality_data(k)
                
        # Verify data
        self.verify_data()
        
    @abc.abstractmethod
    def list_primary_data(self) -> list:
        """Lists primary modality data files in the dataset."""
        pass
    
    def list_modality_data(self, key: str) -> list:
        """Lists other modalities data files in the dataset."""
        pk, pk_modality = self.primary_modality
        pk_name  = pk_modality.name
        pk_files = self.datapoints[pk]
        
        modality = self.modalities[key]
        name     = modality.name
        module   = modality.module
        files    = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            for file in pbar.track(
                sequence    = pk_files,
                description = f"Listing {self.__class__.__name__} {self.split_str} {key}(s)"
            ):
                path = file.path.replace_part(f"{os.sep}{pk_name}{os.sep}", f"{os.sep}{name}{os.sep}")
                files.append(module(path=path, root=file.root))
                
        return files
    
    def verify_data(self):
        """Verifies dataset integrity.
        
        Raises:
            RuntimeError: If no datapoints or attributes invalid.
        """
        if self.__len__() <= 0:
            raise RuntimeError("No datapoints in the dataset.")
        
        for k, v in self.datapoints.items():
            if k not in self.modalities:
                raise RuntimeError(f"Modality ``{k}`` is not defined in ``modalities``. "
                                   f"Define it in the class if intentional.")
            if self.modalities[k]:
                if v is None:
                    raise RuntimeError(f"No ``{k}`` attributes defined!")
                elif len(v) != self.__len__():
                    raise RuntimeError(f"Number of ``{k}`` attributes ({len(v)}) does not "
                                       f"match datapoints ({self.__len__()}).")
                
        if self.verbose:
            log(f"Number of {self.split_str} datapoints: {self.__len__()}.")
    
    @abc.abstractmethod
    def reset(self):
        """Resets the dataset."""
        pass
    
    @abc.abstractmethod
    def close(self):
        """Closes and releases the dataset."""
        pass
    
    # ----- Data Retrieval -----
    @abc.abstractmethod
    def get_datapoint(self, index: int) -> dict:
        """Gets a datapoint at the specified ``index``.

        Args:
            index: Index of datapoint.

        Returns:
            A ``dict`` containing the datapoint.
        """
        pass
    
    @abc.abstractmethod
    def get_meta(self, index: int) -> dict:
        """Gets metadata at the specified ``index``.

        Args:
            index: Index of metadata.

        Returns:
            A ``dict`` containing the metadata.
        """
        pass
    
    def collate_fn(self, batch: list[dict]) -> dict:
        """Collates a batch of input items for ``torch.utils.data.dataset.DataLoader``.
        
        By default, ``batch`` is a ``list`` of dicts, where each ``dict``
        is a datapoint. We need to collate these into a single ``dict``
        where each key corresponds to a modality and the values are stacked
        tensors or arrays.

        Args:
            batch: List of dicts, each ``dict`` is a datapoint.

        Returns:
            Collated ``dict`` for ``torch.utils.data.dataset.DataLoader``.
        """
        zipped = {
            k: list(v)
            for k, v in zip(batch[0].keys(), zip(*[b.values() for b in batch]))
        }

        for k, v in zipped.items():
            if k not in self.modalities:  # i.e., metadata
                continue
            if v is None:
                zipped[k] = None
            elif isinstance(v[0], torch.Tensor):
                zipped[k] = torch.stack(v, dim=0)
            elif isinstance(v[0], np.ndarray):
                zipped[k] = np.stack(v, axis=0)

        return zipped


# ----- Eval Dataset -----
class EvalDataset(dataset.Dataset, abc.ABC):
    """Base class for all evaluation datasets.

    Args:
        input_dir: Absolute path to the input/predict data directory.
        target_dir: Absolute path to the target data directory. Default: ``None``.
        transform: Transformations for input/target. Default: ``None``.
        verbose: If ``True``, enables verbose output. Default: ``False``.
    """
    
    def __init__(
        self,
        input_dir : Path,
        target_dir: Path = None,
        transform : Any  = None,
        verbose   : bool = True,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.input_dir  = input_dir
        self.target_dir = target_dir
        self.transform  = None
        self.verbose    = verbose
        self.index      = 0  # Used with `__iter__` and `__next__`
        self.datapoints = {}
        # Order-specific, DO NOT CHANGE
        self.init_transform(transform)
        self.init_data()
        
    # ----- Magic Methods -----
    def __del__(self):
        """Closes the dataset."""
        self.close()
    
    @abc.abstractmethod
    def __getitem__(self, index: int) -> dict:
        """Retrieves a datapoint and metadata at given ``index`` as a ``dict``."""
        pass
    
    def __iter__(self):
        """Initializes the dataset iterator."""
        self.reset()
        return self
    
    @abc.abstractmethod
    def __len__(self) -> int:
        """Retrieves the total number of datapoints."""
        pass
    
    def __next__(self) -> dict:
        """Retrieves the next datapoint and metadata as a ``dict``.

        Raises:
            StopIteration: If index exceeds the dataset length.
        """
        if self.index >= self.__len__():
            raise StopIteration
        result = self.__getitem__(self.index)
        self.index += 1
        return result
    
    # ----- Properties -----
    @property
    def has_target(self) -> bool:
        return self.target_dir is not None and self.target_dir.is_dir()
    
    @property
    def disable_pbar(self) -> bool:
        """Returns ``True`` if progress bar disabled, ``False`` otherwise."""
        return not self.verbose

    # ----- Initialize -----
    @abc.abstractmethod
    def init_transform(self, transform: Any = None):
        """Initializes transformation operations.

        Args:
            transform: Transformations to apply. Default: ``None``.
        """
        pass
    
    @abc.abstractmethod
    def init_data(self):
        """Initializes all datapoints in the dataset."""
        pass
    
    @abc.abstractmethod
    def reset(self):
        """Resets the dataset."""
        pass
    
    @abc.abstractmethod
    def close(self):
        """Closes and releases the dataset."""
        pass
    
    # ----- Data Retrieval -----
    @abc.abstractmethod
    def get_datapoint(self, index: int) -> dict:
        """Gets a datapoint at the specified ``index``.

        Args:
            index: Index of datapoint.

        Returns:
            A ``dict`` containing the datapoint.
        """
        pass
    
    @abc.abstractmethod
    def get_meta(self, index: int) -> dict:
        """Gets metadata at the specified ``index``.

        Args:
            index: Index of metadata.

        Returns:
            A ``dict`` containing the metadata.
        """
        pass
    
    def collate_fn(self, batch: list[dict]) -> dict:
        """Collates a batch of input items for ``torch.utils.data.dataset.DataLoader``.
        
        By default, ``batch`` is a ``list`` of dicts, where each ``dict``
        is a datapoint. We need to collate these into a single ``dict``
        where each key corresponds to a modality and the values are stacked
        tensors or arrays.

        Args:
            batch: List of dicts, each ``dict`` is a datapoint.

        Returns:
            Collated ``dict`` for ``torch.utils.data.dataset.DataLoader``.
        """
        zipped = {
            k: list(v)
            for k, v in zip(batch[0].keys(), zip(*[b.values() for b in batch]))
        }

        for k, v in zipped.items():
            if v is None:
                zipped[k] = None
            elif isinstance(v[0], torch.Tensor):
                zipped[k] = torch.stack(v, dim=0)
            elif isinstance(v[0], np.ndarray):
                zipped[k] = np.stack(v, axis=0)

        return zipped


# ----- Dual-Domain Dataset -----
class DualDomainDataset(dataset.Dataset, abc.ABC):
    """Base class for all dual-domain datasets.
    
    It is mainly used in Image-to-Image translation tasks. It requires two directories
    to host data from two domains A and B. The number of items in each directory
    can be the same (paired) or different (unpaired/unaligned).
    
    Attributes:
        root_name: Dataset's root directory name.
        tasks: List of supported tasks.
        splits: List of supported splits.
        modalities_A: Dictionary of datapoint modalities in domain A.
        modalities_B: Dictionary of datapoint modalities in domain B.
        classes: List of class-labels. Default: ``None``.
    
    Args:
        root: Absolute path to the dataset root directory.
        split: Data split subset to use. One of: ``Split.TRAIN``, ``Split.VAL``,
            ``Split.TEST``, or ``Split.PREDICT``. Default: ``Split.TRAIN``.
        transform: Transformations for input/target. Default: ``None``.
        verbose: If ``True``, enables verbose output. Default: ``False``.
    """
    
    root_name   : str         = None
    tasks       : list[Task]  = []
    splits      : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST, Split.PREDICT]
    modalities_A: Modalities  = {}
    modalities_B: Modalities  = {}
    classes     : Classes     = None
    
    def __init__(
        self,
        root     : Path,
        split    : Split = Split.TRAIN,
        transform: Any   = None,
        verbose  : bool  = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not self.modalities_A:
            raise ValueError("``modalities_A`` has no defined attributes.")
        if not self.modalities_B:
            raise ValueError("``modalities_B`` has no defined attributes.")
        
        # Set attributes
        self.root         = root
        self.split        = split
        self.transform    = None
        self.verbose      = verbose
        self.index        = 0  # Used with `__iter__` and `__next__`
        self.datapoints_A = {}
        self.datapoints_B = {}
        # Order-specific, DO NOT CHANGE
        self.init_transform(transform)
        self.init_data()
        
    # ----- Magic Methods -----
    def __del__(self):
        """Closes the dataset."""
        self.close()
    
    @abc.abstractmethod
    def __getitem__(self, index: int) -> dict:
        """Retrieves a datapoint and metadata at given ``index`` as a ``dict``."""
        pass
    
    def __iter__(self):
        """Initializes the dataset iterator."""
        self.reset()
        return self
    
    @abc.abstractmethod
    def __len__(self) -> int:
        """Retrieves the total number of datapoints.
        
        As we have two datasets with potentially different numbers of images,
        we take a maximum of.
        """
        # max_size = 0
        # for k, v in self.datapoints.items():
        #     max_size = max(max_size, len(v))
        # return max_size
        pass
        
    def __next__(self) -> dict:
        """Retrieves the next datapoint and metadata as a ``dict``.

        Raises:
            StopIteration: If index exceeds the dataset length.
        """
        if self.index >= self.__len__():
            raise StopIteration
        result = self.__getitem__(self.index)
        self.index += 1
        return result
    
    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [
            f"Number of datapoints in domain A: {self.size(domain="A")}",
            f"Number of datapoints in domain B: {self.size(domain="B")}",
        ]
        if self.root:
            body.append(f"Root location: {self.root}")
        if hasattr(self, "transform") and self.transform:
            body += [repr(self.transform)]
        lines = [head]
        return "\n".join(lines)
    
    # ----- Properties -----
    @property
    def root(self) -> Path:
        """Returns the dataset root directory."""
        return self._root
    
    @root.setter
    def root(self, root: Path):
        root = Path(root)
        if self.root_name not in [None, ""] and root.name != self.root_name:
            root = root / self.root_name
        if not root.is_dir():
            raise FileNotFoundError(f"``root`` directory not found: {root}.")
        self._root = root
    
    @property
    def split(self) -> Split:
        """Return the current dataset ``Split``."""
        return self._split
    
    @split.setter
    def split(self, split: Split):
        split = Split.from_str(split) if isinstance(split, str) else split
        if split in self.splits:
            self._split = split
        else:
            raise ValueError(f"``split`` must be one of {self.splits}, got {split}.")
    
    @property
    def split_str(self) -> str:
        """Returns the ``str`` representation of the current dataset ``Split``."""
        return self.split.value
    
    def primary_modality(self, domain: str) -> tuple[str, Modality]:
        """Returns the primary modality in the ``domain``, which is the first key
        in ``modalities`` that is marked as ``"primary"``.
        """
        modalities = self.modalities_A if domain == "A" else self.modalities_B
        for k, v in modalities.items():
            if v.primary:
                return k, v
        raise ValueError(f"``modalities`` in domain {domain} has no primary modality. "
                         f"Please set `primary=True` for one of the modalities.")
    
    @abc.abstractmethod
    def size(self, domain: str) -> int:
        """Returns the number of items in the ``domain``."""
        pass
    
    @property
    def disable_pbar(self) -> bool:
        """Returns ``True`` if progress bar disabled, ``False`` otherwise."""
        return not self.verbose
    
    # ----- Initialize -----
    @abc.abstractmethod
    def init_transform(self, transform: Any = None):
        """Initializes transformation operations.

        Args:
            transform: Transformations to apply. Default: ``None``.
        """
        pass
    
    def init_data(self):
        """Initializes all datapoints in the dataset.
        
        Raises:
            ValueError: If ``modalities`` has no attributes.
        """
        # ----- Domain A -----
        # Initialize empty datapoints dictionary with modalities
        datapoints_A = {}
        for k, v in self.modalities_A.items():
            if ((v.type  is None  or  v.module is None) or
                (v.train is False and self.split in [Split.TRAIN, Split.VAL]) or
                (v.test  is False and self.split in [Split.TEST,  Split.PREDICT])):
                continue
            datapoints_A[k] = []
        self.datapoints_A = datapoints_A
        
        # List data
        pk, _ = self.primary_modality(domain="A")
        self.datapoints_A[pk] = self.list_primary_data_A()  # List primary modality
        for k, v in self.datapoints_A.items():              # List other modalities
            if k != pk:
                self.datapoints_A[k] = self.list_modality_data(domain="A", key=k)
        
        # Verify data
        self.verify_data(domain="A")
        
        # ----- Domain B -----
        datapoints_B = {}
        for k, v in self.modalities_B.items():
            if ((v.type  is None  or  v.module is None) or
                (v.train is False and self.split in [Split.TRAIN, Split.VAL]) or
                (v.test  is False and self.split in [Split.TEST,  Split.PREDICT])):
                continue
            datapoints_B[k] = []
        self.datapoints_B = datapoints_B
        
        # List data in each domain
        pk, _ = self.primary_modality(domain="B")
        self.datapoints_B[pk] = self.list_primary_data_B()  # List primary modality
        for k, v in self.datapoints_B.items():              # List other modalities
            if k != pk:
                self.datapoints_B[k] = self.list_modality_data(domain="B", key=k)
        
        # Verify data
        self.verify_data(domain="B")
    
    @abc.abstractmethod
    def list_primary_data_A(self) -> list:
        """Lists primary modality data files in domain A."""
        pass
    
    @abc.abstractmethod
    def list_primary_data_B(self) -> list:
        """Lists primary modality data files in domain B."""
        pass
    
    def list_modality_data(self, domain: str, key: str) -> list:
        """Lists other modalities data files in the ``domain``."""
        modalities = self.modalities_A if domain == "A" else self.modalities_B
        datapoints = self.datapoints_A if domain == "A" else self.datapoints_B
        
        pk, pk_modality = self.primary_modality(domain)
        pk_name  = pk_modality.name
        pk_files = datapoints[domain][pk]
        
        modality = modalities[domain][key]
        name     = modality.name
        module   = modality.module
        files    = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            for file in pbar.track(
                sequence    = pk_files,
                description = f"Listing {self.__class__.__name__} {self.split_str} {key}(s)"
            ):
                path = file.path.replace_part(f"{os.sep}{pk_name}{os.sep}", f"{os.sep}{name}{os.sep}")
                files.append(module(path=path, root=file.root))
                
        return files
    
    def verify_data(self, domain: str):
        """Verifies ``domain``'s dataset integrity.
        
        Args:
            domain: Domain name.
        
        Raises:
            RuntimeError: If no datapoints or attributes are invalid.
        """
        modalities = self.modalities_A if domain == "A" else self.modalities_B
        datapoints = self.datapoints_A if domain == "A" else self.datapoints_B
        
        if self.size(domain=domain) <= 0:
            raise RuntimeError(f"No datapoints in domain {domain}.")
        
        for k, v in datapoints.items():
            if k not in modalities:
                raise RuntimeError(f"Modality ``{k}`` is not defined in ``modalities`` of domain {domain}. "
                                   f"Define it in the class if intentional.")
            if modalities[k]:
                if v is None:
                    raise RuntimeError(f"No ``{k}`` attributes defined in domain {domain}!")
                elif len(v) != self.__len__():
                    raise RuntimeError(f"Number of ``{k}`` attributes ({len(v)}) does not "
                                       f"match datapoints ({self.__len__()}) in domain {domain}.")
    
        if self.verbose:
            log(f"Number of {self.split_str} datapoints: {self.__len__()}.")
    
    @abc.abstractmethod
    def reset(self):
        """Resets the dataset."""
        pass
    
    @abc.abstractmethod
    def close(self):
        """Closes and releases the dataset."""
        pass
    
    # ----- Data Retrieval -----
    @abc.abstractmethod
    def get_datapoint(self, domain: str, index: int) -> dict:
        """Gets a datapoint in the ``domain`` at the specified ``index``.

        Args:
            domain: Domain name.
            index: Index of datapoint.

        Returns:
            A ``dict`` containing the datapoint.
        """
        pass
    
    @abc.abstractmethod
    def get_meta(self, domain: str, index: int) -> dict:
        """Gets metadata in the ``domain`` at the specified ``index``.

        Args:
            domain: Domain name.
            index: Index of metadata.
            
        Returns:
            A ``dict`` containing the metadata.
        """
        pass
    
    def collate_fn(self, batch: list[dict]) -> dict:
        """Collates a batch of input items for ``torch.utils.data.dataset.DataLoader``.
        
        By default, ``batch`` is a ``list`` of dicts, where each ``dict``
        is a datapoint. We need to collate these into a single ``dict``
        where each key corresponds to a modality and the values are stacked
        tensors or arrays.

        Args:
            batch: List of dicts, each ``dict`` is a datapoint.

        Returns:
            Collated ``dict`` for ``torch.utils.data.dataset.DataLoader``.
        """
        zipped = {
            k: list(v)
            for k, v in zip(batch[0].keys(), zip(*[b.values() for b in batch]))
        }
        
        for k, v in zipped.items():
            # if k not in self.modalities_a:  # i.e., metadata
            #     continue
            if v is None:
                zipped[k] = None
            elif isinstance(v[0], torch.Tensor):
                zipped[k] = torch.stack(v, dim=0)
            elif isinstance(v[0], np.ndarray):
                zipped[k] = np.stack(v, axis=0)

        return zipped
