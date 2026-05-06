#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dataset Data Structures.

This module provides generic data structures for handling datasets.
"""

from __future__ import annotations

__all__ = [
    "Dataset",
    "InputTargetDataset",
    "StandardDataset",
]

import os
from abc import ABC, abstractmethod
from typing import Any, override

from tensordict import TensorDict
from torch.utils.data.dataset import Dataset as Dataset_

from mon.core import (
    ClassList,
    create_progress_bar,
    is_valid_str,
    Metadata,
    MetadataDictList,
    Path,
    Split,
)
from .modality import Modality, ModalityList


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class Dataset(Dataset_, ABC):
    """Base class for all datasets.

    Provide a unified interface for handling modalities, class definitions,
    metapoints, and datapoints.

    Notes:
        In the datasets, we only persist the metapoints (i.e., metadata) of all
        modalities. When ``__getitem__()`` is called, we load the data from the
        metadata using the modality-specific loaders. This allows us to keep the
        dataset lightweight and only load data on demand.

    Attributes:
        modalities (ModalityList): A list of ``Modality`` definitions.
            By default, the first modality is considered the primary one.
            `Must be defined in subclasses or set during initialization.`
        classes (ClassList, optional): Class definitions associated with the
            dataset. Defaults to empty ClassList.
    """

    modalities: ModalityList
    classes: ClassList = ClassList()

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        metapoints: MetadataDictList | None = None,
        modalities: ModalityList | None = None,
        classes: ClassList | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        If no ``metapoints`` are provided, automatically call the following
        methods during initialization:
        ::

            1.   ``list_metapoints()``       : Retrieve a list of metapoints (extensible).
            1.1. ``list_modality_from_dir()``: Retrieve modality data files from a base directory.
            1.2. ``list_modality_from_ref()``: Retrieve modality data files based on a reference modality.
            2.   ``verify()``                : Validate the integrity of the loaded data.

        Args:
            metapoints (MetadataDictList | None, optional): A dictionary
                containing the modalities' keys and lists of metadata associated
                with each datapoint. Defaults to None.
            modalities (ModalityList | None, optional): A list of ``Modality``
                definitions. By default, the first modality is considered the
                primary one. If provided, it overrides the class-level default.
                Defaults to None.
            classes (ClassList | None, optional): Class definitions associated
                with the dataset. If provided, it overrides the class-level
                default. Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
            *args: Positional arguments for ``Dataset`` constructor.
            **kwargs: Keyword arguments for ``Dataset`` constructor.

        Raises:
            TypeError: If ``metapoints`` is not a MetadataDictList or None.
        """
        super().__init__()

        # Assign attributes
        self.verbose = verbose

        # Override class-level defaults if provided
        if isinstance(modalities, ModalityList) and len(modalities) > 0:
            self.modalities = modalities
        if isinstance(classes, ClassList) and len(classes) > 0:
            self.classes = classes

        if metapoints:
            # If metapoints are provided, use them directly
            if isinstance(metapoints, MetadataDictList):
                self.metapoints = metapoints
            else:
                raise TypeError(f"expected metapoints to be a MetadataDictList, "
                                f"got {type(metapoints).__name__}.")
        else:
            # Else, retrieve metapoints
            self.metapoints = MetadataDictList()
            self.list_metapoints()

        # Validate inputs
        self.verify()

    def __del__(self):
        """Finalizer called when the object is about to be destroyed."""
        pass

    # --- Representation ---
    @override
    def __repr__(self) -> str:
        """Return the official string representation for developers."""
        lines = [f"Dataset {self.__class__.__name__}"]
        lines += [f"Number of datapoints: {len(self)}"]
        return "\n".join(lines)

    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the length of the container."""
        pk, _ = self.primary
        return len(self.metapoints[pk])

    @abstractmethod
    def __getitem__(self, index: int) -> TensorDict:
        """Return an item at the given ``index``.

        Args:
            index (int): Index of datapoint.

        Returns:
            TensorDict: A datapoint dictionary containing all modalities, each
                associated with a 'key'.
        """
        pass

    def __iter__(self):
        """Return an iterator for the container."""
        for i in range(len(self)):
            yield self[i]

    # --- Properties ---
    @property
    def primary(self) -> tuple[str, Modality]:
        """Return the primary modality and its key."""
        modality: Modality = self.modalities[0]
        return modality.name, modality

    @property
    def disable_pbar(self) -> bool:
        """Check if progress bars are disabled."""
        return not self.verbose

    # --- Discovery ---
    @abstractmethod
    def list_metapoints(self):
        """List all metapoints available for loading.

        After calling this method, ``self.metapoints`` must be populated.
        """
        pass

    def list_modality_from_dir(
        self,
        modality: Modality,
        base_dir: Path,
    ) -> list[Metadata]:
        """List modality data files from a base directory.

        Args:
            modality (Modality): Primary modality definition.
            base_dir (Path): Base directory to search for data files.

        Returns:
            list[Metadata]: List of metadata for the modality.
        """
        # Build the pattern for the current modality
        name = modality.name
        dirname = modality.dirname
        ext = modality.ext
        input_dir = Path(base_dir).normalize().resolve_subdir(dirname)

        # Retrieve the list of metadata for the current modality
        metadata: list[Metadata] = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            paths: list[Path] = sorted(input_dir.rglob("*"))
            desc = f"Listing {self.__class__.__name__} {name}(s)"
            for path in pbar.track(sequence=paths, description=desc):
                path = path.normalize()
                if path.has_ext(ext):
                    metadata.append(Metadata(path=path, base_dir=base_dir))

        return metadata

    def list_modality_from_ref(
        self,
        modality: Modality,
        ref_modality: Modality,
        ref_metadata: list[Metadata],
        base_dir: Path | None = None
    ) -> list[Metadata]:
        """List modality data files based on a reference modality.

        Args:
            modality (Modality): Modality definition.
            ref_modality (Modality): Reference modality definition.
            ref_metadata (list[Metadata]): List of metadata for the reference
                modality.
            base_dir (Path | None, optional): Base directory to search for data
                files. Defaults to None.

        Returns:
            list[Metadata]: List of metadata for the modality.
        """
        # Replace the primary modality directory part with the current modality
        name = modality.name
        dirname = modality.dirname
        ext = modality.ext

        if base_dir:
            input_dir = Path(base_dir).normalize().resolve_subdir(dirname)
            replace_kwargs = {}
        else:
            pk_dirname = ref_modality.dirname
            old_part = f"{os.sep}{pk_dirname}{os.sep}"
            new_part = f"{os.sep}{dirname}{os.sep}"
            input_dir = None
            replace_kwargs = dict(old=old_part, new=new_part)

        # Retrieve the list of metadata for the current modality
        metadata: list[Metadata] = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            desc = f"Listing {self.__class__.__name__} {name}(s)"
            for meta in pbar.track(sequence=ref_metadata, description=desc):
                # Replace directory part
                if input_dir:
                    path = input_dir / meta.path.name
                else:
                    path = meta.path.replace_part(**replace_kwargs)
                # Replace extension
                path = path.normalize().sibling(ext=ext)
                if path.has_ext(ext):
                    metadata.append(Metadata(path=path, base_dir=meta.base_dir))

        return metadata

    # --- Creation ---
    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> Dataset:
        """Create a new instance from a configuration dictionary."""
        pass

    # --- Validation ---
    def verify(self):
        """Verify dataset integrity after loading.

        Raises:
            RuntimeError: If no metapoints are found or if modality lengths are
                inconsistent.
        """
        modalities = self.modalities
        metapoints = self.metapoints
        keys = modalities.names
        num_datapoints = len(self)

        if not isinstance(metapoints, MetadataDictList):
            raise TypeError(f"expected metapoints to be a MetadataDictList, "
                            f"got {type(metapoints).__name__}.")
        if keys != list(metapoints.keys()):
            raise ValueError(f"modalities keys {keys} do not match with "
                             f"metapoints keys {list(metapoints.keys())}.")

        for k, v in metapoints.items():
            if len(v) > 0 and len(v) != num_datapoints:
                raise ValueError(f"modality {k} has inconsistent length with the "
                                 f"rest of the dataset {len(v)} != {num_datapoints}.")

    # --- Retrieval ---
    def get_metapoint(self, index: int) -> dict[str, Metadata]:
        """Get a metapoint at the specified ``index``.

        Args:
            index (int): Index of metapoint.

        Returns:
            dict[str, Metadata]: A metapoint dictionary containing all
                modalities, each associated with a 'key'.
        """
        return {k: m[index] for k, m in self.metapoints.items()}

    def get_datapoint(self, index: int) -> dict[str, Any]:
        """Get a datapoint at the specified ``index``.

        Args:
            index (int): Index of datapoint.

        Returns:
            dict[str, Any]: A datapoint dictionary containing all modalities,
                each associated with a 'key'.
        """
        # Get all metadata of the datapoint at the current index
        metapoint = self.get_metapoint(index)
        # Create an empty datapoint
        datapoint = {k: None for k in self.modalities.keys}

        # For each modality, use the corresponding loader to load the data from
        # the metadata
        for k, metadata in metapoint.items():
            modality: Modality = self.modalities[k]
            loader = modality.loader
            if metadata is not None and loader is not None:
                datapoint[k] = loader(metadata)

        # Add the primary modality metadata to the datapoint
        pk, _ = self.primary
        pk_data = datapoint[pk]
        if hasattr(pk_data, "meta"):
            datapoint["meta"] = pk_data.meta
        else:
            datapoint["meta"] = None

        return datapoint

    def get_underlying_data(self, index: int) -> dict[str, Any]:
        """Get the underlying data of a datapoint at the specified ``index``."""
        datapoint = self.get_datapoint(index)
        for k, v in datapoint.items():
            if hasattr(v, "data"):
                datapoint[k] = v.data
        return datapoint


class StandardDataset(Dataset, ABC):
    """A standard dataset that contains several splits (e.g., train, val, test)
    placed under a common root directory.

    Attributes:
        dirname (str, optional): Name of the dataset directory within the root.
            Use this if the given ``root`` directory does not contain the
            dataset directory itself. Defaults to an empty string, meaning the
            dataset is located directly at the given ``root``.
        subdir (str, optional): Name of the subdirectory within the dataset's
            ``root``. (i.e., ``root/subdir``). Use this if the current dataset
            is a subset of another dataset. If provided, it will be automatically
            appended to the ``root`` path. Defaults to an "", meaning no subdirectory.
        splits (list[Split], optional): List of supported data splits.
            Defaults to an empty list, which must be overridden in subclasses.
    """

    dirname: str = ""
    subdir: str = ""
    splits: list[Split] = []

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        root: Path,
        split: Split,
        dirname: str = "",
        subdir: str = "",
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            root (Path): Path to the root directory of the dataset.
            split (Split): Data split subset to use. Must be one of the options
                defined in ``splits``.
            dirname (str, optional): Name of the dataset directory within the
                root path. Use this if the given ``root`` path does not contain
                the dataset directory itself. Defaults to "".
            subdir (str, optional): Name of the subdirectory within the dataset's
                ``root`` (i.e., ``root/subdir``). Use this if the current
                dataset is a subset of another dataset. If provided, it
                overrides the class-level default. Defaults to "".
            *args: Positional arguments for ``Dataset`` constructor.
            **kwargs: Keyword arguments for ``Dataset`` constructor.

        Raises:
            ValueError: If ``splits`` is not defined in the subclass.
        """
        # Assign attributes
        # Override class-level default if provided
        if is_valid_str(dirname):
            self.dirname = dirname
        if is_valid_str(subdir):
            self.subdir = subdir

        self.root = root
        self.split = split

        # Validate inputs
        if not self.splits:
            raise ValueError("no splits is defined for the dataset.")

        # Continue the initialization chain
        super().__init__(*args, **kwargs)

    def __init_subclass__(cls, *args, **kwargs):
        """Validate subclass attributes on inheritance.

        Raises:
            AttributeError: If ``name`` or ``splits`` is not defined in the subclass.
        """
        super().__init_subclass__(*args, **kwargs)

        # Check for EXPLICIT definition in the subclass (not inherited)
        for attr in ["splits"]:
            if not getattr(cls, attr):
                raise AttributeError(f"{cls.__name__} must define '{attr}' attribute.")

    # --- Properties ---
    @property
    def root(self) -> Path:
        """Return the path to the dataset root directory."""
        return self._root

    @root.setter
    def root(self, value: Path):
        """Set the dataset root directory.

        Args:
            value (Path): Path to the root directory of the dataset.

        Raises:
            FileNotFoundError: If the ``root`` directory does not exist.
        """
        value = Path(value).normalize()

        # Append dirname if specified
        if is_valid_str(self.dirname):
            # Check if the current root ends with a dataset directory; if not,
            # try to append
            dataset_dir = value.resolve_subdir(self.dirname)
            if dataset_dir.is_dir():
                value = dataset_dir

        # Append subdir if specified
        if is_valid_str(self.subdir):
            # Check if the current root ends with a subset; if not, try to append
            subdir = value.resolve_subdir(self.subdir)
            if subdir.is_dir():
                value = subdir

        # Validate inputs
        if not value.is_dir():
            raise FileNotFoundError(f"dataset root not found at {value.as_posix()}")

        self._root = value

    @property
    def split(self) -> Split:
        """Return the current dataset split."""
        return self._split

    @split.setter
    def split(self, value: Split):
        """Set the current dataset split.

        Args:
            value (Split): Data split subset to use. Must be one of the options
                defined in ``splits``.

        Raises:
            ValueError: If ``split`` is not one of the supported ``splits``.
        """
        value = Split(value)

        # Validate inputs
        if value not in self.splits:
            raise ValueError(f"unsupported split {value}, must be one of {self.splits}.")

        self._split = value

    @property
    def base_dir(self) -> Path:
        """Return the common path for all modalities in the current split."""
        return self.root / self.split

    # --- Discovery ---
    @override
    def list_metapoints(self):
        """List all metapoints available for loading.

        After calling this method, ``self.metapoints`` must be populated.
        """
        # Initialize empty metapoints dictionary with modalities
        metapoints = MetadataDictList.from_keys(keys=self.modalities.keys)

        # List primary modality metadata files
        pk, pm = self.primary
        pk_metadata = self.list_modality_from_dir(modality=pm, base_dir=self.base_dir)
        metapoints[pk] = pk_metadata

        # List other modality metadata files
        for m in self.modalities:
            if m.name == pk:
                continue
            if (
                (self.split == Split.TRAIN and m.train)
                or (self.split == Split.VAL and m.val)
                or (self.split in [Split.TEST, Split.PREDICT] and m.test)
            ):
                # If in test/predict split, include the modality if marked as test
                metapoints[m.name] = self.list_modality_from_ref(
                    modality=m,
                    ref_modality=pm,
                    ref_metadata=pk_metadata,
                    base_dir=None
                )
            else:
                # Else, set all metapoints to None
                metapoints[m.name] = [None] * len(pk_metadata)

        self.metapoints = metapoints


class InputTargetDataset(Dataset, ABC):
    """A dataset structure that contains input and target directories."""

    # --- Lifecycle & Initialization ---
    def __init__(self, input_dir: Path, target_dir: Path, *args, **kwargs):
        """Initialize a new instance.

        Args:
            input_dir (Path): Path to the input directory.
            target_dir (Path): Path to the target directory.
            *args: Positional arguments for ``Dataset`` constructor.
            **kwargs: Keyword arguments for ``Dataset`` constructor.
        """
        # Assign attributes
        self.input_dir = input_dir
        self.target_dir = target_dir

        # Continue the initialization chain
        super().__init__(*args, **kwargs)

    # --- Properties ---
    @property
    def input_dir(self) -> Path:
        """Return the path to the input directory."""
        return self._input_dir

    @input_dir.setter
    def input_dir(self, value: Path):
        """Set the input directory.

        Args:
            value (Path): Path to the input directory.

        Raises:
            TypeError: If ``input_dir`` is None.
            FileNotFoundError: If the ``input_dir`` directory does not exist.
        """
        if value is None:
            raise TypeError(f"expected input_dir to be a valid Path, "
                            f"got {type(value).__name__}.")

        value = Path(value).normalize()
        if not value.is_dir():
            raise FileNotFoundError(f"input directory not found at {value.as_posix()}")

        self._input_dir = value

    @property
    def target_dir(self) -> Path | None:
        """Return the path to the target directory."""
        return self._target_dir

    @target_dir.setter
    def target_dir(self, value: Path | None):
        """Set the target directory.

        Args:
            value (Path | None): Path to the target directory.

        Raises:
            FileNotFoundError: If the ``target_dir`` directory does not exist.
        """
        if value is not None:
            value = Path(value).normalize()
            if isinstance(value, Path) and not value.is_dir():
                raise FileNotFoundError(f"target directory not found at {value.as_posix()}")
            self._target_dir = value
        else:
            self._target_dir = None

    @property
    def has_target(self) -> bool:
        """Check if the dataset has target data."""
        return self._target_dir is not None

    @property
    def label_dir(self) -> Path | None:
        """An alias to ``target_dir`` for better readability in certain contexts."""
        return self._target_dir

    # --- Discovery ---
    @override
    def list_metapoints(self):
        """List all metapoints available for loading.

        After calling this method, ``self.metapoints`` must be populated.
        """
        # Initialize empty metapoints dictionary with modalities
        metapoints = MetadataDictList.from_keys(self.modalities.keys)

        # List primary modality metadata files (inputs)
        pk, pm = self.primary
        pk_metadata = self.list_modality_from_dir(modality=pm, base_dir=self.input_dir)
        metapoints[pk] = pk_metadata

        # List secondary modality metadata files (targets)
        for m in self.modalities:
            if m.name == pk:
                continue
            if self.has_target:
                metapoints[m.name] = self.list_modality_from_ref(
                    modality=m,
                    ref_modality=pm,
                    ref_metadata=pk_metadata,
                    base_dir=self.target_dir
                )
            else:
                # If no target data is available, set all metapoints to None
                metapoints[m.name] = [None] * len(pk_metadata)

        self.metapoints = metapoints

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
