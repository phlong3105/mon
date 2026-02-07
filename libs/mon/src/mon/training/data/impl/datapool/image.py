#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image-based data pools.

This module provides a base class for data pools where one image can have
multiple labels/annotations.
"""

__all__ = [
    "ImageDataPool",
]

from typing import Any

from mon.core import BBoxFormat, create_progress_bar, log, Path
from mon.core.types import bbox as B, ClassList, Image, Instance
from ...base import Dataset
from ...comp import BatchCollateMixin, InputTargetLoadMixin


# ==============================================================================
# DATA POOLS
# ==============================================================================

class ImageDataPool(Dataset, InputTargetLoadMixin, BatchCollateMixin):
    """A concrete class for data pools where one image can have multiple
    annotations.

    Define two main modalities: ``image`` and ``label``. Primarily used for
    separated evaluation pipelines outside the train/eval/test loop.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        input_dir: Path,
        label_dir: Path,
        classlist: Path | ClassList = None,
        verbose  : bool             = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            input_dir: Absolute path to the input/predict data directory.
            label_dir: Absolute path to the label directory.
            classlist: Either a .yaml file containing the classes definitions,
                or a ``ClassList`` instance. If given, this will override any
                ``classes`` defined in the subclass. Defaults to None.
            verbose: Verbosity mode. Defaults to True.
        """
        super().__init__(
            input_dir  = input_dir,
            target_dir = label_dir,
            classlist  = classlist,
            verbose    = verbose,
            *args, **kwargs
        )

    def __del__(self):
        """Close the dataset loading mechanism and releases resources."""
        pass

    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the length of the dataset (i.e., number of datapoints)."""
        return len(self.datapoints["image"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the datapoint at the specified ``index`` in ``_datapoints``.

        Args:
            index: Index of datapoint.

        Returns:
            A dictionary containing the datapoint and its metadata.
        """
        data = self.get_underlying_data(index=index)
        return data

    # --- Data Loading ---
    def _load_data(self) -> dict[str, Any]:
        """Core data loading mechanism for the dataset."""
        # Initialize empty datapoints dictionary with modalities
        datapoints = {}

        # List image
        images: list[Image] = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            paths = sorted(self.input_dir.rglob("*"))
            desc  = f"Listing {self.__class__.__name__} input image(s)"
            for path in pbar.track(sequence=paths, description=desc):
                if path.is_image_file(exist=True):
                    images.append(Image(data=path, root=self.input_dir))
        datapoints["image"] = images

        # List label
        labels: list[list[Instance]] = []
        with create_progress_bar(disable=self.disable_pbar) as pbar:
            desc = f"Listing {self.__class__.__name__} target label(s)"
            for image in pbar.track(sequence=images, description=desc):
                # Here, we assume that each label file is a YOLO-format .txt file
                # where each line corresponds to one instance/annotation in the
                # image.
                label_file = self.label_dir / f"{image.path.stem}.txt"
                if label_file.is_txt_file(exist=True):
                    labels.append(self._load_label_file(label_file=label_file, image=image))
        datapoints["label"] = labels

        # List metadata
        datapoints["meta"] = [i.meta for i in images]

        return datapoints

    def _load_label_file(self, label_file: Path, image: Image) -> list[Instance]:
        """Load all label instances from a label file.

        Args:
            label_file: Path to the label file.
            image: The corresponding image object.

        Returns:
            A list of Instance objects representing the labels.
        """
        # For now, we only support loading YOLO bounding boxes
        # Todo: Implement a unified ``load()`` function for ``Instance``
        lines  = B.load(path=label_file, fmt=BBoxFormat.CXCYWHN, imgsz=image.imgsz)
        labels = []
        for l in lines:
            label = Instance(
                data       = l,
                imgsz      = image.imgsz,
                image_path = image.path,
                root       = self.label_dir,
            )
            labels.append(label)
        return labels

    def verify(self):
        """Verify dataset integrity.

        Raises:
            RuntimeError: If no datapoints or attributes invalid.
        """
        if self.__len__() <= 0:
            raise RuntimeError("No datapoints in the dataset!")

        for k, v in self.datapoints.items():
            if v in [None, []]:
                raise RuntimeError(f"``datapoints`` has no ``{k}`` attributes!")
            elif len(v) != self.__len__():
                raise RuntimeError(f"Number of ``{k}`` items does not match number "
                                   f"of ``image``, got: {len(v)} != {self.__len__()}")

        if self.verbose:
            log(f"Number of datapoints: {self.__len__()}.")

    # --- Access ---
    def get_datapoint(self, index: int) -> dict[str, Any]:
        """Get a datapoint at the specified ``index``.

        Args:
            index: Index of datapoint.

        Returns:
            A dictionary containing all modalities for the specified datapoint.
        """
        datapoint = {}
        for k, v in self.datapoints.items():
            if v is not None:
                datapoint[k] = v[index]
            else:
                datapoint[k] = None
        return datapoint


# ==============================================================================
# UTILITIES
# ==============================================================================

# --- Validation & Sanitization ---


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
