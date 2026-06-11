from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple

import numpy as np
from pydantic import BaseModel


class Primitive(BaseModel, metaclass=ABCMeta):
    def model_post_init(self, context) -> None:
        if self.task_name == "":  # type: ignore[has-type] # Hack because 'task_name' doesn't exist in base-class yet.
            self.task_name = self.default_task_name()

    @staticmethod
    @abstractmethod
    def default_task_name() -> str:
        # E.g. "return bboxes" for Bbox
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def column_name() -> str:
        """
        Name of field used in hugging face datasets for storing annotations
        E.g. "bboxes" for Bbox.
        """
        pass

    @abstractmethod
    def calculate_area(self, image_height: int, image_width: int) -> float:
        # Calculate the area of the primitive
        pass

    @abstractmethod
    def draw(self, image: np.ndarray, inplace: bool = False) -> np.ndarray:
        pass

    @abstractmethod
    def mask(
        self,
        image: np.ndarray,
        inplace: bool = False,
        color: Optional[Tuple[np.uint8, np.uint8, np.uint8]] = None,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def anonymize_by_blurring(self, image: np.ndarray, inplace: bool = False, max_resolution: int = 20) -> np.ndarray:
        pass
