from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import Field

from hafnia.dataset.colors import get_n_colors
from hafnia.dataset.primitives.primitive import Primitive
from hafnia.dataset.primitives.utils import get_class_name


class Segmentation(Primitive):
    # WARNING: Segmentation masks have not been fully implemented yet
    class_names: Optional[List[str]] = Field(default=None, description="Class names of the segmentation")
    ground_truth: bool = Field(default=True, description="Whether this is ground truth or a prediction")

    task_name: str = Field(
        default="",
        description="Task name to support multiple Segmentation tasks in the same dataset. Defaults to 'segmentation'",
    )
    meta: Optional[Dict[str, Any]] = Field(
        default=None, description="This can be used to store additional information about the segmentation"
    )

    @staticmethod
    def default_task_name() -> str:
        return "semantic_segmentation"

    @staticmethod
    def column_name() -> str:
        return "segmentations"

    def calculate_area(self, image_height: int, image_width: int) -> float:
        raise NotImplementedError()

    def draw(self, image: np.ndarray, inplace: bool = False) -> np.ndarray:
        if not inplace:
            image = image.copy()

        color_mapping = np.asarray(get_n_colors(len(self.class_names)), dtype=np.uint8)  # type: ignore[arg-type]
        label_image = color_mapping[self.mask]
        blended = cv2.addWeighted(image, 0.5, label_image, 0.5, 0)
        return blended

    def mask(
        self, image: np.ndarray, inplace: bool = False, color: Optional[Tuple[np.uint8, np.uint8, np.uint8]] = None
    ) -> np.ndarray:
        return image

    def anonymize_by_blurring(self, image: np.ndarray, inplace: bool = False, max_resolution: int = 20) -> np.ndarray:
        return image

    def get_class_name(self) -> str:
        return get_class_name(self.class_name, self.class_idx)
