from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import Field

from hafnia.dataset.primitives.primitive import Primitive
from hafnia.dataset.primitives.utils import anonymize_by_resizing, get_class_name


class Classification(Primitive):
    # Names should match names in FieldName
    class_name: Optional[str] = Field(default=None, description="Class name, e.g. 'car'")
    class_idx: Optional[int] = Field(default=None, description="Class index, e.g. 0 for 'car' if it is the first class")
    object_id: Optional[str] = Field(default=None, description="Unique identifier for the object, e.g. '12345123'")
    confidence: float = Field(
        default=1.0, description="Confidence score (0-1.0) for the primitive, e.g. 0.95 for Classification"
    )
    ground_truth: bool = Field(default=True, description="Whether this is ground truth or a prediction")

    task_name: str = Field(
        default="",
        description="To support multiple Classification tasks in the same dataset. '' defaults to 'classification'",
    )
    created_at: Optional[datetime] = Field(default=None, description="Date when the primitive was created")
    updated_at: Optional[datetime] = Field(default=None, description="Date when the primitive was last updated")
    meta: Optional[Dict[str, Any]] = Field(
        default=None, description="This can be used to store additional information about the classification"
    )

    # Attributes - allow nesting
    classifications: Optional[List["Classification"]] = None

    @staticmethod
    def default_task_name() -> str:
        return "image_classification"

    @staticmethod
    def column_name() -> str:
        return "classifications"

    def calculate_area(self, image_height: int, image_width: int) -> float:
        return 1.0

    def draw(self, image: np.ndarray, inplace: bool = False, draw_label: bool = True) -> np.ndarray:
        if draw_label is False:
            return image
        from hafnia.dataset import image_visualizations

        class_name = self.get_class_name()
        if self.task_name == self.default_task_name():
            text = class_name
        else:
            text = f"{self.task_name}: {class_name}"
        image = image_visualizations.append_text_below_frame(image, text=text, text_size_ratio=0.05)

        return image

    def mask(
        self, image: np.ndarray, inplace: bool = False, color: Optional[Tuple[np.uint8, np.uint8, np.uint8]] = None
    ) -> np.ndarray:
        # Classification does not have a mask effect, so we return the image as is
        return image

    def anonymize_by_blurring(self, image: np.ndarray, inplace: bool = False, max_resolution: int = 20) -> np.ndarray:
        # Classification does not have a blur effect, so we return the image as is
        return anonymize_by_resizing(image, max_resolution=max_resolution)

    def get_class_name(self) -> str:
        return get_class_name(self.class_name, self.class_idx)
