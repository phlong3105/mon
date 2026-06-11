from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
from PIL import Image

from hafnia.dataset.hafnia_dataset_types import ModelInfo
from hafnia.dataset.primitives import Primitive

ImageType = Union[str, Image.Image, np.ndarray]


class InferenceModel(ABC):
    """Abstract base class for inference models."""

    @abstractmethod
    def predict(
        self,
        images: Union[ImageType, List[ImageType]],
        sample_dict: Union[dict, List[dict], None] = None,
    ) -> List[Primitive]:
        """
        Perform prediction on input data.

        Args:
            images: Input images for prediction. Can be a single image or a list of images.
            sample_dict: Optional dictionary or list of dictionaries containing additional information about the sample(s).

        Returns:
            Prediction results.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """
        Get the tasks that this model can perform.

        Returns:
            Tasks supported by the model.
        """
        pass
