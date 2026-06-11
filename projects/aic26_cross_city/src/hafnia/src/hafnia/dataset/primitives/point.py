from typing import Any, Tuple

from pydantic import BaseModel, Field

from hafnia.dataset.primitives.utils import clip


class Point(BaseModel):
    x: float = Field(
        description="Normalized x-coordinate (0.0=left edge, 1.0=right edge) relative to image width",
    )
    y: float = Field(
        description="Normalized y-coordinate (0.0=top edge, 1.0=bottom edge) relative to image height",
    )

    def to_pixel_coordinates(
        self, image_shape: Tuple[int, int], as_int: bool = True, clip_values: bool = True
    ) -> Tuple[Any, Any]:
        x = self.x * image_shape[1]
        y = self.y * image_shape[0]

        if as_int:
            x, y = int(round(x)), int(round(y))  # noqa: RUF046

        if clip_values:
            x = clip(value=x, v_min=0, v_max=image_shape[1])
            y = clip(value=y, v_min=0, v_max=image_shape[0])

        return x, y
