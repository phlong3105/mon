from typing import Dict, List, Optional, Tuple, Type, Union

import cv2
import numpy as np
import numpy.typing as npt

from hafnia.dataset.primitives import (
    Bbox,
    Bitmask,
    Classification,
    Polygon,
    Segmentation,
)
from hafnia.dataset.primitives.primitive import Primitive


def draw_anonymize_by_blurring(
    image: np.ndarray,
    primitives: List[Primitive],
    inplace: bool = False,
    class_names: Union[List[str], str] = "all",
    max_resolution: int = 20,  # Use int for pixel, float for ratio of the image size
) -> np.ndarray:
    if not inplace:
        image = image.copy()

    if isinstance(class_names, str) and class_names == "all":
        primitives = primitives
    elif isinstance(class_names, list):
        primitives = [primitive for primitive in primitives if primitive.class_name in class_names]
    else:
        raise ValueError(f"Invalid class_names type: {type(class_names)}. Expected 'all' or a list of class names.")
    for primitive in primitives:
        image = primitive.anonymize_by_blurring(image, max_resolution=max_resolution, inplace=inplace)
    return image


def draw_masks(
    image: np.ndarray,
    primitives: List[Primitive],
    inplace: bool = False,
    color: Optional[Tuple[np.uint8, np.uint8, np.uint8]] = None,
) -> np.ndarray:
    if not inplace:
        image = image.copy()

    for primitive in primitives:
        primitive.mask(image, inplace=True, color=color)
    return image


def draw_annotations(
    image: np.ndarray,
    primitives: List[Primitive],
    inplace: bool = False,
    draw_settings: Optional[Dict[Type[Primitive], Dict]] = None,
) -> np.ndarray:
    if not inplace:
        image = image.copy()
    draw_settings = draw_settings or {}
    primitives_order = [Segmentation, Bitmask, Bbox, Polygon, Classification]
    primitives = sorted(primitives, key=lambda x: primitives_order.index(type(x)))
    for primitive in primitives:
        draw_settings_for_primitive = draw_settings.get(type(primitive), {})
        image = primitive.draw(image, **draw_settings_for_primitive)
    return image


def concatenate_below(img0: np.ndarray, below_img: np.ndarray) -> np.ndarray:
    scale_factor = img0.shape[1] / below_img.shape[1]
    new_height = int(below_img.shape[0] * scale_factor)
    text_region_resized = cv2.resize(below_img, (img0.shape[1], new_height))
    if len(img0.shape) == 2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    if len(text_region_resized.shape) == 2:
        text_region_resized = cv2.cvtColor(text_region_resized, cv2.COLOR_GRAY2BGR)
    frame_visualized = np.concatenate([img0, text_region_resized], axis=0)
    return frame_visualized


def concatenate_below_resize_by_padding(img0: np.ndarray, below_img: np.ndarray):
    max_width = max(img0.shape[1], below_img.shape[1])

    if len(img0.shape) == 2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
    if len(below_img.shape) == 2:
        below_img = cv2.cvtColor(below_img, cv2.COLOR_GRAY2RGB)
    img0_padded = resize_width_by_padding(img0, new_width=max_width)
    below_img_padded = resize_width_by_padding(below_img, new_width=max_width)

    return np.concatenate([img0_padded, below_img_padded], axis=0)


def resize_width_by_padding(img0: np.ndarray, new_width: int) -> np.ndarray:
    img0_new_shape = list(img0.shape)
    img0_new_shape[1] = new_width
    img0_padded = np.zeros(img0_new_shape, dtype=img0.dtype)
    extra_width = new_width - img0.shape[1]
    left_margin = extra_width // 2
    img0_padded[:, left_margin : left_margin + img0.shape[1]] = img0
    return img0_padded


def append_text_below_frame(frame: np.ndarray, text: str, text_size_ratio: float = 0.1) -> np.ndarray:
    font_size_px = int(frame.shape[0] * text_size_ratio)  # 10% of the frame height
    font_size_px = max(font_size_px, 7)  # Ensure a minimum font size
    font_size_px = min(font_size_px, 25)  # Ensure a maximum font size

    text_region = create_text_img(text, font_size_px=font_size_px)
    frame_with_text = concatenate_below_resize_by_padding(frame, text_region)
    return frame_with_text


def create_text_img(
    text_strings: Union[List[str], str],
    font_size_px: int,
    text_width: Optional[int] = None,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> npt.NDArray[np.uint8]:
    """Private code borrowed by Peter."""
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    if font_size_px < 15:
        thickness = 1
    line_type = cv2.LINE_AA
    if isinstance(text_strings, str):
        text_strings = [text_strings]
    font_scale = cv2.getFontScaleFromHeight(fontFace=font_face, pixelHeight=font_size_px, thickness=thickness)

    text_w_max = 0
    for text in text_strings:
        (text_w, text_h), baseline = cv2.getTextSize(
            text=text, fontFace=font_face, fontScale=font_scale, thickness=thickness
        )
        text_w_max = max(text_w_max, text_w)

    text_width = text_width or text_w_max + baseline * 2
    text_height = text_h + baseline * 2

    y_pos = text_h + baseline
    text_imgs = []
    for text in text_strings:
        shape_color_image = (text_height, text_width, 3)
        img = np.full(shape_color_image, bg_color, dtype=np.uint8)
        text_img = cv2.putText(
            img=img,
            text=text,
            org=(baseline, y_pos),
            fontFace=font_face,
            fontScale=font_scale,
            color=color,
            thickness=thickness,
            lineType=line_type,
        )
        text_imgs.append(text_img)
    img_text = np.vstack(text_imgs)
    return img_text


def concatenate_right(img0: np.ndarray, below_img: np.ndarray) -> np.ndarray:
    scale_factor = img0.shape[0] / below_img.shape[0]
    new_width = int(below_img.shape[1] * scale_factor)
    text_region_resized = cv2.resize(below_img, (new_width, img0.shape[0]))

    frame_visualized = np.concatenate([img0, text_region_resized], axis=1)
    return frame_visualized
