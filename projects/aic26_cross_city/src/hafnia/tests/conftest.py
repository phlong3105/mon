from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pytest
from PIL import Image

from hafnia.dataset import image_visualizations
from tests.helper_testing import get_path_expected_images


@pytest.fixture
def compare_to_expected_image(request, cache) -> Callable:
    # Fixture to automatically compare actual image to expected image
    # If no expected image exists, it will automatically create one in "tests/data/expected_images/[NAME_OF_TEST].png"
    # If the actual image does not match the expected image, it will raise an exception with a path to the debug image
    # that shows the actual and expected image side-by-side
    test_name: str = request.node.name
    pytest_cache_path: Path = cache._cachedir
    node = request.node

    def callable_function(actual_image: np.ndarray, force_update: bool = False):
        test_script_name = node.nodeid.split("::")[0].replace(".py", "").split("/")[-1]
        path_expected_images = get_path_expected_images() / test_script_name
        path_expected_images.mkdir(exist_ok=True, parents=True)
        path_expected_image = path_expected_images / f"{test_name}.png"

        if actual_image.dtype == bool:
            actual_image = (actual_image.astype(np.uint8)) * 255

        if actual_image.ndim == 2:
            actual_image = cv2.cvtColor(actual_image, cv2.COLOR_GRAY2RGB)

        if force_update:
            Image.fromarray(actual_image).save(path_expected_image)
            pytest.fail("Expected image has been updated with 'force_update=True'. Rerun the test to pass")

        if not path_expected_image.exists():
            Image.fromarray(actual_image).save(path_expected_image)
            pytest.fail(
                "Expected image does not exist. An expected image has now "
                "been created automatically. Rerun the test to pass"
            )

        expected_image = np.array(Image.open(path_expected_image))
        is_equal = np.array_equal(actual_image, expected_image)

        if not is_equal:
            org = (10, 30)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1
            color = (255, 255, 255)
            thickness = 2
            cv2.putText(
                actual_image, text="Actual", org=org, fontFace=font, fontScale=scale, color=color, thickness=thickness
            )
            cv2.putText(
                expected_image,
                text="Expected",
                org=org,
                fontFace=font,
                fontScale=scale,
                color=color,
                thickness=thickness,
            )
            if actual_image.shape[0] != expected_image.shape[0]:
                debug_image = image_visualizations.concatenate_right(expected_image, actual_image)
            else:
                debug_image = np.hstack([expected_image, actual_image])

            # Parameterized test names include [ and ] e.g. 'test_check_dataset[coco-2017].png'
            # These characters makes the file path unclickable when pytest prints the error message.
            # Remove and replace with '---' to make the path clickable --> 'test_check_dataset---coco-2017.png'
            clickable_test_name = test_name.replace("[", "---").replace("]", "---")
            pytest_cache_path.mkdir(parents=True, exist_ok=True)
            path_debug_image = pytest_cache_path / f"{clickable_test_name}debug.png"

            Image.fromarray(debug_image).save(path_debug_image)

            pytest.fail(
                f"Actual image does not match expected image. To see actual and expected image side-by-side \ncheck this image: {path_debug_image} "
            )

    return callable_function
