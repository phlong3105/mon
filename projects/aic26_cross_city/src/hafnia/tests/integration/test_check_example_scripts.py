import subprocess
import sys
from pathlib import Path

import pytest

from hafnia.utils import is_hafnia_configured


@pytest.mark.parametrize(
    "script_path_str",
    [
        "examples/example_torchvision_dataloader.py",
        "examples/example_hafnia_dataset.py",
        "examples/example_logger.py",
        "examples/example_dataset_recipe.py",
        "examples/example_custom_dataset.py",
        "examples/example_benchmark.py",
        # Add other example scripts here
    ],
)
@pytest.mark.slow
def test_example_scripts(script_path_str: str):
    if not is_hafnia_configured():
        pytest.skip("Not logged in to Hafnia")

    script_path = Path(script_path_str)
    if not script_path.exists():
        pytest.fail(f"Script {script_path} does not exist")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=120,  # 2-minute timeout
        )

        # Print output for debugging if there was an error
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")

        assert result.returncode == 0, f"Script {script_path} failed with return code {result.returncode}"

    except subprocess.TimeoutExpired:
        pytest.fail(f"Script {script_path} timed out")
    except Exception as e:
        pytest.fail(f"Failed to run script {script_path}: {str(e)}")
