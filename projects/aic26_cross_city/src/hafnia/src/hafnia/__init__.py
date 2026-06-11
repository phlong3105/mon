from importlib.metadata import version

__package_name__ = "hafnia"
__version__ = version(__package_name__)  # Returns version from 'pyproject.toml'

__dataset_format_version__ = "0.3.1"  # Hafnia dataset format version
