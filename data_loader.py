"""Data Loader Utilities.

This module provides data loading utilities for various data formats.
"""

import numpy as np

class DataLoader:
    """A container for data loading operations.

    Attributes:
        data (np.ndarray): Loaded data array.
        file_path (str): Path to the data file.
    """

    def __init__(self, file_path: str):
        """Initialize the DataLoader with a file path.

        Args:
            file_path: Path to the data file.
        """
        self.file_path = file_path
        self.data = None

    def load_csv(self):
        """Load data from a CSV file."""
        self.data = np.genfromtxt(self.file_path, delimiter=',')

    def get_data(self):
        """Return the loaded data."""
        return self.data

    def normalize(self):
        """Normalize the loaded data."""
        if self.data is not None:
            self.data = (self.data - np.mean(self.data)) / np.std(self.data)
        else:
            raise ValueError("Data not loaded. Call load_csv() first.")

def concatenate_data(data1, data2):
    """Concatenate two data arrays.

    Args:
        data1: First data array.
        data2: Second data array.

    Returns:
        Concatenated data array.
    """
    return np.concatenate((data1, data2))

