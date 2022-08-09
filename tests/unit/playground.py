from os import PathLike
from pathlib import Path

import torch

a = torch.Tensor([10])
b = torch.Tensor([1, 2, 3, 4, 5])
print(1 - b)
