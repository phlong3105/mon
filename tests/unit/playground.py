from os import PathLike
from pathlib import Path

import torch

max_n                = 5
top_k                = 3
pred                 = torch.randn(3, 10)
pred_scores, indices = torch.sort(pred, dim=1)
pred_labels          = torch.argsort(pred, dim=1)
pred_1_labels        = torch.argsort(pred, dim=1)

print(pred)
print("")
print(pred_scores[:max_n, -top_k:].tolist())
print("")
print(indices[:max_n, -top_k:][:, -1].tolist())
