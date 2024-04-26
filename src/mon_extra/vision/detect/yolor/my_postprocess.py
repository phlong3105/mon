#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging

from mon import core
from utils.general import (
    strip_optimizer,
)

logger        = logging.getLogger(__name__)
console       = core.console
_current_file = core.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")


# region Post-processing

def main():
    for f in [
        "/home/longpham/10_workspace/11_code/mon/project/aicity_2024_fisheye8k/run/train/yolor_d6_aicity_2024_fisheye8k_1536_epoch_100/weights/best_p.pt",
        "/home/longpham/10_workspace/11_code/mon/project/aicity_2024_fisheye8k/run/train/yolor_d6_aicity_2024_fisheye8k_1536_epoch_100/weights/best_r.pt",
        "/home/longpham/10_workspace/11_code/mon/project/aicity_2024_fisheye8k/run/train/yolor_d6_aicity_2024_fisheye8k_1536_epoch_100/weights/best_f1.pt",
        "/home/longpham/10_workspace/11_code/mon/project/aicity_2024_fisheye8k/run/train/yolor_d6_aicity_2024_fisheye8k_1536_epoch_100/weights/best_ap50.pt",
        "/home/longpham/10_workspace/11_code/mon/project/aicity_2024_fisheye8k/run/train/yolor_d6_aicity_2024_fisheye8k_1536_epoch_100/weights/best_ap.pt",
        "/home/longpham/10_workspace/11_code/mon/project/aicity_2024_fisheye8k/run/train/yolor_d6_aicity_2024_fisheye8k_1536_epoch_100/weights/best.pt",
        "/home/longpham/10_workspace/11_code/mon/project/aicity_2024_fisheye8k/run/train/yolor_d6_aicity_2024_fisheye8k_1536_epoch_100/weights/last.pt",
    ]:
        f = core.Path(f)
        if f.exists():
            strip_optimizer(f)  # strip optimizers

# endregion


# region Main

if __name__ == "__main__":
    main()

# endregion
