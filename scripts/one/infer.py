#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import socket

from one import console
from one import Inference
from one import load_config
from one import MODELS
from one import print_dict
from one import set_distributed_backend
from scripts.host import hosts


# MARK: - Main

def main():
    """Main function."""
    # NOTE: Initialization
    console.rule("[bold red]1. INITIALIZATION")
    hostname = socket.gethostname().lower()
    host     = hosts[hostname]
    console.log(f"Host: {hostname}")
    
    # Distributed backend
    set_distributed_backend(strategy=host.strategy)
    
    # Configs
    config = load_config(config=host.config_file.config)
   
    # Model
    model = MODELS.build_from_dict(cfg=config.model)
    model = model.load_from_checkpoint(checkpoint_path=host.ckpt, **config.model)

    print_dict(config, title=host.config.model_fullname)
    console.log("[green]Done")
    
    # NOTE: Inference
    console.rule("[bold red]2. MODEL INFERENCE")
    inference_cfg                  = config.inference
    inference_cfg.default_root_dir = "inference/outputs"
    
    inference = Inference(**inference_cfg)
    
    inference.run(model=model, data=host.infer_data)
    console.log("[green]Done")
    

if __name__ == "__main__":
    main()
