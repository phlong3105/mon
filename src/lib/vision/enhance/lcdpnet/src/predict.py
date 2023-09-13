#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import time

import hydra
import pytorch_lightning as pl
from omegaconf import open_dict
from pytorch_lightning import Trainer

import mon
from globalenv import *
from utils.util import parse_config

console = mon.console

pl.seed_everything(GLOBAL_SEED)


@hydra.main(config_path="config", config_name="config")
def main(opt):
    opt   = parse_config(opt, TEST)
    # print("Running config:", opt)
    from model.lcdpnet import LitModel as ModelClass
    ckpt  = opt[CHECKPOINT_PATH]
    # ckpt  = mon.RUN_DIR/"train/ie/llie/lcdpnet-lol/log/lcdpnet/lcdpnet:lcdpnet-lol@lcdp_data.train/last.ckpt"
    # opt[CHECKPOINT_PATH] = ckpt
    assert ckpt
    model = ModelClass.load_from_checkpoint(ckpt, opt=opt)
    # model.opt = opt
    with open_dict(opt):
        model.opt[IMG_DIRPATH] = model.build_test_res_dir()
        opt.mode = "test"
    console.log(f"Loading model from: {ckpt}")
    
    # Measure efficiency score
    flops, params, avg_time = mon.calculate_efficiency_score(
        model=model,
        image_size=512,
        channels=3,
        runs=100,
        use_cuda=True,
        verbose=False,
    )
    console.log(f"FLOPs (G)  = {flops:.4f}")
    console.log(f"Params (M) = {params:.4f}")
    console.log(f"Time (s)   = {avg_time:.4f}")
    
    from data.img_dataset import DataModule
    datamodule = DataModule(opt)
    trainer    = Trainer(
        # gpus      = opt[GPU],
        strategy  = opt[BACKEND],
        precision = opt[RUNTIME_PRECISION]
    )
    
    start_time = time.time()
    trainer.predict(model, datamodule)
    run_time   = (time.time() - start_time)
    console.log(f"[ TIMER ] Total time usage: {run_time}")
    # print("[ PATH ] The results are in :")
    # print(model.opt[IMG_DIRPATH])


if __name__ == "__main__":
    main()
