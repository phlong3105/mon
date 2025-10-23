#!/usr/bin/env python
# -*- coding: utf-8 -*-

import lightning as L
from torch import optim
from torchmetrics.image import PeakSignalNoiseRatio

from .indi import get_indi_step, indi_transform, sample
from .utils import write_images


class IndiDeband(L.LightningModule):
    
    def __init__(self, net, setup, loss, steps: int = 10):
        super().__init__()
        self.net   = net
        self.setup = setup
        self.loss  = loss
        self.steps = steps
        self.psnr  = PeakSignalNoiseRatio()
        
    def training_step(self, batch, batch_idx):
        dry, wet    = batch["dry"], batch["wet"]
        # Get features from network
        fct, t      = get_indi_step(dry, deterministic = False, steps=self.steps)
        transformed = indi_transform(fct.to(self.setup.trainDevice), clean=dry.to(self.setup.trainDevice), dirty=wet.to(self.setup.trainDevice))
        repair      = self.net(transformed.to(self.setup.trainDevice), t.to(self.setup.trainDevice))
        if self.loss.transform is not None:
            loss = (self.loss.fn(self.loss.transform(repair).to(self.setup.trainDevice), self.loss.transform(dry).to(self.setup.trainDevice))) * self.loss.weight
        else:
            loss = (self.loss.fn(repair.to(self.setup.trainDevice), dry.to(self.setup.trainDevice))) * self.loss.weight
        total = loss
        if self.setup.args.debug:
            print(total)
        self.log("train_loss", total, on_step=False, on_epoch=True)
        psnr = self.psnr(repair, dry)
        self.log("train_psnr", psnr, on_step=False, on_epoch=True)
        return total

    def validation_step(self, batch, batch_idx):
        dry, wet = batch['dry'], batch['wet']
        # get features from network
        # test returns all 1s for proper evaluation
        fct, t = get_indi_step(dry, steps=self.steps, test=True)
        transformed = indi_transform(fct.to(self.setup.trainDevice), clean=dry.to(self.setup.trainDevice), dirty=wet.to(self.setup.trainDevice))
        repair = self.net(transformed.to(self.setup.trainDevice), t.to(self.setup.trainDevice))
        if self.loss.transform is not None:
            loss = (self.loss.fn(self.loss.transform(repair).to(self.setup.trainDevice), self.loss.transform(dry).to(self.setup.trainDevice))) * self.loss.weight
        else:
            loss = self.loss.fn(repair, dry) * self.loss.weight
        total = loss
        d = dry.clone()
        w = wet.clone()
        # set each time, we only care about the last (for now)
        self.image_map = {"dry": d, "wet": w}
        if self.setup.args.debug:
            print(total)
        self.log("val_loss", total, on_step=False, on_epoch=True)
        psnr = self.psnr(repair, dry)
        self.log("test_psnr", psnr, on_step=False, on_epoch=True)
        return total

    def on_validation_epoch_end(self):
        repair = sample(self.net, self.image_map["wet"], self.steps)
        self.image_map["repair"] = repair
        writer = self.logger.experiment
        # writes audio to whichever experiment logger we are using.
        write_images(self.image_map, writer=writer, epoch=self.current_epoch)
        return None

    # default to adam.
    def configure_optimizers(self, optimizer=optim.Adam):
        #opt = optim.Adamax(self.parameters(), lr=self.lr)
        optimizer = optimizer(self.parameters(), lr= self.setup.args.learningRate)
        return optimizer
