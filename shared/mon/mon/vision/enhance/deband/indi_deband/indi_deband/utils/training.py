#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import random
import shutil
import time

import numpy as np
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


# Add all args across images, audio, video, etc.
def add_parser_args(parser):
    parser.add_argument("model", help="Name of the model being trained")
    # Optional args
    parser.add_argument("-b",   "--batchSize",    type=int,   default=16,        help="Specify size of batch for training set, default is 32")
    parser.add_argument("-cpu", "--cpu",          action="store_true",           help="Flag to use cpu for training, default device is CUDA")
    parser.add_argument("-d",   "--debug",        action="store_true",           help="Flag to output loss at each training step for debugging worker count")
    parser.add_argument("-dt",  "--data_type",                default="float",   help="Data type for training, default is float32")
    parser.add_argument("-e",   "--epochs",       type=int,   default=500000000, help="Specify number of epochs, default is 50")
    parser.add_argument("-id",  "--device_id",    type=int,   default=0,         help="Specify which GPU to use, default is 0")
    parser.add_argument("-lr",  "--learningRate", type=float, default=0.0004,    help="Specify initial learning rate for training")
    parser.add_argument("-r",   "--runDir",       type=str,   default="runs/",   help="Specify directory for tensorboard events")
    parser.add_argument("-p",   "--precision",    type=str,   default=None,      help="Specify precision for weights")
    parser.add_argument("-rm",  "--remove",       action="store_true",           help="Flag to remove old Tensorboard folder for a model if it exists")
    parser.add_argument("-s",   "--script",       action="store_true",           help="Flag to save model as a script module")
    parser.add_argument("-w",   "--workers",      type=int,   default=0 ,        help="Specify number of workers for training, default is 0")
    return parser


def get_model_callback(setup, monitor: str = "val_loss", top_k: int = 1, mode: str = "min"):
    checkpoint_callback = ModelCheckpoint(
        save_top_k = top_k,
        monitor    = monitor,
        mode       = mode,
        dirpath    = setup.model_dir + "/",
        filename   = "best-" + setup.args.model,
    )
    return checkpoint_callback


# Eventually be able to select different loggers!
def get_logger(setup):
    logger = TensorBoardLogger(setup.args.runDir, name=setup.args.model)
    return logger


def create_model_card(net, name, batch_size, epochs, opt):
    print("Creating model training config...")
    # Save model training configuration in a text file incase we crash.
    model_card = open("configs/trainConfig-" + name + ".txt", "w+")
    model_card.write("MODEL CARD FOR : " + name + "\n")
    model_card.write("EPOCHS : " + str(epochs) + "\n")
    model_card.write("BATCH SIZE USED : " + str(batch_size) + "\n")
    model_card.write(str(opt) + "\n")
    model_card.write("LAYERS :")
    model_card.write(str(net) + "\n")
    model_card.close()
    print(net)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_lightning_seed(setup):
    workers = True
    if setup.args.workers == 0:
        workers = False
    seed_everything(setup.seed, workers=workers)


def set_device(args):
    device = "cuda"
    device_count = torch.cuda.device_count()
    # No GPUS available
    if device_count == 0:
        device = "cpu"
    train_device = torch.device(device + ":" + str(args.device_id))
    return train_device, device


def create_model_dirs(models = "models", configs = "configs"):
    if os.path.isdir(models):
        print(models + " already exists")
    else:
        os.makedirs(models)

    if os.path.isdir(configs):
        print(configs + " already exists")
    else:
        os.makedirs(configs)


def remove_log_events(run_dir, model_name):
    if os.path.isdir(run_dir + "/" + model_name):
        print("Removing old t-board events for " + model_name)
        shutil.rmtree(os.path.abspath(run_dir + "/" + model_name + "/"))
        # delete may take some time
        time.sleep(7)


def save_checkpoint(best,trainer, net, remote_dir, epoch, multiple=200):
    if best:
        if trainer.args.local:
            torch.save(net.state_dict(), "models/" + "best-" + trainer.args.model + ".pt")
        else:
            torch.save(net.state_dict(), remote_dir + "best-" + trainer.args.model + ".pt")
    elif (epoch + 1) % multiple == 0:
        if trainer.args.local:
            torch.save(net.state_dict(), "models/" + trainer.args.model + "_" + str(epoch + 1) + ".pt")
        else:
            torch.save(net.state_dict(), remote_dir + trainer.args.model + "_" + str(epoch + 1) + ".pt")


class SetupTrain:
    
    def __init__(
        self,
        net,
        seed       = 196588,
        model_dir  = "models",
        config_dir = "configs",
    ):
        self.parser = argparse.ArgumentParser()
        self.seed   = seed
        # get parser args from command line
        add_parser_args(self.parser)
        self.args = self.parser.parse_args()
        # set seed
        #set_seed(self.seed)
        # create model dirs
        # always use relative path of models, unless we change it
        self.model_dir  = model_dir
        self.config_dir = config_dir
        create_model_dirs(self.model_dir, self.config_dir)
        # remove older run if it exists
        if self.args.remove:
            remove_log_events(self.args.runDir, self.args.model)
        # set the train device based on config!
        self.trainDevice, self.device = set_device(self.args)
        create_model_card(net, self.args.model, self.args.batchSize, self.args.epochs, "")


class Loss:
    
    def __init__(
        self,
        name,
        fn,
        weight    = 1.0,
        save      = True,
        transform = None
    ):
        self.name      = name
        self.fn        = fn
        self.weight    = weight
        self.save      = save
        self.loss      = 0
        self.transform = transform
