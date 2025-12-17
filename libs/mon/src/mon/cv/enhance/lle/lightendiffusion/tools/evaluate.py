import argparse
import os

import torch
import yaml

import datasets
from models import DenoisingDiffusion, DiffusiveRestoration


def parse_args_and_config():
    parser = argparse.ArgumentParser(description="Latent-Retinex Diffusion Models")
    parser.add_argument("--config",       type=str, default="unsupervised.yml", help="Path to the config file")
    parser.add_argument("--mode",         type=str, default="evaluation",       help="training or evaluation")
    parser.add_argument("--resume",       type=str, default="ckpt/stage2/stage2_weight.pth.tar", help="Path for the diffusion model checkpoint to load for evaluation")
    parser.add_argument("--image_folder", type=str, default="results/",         help="Location to save restored images")
    args = parser.parse_args()

    with open(os.path.join("options", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print("Note: Currently supports evaluations (restoration) when run only on a single GPU!")

    print("=> using dataset '{}'".format(config.data.val_dataset))
    DATASET = datasets.__dict__[config.data.type](config)
    _, val_loader = DATASET.get_loaders()

    # create model
    print("=> creating denoising-diffusion model")
    diffusion = DenoisingDiffusion(args, config)
    model = DiffusiveRestoration(diffusion, args, config)
    model.restore(val_loader)


if __name__ == "__main__":
    main()
