#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements prediction pipeline."""

from __future__ import annotations

import sys

sys.path.append("..")

import argparse
from collections import namedtuple

import torch.nn as nn
from cv2.ximgproc import guidedFilter

import mon
from net import *
from net.losses import StdLoss
from net.vae_model import VAE
from utils.dcp import get_atmosphere
from utils.image_io import *
from utils.imresize import np_imresize

console      = mon.console
DehazeResult = namedtuple("DehazeResult", ["learned", "t", "a"])


class Dehaze(object):
    def __init__(self, image_name, image, num_iter=500, clip=True, output_path="output/"):
        self.image_name       = image_name
        self.image            = image
        self.num_iter         = num_iter
        self.ambient_net      = None
        self.image_net        = None
        self.mask_net         = None
        self.ambient_val      = None
        self.mse_loss         = None
        self.learning_rate    = 0.001
        self.parameters       = None
        self.current_result   = None
        self.output_path      = output_path

        self.clip             = clip
        self.blur_loss        = None
        self.best_result_psnr = None
        self.best_result_ssim = None
        self.image_net_inputs = None
        self.mask_net_inputs  = None
        self.image_out        = None
        self.mask_out         = None
        self.ambient_out      = None
        self.total_loss       = None
        self.input_depth      = 3
        self.post             = None
        self._init_all()

    def _init_images(self):
        self.original_image = self.image.copy()
        self.images_torch = np_to_torch(self.image).type(torch.cuda.FloatTensor)

    def _init_nets(self):
        input_depth = self.input_depth
        data_type   = torch.cuda.FloatTensor
        pad         = "reflection"

        image_net = skip(
            input_depth,
            3,
            num_channels_down = [8, 16, 32, 64, 128],
            num_channels_up   = [8, 16, 32, 64, 128],
            num_channels_skip = [0, 0 , 0 , 4 , 4],
            upsample_mode     = "bilinear",
            need_sigmoid      = True,
            need_bias         = True,
            pad               = pad,
            act_fun           = "LeakyReLU"
        )
        self.image_net = image_net.type(data_type)

        mask_net = skip(
            input_depth,
            1,
            num_channels_down = [8, 16, 32, 64, 128],
            num_channels_up   = [8, 16, 32, 64, 128],
            num_channels_skip = [0, 0 , 0 , 4 , 4],
            upsample_mode     = "bilinear",
            need_sigmoid      = True,
            need_bias         = True,
            pad               = pad,
            act_fun           = "LeakyReLU"
        )

        self.mask_net = mask_net.type(data_type)

    def _init_ambient(self):
        ambient_net      = VAE(self.image.shape)
        self.ambient_net = ambient_net.type(torch.cuda.FloatTensor)

        atmosphere       = get_atmosphere(self.image)
        self.ambient_val = nn.Parameter(data=torch.cuda.FloatTensor(atmosphere.reshape((1, 3, 1, 1))), requires_grad=False)
        self.at_back     = atmosphere

    def _init_parameters(self):
        parameters  = [p for p in self.image_net.parameters()] + [p for p in self.mask_net.parameters()]
        parameters += [p for p in self.ambient_net.parameters()]
        self.parameters = parameters

    def _init_loss(self):
        data_type      = torch.cuda.FloatTensor
        self.mse_loss  = torch.nn.MSELoss().type(data_type)
        self.blur_loss = StdLoss().type(data_type)

    def _init_inputs(self):
        self.image_net_inputs  = np_to_torch(self.image).cuda()
        self.mask_net_inputs   = np_to_torch(self.image).cuda()
        self.ambient_net_input = np_to_torch(self.image).cuda()

    def _init_all(self):
        self._init_images()
        self._init_nets()
        self._init_ambient()
        self._init_inputs()
        self._init_parameters()
        self._init_loss()

    def optimize(self):
        torch.backends.cudnn.enabled   = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure()
            self._obtain_current_result(j)
            self._plot_closure(j)
            optimizer.step()

    def _optimization_closure(self):
        self.image_out   = self.image_net(self.image_net_inputs)
        self.ambient_out = self.ambient_net(self.ambient_net_input)

        self.mask_out    = self.mask_net(self.mask_net_inputs)

        self.blur_out    = self.blur_loss(self.mask_out)
        self.mseloss     = self.mse_loss(self.mask_out * self.image_out + (1 - self.mask_out) * self.ambient_out, self.images_torch)

        vae_loss         = self.ambient_net.getLoss()
        self.total_loss  = self.mseloss + vae_loss
        self.total_loss += 0.005 * self.blur_out

        dcp_prior        = torch.min(self.image_out.permute(0, 2, 3, 1), 3)[0]
        self.dcp_loss    = self.mse_loss(dcp_prior, torch.zeros_like(dcp_prior)) - 0.05
        self.total_loss += self.dcp_loss

        self.total_loss += 0.1 * self.blur_loss(self.ambient_out)
        self.total_loss += self.mse_loss(self.ambient_out, self.ambient_val * torch.ones_like(self.ambient_out))

        self.total_loss.backward(retain_graph=True)

    def _obtain_current_result(self, step):
        if step % 5 == 0:
            image_out_np        = np.clip(torch_to_np(self.image_out), 0, 1)
            mask_out_np         = np.clip(torch_to_np(self.mask_out), 0, 1)
            ambient_out_np      = np.clip(torch_to_np(self.ambient_out), 0, 1)
            mask_out_np         = self.t_matting(mask_out_np)
            self.current_result = DehazeResult(learned=image_out_np, t=mask_out_np, a=ambient_out_np)

    def _plot_closure(self, step):
        """
         :param step: the number of the iteration

         :return:
         """
        console.log('Iteration %05d    Loss %f  %f' % (step, self.total_loss.item(), self.blur_out.item()), '\r', end='')

    def finalize(self):
        self.final_t_map = np_imresize(self.current_result.t, output_shape=self.original_image.shape[1:])
        self.final_a     = np_imresize(self.current_result.a, output_shape=self.original_image.shape[1:])
        mask_out_np      = self.t_matting(self.final_t_map)
        post             = np.clip((self.original_image - ((1 - mask_out_np) * self.final_a)) / mask_out_np, 0, 1)
        save_image(self.image_name + "-final", post, self.output_path)
        save_image(self.image_name + "-t", self.final_t_map, self.output_path)
        save_image(self.image_name + "-a", self.final_a, self.output_path)
        save_image(self.image_name + "-mask", mask_out_np, self.output_path)

    def t_matting(self, mask_out_np):
        refine_t = guidedFilter(self.original_image.transpose(1, 2, 0).astype(np.float32), mask_out_np[0].astype(np.float32), 50, 1e-4)
        if self.clip:
            return np.array([np.clip(refine_t, 0.1, 1)])
        else:
            return np.array([np.clip(refine_t, 0, 1)])


def dehaze(args: dict):
    assert args["image"] is not None and mon.Path(args["image"]).is_dir()

    image_dir = mon.Path(args["image"])
    if args["output"] is None:
        output_dir = image_dir.parent / f"{image_dir.stem}-hazefree"
    else:
        output_dir = mon.Path(args["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(image_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)

    num_iter = args["num_iter"]
    with mon.get_progress_bar() as pbar:
        for f in pbar.track(
            sequence    = image_files,
            total       = len(image_files),
            description = f"[bright_yellow] Dehazing"
        ):
            image = prepare_hazy_image(str(f))
            dh    = Dehaze(str(f.stem), image, num_iter, clip=True, output_path=str(output_dir) + "/")
            dh.optimize()
            dh.finalize()
            # save_image(name + "_original", np.clip(image, 0, 1), dh.output_path)


# region Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",    type=str, default="data/",   help="Image directory.")
    parser.add_argument("--output",   type=str, default="output/", help="Output directory.")
    parser.add_argument("--num-iter", type=int, default=500,       help="Number of iterations per image.")
    parser.add_argument("--verbose",  action="store_true",         help="Display results.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = vars(parse_args())
    dehaze(args=args)


"""
if __name__ == "__main__":
    torch.cuda.set_device(0)

    hazy_add = "data/DSC00574.png"
    name     = "DSC00574"
    print(name)

    hazy_img = prepare_hazy_image(hazy_add)

    args = {
        "image"   : "data/",
        "output"  : "output/",
        "num_iter": 500,
    }
    dehaze(args)
"""

# endregion
