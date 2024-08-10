#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import namedtuple

import torch.nn as nn
import torch.optim
from cv2.ximgproc import guidedFilter

import mon
from net import *
from net.losses import StdLoss
from net.vae_model import VAE
from utils.dcp import get_atmosphere
from utils.image_io import *
from utils.imresize import np_imresize

_root = pathlib.Path(__file__).resolve().parents[0]  # root directory
if str(_root) not in sys.path:
    sys.path.append(str(_root))  # add ROOT to PATH

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
DehazeResult = namedtuple("DehazeResult", ["learned", "t", "a"])


# region Predict

class Dehaze:
    
    def __init__(self, image_name, image, num_iter=500, clip=True, output_path="output"):
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
        self.images_torch   = np_to_torch(self.image).type(torch.cuda.FloatTensor)
    
    def _init_nets(self):
        input_depth = self.input_depth
        data_type   = torch.cuda.FloatTensor
        pad         = "reflection"
        
        image_net = skip(
            num_input_channels  = input_depth,
            num_output_channels = 3,
            num_channels_down   = [8, 16, 32, 64, 128],
            num_channels_up     = [8, 16, 32, 64, 128],
            num_channels_skip   = [0, 0 , 0 , 4 , 4],
            upsample_mode       = "bilinear",
            need_sigmoid        = True,
            need_bias           = True,
            pad                 = pad,
            act_fun             = "LeakyReLU"
        )
        self.image_net = image_net.type(data_type)
        
        mask_net = skip(
            num_input_channels  = input_depth,
            num_output_channels = 1,
            num_channels_down   = [8, 16, 32, 64, 128],
            num_channels_up     = [8, 16, 32, 64, 128],
            num_channels_skip   = [0, 0 , 0 , 4 , 4],
            upsample_mode       = "bilinear",
            need_sigmoid        = True,
            need_bias           = True,
            pad                 = pad,
            act_fun             = "LeakyReLU"
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
        # print(f"final_t_map: {self.final_t_map.shape}")
        mask_out_np      = self.t_matting(self.final_t_map)
        # print(f"final_t_map: {self.final_t_map.shape}")
        # print(f"original_image: {self.original_image.shape}")
        post             = np.clip((self.original_image - ((1 - mask_out_np) * self.final_a)) / mask_out_np, 0, 1)
        save_image(self.image_name, post            , self.output_path)
        save_image(self.image_name, self.final_t_map, self.output_path + "_debug/" + "t/")
        save_image(self.image_name, self.final_a    , self.output_path + "_debug/" + "a/")
        save_image(self.image_name, mask_out_np     , self.output_path + "_debug/" + "mask/")
    
    def t_matting(self, mask_out_np):
        refine_t = guidedFilter(
            guide  = self.original_image.transpose(1, 2, 0).astype(np.float32),
            src    = mask_out_np[0].astype(np.float32),
            radius = 50,
            eps    = 1e-4,
        )
        if self.clip:
            return np.array([np.clip(refine_t, 0.1, 1)])
        else:
            return np.array([np.clip(refine_t, 0, 1)])


def predict(args: argparse.Namespace):
    # General config
    data       = args.data
    save_dir   = args.save_dir
    weights    = args.weights
    device     = mon.set_device(args.device)
    epochs     = args.epochs
    imgsz      = args.imgsz
    resize     = args.resize
    benchmark  = args.ben
    save_debug = args.save_debug
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(
        src         = data,
        dst         = save_dir,
        to_tensor   = False,
        denormalize = True,
        verbose     = False,
    )
    debug_dir = save_debug / f"{data_name}_debug"
    save_dir  = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir /    "t").mkdir(parents=True, exist_ok=True)
    (debug_dir /    "a").mkdir(parents=True, exist_ok=True)
    (debug_dir / "mask").mkdir(parents=True, exist_ok=True)
    
    # Predicting
    timer = mon.Timer()
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(data_loader),
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                meta       = datapoint.get("meta")
                image_path = meta["path"]
                image      = prepare_hazy_image(str(image_path))
                timer.tick()
                dh = Dehaze(str(image_path.stem), image, epochs, clip=True, output_path=str(save_dir))
                dh.optimize()
                dh.finalize()
                timer.tock()
        avg_time = float(timer.avg_time)
        console.log(f"Average time: {avg_time}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
    
# endregion
