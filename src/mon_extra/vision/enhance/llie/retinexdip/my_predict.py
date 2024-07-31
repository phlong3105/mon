#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # https://github.com/zhaozunjin/RetinexDIP

from __future__ import annotations

import argparse
import math
import time
from collections import namedtuple

import cv2
from torchvision import transforms

import mon
from mon import nn
from net import *
from net.losses import ExclusionLoss, GradientLoss, TVLoss
from net.noise import get_noise
from utils.image_io import *

console           = mon.console
_current_file     = mon.Path(__file__).absolute()
_current_dir      = current_file.parents[0]
EnhancementResult = namedtuple("EnhancementResult", ["reflection", "illumination"])

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


# region Predict

class Enhancement(object):
    
    def __init__(
        self,
        image_name,
        image,
        plot_during_training: bool = True,
        show_every          : int  = 10,
        num_iter            : int  = 300
    ):
        self.image                   = image
        self.img                     = image
        self.size                    = image.size
        self.image_np                = None
        self.images_torch            = None
        self.plot_during_training    = plot_during_training
        # self.ratio                 = ratio
        self.psnrs                   = []
        self.show_every              = show_every
        self.image_name              = image_name
        self.num_iter                = num_iter
        self.loss_function           = None
        # self.ratio_net             = None
        self.parameters              = None
        self.learning_rate           = 0.01
        self.input_depth             = 8
        # This value could affect the performance. 3 is ok for natural image,
        # if your images are extremely dark, you may consider 8 for the value.
        self.data_type               = torch.cuda.FloatTensor
        # self.data_type             = torch.FloatTensor
        self.reflection_net_inputs   = None
        self.illumination_net_inputs = None
        self.original_illumination   = None
        self.original_reflection     = None
        self.reflection_net          = None
        self.illumination_net        = None
        self.total_loss              = None
        self.reflection_out          = None
        self.illumination_out        = None
        self.current_result          = None
        self.best_result             = None
        self._init_all()

    def _init_all(self):
        self._init_images()
        self._init_decomposition()
        self._init_nets()
        self._init_inputs()
        self._init_parameters()
        self._init_losses()

    def _maxRGB(self):
        """
        self.image: pil image, input low-light image
        :return: np, initial illumination.
        """
        (R, G, B) = self.image.split()
        I_0       = np.array(np.maximum(np.maximum(R, G), B))
        return I_0

    def _init_decomposition(self):
        temp = self._maxRGB()  # numpy
        # get initial illumination map
        self.original_illumination = np.clip(np.asarray([temp for _ in range(3)]), 1, 255) / 255
        # self.original_illumination = np.clip(temp,1, 255) / 255
        # get initial reflection.
        self.original_reflection   = self.image_np / self.original_illumination
        self.original_illumination = np_to_torch(self.original_illumination).type(self.data_type)
        self.original_reflection   = np_to_torch(np.asarray(self.original_reflection)).type(self.data_type)
        # print(self.original_reflection.shape)namedtuple
        # print(self.original_illumination.shape)

    def _init_images(self):
        # self.images       = create_augmentations(self.image)
        # self.images_torch = [np_to_torch(image).type(torch.cuda.FloatTensor) for image in self.images]
        self.image       = transforms.Resize((512, 512))(self.image)
        self.image_np    = pil_to_np(self.image)  # pil image to numpy
        self.image_torch = np_to_torch(self.image_np).type(self.data_type)
        # print(self.size)
        # print((self.image_torch.shape[2], self.image_torch.shape[3]))

    def _init_inputs(self):
        if self.image_torch is not None:
            size = (self.image_torch.shape[2], self.image_torch.shape[3])
        input_type = "noise"
        # input_type = "meshgrid"
        self.reflection_net_inputs   = get_noise(self.input_depth, input_type, size).type(self.data_type).detach()
        # misc.imsave("out/input_illumination.png", misc.imresize(torch_to_np(self.reflection_net_inputs).transpose(1,  2, 0), (self.size[1], self.size[0])))
        self.illumination_net_inputs = get_noise(self.input_depth, input_type, size).type(self.data_type).detach()

    def _init_parameters(self):
        self.parameters = [p for p in self.reflection_net.parameters()] + \
                          [p for p in self.illumination_net.parameters()]

    def weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0.0, 0.5 * math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find("BatchNorm") != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif classname.find("Linear") != -1:
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data = torch.ones(m.bias.data.size())

    def _init_nets(self):
        pad = "zero"
        self.reflection_net = skip(
            self.input_depth,
            num_output_channels = 3,
            num_channels_down   = [8, 16, 32, 64, 128],
            num_channels_up     = [8, 16, 32, 64, 128],
            num_channels_skip   = [0, 0 , 0 , 0 , 0],
            filter_size_down    = 3,
            filter_size_up      = 3,
            filter_skip_size    = 1,
            upsample_mode       = "bilinear",
            downsample_mode     = "avg",
            need_sigmoid        = True,
            need_bias           = True,
            pad                 = pad
        )
        self.reflection_net.apply(self.weight_init).type(self.data_type)
        self.illumination_net = skip(
            self.input_depth,
            num_output_channels = 3,
            num_channels_down   = [8, 16, 32, 64],
            num_channels_up     = [8, 16, 32, 64],
            num_channels_skip   = [0, 0 , 0 , 0],
            filter_size_down    = 3,
            filter_size_up      = 3,
            filter_skip_size    = 1,
            upsample_mode       = "bilinear",
            downsample_mode     = "avg",
            need_sigmoid        = True,
            need_bias           = True,
            pad                 = pad
        )
        self.illumination_net.apply(self.weight_init).type(self.data_type)

    def _init_losses(self):
        self.l1_loss        = nn.SmoothL1Loss().type(self.data_type)  # for illumination
        self.mse_loss       = nn.MSELoss().type(self.data_type)     # for reflection and reconstruction
        self.exclusion_loss =  ExclusionLoss().type(self.data_type)
        self.tv_loss        = TVLoss().type(self.data_type)
        self.gradient_loss  = GradientLoss().type(self.data_type)

    def optimize(self):
        # torch.backends.cudnn.enabled   = True
        # torch.backends.cudnn.benchmark = True
        # optimizer = SGLD(self.parameters, lr=self.learning_rate)
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        # print("Processing: {}".format(self.image_name.split("/")[-1]))
        start     = time.time()
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure(500, 499)
            if j == 499:
                self._obtain_current_result(499)
            if self.plot_during_training:
                self._plot_closure(j)
            optimizer.step()
        end      = time.time()
        run_time = end - start
        # print("time:%.4f" % (end - start))
        # cv2.imwrite(str(self.image_name), self.best_result)
        self.get_enhanced(self.num_iter - 1, flag=True)
        return self.best_result, run_time
    
    def _get_augmentation(self, iteration):
        if iteration % 2 == 1:
            return 0
        # return 0
        iteration //= 2
        return iteration % 8

    def _optimization_closure(self, num_iter, step):
        reg_noise_std = 1 / 10000.0
        aug = self._get_augmentation(step)
        if step == num_iter - 1:
            aug = 0

        illumination_net_input = (
            self.illumination_net_inputs +
            (self.illumination_net_inputs.clone().normal_() * reg_noise_std)
        )
        reflection_net_input = (
            self.reflection_net_inputs +
            (self.reflection_net_inputs.clone().normal_() * reg_noise_std)
        )
        self.illumination_out = self.illumination_net(illumination_net_input)
        self.reflection_out   = self.reflection_net(reflection_net_input)

        # Weighted with the gradient of latent reflectance
        self.total_loss  = 0.5    * self.tv_loss(self.illumination_out, self.reflection_out)
        self.total_loss += 0.0001 * self.tv_loss(self.reflection_out)
        self.total_loss += self.l1_loss(self.illumination_out, self.original_illumination)
        self.total_loss += self.mse_loss(self.illumination_out*self.reflection_out, self.image_torch)
        self.total_loss.requires_grad = True
        self.total_loss.backward()
    
    def _obtain_current_result(self, step):
        """Puts in self.current result the current result.
        Also updates the best result.
        """
        if step == self.num_iter - 1 or step % 8 == 0:
            reflection_out_np   = np.clip(torch_to_np(self.reflection_out), 0, 1)
            illumination_out_np = np.clip(torch_to_np(self.illumination_out), 0, 1)
            # psnr = compare_psnr(np.clip(self.image_np,0,1), reflection_out_np * illumination_out_np)
            # self.psnrs.append(psnr)
            self.current_result = EnhancementResult(reflection=reflection_out_np, illumination=illumination_out_np)
            # if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
            #     self.best_result = self.current_result
    
    def _plot_closure(self, step):
        # print("Iteration {:5d}    Loss {:5f}".format(step, self.total_loss.item()))
        if step % self.show_every == self.show_every - 1:
            # plot_image_grid("left_right_{}".format(step),
            #                 [self.current_result.reflection, self.current_result.illumination])
            # misc.imsave('out/illumination.png',
            #             misc.imresize(torch_to_np(self.illumination_out).transpose(1, 2, 0),(self.size[1],self.size[0])))

            # misc.imsave(
            #     "output/reflection/reflection-{}.png".format(step),
            #     misc.imresize(torch_to_np(self.reflection_out).transpose(1, 2, 0), (self.size[1], self.size[0]))
            # )
            self.get_enhanced(step, flag=True)
        
    def gamma_trans(self, image, gamma):
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        output      = cv2.LUT((255 * image).astype(np.uint8), gamma_table)
        output      = (output.astype(np.float32)) / 255
        return output

    def adjust_gammma(self, image_gray):
        # mean = np.mean(img_gray)
        # gamma_val = math.log10(0.5) / math.log10(mean / 255)
        # print(gamma_val)
        image_gamma_correct = self.gamma_trans(image_gray, 0.5)
        return image_gamma_correct

    def get_enhanced(self, step, flag: bool = False):
        (R, G, B)        = self.img.split()
        ini_illumination = torch_to_np(self.illumination_out).transpose(1, 2, 0)
        ini_illumination = cv2.resize(ini_illumination, (self.size[0], self.size[1]))
        # print(ini_illumination.shape)
        ini_illumination = np.max(ini_illumination, axis=2)
        # cv2.imwrite("output/illumination/illumination-{}.png".format(step), ini_illumination)
        # If the input image is extremely dark, setting the flag as True can produce promising result.
        if flag:
            # ini_illumination = np.clip(np.max(ini_illumination, axis=2), 0.0000002, 255)
            ini_illumination = np.clip(ini_illumination, 0.0000002, 255)
        else:
            ini_illumination = np.clip(self.adjust_gammma(ini_illumination), 0.0000002, 255)
        R = R / ini_illumination
        G = G / ini_illumination
        B = B / ini_illumination
        self.best_result = np.clip(cv2.merge([B, G, R]) * 255, 0.02, 255).astype(np.uint8)
        # cv2.imwrite(str(self.image_name), self.best_result)
    
    def calculate_efficiency_score(
        self,
        image_size: int | list[int] = 512,
        channels  : int             = 8,
        runs      : int             = 100,
    ):
        flops_1, params_1, avg_time_1 = mon.calculate_efficiency_score(
            model      = self.illumination_net,
            image_size = image_size,
            channels   = channels,
            runs       = runs,
            use_cuda   = True,
            verbose    = False,
        )
        flops_2, params_2, avg_time_2 = mon.calculate_efficiency_score(
            model      = self.reflection_net,
            image_size = image_size,
            channels   = channels,
            runs       = runs,
            use_cuda   = True,
            verbose    = False,
        )
        flops    = flops_1 + flops_2
        params   = params_1 + params_2
        avg_time = (avg_time_1 + avg_time_2) / 2
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.4f}")
    

def lowlight_enhancer(image_name, image):
    s = Enhancement(
        image_name           = image_name,
        image                = image,
        plot_during_training = True,
        show_every           = 10,
        num_iter             = 500,
    )
    return s.optimize()


def predict(args: argparse.Namespace):
    # General config
    data      = args.data
    save_dir  = args.save_dir
    # weights   = args.weights
    # weights   = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    device    = mon.set_device(args.device)
    imgsz     = args.imgsz
    resize    = args.resize
    benchmark = args.benchmark
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(src=data, dst=save_dir, denormalize=True)
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicting
    sum_time = 0
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for i, (images, target, meta) in pbar.track(
                sequence    = enumerate(data_loader),
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                image_path   = meta["path"]
                image        = Image.open(image_path).convert("RGB")
                enhanced_image, run_time = lowlight_enhancer(str(image_path), image)
                output_path  = save_dir / image_path.name
                # torchvision.utils.save_image(enhanced_image, str(output_path))
                cv2.imwrite(str(output_path), enhanced_image)
                sum_time    += run_time
                
                # Benchmark
                if i == 0:
                    s = Enhancement(
                        image_name           = output_path,
                        image                = image,
                        plot_during_training = True,
                        show_every           = 10,
                        num_iter             = 500,
                    )
                    if benchmark:
                        s.calculate_efficiency_score(image_size=imgsz, channels=8, runs=100)

        avg_time = float(sum_time / len(data_loader))
        console.log(f"Average time: {avg_time}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=_current_dir)
    predict(args)


if __name__ == "__main__":
    main()

# endregion
