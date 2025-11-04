#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements RetinexNet model for low-light image enhancement.

References:
    - Paper: "Deep Retinex Decomposition for Low-Light Enhancement," BMCV 2018.
    - Code: https://github.com/aasharma90/RetinexNet_PyTorch
"""

__all__ = [
    "RetinexNet",
]

import os
import random
import time

import box
import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable

from mon.constants import MODELS
from mon.core import get_model_device, log, MLType, nn, Path, Task
from mon.core.nn import functional as F
from .module import (
    calculate_efficiency_score_decomnet,
    calculate_efficiency_score_enhancenet,
    DecomNet,
    RelightNet,
)

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


@MODELS.register(name="retinexnet", arch="retinexnet")
class RetinexNet(nn.Module):
    """RetinexNet model for low-light image enhancement.
    
    References:
        - Paper: "Deep Retinex Decomposition for Low-Light Enhancement," BMCV 2018.
        - Code: https://github.com/aasharma90/RetinexNet_PyTorch
    """
    
    arch     : str          = "retinexnet"
    name     : str          = "retinexnet"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, imgsz: int = 512, benchmark: bool = False):
        super().__init__()
        self.DecomNet   = DecomNet()
        self.RelightNet = RelightNet()
        if benchmark:
            flops1, params1 =   calculate_efficiency_score_decomnet(model=self.DecomNet,   imgsz=imgsz)
            flops2, params2 = calculate_efficiency_score_enhancenet(model=self.RelightNet, imgsz=imgsz)
            log(f"FLOPs  = {flops1  + flops2:.4f}")
            log(f"Params = {params1 + params2:.4f}")

    def forward(self, input_low, input_high):
        # Forward DecompNet
        device         = get_model_device(self)
        input_low      = Variable(torch.FloatTensor(torch.from_numpy(input_low))).to(device)
        input_high     = Variable(torch.FloatTensor(torch.from_numpy(input_high))).to(device)
        R_low, I_low   = self.DecomNet(input_low)
        R_high, I_high = self.DecomNet(input_high)

        # Forward RelightNet
        I_delta   = self.RelightNet(I_low, R_low)

        # Other variables
        I_low_3   = torch.cat((I_low,   I_low,   I_low),   dim=1)
        I_high_3  = torch.cat((I_high,  I_high,  I_high),  dim=1)
        I_delta_3 = torch.cat((I_delta, I_delta, I_delta), dim=1)

        # Compute losses
        self.recon_loss_low        = F.l1_loss(R_low * I_low_3,  input_low)
        self.recon_loss_high       = F.l1_loss(R_high * I_high_3, input_high)
        self.recon_loss_mutal_low  = F.l1_loss(R_high * I_low_3, input_low)
        self.recon_loss_mutal_high = F.l1_loss(R_low * I_high_3, input_high)
        self.equal_R_loss          = F.l1_loss(R_low,  R_high.detach())
        self.relight_loss          = F.l1_loss(R_low * I_delta_3, input_high)

        self.Ismooth_loss_low      = self.smooth(I_low, R_low)
        self.Ismooth_loss_high     = self.smooth(I_high, R_high)
        self.Ismooth_loss_delta    = self.smooth(I_delta, R_low)

        self.loss_Decom = self.recon_loss_low + \
                          self.recon_loss_high + \
                          0.001 * self.recon_loss_mutal_low + \
                          0.001 * self.recon_loss_mutal_high + \
                          0.1   * self.Ismooth_loss_low + \
                          0.1   * self.Ismooth_loss_high + \
                          0.01  * self.equal_R_loss
        self.loss_Relight = self.relight_loss + 3 * self.Ismooth_loss_delta

        self.output_R_low   = R_low.detach().cpu()
        self.output_I_low   = I_low_3.detach().cpu()
        self.output_I_delta = I_delta_3.detach().cpu()
        self.output_S       = R_low.detach().cpu() * I_delta_3.detach().cpu()

    def gradient(self, input_tensor, direction):
        device = get_model_device(self)
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).to(device)
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction), kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R = 0.299 * input_R[:, 0, :, :] + 0.587 * input_R[:, 1, :, :] + 0.114 * input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(
            self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
            self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y"))
        )

    def evaluate(self, epoch_num, eval_low_data_names, vis_dir, train_phase):
        print("Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data_names)):
            eval_low_img   = Image.open(eval_low_data_names[idx])
            eval_low_img   = np.array(eval_low_img, dtype="float32") / 255.0
            eval_low_img   = np.transpose(eval_low_img, (2, 0, 1))
            input_low_eval = np.expand_dims(eval_low_img, axis=0)

            if train_phase == "decom":
                self.forward(input_low_eval, input_low_eval)
                result_1  = self.output_R_low
                result_2  = self.output_I_low
                input     = np.squeeze(input_low_eval)
                result_1  = np.squeeze(result_1)
                result_2  = np.squeeze(result_2)
                cat_image = np.concatenate([input, result_1, result_2], axis=2)
            if train_phase == "relight":
                self.forward(input_low_eval, input_low_eval)
                result_1  = self.output_R_low
                result_2  = self.output_I_low
                result_3  = self.output_I_delta
                result_4  = self.output_S
                input     = np.squeeze(input_low_eval)
                result_1  = np.squeeze(result_1)
                result_2  = np.squeeze(result_2)
                result_3  = np.squeeze(result_3)
                result_4  = np.squeeze(result_4)
                cat_image = np.concatenate([input, result_1, result_2, result_3, result_4], axis=2)

            cat_image = np.transpose(cat_image, (1, 2, 0))
            # print(cat_image.shape)
            im       = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype("uint8"))
            filepath = os.path.join(vis_dir, "eval_%s_%d_%d.png" % (train_phase, idx + 1, epoch_num))
            im.save(filepath[:-4] + ".jpg")

    def save(self, iter_num, ckpt_dir):
        save_dir  = ckpt_dir + "/" + self.train_phase + "/"
        save_name = save_dir + "/" + str(iter_num) + ".tar"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.train_phase == "decom":
            torch.save(self.DecomNet.state_dict(), save_name)
        elif self.train_phase == "relight":
            torch.save(self.RelightNet.state_dict(),save_name)

    def load(self, ckpt_dir):
        load_dir = ckpt_dir + "/" + self.train_phase + "/"
        if os.path.exists(load_dir):
            load_ckpts = os.listdir(load_dir)
            load_ckpts.sort()
            load_ckpts = sorted(load_ckpts, key=len)
            if len(load_ckpts) > 0:
                load_ckpt   = load_ckpts[-1]
                global_step = int(load_ckpt[:-4])
                ckpt_dict   = torch.load(load_dir + load_ckpt)
                if self.train_phase == "decom":
                    self.DecomNet.load_state_dict(ckpt_dict)
                elif self.train_phase == "relight":
                    self.RelightNet.load_state_dict(ckpt_dict)
                return True, global_step
            else:
                return False, 0
        else:
            return False, 0
    
    def train(
        self,
        train_low_data_names,
        train_high_data_names,
        eval_low_data_names,
        batch_size,
        patch_size,
        epoch,
        lr,
        vis_dir,
        ckpt_dir,
        eval_every_epoch,
        train_phase
    ):
        assert len(train_low_data_names) == len(train_high_data_names)
        numBatch = len(train_low_data_names) // int(batch_size)

        # Create the optimizers
        self.train_op_Decom   = nn.Adam(self.DecomNet.parameters(), lr=lr[0], betas=(0.9, 0.999))
        self.train_op_Relight = nn.Adam(self.RelightNet.parameters(), lr=lr[0], betas=(0.9, 0.999))

        # Initialize a network if its checkpoint is available
        self.train_phase = train_phase
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num    = global_step
            start_epoch = global_step // numBatch
            start_step  = global_step % numBatch
            print("Model restore success!")
        else:
            iter_num    = 0
            start_epoch = 0
            start_step  = 0
            print("No pretrained model to restore!")

        print("Start training for phase %s, with start epoch %d start iter %d : " %
             (self.train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id   = 0
        for epoch in range(start_epoch, epoch):
            self.lr = lr[epoch]
            # Adjust learning rate
            for param_group in self.train_op_Decom.param_groups:
                param_group["lr"] = self.lr
            for param_group in self.train_op_Relight.param_groups:
                param_group["lr"] = self.lr
            for batch_id in range(start_step, numBatch):
                # Generate training data for a batch
                batch_input_low  = np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")
                batch_input_high = np.zeros((batch_size, 3, patch_size, patch_size,), dtype="float32")
                for patch_id in range(batch_size):
                    # Load images
                    train_low_img  = Image.open(train_low_data_names[image_id])
                    train_low_img  = np.array(train_low_img, dtype="float32")/255.0
                    train_high_img = Image.open(train_high_data_names[image_id])
                    train_high_img = np.array(train_high_img, dtype="float32")/255.0
                    # Take random crops
                    h, w, _        = train_low_img.shape
                    x              = random.randint(0, h - patch_size)
                    y              = random.randint(0, w - patch_size)
                    train_low_img  = train_low_img[x: x + patch_size, y: y + patch_size, :]
                    train_high_img = train_high_img[x: x + patch_size, y: y + patch_size, :]
                    # Data augmentation
                    if random.random() < 0.5:
                        train_low_img  = np.flipud(train_low_img)
                        train_high_img = np.flipud(train_high_img)
                    if random.random() < 0.5:
                        train_low_img  = np.fliplr(train_low_img)
                        train_high_img = np.fliplr(train_high_img)
                    rot_type = random.randint(1, 4)
                    if random.random() < 0.5:
                        train_low_img  = np.rot90(train_low_img, rot_type)
                        train_high_img = np.rot90(train_high_img, rot_type)
                    # Permute the images to tensor format
                    train_low_img  = np.transpose(train_low_img, (2, 0, 1))
                    train_high_img = np.transpose(train_high_img, (2, 0, 1))
                    # Prepare the batch
                    batch_input_low[patch_id, :, :, :]  = train_low_img
                    batch_input_high[patch_id, :, :, :] = train_high_img
                    self.input_low  = batch_input_low
                    self.input_high = batch_input_high

                    image_id = (image_id + 1) % len(train_low_data_names)
                    if image_id == 0:
                        tmp = list(zip(train_low_data_names, train_high_data_names))
                        random.shuffle(list(tmp))
                        train_low_data_names, train_high_data_names = zip(*tmp)

                # Feed-Forward to the network and obtain loss
                self.forward(self.input_low,  self.input_high)
                if self.train_phase == "decom":
                    self.train_op_Decom.zero_grad()
                    self.loss_Decom.backward()
                    self.train_op_Decom.step()
                    loss = self.loss_Decom.item()
                elif self.train_phase == "relight":
                    self.train_op_Relight.zero_grad()
                    self.loss_Relight.backward()
                    self.train_op_Relight.step()
                    loss = self.loss_Relight.item()

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1

            # Evaluate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data_names, vis_dir=vis_dir, train_phase=train_phase)
                self.save(iter_num, ckpt_dir)

        print("Finished training for phase %s." % train_phase)
    
    def predict(
        self,
        test_low_data_names,
        res_dir,
        ckpt_dir,
        imgsz : int  = 512,
        resize: bool = False,
    ):
        # Load the network with a pre-trained checkpoint
        self.train_phase     = "decom"
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            # print(self.train_phase, "  : Model restore success!")
            pass
        else:
            weights_file = str(Path(ckpt_dir) / "retinexnet_lolv1_decom_9200.tar")
            ckpt_dict    = torch.load(weights_file, weights_only=True)
            self.DecomNet.load_state_dict(ckpt_dict)
            # print(f"No pretrained model to restore! Use default pretrained weights: {weights_file}")
        
        self.train_phase     = "relight"
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            # print(self.train_phase, ": Model restore success!")
            pass
        else:
            weights_file = str(Path(ckpt_dir) / "retinexnet_lolv1_relight_9200.tar")
            ckpt_dict    = torch.load(weights_file, weights_only=True)
            self.RelightNet.load_state_dict(ckpt_dict)
            # print(f"No pretrained model to restore! Use default pretrained weights: {weights_file}")
            
        # Set this switch to True to also save the reflectance and shading maps
        save_R_L = False
        
        # Predict for the test images
        for idx in range(len(test_low_data_names)):
            test_img_path = test_low_data_names[idx]
            # test_img_name = test_img_path.stem
            # test_img_name  = test_img_path.split("/")[-1]
            # print("Processing ", test_img_name)
            test_low_img   = Image.open(test_img_path)
            test_low_img   = np.array(test_low_img, dtype="float32") / 255.0
            if resize:
                h, w, c      = test_low_img.shape
                imgsz        = (imgsz, imgsz) if isinstance(imgsz, (int, float)) else imgsz
                test_low_img = cv2.resize(test_low_img, imgsz)
            test_low_img   = np.transpose(test_low_img, (2, 0, 1))
            input_low_test = np.expand_dims(test_low_img, axis=0)

            self.forward(input_low_test, input_low_test)
            result_1 = self.output_R_low
            result_2 = self.output_I_low
            result_3 = self.output_I_delta
            result_4 = self.output_S
            input    = np.squeeze(input_low_test)
            result_1 = np.squeeze(result_1)
            result_2 = np.squeeze(result_2)
            result_3 = np.squeeze(result_3)
            result_4 = np.squeeze(result_4)
            if save_R_L:
                cat_image = np.concatenate([input, result_1, result_2, result_3, result_4], axis=2)
            else:
                cat_image = np.concatenate([result_4], axis=2)

            cat_image = np.transpose(cat_image, (1, 2, 0))
            if resize:
                cat_image = cv2.resize(cat_image, (w, h))
                # print(cat_image.shape)
            im       = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype("uint8"))
            filepath = Path(res_dir) / test_img_path.name
            im.save(str(filepath))
