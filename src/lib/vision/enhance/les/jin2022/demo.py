#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

import cv2
import torch.optim as optim
import torch.utils.data as Data
from guided_filter_pytorch.guided_filter import GuidedFilter
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch.autograd import Variable
from torchvision import utils as vutils

import load_data as DA
from mon import RUN_DIR
from net import *

console = mon.console


def get_LFHF(image, rad_list=[4, 8, 16, 32], eps_list=[0.001, 0.0001]):

    def decomposition(guide, inp, rad_list, eps_list):
        LF_list = []
        HF_list = []
        for radius in rad_list:
            for eps in eps_list:
                gf = GuidedFilter(radius, eps)
                LF = gf(guide, inp)
                LF[LF > 1] = 1
                LF_list.append(LF)
                HF_list.append(inp - LF)
        LF = torch.cat(LF_list, dim=1)
        HF = torch.cat(HF_list, dim=1)
        return LF, HF

    image = torch.clamp(image, min=0.0, max=1.0)
    # Compute the LF-HF features of the image
    img_lf, img_hf = decomposition(guide=image, inp=image, rad_list=rad_list, eps_list=eps_list)
    return img_lf, img_hf


class MeanShift(nn.Conv2d):

    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False
        

'''     
class Vgg16ExDark(torch.nn.Module):
    def __init__(self, load_model=None, requires_grad=False):
        super(Vgg16ExDark, self).__init__()
        # Create the model
        self.vgg_pretrained_features = visionmodels.vgg16(pretrained=True).features
        if load_model is None:
            console.log("Vgg16ExDark needs a pre-trained checkpoint!")
            raise Exception
        else:
            console.log("Vgg16ExDark initialized with %s"% load_model)
            model_state_dict = torch.load(load_model)
            model_dict       = self.vgg_pretrained_features.state_dict()
            model_state_dict = {k[16:]: v for k, v in model_state_dict.items() if k[16:] in model_dict}
            self.vgg_pretrained_features.load_state_dict(model_state_dict)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, indices=None):
        if indices is None:
            indices = [3, 8, 15, 22] 
        out = []
        for i in range(indices[-1] + 1):
            X = self.vgg_pretrained_features[i](X)
            if i in indices:
                out.append(X)
        return out
'''

'''
class PerceptualLossVgg16ExDark(nn.Module):
    def __init__(
            self,
            vgg        = None,
            load_model = None,
            weights    = None,
            indices    = None,
            normalize  = True
    ):
        super(PerceptualLossVgg16ExDark, self).__init__()        
        if vgg is None:
            self.vgg = Vgg16ExDark(load_model)
        else:
            self.vgg = vgg
        self.vgg     = self.vgg.cuda()
        self.criter  = nn.L1Loss()
        self.weights = weights or [1.0, 1.0, 1.0, 1.0]
        self.indices = indices or [3, 8, 15, 22]
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criter(x_vgg[i], y_vgg[i].detach())
        return loss
'''

'''
class StdLoss(nn.Module):
    def __init__(self):
        super(StdLoss, self).__init__()
        blur            = (1 / 25) * np.ones((5, 5))
        blur            = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
        self.mse        = nn.MSELoss()
        self.blur       = nn.Parameter(data=torch.cuda.FloatTensor(blur), requires_grad=False)
        image           = np.zeros((5, 5))
        image[2, 2]     = 1
        image           = image.reshape(1, 1, image.shape[0], image.shape[1])
        self.image      = nn.Parameter(data=torch.cuda.FloatTensor(image), requires_grad=False)
        self.gray_scale = GrayscaleLayer()

    def forward(self, x):
        x = self.gray_scale(x)
        return self.mse(functional.conv2d(x, self.image), functional.conv2d(x, self.blur))
'''


class ExclusionLoss(nn.Module):

    def __init__(self, level=3):
        super(ExclusionLoss, self).__init__()
        self.level    = level
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2).type(torch.cuda.FloatTensor)
        self.sigmoid  = nn.Sigmoid().type(torch.cuda.FloatTensor)

    def get_gradients(self, img1, img2):
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)
            alphay         = 1
            alphax         = 1
            gradx1_s       = (self.sigmoid(gradx1) * 2) - 1
            grady1_s       = (self.sigmoid(grady1) * 2) - 1
            gradx2_s       = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s       = (self.sigmoid(grady2 * alphay) * 2) - 1
            gradx_loss    += self._all_comb(gradx1_s, gradx2_s)
            grady_loss    += self._all_comb(grady1_s, grady2_s)
            img1           = self.avg_pool(img1)
            img2           = self.avg_pool(img2)
        return gradx_loss, grady_loss

    def _all_comb(self, grad1_s, grad2_s):
        v = []
        for i in range(3):
            for j in range(3):
                v.append(torch.mean(((grad1_s[:, j, :, :] ** 2) * (grad2_s[:, i, :, :] ** 2))) ** 0.25)
        return v

    def forward(self, img1, img2):
        gradx_loss, grady_loss = self.get_gradients(img1, img2)
        loss_gradxy = sum(gradx_loss) / (self.level * 9) + sum(grady_loss) / (self.level * 9)
        return loss_gradxy / 2.0

    def compute_gradient(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady


def gradient(pred):
    D_dy = pred[:, :, 1:]    - pred[:, :, :-1]
    D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    return D_dx, D_dy


class GradientLoss(nn.Module):

    def __init__(self):
        super(GradientLoss, self).__init__()
        
    def forward(self, a):
        gradient_a_x = torch.abs(a[:, :, :, :-1] - a[:, :, :, 1:])
        gradient_a_y = torch.abs(a[:, :, :-1, :] - a[:, :, 1:, :])
        return torch.mean(gradient_a_x) + torch.mean(gradient_a_y)


def smooth_loss(pred_map):
    dx, dy    = gradient(pred_map)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    loss      =  (dx2.abs().mean()  + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())
    return loss


def rgb2gray(rgb):
    gray = 0.2989 * rgb[:, :, 0:1, :] + \
           0.5870 * rgb[:, :, 1:2, :] + \
           0.1140 * rgb[:, :, 2:3, :]
    return gray


def validate(dle_net, inputs):
    console.log("Validation not possible since there are no labels!")
    raise Exception


def calc_psnr(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_psnr(im1_y, im2_y)


def calc_ssim(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_ssim(im1_y, im2_y)


def calc_psnr_masked(im1, im2, mask):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_psnr(im1_y[mask], im2_y[mask])


def calc_ssim_masked(im1, im2, mask):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_ssim(im1_y[mask], im2_y[mask])


def demo(args, dle_net, optimizer_dle_net, inputs):
    dle_net.train()
    img_in = Variable(torch.FloatTensor(inputs["img_in"])).cuda()
    optimizer_dle_net.zero_grad()

    le_pred  = dle_net(img_in)
    dle_pred = img_in + le_pred

    lambda_cc      = 1.0
    dle_pred_cc    = torch.mean(dle_pred, dim=1, keepdims=True)
    cc_loss        = (F.l1_loss(dle_pred[:, 0:1, :, :], dle_pred_cc) + \
                     F.l1_loss(dle_pred[:, 1:2, :, :], dle_pred_cc) + \
                     F.l1_loss(dle_pred[:, 2:3, :, :], dle_pred_cc)) * (1/3)  # Color Constancy Loss
                   
    lambda_recon   = 1.0
    recon_loss     = F.l1_loss(dle_pred, img_in)

    lambda_excl    = 0.01
    data_type      = torch.cuda.FloatTensor
    excl_loss      = ExclusionLoss().type(data_type)

    lambda_smooth  = 1.0
    le_smooth_loss = smooth_loss(le_pred)

    loss  = lambda_recon  * recon_loss + lambda_cc * cc_loss
    loss += lambda_excl   * excl_loss(dle_pred, le_pred)
    loss += lambda_smooth * le_smooth_loss
    loss.backward()

    optimizer_dle_net.step()

    imgs_dict = {
        "dle_pred": dle_pred.detach().cpu(),
    }
    return imgs_dict


def main(args: args.Namespace):
    args.use_gray       = False
    args.input_dir      = mon.Path(args.input_dir)
    args.output_dir     = mon.Path(args.output_dir)
    args.checkpoint_dir = mon.Path(args.checkpoint_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)

    channels = 1 if args.use_gray else 3
    dle_net  = Net(input_nc=channels, output_nc=channels)

    # Measure efficiency score
    if args.benchmark:
        flops, params, avg_time = mon.calculate_efficiency_score(
            model      = dle_net,
            image_size = args.image_size,
            channels   = 3,
            runs       = 100,
            use_cuda   = True,
            verbose    = False,
        )
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.4f}")

    dle_net = nn.DataParallel(dle_net).cuda()
    if args.weights is not None:
        dle_net.load_state_dict(torch.load(str(args.weights))["state_dict"])

    optimizer_dle_net = optim.Adam(dle_net.parameters(), lr=args.lr, betas=(0.9, 0.999))

    #
    image_paths = list(args.input_dir.rglob("*"))
    image_paths = [str(path) for path in image_paths if path.is_image_file()]
    demo_list   = image_paths * args.iters
    loader      = torch.utils.data.DataLoader(
        DA.LoadImgs(args, demo_list, mode="demo"),
        batch_size  = 1,
        shuffle     = True,
        num_workers = 16,
        drop_last   = False
    )
    count_idx   = 0
    sum_time    = 0
    with mon.get_progress_bar() as pbar:
        for inputs, img_in_path in pbar.track(
            sequence    = loader,
            total       = len(loader),
            description = f"[bright_yellow] Inferring"
        ):
            count_idx = count_idx + 1
            imgs_dict = demo(args, dle_net, optimizer_dle_net, inputs)

            if count_idx % 60 == 0:
                img_in_path = mon.Path(img_in_path[0])
                inout       = args.output_dir / f"{img_in_path.stem}_in_out.png"
                out         = args.output_dir / f"{img_in_path.stem}_out.png"
                save_img    = torch.cat((inputs["img_in"][0, :, :, :], imgs_dict["dle_pred"][0, :, :, :]), dim=2)
                out_img     = imgs_dict["dle_pred"][0, :, :, :]
                vutils.save_image(save_img, str(inout))
                vutils.save_image(out_img,  str(out))
                torch.save(dle_net.state_dict(), str(args.checkpoint_dir / "best.pt"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",      type=str,   default="GOPR0364_frame_000939_rgb_anon.png",           help="Image to be used for demo")
    parser.add_argument("--output-dir",     type=str,   default=RUN_DIR / "predict/vision/enhance/les/jin2022", help="Location at which to save the light-effects suppression results.")
    parser.add_argument("--weights",        type=str,   default=None,                                           help="model to initialize with")
    parser.add_argument("--image-size",     type=int,   default=512,                                            help="The training size of image")
    parser.add_argument("--load-size",      type=str,   default="Resize",                                       help="Width and height to resize training and testing frames. None for no resizing, only [512, 512] for no resizing")
    parser.add_argument("--crop-size",      type=str,   default="[512, 512]",                                   help="Width and height to crop training and testing frames. Must be a multiple of 16")
    parser.add_argument("--iters",          type=int,   default=60,                                             help="No of iterations to train the model.")
    parser.add_argument("--lr",             type=float, default=1e-4,                                           help="Learning rate for the model.")
    parser.add_argument("--checkpoint-dir", type=str,   default=RUN_DIR / "train/vision/enhance/les/jin2022",   help="Location at which to save the light-effects suppression results.")
    parser.add_argument("--benchmark",      action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
