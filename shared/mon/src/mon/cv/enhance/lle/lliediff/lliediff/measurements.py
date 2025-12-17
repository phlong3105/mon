'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

import sys
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

sys.path.append('../diffusion-posterior-sampling/')
from LIME import LIME
import torch
import torch.nn as nn
import torch.nn.functional as F

# =================
# Operation classes
# =================

__OPERATOR__ = {}

# class L_color(nn.Module):
    
#     def __init__(self):
#         super(L_color, self).__init__()

#     def forward(self, x):
#         x = (x + 1.0)/2.0
#         b, c, h, w = x.shape
#         mean_rgb = torch.mean(x, [2, 3], keepdim=True)
#         mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
#         Drg = torch.pow(mr - mg, 2)
#         Drb = torch.pow(mr - mb, 2)
#         Dgb = torch.pow(mb - mg, 2)
#         k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
#         return k.sum()

class L_color(nn.Module):
    
    def __init__(self):
        super(L_color, self).__init__()
        self.mr = None
        self.mg = None
        self.mb = None

    def forward(self, x):
        x = (x + 1.0) / 2.0
        b, c, h, w = x.shape
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        # print(mean_rgb.shape)
        self.mr, self.mg, self.mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(self.mr - self.mg, 2)
        Drb = torch.pow(self.mr - self.mb, 2)
        Dgb = torch.pow(self.mb - self.mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        return k.sum()

    def save_images(self, file_prefix):
    # def save_images(self):
        # Helper function to convert tensor to image
        def tensor_to_image(tensor, filename):
            tensor = tensor.detach().squeeze().cpu().numpy()  # Remove batch and channel dimensions, convert to NumPy
            tensor = (tensor * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
            # print(tensor.shape)
            img = Image.fromarray(tensor)
            img.save(filename)
        # print(self.mr.shape)
        tensor_to_image(self.mr, f'{file_prefix}_mr.png')
        tensor_to_image(self.mg, f'{file_prefix}_mg.png')
        tensor_to_image(self.mb, f'{file_prefix}_mb.png')


class L_exp(nn.Module):
    
    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    
    def forward(self, x):
        x = (x + 1.0)/2.0
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d


class CosineAngleLoss(nn.Module):
    
    def __init__(self):
        super(CosineAngleLoss, self).__init__()

    def forward(self, img1, img2):
        # (B, C, H, W)
        img1 = (img1 + 1.0)/2.0
        img2 = (img2 + 1.0)/2.0
        # (B, H, W, C)
        img1_flat = img1.permute(0, 2, 3, 1).contiguous()
        img2_flat = img2.permute(0, 2, 3, 1).contiguous()
        
        cosine_similarity = F.cosine_similarity(img1_flat, img2_flat, dim=-1)
        
        loss = 1 - cosine_similarity
        
        loss = loss.mean()
        
        return loss
    
    
def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls

    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data)

# class L_exp(nn.Module):

#     def __init__(self,patch_size,mean_val):
#         super(L_exp, self).__init__()
#         # print(1)
#         self.pool = nn.AvgPool2d(patch_size)
#         self.mean_val = mean_val

#     def get_mean(self, x):
#         b,c,h,w = x.shape
#         x = torch.mean(x,1,keepdim=True)
#         mean = self.pool(x)
#         return mean

#     def forward(self, x ):

#         b,c,h,w = x.shape
#         x = torch.mean(x,1,keepdim=True)
#         mean = self.pool(x)

#         d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val] ).cuda(),2))
#         return d

# class L_color(nn.Module): 

#     def __init__(self):
#         super(L_color, self).__init__()

#     def forward(self, x ):

#         b,c,h,w = x.shape

#         mean_rgb = torch.mean(x,[2,3],keepdim=True)
#         mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
#         Drg = torch.pow(mr-mg,2)
#         Drb = torch.pow(mr-mb,2)
#         Dgb = torch.pow(mb-mg,2)
#         k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
#         return k


@register_operator(name='darken')
class DarkenOperator(NonLinearOperator):
    def __init__(self, img_path, w=256, h=256, dtype=torch.float32, device='cuda', beta=0.6, sigma=0.3, a=0.0, b=0.0):
        self.device = device
        self.dtype = dtype
        self.beta = beta
        self.sigma = sigma
        self.illum = None
        self.a = a
        self.b = b

        pic = Image.open(img_path).convert('RGB')
        self.origin_img = np.array(pic.resize((w, h), resample=Image.LANCZOS)).astype(np.float32) / 255.0
        self.lime = LIME(img=self.origin_img)
        # self.beta_map = nn.Parameter(torch.full((3, 256, 256), 0.75).to(self.device), requires_grad = True)
        self.beta_map = torch.full((3, 256, 256), 0.75).to(self.device)
        self.loss_exp = L_exp(32, 0.5)
        self.loss_color = L_color()

    def get_illum(self, step=1):
        with torch.no_grad():
            T_numpy = self.lime.run(step)
            T_ = torch.from_numpy(T_numpy).to(self.device, self.dtype)
            self.illum = T_.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
            return self.illum

    def get_new_illum(self, x0, step=1):
        with torch.no_grad():
            # x0_numpy = x0.data
            x0_numpy = x0.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            x0_numpy = (x0_numpy / 2.0)+0.5
            x0_lime = LIME(img=x0_numpy)
            T_x0_numpy = x0_lime.run(1)
            T_x0 = torch.from_numpy(T_x0_numpy).to(self.device)
            T_x0 = T_x0.permute(2,0,1).unsqueeze(0).clamp(0, 1)

            return T_x0, x0/T_x0

    def calculate_beta(self):
        delta = 1e-3
        delta = delta * torch.ones_like(self.illum)
        Lav = torch.mean(torch.log(self.illum + delta))
        maxL = torch.log(torch.min(self.illum) + delta)
        minL = torch.log(torch.max(self.illum) + delta)
        key = (Lav - minL) / maxL - minL
        # a_var = 3.0  = self.sigma
        self.beta = 1 - self.sigma * (key - 0.6)

    # def forward(self, data, illum, is_latent=False, adaptive_beta=False):
    def forward(self, data, is_latent=False, adaptive_beta=False):
        if is_latent:
            data = (data + 1) / 2.0
        if adaptive_beta:
            meas = data * self.illum ** (torch.exp(self.illum) - self.sigma)
        else:
            meas = data * self.illum ** (self.beta)
        if is_latent:
            meas = meas * 2.0 - 1.0
        return meas

    def gamma_correction(self, image, gamma):
        # Ensure the image is in the right format (uint8)
        image = (image * 255).astype(np.uint8)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        import cv2
        return cv2.LUT(image, table)
    
    def apply_channel_ratio(self, image, target_ratios):
        mean_r = np.mean(image[0, :, :])
        mean_g = np.mean(image[1, :, :])
        mean_b = np.mean(image[2, :, :])
        total_mean = mean_r + mean_g + mean_b
        
        # Current channel ratios
        current_ratios = [mean_r / total_mean, mean_g / total_mean, mean_b / total_mean]
        
        # Adjusting the channels in image to match the target ratios
        for i in range(3):
            if current_ratios[i] != 0:
                scale_factor = target_ratios[i] / current_ratios[i]
                image[i, :, :] *= scale_factor
        return image

    def color_channel_adjust(self, meas, is_latent=False, adaptive_beta=False):
        img = self.origin_img
        l_img_used_for_gamma = torch.from_numpy(img.transpose((2, 0, 1))).contiguous().unsqueeze(0).to(device=self.device, dtype=torch.float32)
        l_img_np = l_img_used_for_gamma.squeeze(0).cpu().numpy().transpose((1, 2, 0)) * 255.0
        l_img_gamma_corrected = self.gamma_correction(l_img_np, gamma=2.5).astype(np.float32) / 255.0
        # 计算 gamma 校正后的三通道比例
        mean_r = np.mean(l_img_gamma_corrected[:, :, 0])
        mean_g = np.mean(l_img_gamma_corrected[:, :, 1])
        mean_b = np.mean(l_img_gamma_corrected[:, :, 2])
        total_mean = mean_r + mean_g + mean_b
        r_ratio = mean_r / total_mean
        g_ratio = mean_g / total_mean
        b_ratio = mean_b / total_mean
        
        I0_img_ = meas * self.illum ** (self.sigma - torch.exp(self.illum))
        I0_img_np = I0_img_.squeeze(0).cpu().numpy()
        I0_img_np_adjusted = self.apply_channel_ratio(I0_img_np, (r_ratio, g_ratio, b_ratio))
        I0_img_final = torch.from_numpy(I0_img_np_adjusted).unsqueeze(0).to(device=self.device, dtype=torch.float32)
        
        return I0_img_final
        
    # def transpose(self, meas, illum, is_latent=False, adaptive_beta=False):
    def transpose(self, meas, is_latent=False, adaptive_beta=False):
        if is_latent:
            meas = (meas + 1) / 2.0
        if adaptive_beta:
            # data = meas * self.illum ** (1-torch.exp(self.illum)-0.7)
            data = meas * self.illum ** (self.sigma - torch.exp(self.illum))
        else:
            data = meas * self.illum ** (-self.beta)
        if is_latent:
            data = data * 2.0 - 1.0
        return data


    # def calculate_alpha(self, input):
    #     with torch.no_grad():
    #         in_  = (input + 1) / 2.0
    #         I_tmp = in_ * self.illum ** (-self.cal_beta)
    #         r, g, b = torch.split(I_tmp, 1, dim=1)
    #         mean_r = torch.mean(r)
    #         mean_g = torch.mean(g)
    #         mean_b = torch.mean(b)
    #         print('   ', mean_r, ' ', mean_g, ' ', mean_b)
    #         self.alpha = torch.ones(3)
    #         self.alpha[0] =  mean_g / mean_r
    #         self.alpha[1] = 1.0
    #         self.alpha[2] =  mean_g / mean_b 
    #         print('alpha ', self.alpha)

    #         # mean_r_ = mean_r*self.alpha[0]
    #         # mean_g_ = mean_g*self.alpha[1]
    #         # mean_b_ = mean_b*self.alpha[2]
    #         # print('is equal? ', mean_r_, ' ', mean_g_, ' ', mean_b_)
    #         self.alpha = (self.alpha.view(1, 3, 1, 1)).to('cuda')
            
    #         return self.alpha

    # def calculate_beta(self, input, normal_value):
    #     with torch.no_grad():
    #         exposure_oprator = L_exp(16, normal_value)
    #         local_mean = exposure_oprator.get_mean((input+1.0)/2.0)
    #         u_logL = torch.log(local_mean)
    #         u_logE = torch.log(exposure_oprator.get_mean(self.illum))
    #         u_normal = torch.log(normal_value*torch.ones_like(local_mean))
    #         self.cal_beta = torch.mean((u_logL - u_normal) / (u_logE+1e-5))
    #         in_ = (input + 1) / 2.0
    #         # print('is 0.5? ', torch.mean(in_ * self.illum ** (-self.cal_beta)))
    #         print('beta ', self.cal_beta)
    #         return self.cal_beta

    # def forward(self, data, is_latent=False):
    #     if is_latent:
    #         data = (data + 1) / 2.0
    #     # meas = data * self.illum ** self.beta   
    #     meas = data * (self.illum ** self.cal_beta ) * self.alpha
    #     if is_latent:
    #         meas = meas * 2.0 - 1.0
    #     return meas

    # # def transpose(self, meas, illum, is_latent=False):
    # def transpose(self, meas, is_latent=False):
    #     if is_latent:
    #         meas = (meas + 1) / 2.0
    #     # data = meas * self.illum ** (-self.beta)
    #     data = meas * (self.illum ** (-self.cal_beta)) / self.alpha
    #     if is_latent:
    #         data = data * 2.0 - 1.0
    #     return data

    # def forward(self, data, illum, is_latent=False):
    #     if is_latent:
    #         data = (data + 1) / 2.0
    #     # meas = data * illum ** self.cal_beta * self.alpha
    #     # meas = data * illum ** self.cal_beta 
    #     meas = data * illum ** self.beta 
    #     if is_latent:
    #         meas = meas * 2.0 - 1.0
    #     return meas

    # def transpose(self, meas, illum, is_latent=False):
    #     if is_latent:
    #         meas = (meas + 1) / 2.0
    #     # data = meas * illum ** (-self.cal_beta) / self.alpha
    #     # data = meas * illum ** (-self.cal_beta) 
    #     data = meas * illum ** (-self.beta) 
    #     if is_latent:
    #         data = data * 2.0 - 1.0
    #     return data

    # def forward(self, data, is_latent=False): ssh -p 22080 root@connect.westb.seetacloud.com PNXHPMwCPzlc
    #     if is_latent:
    #         data = (data + 1) / 2.0
    #     meas = data * self.illum ** self.beta_map 
    #     meas = data * self.illum ** self.beta_map 
    #     if is_latent:
    #         meas = meas * 2.0 - 1.0
    #     return meas

    # def transpose(self, meas, is_latent=False):
    #     if is_latent:
    #         meas = (meas + 1) / 2.0
    #     data = meas * self.illum * (-1 ** self.beta_map) 
    #     if is_latent:
    #         data = data * 2.0 - 1.0
    #     return data
    
    # def update_beta(self, data):
        
    #     beta_grad = torch.clone(self.beta_map.detach())
    #     beta_grad.requires_grad = True
        
    #     tmp = ((data + 1) / 2.0) * self.illum * (-1 * beta_grad)
    #     loss = self.loss_exp(tmp) + self.loss_color(tmp)
    #     loss.backward()
    #     # print(self.beta_map.grad)
    #     # with torch.no_grad():
    #     #     self.beta_map = self.beta_map - 1.0 * beta_grad.grad


# =============
# Noise classes
# =============


__NOISE__ = {}


def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls

    return wrapper


def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser


class Noise(ABC):
    
    def __call__(self, data):
        return self.forward(data)

    @abstractmethod
    def forward(self, data):
        pass


@register_noise(name='clean')
class Clean(Noise):
    
    def forward(self, data):
        return data


@register_noise(name='gaussian')
class GaussianNoise(Noise):
    
    def __init__(self, sigma):
        self.sigma = sigma

    def forward(self, data):
        return data + torch.randn_like(data, device=data.device, dtype=data.dtype) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson

        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0

        # return data.clamp(low_clip, 1.0)
