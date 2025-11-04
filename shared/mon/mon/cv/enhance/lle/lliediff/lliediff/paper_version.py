# import tensorflow
# import pytorch_lightning as pl

import argparse
import inspect
import os
import time
from itertools import islice
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
# from MyLoss import PerceptualLoss, SimplePerceptualLoss
# from metrics import calculate_metrics
import kornia
import torch
import torch.nn as nn
import torchvision.utils as vutils
import yaml
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    ImagePipelineOutput,
    UNet2DConditionModel,
)  # DDPMScheduler,
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from pytorch_lightning import seed_everything
from transformers import CLIPTextModel, CLIPTokenizer

from measurements import DarkenOperator, get_noise

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


# Load configurations
def load_yaml(file_path:str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def img_save(image, path):
    image = (image.detach() / 2 + 0.5).clamp(0, 1)
    image = (image.cpu().permute(0, 2, 3, 1)).squeeze(0).numpy()
    images = (image * 255).round().astype("uint8")
    images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, images)


# import torch.nn.functional as F

class L2Loss(nn.Module):
    
    def forward(self, input, target):
        return torch.linalg.norm(input - target)
        # return torch.norm(torch.sum(torch.abs(input - target)), p=1, dim=0)
        # return torch.sqrt(F.mse_loss(input, target, reduction='mean')) * 255.
        # return torch.sqrt(F.mse_loss(input, target, reduction='sum'))
        # return torch.sqrt(torch.sum((input - target) ** 2))


class SmoothL1Loss(nn.Module):
    
    def forward(self, input, target):
        loss_function = nn.SmoothL1Loss(beta=1.0)
        return loss_function(input , target)


class L1Loss(nn.Module):
    
    def forward(self, input, target):
        loss_function = nn.L1Loss()
        return loss_function(input , target)


class LaplacianPyramid(nn.Module):
    
    # def get(self, input):
    #     input_numpy = input.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0
    #     I0 = input_numpy
    #     I1 = cv2.pyrDown(I0)
    #     I2 = cv2.pyrDown(I1)

    #     G1 = cv2.pyrUp(I1)
    #     G2 = cv2.pyrUp(I2)
        
    #     H1 = I0 - G1
    #     H2 = I1 - G2

    #     G1 = torch.from_numpy(G1).to('cuda').permute(2, 0, 1).unsqueeze(0) / 255.0 
    #     G2 = torch.from_numpy(G2).to('cuda').permute(2, 0, 1).unsqueeze(0) / 255.0
    #     H1 = torch.from_numpy(H1).to('cuda').permute(2, 0, 1).unsqueeze(0) / 255.0
    #     H2 = torch.from_numpy(H2).to('cuda').permute(2, 0, 1).unsqueeze(0) / 255.0
        
    #     return  G1, G2, H1, H2

    def get(self, input):
        
        I1 = input
        # 首先对图像进行高斯模糊
        G1 = kornia.filters.gaussian_blur2d(I1, (5, 5), (1.5, 1.5))
        H1 = I1 - G1

        # 然后对模糊后的图像进行下采样
        I2 = kornia.geometry.transform.resize(G1, (128, 128))
        G2 = kornia.filters.gaussian_blur2d(I2, (5, 5), (1.5, 1.5))
        H2 = I2 - G2
        
        return  G1, G2, H1, H2


class GuidanceScheduler(DDIMScheduler):
    # Copied from diffusers.schedulers.scheduling_ddim.DDIMScheduler.step
    
    def step(
            self,
            model_output: torch.FloatTensor,
            timestep: int,
            sample: torch.FloatTensor,
            eta: float = 0.0,
            use_clipped_model_output: bool = False,
            generator=None,
            variance_noise: Optional[torch.FloatTensor] = None,
            return_dict: bool = True,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_x0: Optional[Callable[[int, int, Dict], None]] = None,
            callback_kwargs: dict = None,
            callback_for_original_sample: bool = True,
    ) -> Union[DDIMSchedulerOutput, Tuple]:
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
            pred_epsilon = (alpha_prod_t ** 0.5) * model_output + (beta_prod_t ** 0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        #########################################################

        # error = callback_on_x0(pred_original_sample, callback_kwargs['measurement'])
        # gradients = torch.autograd.grad(error, inputs=sample)[0]
        # pred_original_sample = pred_original_sample.detach() - 1.0*gradients
        # callback_kwargs['modify_original_sample'] = pred_original_sample.requires_grad_()

        #########################################################

        callback_kwargs['timestep'] = timestep
        callback_kwargs['sample_w_grad'] = sample
        callback_kwargs['pred_original_sample'] = pred_original_sample

        #########################################################

        if callback_kwargs['use_adpative_stepsize']==True and 'last_grad' not in callback_kwargs:
            callback_kwargs['last_grad'] = sample*0

        if callback_on_step_end is not None and callback_for_original_sample:
            callback_kwargs['sample_original'] = pred_original_sample
            pred_original_sample = callback_on_step_end(**callback_kwargs)
        #########################################################

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                )
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        #########################################################
        if callback_on_step_end is not None and not callback_for_original_sample:
            callback_kwargs['sample_original'] = prev_sample

            if callback_kwargs['use_adpative_stepsize']==True:
                callback_kwargs['last_sample'] = sample
                prev_sample, last_grad = callback_on_step_end(**callback_kwargs)
                callback_kwargs['last_grad'] = last_grad
            else:
                callback_kwargs['last_sample'] = None
                callback_kwargs['last_grad'] = None
                prev_sample = callback_on_step_end(**callback_kwargs)
        #########################################################

        if not return_dict:
            return (prev_sample,)

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
    
    # Copied from diffusers.schedulers.scheduling_repaint.set_timesteps
    # def set_timesteps(
    #     self,
    #     num_inference_steps: int,
    #     jump_length: int = 10,
    #     jump_n_sample: int = 10,
    #     device: Union[str, torch.device] = None,
    # ):
    #     """
    #     Sets the discrete timesteps used for the diffusion chain (to be run before inference).

    #     Args:
    #         num_inference_steps (`int`):
    #             The number of diffusion steps used when generating samples with a pre-trained model. If used,
    #             `timesteps` must be `None`.
    #         jump_length (`int`, defaults to 10):
    #             The number of steps taken forward in time before going backward in time for a single jump (“j” in
    #             RePaint paper). Take a look at Figure 9 and 10 in the paper.
    #         jump_n_sample (`int`, defaults to 10):
    #             The number of times to make a forward time jump for a given chosen time sample. Take a look at Figure 9
    #             and 10 in the paper.
    #         device (`str` or `torch.device`, *optional*):
    #             The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.

    #     """
    #     num_inference_steps = min(self.config.num_train_timesteps, num_inference_steps)
    #     self.num_inference_steps = num_inference_steps

    #     timesteps = []

    #     jumps = {}
    #     for j in range(0, num_inference_steps - jump_length, jump_length):
    #         jumps[j] = jump_n_sample - 1

    #     t = num_inference_steps
    #     while t >= 1:
    #         t = t - 1
    #         timesteps.append(t)

    #         if jumps.get(t, 0) > 0:
    #             jumps[t] = jumps[t] - 1
    #             for _ in range(jump_length):
    #                 t = t + 1
    #                 timesteps.append(t)

    #     import numpy as np
    #     timesteps = np.array(timesteps) * (self.config.num_train_timesteps // self.num_inference_steps)
    #     # print('scheduler ', timesteps)
    #     self.timesteps = torch.from_numpy(timesteps).to(device)
    
    # Copied from diffusers.schedulers.scheduling_repaint.undo_step
    def undo_step(self, sample, timestep, generator=None):
        n = self.config.num_train_timesteps // self.num_inference_steps

        for i in range(n):
            beta = self.betas[timestep + i]
            if sample.device.type == "mps":
                # randn does not work reproducibly on mps
                noise = randn_tensor(sample.shape, dtype=sample.dtype, generator=generator)
                noise = noise.to(sample.device)
            else:
                noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)

            # 10. Algorithm 1 Line 10 https://arxiv.org/pdf/2201.09865.pdf
            sample = (1 - beta) ** 0.5 * sample + beta**0.5 * noise

        return sample


class PLSDPipeline(DiffusionPipeline):
    
    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    
    def __init__(
        self,
        vae         : AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer   : CLIPTokenizer,
        unet        : UNet2DConditionModel,
        scheduler   : GuidanceScheduler,
    ):
        super().__init__()
        self.register_modules(
            vae          = vae,
            text_encoder = text_encoder,
            tokenizer    = tokenizer,
            unet         = unet,
            scheduler    = scheduler,
        )
        set_requires_grad(self.text_encoder, False)
        set_requires_grad(self.vae,  False)
        set_requires_grad(self.unet, False)
        self.loss_fn        = L2Loss()  #SimplePerceptualLoss() #SmoothL1Loss() #L2Loss()
        self.loss_fn_latent = L2Loss()  #SmoothL1Loss() L2Loss() #nn.SmoothL1Loss(reduction='sum')
        self.pyramid        = LaplacianPyramid()
    
    def x0_callback(
        self,
        pred_original_sample: torch.FloatTensor,
        measurement         : torch.Tensor = None,
    ):
        origi_pred = self.vae.decode(1 / self.vae.config.scaling_factor * pred_original_sample).sample
        meas_pred  = operator.forward(origi_pred, is_latent=True, adaptive_beta=False)
        meas_error = self.loss_fn(measurement, meas_pred)
        return meas_error

    def guidance_callback(
            self,
            timestep: int,
            index: int,
            last_grad: torch.FloatTensor,
            sample_w_grad: torch.FloatTensor,
            pred_original_sample: torch.FloatTensor,
            # modify_original_sample: torch.FloatTensor,
            last_sample: torch.FloatTensor,
            sample_original: torch.FloatTensor,
            measurement: torch.Tensor = None,
            operator=None,
            save_path: Optional[str] = None,
            step_size: Optional[float] = 10.0,
            w_inpaint: Optional[float] = 0.5,
            w_inv: Optional[float] = 0.5,
            w_meas: Optional[float] = 0.5,
            w_stepsize_max: Optional[float] = 1.0,
            w_factor: Optional[float] = 1.0,
            print_freq: Optional[int] = 10,
            noiser=None,
            use_adpative_stepsize = False,
            use_adpative_beta = False,
    ):
        # 1. decode latent
        # alpha_prod_t = self.scheduler.alphas_cumprod[t]
        # beta_prod_t = 1 - alpha_prod_t
        # pred_original_sample = (latents_diff - beta_prod_t ** (0.5) * noise_pred_diff) / alpha_prod_t ** (0.5)
        origi_pred = self.vae.decode(1 / self.vae.config.scaling_factor * pred_original_sample).sample
        # modify_origi_pred = self.vae.decode(1 / self.vae.config.scaling_factor * modify_original_sample).sample
        sample_prev = self.vae.decode(1 / self.vae.config.scaling_factor * sample_original).sample
        if use_adpative_stepsize:
            sample_last = self.vae.decode(1 / self.vae.config.scaling_factor * last_sample).sample
        # fac = torch.sqrt(beta_prod_t)
        # sample = pred_original_sample * (fac) + latents_diff * (1 - fac)
        # origi_pred = self.vae.decode(1 / self.vae.config.scaling_factor * sample).sample

        # 2. compute loss and gradient
        loss = 0
        meas_error = None
        inv_error = None
        inpaint_error = None
        # meas_inv = operator.transpose(measurement, is_latent=True, adaptive_beta=use_adpative_beta)

        IG1, IG2, IH1, IH2 = self.pyramid.get(origi_pred)
        LG1, LG2, LH1, LH2 = self.pyramid.get(measurement)  #要在像素空间还是别的空间做?
        
        
        # if timestep.item()>400:
        #     illum = operator.illum
        # else:
        #     illum, _ = operator.get_new_illum(origi_pred)
    
        meas_inv = operator.transpose(measurement,  is_latent=True, adaptive_beta=use_adpative_beta)
        meas_pred = operator.forward(origi_pred,  is_latent=True, adaptive_beta=use_adpative_beta)
        invG1, invG2, invH1, invH2 = self.pyramid.get(meas_inv)

        if w_meas > 0:  #alpha
            # use original calculate method
            compare = operator.illum * ((IG1+1.0)/2.0)

            # self.loss_fn(measurement, meas_pred) + self.loss_fn(compare, meas_pred)                     
            self.loss_fn(measurement, meas_pred) + self.loss_fn(compare, LG1)                     
            
            meas_error = self.loss_fn(measurement, meas_pred)

            # loss = loss + meas_error * w_meas
            loss = loss + meas_error  * w_meas
            
        if w_inv > 0:  #omega
            
            # inv_error = self.loss_fn(meas_inv, origi_pred) + 0.05 * self.loss_fn(invG2, IG2)
            inv_error = self.loss_fn(meas_inv, origi_pred)

            # w_inv=1.0
            loss = loss + inv_error  * w_inv
            
        if w_inpaint > 0:  #gamma
            # ortho_pred = operator.transpose(operator.forward(origi_pred, is_latent=True), is_latent=True)
            # ortho_project = origi_pred - ortho_pred
            # parallel_project = operator.transpose(measurement, is_latent=True)
            # inpainted_image = parallel_project + ortho_project
            encoded_z_0 = self.vae.config.scaling_factor * vae.encode(
                # inpainted_image).latent_dist.mean
                # meas_inv).latent_dist.sample(generator)
                invG1).latent_dist.mean
                # meas_inv).latent_dist.mean
         
            inpaint_error = self.loss_fn_latent(encoded_z_0, pred_original_sample)
            
            loss = loss + inpaint_error * w_inpaint

        loss.backward()
        # print(sample_w_grad.grad)

        # 3. print
        if index % print_freq == 0:
            # print(f"[{index:04d}:{timestep:04d}] loss: {loss.item():.6f} = "
            #       f"w_meas: {w_meas:.2f} * meas_error: {0 if meas_error is None else meas_error.item():.6f} + "
            #       f"w_inv: {w_inv:.2f} * inv_error: {0 if inv_error is None else inv_error.item():.6f} + "
            #       f"w_inpaint: {w_inpaint:.2f} * inpaint_error: {0 if inpaint_error is None else inpaint_error.item():.6f}")
            name, ext = os.path.splitext(save_path)
            img_save(origi_pred, f"{name}_x0_pred_{timestep.item():04d}{ext}")
            # img_save(modify_origi_pred, f"{name}_modify_x0_pred_{timestep.item():04d}{ext}")
            img_save(sample_prev, f"{name}_xt_pred_{timestep.item():04d}{ext}")
            # if use_adpative_stepsize:
                # img_save(sample_last, f"{name}_xt_last_{timestep.item():04d}{ext}")
            if w_inv > 0:
                img_save(meas_inv, f"{name}_meas_inv_{timestep.item():04d}{ext}")
             
            # img_save(invG1, f"{name}_invG1_{timestep.item():04d}{ext}")
            # img_save(invG2, f"{name}_invG2_{timestep.item():04d}{ext}")
            # img_save(invH1, f"{name}_invH1_{timestep.item():04d}{ext}")
            # img_save(IH1, f"{name}_IH1_{timestep.item():04d}{ext}")
            # img_save(invH2, f"{name}_invH2_{timestep.item():04d}{ext}")

            # if w_inpaint > 0:
            #     img_save(ortho_pred, f"{name}_ortho_pred_{timestep.item():04d}{ext}")
            #     img_save(inpainted_image, f"{name}_inpainted_image_{timestep.item():04d}{ext}")

        # upgrade step_size
        if use_adpative_stepsize:
            if timestep.item() >= 999:
                step_size = 1.0
            else:
                # print(w_stepsize_max)
                step_size = torch.mul(sample_original - last_sample, sample_original - last_sample) / torch.mul(sample_original - last_sample, (sample_w_grad.grad - last_grad))
                step_size[torch.isnan(step_size)] = 0
                step_size[torch.isinf(step_size)] = 0
                # 计算最大值和最小值
                max_step_size = torch.max(step_size)
                min_step_size = torch.min(step_size)
                # 防止分母为零
                if max_step_size == min_step_size:
                    step_size = step_size - min_step_size
                else:
                    step_size = ((step_size - min_step_size) / (max_step_size - min_step_size + 1e-5))*w_stepsize_max
                # 重新缩放至[0,1]范围
                step_size = torch.min(torch.max(step_size, torch.zeros_like(step_size)), w_stepsize_max * torch.ones_like(step_size))

            grad_saved = sample_w_grad.grad
        
        # 3. update sample_original
        with torch.no_grad():
            sample_original = sample_original - step_size * sample_w_grad.grad
            # noise_pred = noise_pred - torch.sqrt(beta_prod_t) * step_size * (-latents_diff.grad)
        if use_adpative_stepsize:
            return sample_original.detach(), grad_saved.detach()
        else:
            return sample_original.detach()

    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = 512,
            width: Optional[int] = 512,
            num_inference_steps: Optional[int] = 1000,
            guidance_scale: Optional[float] = 7.5,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[torch.Generator] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            do_measure: bool = True,
            callback_for_original_sample: bool = False,
            callback_kwargs: dict = None,
            weight_dtype=torch.float32,
            use_adpative_stepsize = False,
            use_adpative_beta = False,
            jump_len = 0,
            jump_sample = 0,
    ) -> Union[ImagePipelineOutput, Tuple]:
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(device=self.device))[0]

        text_embeddings = text_embeddings.repeat_interleave(num_images_per_prompt, dim=0).to(dtype=weight_dtype)

        latents_shape = (batch_size * num_images_per_prompt, self.unet.config.in_channels, height // 8, width // 8)

        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            uncond_embeddings = uncond_embeddings.repeat_interleave(num_images_per_prompt, dim=0)
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        # time travel settings
        # jump_length = 5
        # jump_n_sample = 5
        # num_inference_steps = 999
        # self.scheduler.set_timesteps(num_inference_steps+1, jump_len, jump_sample)
        # t_last = self.scheduler.timesteps[0] + 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        

        if latents is None:
            if self.device.type == "mps":
                latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=weight_dtype).to(
                    self.device
                )
            else:
                latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=weight_dtype)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            latents = latents.to(self.device)

        latents = latents * self.scheduler.init_noise_sigma

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            # if do_classifier_free_guidance:
            #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            with torch.enable_grad():
                # if t < t_last:
                #     # predict the noise residual
                    sample_w_grad = latents.detach().requires_grad_()

                    latent_model_input = torch.cat([sample_w_grad] * 2) if do_classifier_free_guidance else sample_w_grad
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    model_output = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                    if do_classifier_free_guidance:
                        noise_pred_uncond_diff, model_output = model_output.chunk(2)
                        model_output = noise_pred_uncond_diff + guidance_scale * (
                                model_output - noise_pred_uncond_diff)
                    # compute previous image: x_t -> x_t-1
                    if do_measure:
                        extra_step_kwargs['callback_on_step_end'] = self.guidance_callback
                        extra_step_kwargs['callback_on_x0'] = self.x0_callback
                        extra_step_kwargs['callback_for_original_sample'] = callback_for_original_sample
                        callback_kwargs['index'] = i
                        extra_step_kwargs['callback_kwargs'] = callback_kwargs

                    out = self.scheduler.step(model_output, t, sample_w_grad, **extra_step_kwargs)
                    latents = out.prev_sample
                    origin = out.pred_original_sample

                # else:
                #     # compute the reverse: x_t-1 -> x_t
                #     sample_w_grad = latents.detach().requires_grad_()
                #     latents = self.scheduler.undo_step(sample_w_grad, t_last)
                # t_last = t

            # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # 14. use the vae to decode the generated latents back into the image
        image = self.vae.decode(1 / self.vae.config.scaling_factor * latents).sample
        image = (image / 2 + 0.5).clamp(0, 1).detach()
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        
        o_image = self.vae.decode(1 / self.vae.config.scaling_factor * origin).sample
        o_image = (o_image / 2 + 0.5).clamp(0, 1).detach()
        o_image = o_image.detach().cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            o_image = self.numpy_to_pil(o_image)

        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=[image[0], o_image[0]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir to read samples from",
        default="lol_samples/"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="",
        # default="8k uhd, soft lighting, high quality, best quality, extremely detailed, clean, tidy",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/psld-samples-llie"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=1000,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default='configs/LLIE_config_psld.yaml',
        help="task config yml file",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default='/root/paddlejob/workspace/env_run/lzx/0227/0301/stable-diffusion-v1-5',
        help="model id",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=10.0,
        help="reconstruction error",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.6,
        help="beta for lime",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="lime step",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=500,
        # L1Smooth:  0499] loss: 5000.00 = w_inpaint: 10000.00 * inpaint_error: 0.50
        # L2Sum: :0499] loss: 22.471407 = w_inpaint: 2.00 * inpaint_error: 11.235703
        # L1SmoothSum: 0499] loss: 3098.617676 = w_inpaint: 1.00 * inpaint_error: 3098.617676
        # L1SmoothSum: 0399] loss: 848.178406 = w_inpaint: 1.00 * inpaint_error: 848.178406
        help="inpainting error",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=200.0,
        # L1Smooth: 0899] loss: 2000.00 = w_inv: 200.00 * inv_error: 10.00
        # L2Sum: 0499] loss: 133.247955 = w_inv: 2.00 * inv_error: 66.623978
        help="measurement error",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0,
        # L1Smooth: 0899] loss: 400.00 = w_meas: 4000.00 * meas_error: 0.10
        # L2Sum: 0459] loss: 33.642025 = w_meas: 2.00 * meas_error: 16.821012
        help="reconstruction error",
    )
    parser.add_argument(
        "--adp_stepsize",
        action='store_true', 
        default=False,
        help="measurements beta adaptive",
    )
    parser.add_argument(
        "--adp_beta",
        action='store_true', 
        default=False,
        help="measurements beta adaptive",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        # L1Smooth: 0899] loss: 400.00 = w_meas: 4000.00 * meas_error: 0.10
        # L2Sum: 0459] loss: 33.642025 = w_meas: 2.00 * meas_error: 16.821012
        help="adaptive step_size scale max value",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.3,
        help="adpative beta",
    )
    parser.add_argument(
        "--a",
        type=float,
        default=0.0,
        help="adpative beta",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.0,
        help="adpative beta",
    )
    parser.add_argument(
        "--jump_len",
        type=int,
        default=0,
        help="time travel",
    )
    parser.add_argument(
        "--jump_sample",
        type=int,
        default=0,
        help="time travel",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=10,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--test_id",
        type=str,
        default=0,
        help="number of test id",
    )

    opt = parser.parse_args()
    
    seed_everything(opt.seed)

    torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(opt.model_id, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(opt.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(opt.model_id, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(opt.model_id, subfolder="unet")
    scheduler = GuidanceScheduler.from_pretrained(opt.model_id, subfolder="scheduler")
    scheduler.set_timesteps(opt.ddim_steps)

    vae.to(device=torch_device, dtype=weight_dtype)
    # vae.enable_xformers_memory_efficient_attention()
    text_encoder.to(device=torch_device, dtype=weight_dtype)
    unet.to(device=torch_device, dtype=weight_dtype)
    # unet.enable_xformers_memory_efficient_attention()

    pipeline = PLSDPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    )
    pipeline.to(torch_device=torch_device, torch_dtype=weight_dtype)

    batch_size = opt.n_samples
    prompt = opt.prompt
    assert prompt is not None

    outdir = os.path.join(opt.outdir, opt.test_id)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, 'measurements'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'inter'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'results'), exist_ok=True)

    # df = pd.DataFrame(data=None, columns=['test_id', 'beta', 'PSNR', 'SSIM', 'LPIPS'])
    # df.to_csv("/home/eileen/Diffusion/DuduZhu/Diff_llie/psnr_ssim_lpips.csv", mode='a', index=False) 

    imgsName = sorted(os.listdir(opt.indir))

    for i, imgName in enumerate(imgsName):

        #########################################################
        # set up operator and measurement
        #########################################################
        img_path = os.path.join(opt.indir, imgName)
        save_path = os.path.join(outdir, 'inter', f'inter_{imgName}')
        print('>>>>>>>>> ', opt.test_id, ' ', img_path, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

        # Prepare Operator and noise
        operator = DarkenOperator(
            device=torch_device, w=opt.W, h=opt.H,
            img_path=img_path, dtype=torch.float32, beta=opt.beta, sigma=opt.sigma, #a=opt.a, b=opt.b
        )
        noiser = get_noise(name='gaussian', sigma=0.01)

        img = operator.origin_img
        if img.ndim == 2:
            img = img[:, :, None]

        L_img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous().unsqueeze(0).to(device=torch_device,
                                                                                        dtype=weight_dtype)
        E0_img = operator.get_illum(opt.step).to(device=torch_device, dtype=weight_dtype)

        vutils.save_image(L_img, os.path.join(outdir, 'measurements', f'L_{imgName}'))
        vutils.save_image(E0_img, os.path.join(outdir, 'measurements', f'E0_{imgName}'))
        # vutils.save_image(L_img * E0_img ** -0.6, os.path.join(outdir, 'measurements', f'I0_{imgName}'))
        E0_img = operator.get_illum(opt.step).to(device=torch_device, dtype=torch.float32)
        vutils.save_image(L_img  * E0_img**(1-torch.exp(E0_img)-0.7), os.path.join(outdir, 'measurements', f'I0_{imgName}'))

        L_img = L_img * 2.0 - 1.0
        # operator.calculate_beta(L_img, 0.5)
        # operator.calculate_alpha(L_img)

        #########################################################

        callback_kwargs = dict(
            measurement=L_img,
            operator=operator,
            save_path=save_path,
            step_size=opt.step_size,
            w_inpaint=opt.gamma,
            w_inv=opt.omega,
            w_meas=opt.alpha,
            w_stepsize_max=opt.eta,
            w_factor=opt.a,
            print_freq=opt.print_freq,
            noiser=noiser,
            use_adpative_stepsize = opt.adp_stepsize,
            use_adpative_beta = opt.adp_beta,
        )

        out = pipeline(
            prompt=prompt,
            height=opt.H,
            width=opt.W,
            num_inference_steps=opt.ddim_steps,
            guidance_scale=opt.scale,
            output_type="pil",
            return_dict=True,
            weight_dtype=weight_dtype,
            do_measure=True,
            callback_for_original_sample=False,
            callback_kwargs=callback_kwargs,
            jump_len=opt.jump_len,
            jump_sample=opt.jump_sample,
        ).images

        image = out[0]
        o_image = out[1]

        image.save(os.path.join(outdir, 'results', f'{imgName}'))
        # o_image.save(os.path.join(outdir, 'results', f'o_{imgName}'))
        resized_image = image.resize((600, 400), Image.Resampling.LANCZOS)

        os.makedirs(os.path.join(outdir, 'resize'), exist_ok=True)
        resized_image.save(os.path.join(outdir, 'resize', 'crop.png'))
        image.save(os.path.join(outdir, 'results', f'{imgName}'))
        # o_image.save(os.path.join(outdir, 'results', f'o_{imgName}'))
        # 指定输出目录和图像名称
        os.makedirs(os.path.join(outdir, 'resize'), exist_ok=True)
        imgName = 'crop.png'

        # 保存调整尺寸后的图像
        resized_image.save(os.path.join(outdir, 'resize', imgName))
    
    # psnr, ssim, lpips = calculate_metrics(outdir)
    # output_list = [opt.test_id, opt.gamma, opt.omega, opt.alpha, psnr, ssim, lpips]
    # new_data = pd.DataFrame([output_list])
    # new_data.to_csv("psnr_ssim_lpips.csv", mode='a', header=False, index=False)
