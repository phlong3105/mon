from __future__ import print_function
import os, warnings, cv2
import numpy as np
from colorama import Fore, Back, Style

from libs.FULL.src.v8.model import RRNet
from libs.FULL.datasets.datasets import MyDataset

import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
from kornia.filters import bilateral_blur
from libs.FULL.utils.helpers import bgr2ycbcr
# from kornia.color import rgb_to_ycbcr, ycbcr_to_rgb, rgb_to_grayscale, rgb_to_lab, lab_to_rgb
from tqdm import tqdm
# import pyiqa
eps = np.finfo(np.float32).eps
warnings.filterwarnings("ignore", category=FutureWarning)


def test(config):
    if config.p_resDir==None:
        config.p_resDir = os.path.split(config.p_model)[0]
    p_resDir    = os.path.join(config.p_resDir,config.dataset)
    if not os.path.isdir(p_resDir): os.makedirs(p_resDir)
    device      = torch.device("cuda") if (config.device=='cuda' and torch.cuda.is_available()) else torch.device("cpu")
    config.device = device
    print(f'WORKING ON Device={device}')
    
    dataset_test    = MyDataset(config, 'test')
    loader_test     = DataLoader(dataset_test, num_workers=config.num_workers, batch_size=config.batch_size)
    
    model = RRNet(config)
    print('Loading model from ', config.p_model)
    # model = torch.nn.DataParallel(model, device_ids=[int(t) for t in config.gpuId.split(",")])
    # model.module.load_state_dict(torch.load(config.p_model))
    model.load_state_dict(torch.load(config.p_model))
    model.to(device)
    model.eval()

    # Testing
    dic_Y      = {'psnr':0, 'ssim':0,}
    dic_C      = {'psnr':0, 'ssim':0, 'lpips':0}
    ssim_lis_Y = []
    psnr_lis_Y = []
    ssim_lis_C = []
    psnr_lis_C = []
    lpips_lis_C = []
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    with torch.set_grad_enabled(False):
        for _,data in tqdm(enumerate(loader_test), total=len(loader_test), colour='blue', leave=False):
            imNum, y_labels, imlow  = data['imNum'], data['gtdata'], data['imlow']
            imNum        = imNum[0]
            y_labels, imlow         = y_labels.to(device).type(torch.float32), imlow.to(device).type(torch.float32)
            model.to(device)
            pred,_       = model(imlow, imNum=imNum)
            if config.f_OverExp: pred = 1-pred
            
            # if config.f_denoise: pred = bilateral_blur(pred,(5,5), 0.4, (1.0,1.0))
            # if config.f_denoise: pred = bilateral_blur(pred,(5,5), 0.4, (1.0,1.0))
            
            lpips_lis_C.append(loss_fn_vgg(y_labels,pred).item())
            im          = torch.permute(pred[0].detach().cpu(),(1,2,0)).numpy()
            if config.f_saveRes:
                p_res   = os.path.join(p_resDir,imNum+'.png')
                if not config.f_RGB: cv2.imwrite(p_res, cv2.cvtColor(np.uint8(im*255), cv2.COLOR_YCrCb2RGB)) # as input is YCbCr this is actually BGR output
                else: cv2.imwrite(p_res, cv2.cvtColor(np.uint8(im*255), cv2.COLOR_RGB2BGR))

            if config.f_eval:
                Igt         = torch.permute(y_labels[0].detach().cpu(),(1,2,0)).numpy()
                ssim_lis_C.append(ssim(Igt, im, data_range=1.0, channel_axis=-1))
                # ssim_lis_C.append(ssim(Igt, im, data_range=1.0, multichannel=True))
                psnr_lis_C.append(psnr(Igt, im, data_range=1.0))
                
                #rgb->bgr
                im_t     = np.zeros_like(im)
                Igt_t    = np.zeros_like(Igt)
                im_t[:,:,0]     = im[:,:,2] 
                Igt_t[:,:,0]    = Igt[:,:, 2]
                im_t[:,:,1]     = im[:,:, 1] 
                Igt_t[:,:,1]    = Igt[:,:, 1]
                im_t[:,:,2]     = im[:,:, 0] 
                Igt_t[:,:,2]    = Igt[:,:, 0]
                im      = bgr2ycbcr(np.uint8(im_t*255))  # returns Y only
                Igt     = bgr2ycbcr(np.uint8(Igt_t*255)) # returns Y only
                ssim_lis_Y.append(ssim(Igt, im, data_range=255.0))    # P
                psnr_lis_Y.append(psnr(Igt, im, data_range=255.0))    # P
                print(f"Processed {imNum} \t" + Back.LIGHTGREEN_EX+Fore.BLACK+Style.BRIGHT + f" psnr={psnr_lis_Y[-1]:0.2f} ssim={ssim_lis_Y[-1]:0.3f} "+Style.RESET_ALL)
        if config.f_eval:
            dic_Y['psnr'] = np.mean(psnr_lis_Y)
            dic_Y['ssim'] = np.mean(ssim_lis_Y)
            dic_C['psnr'] = np.mean(psnr_lis_C)
            dic_C['ssim'] = np.mean(ssim_lis_C)
            dic_C['lpips'] = np.mean(lpips_lis_C)
            # dic_C['niqe'] = np.mean(niqe_lis_C)
            # print(Back.LIGHTGREEN_EX+Fore.BLACK+Style.BRIGHT+f' \t\t\t Y : test_psnr={dic_Y["psnr"]:0.3f} test_ssim={dic_Y["ssim"]:0.3f} \t '+Style.RESET_ALL)
            # print(Back.LIGHTGREEN_EX+Fore.BLACK+Style.BRIGHT+f' \t\t\t C : test_psnr={dic_C["psnr"]:0.3f} test_ssim={dic_C["ssim"]:0.3f} test_lpips={dic_C["lpips"]:0.3f} \t '+Style.RESET_ALL)
            print(Back.LIGHTGREEN_EX+Fore.BLACK+Style.BRIGHT+f' \t\t\t {dic_Y["psnr"]:0.2f}-{dic_Y["ssim"]:0.3f} \t '+Style.RESET_ALL)
            print(Back.LIGHTGREEN_EX+Fore.BLACK+Style.BRIGHT+f' \t\t\t {dic_C["psnr"]:0.2f}-{dic_C["ssim"]:0.3f} {dic_C["lpips"]:0.3f} \t '+Style.RESET_ALL)
