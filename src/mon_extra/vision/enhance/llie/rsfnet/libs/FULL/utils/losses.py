import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torch.nn.modules.loss import _Loss
from torchvision.models.vgg import vgg16

warnings.filterwarnings("ignore", message=r"multichannel", category=FutureWarning)
from kornia.filters import spatial_gradient
from kornia.color import ycbcr_to_rgb


class BurstLoss(_Loss):
    def __init__(self, rank=None, size_average=None, reduce=None, reduction='mean'):
        super(BurstLoss, self).__init__(size_average, reduce, reduction)
        self.reduction = reduction
        # use_cuda = torch.cuda.is_available()
        if not rank==None: device = torch.device("cuda:"+str(rank))
        else: device = torch.device("cuda")
        prewitt_filter = 1 / 6 * np.array([[1, 0, -1],
                                           [1, 0, -1],
                                           [1, 0, -1]])
        self.prewitt_filter_horizontal = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                                         kernel_size=prewitt_filter.shape,
                                                         padding=prewitt_filter.shape[0] // 2).to(device).type(torch.float32)
        self.prewitt_filter_horizontal.weight.data.copy_(torch.from_numpy(prewitt_filter).to(device)).type(torch.float32)
        self.prewitt_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])).to(device)).type(torch.float32)
        self.prewitt_filter_vertical = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                                       kernel_size=prewitt_filter.shape,
                                                       padding=prewitt_filter.shape[0] // 2).to(device).type(torch.float32)
        self.prewitt_filter_vertical.weight.data.copy_(torch.from_numpy(prewitt_filter.T).to(device)).type(torch.float32)
        self.prewitt_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])).to(device)).type(torch.float32)

    def get_gradients(self, img):
        img_r    = img[:, 0:1, :, :]
        img_g    = img[:, 1:2, :, :]
        img_b    = img[:, 2:3, :, :]
        grad_x_r = self.prewitt_filter_horizontal(img_r)
        grad_y_r = self.prewitt_filter_vertical(img_r)
        grad_x_g = self.prewitt_filter_horizontal(img_g)
        grad_y_g = self.prewitt_filter_vertical(img_g)
        grad_x_b = self.prewitt_filter_horizontal(img_b)
        grad_y_b = self.prewitt_filter_vertical(img_b)
        grad_x   = torch.stack([grad_x_r[:, 0, :, :], grad_x_g[:, 0, :, :], grad_x_b[:, 0, :, :]], dim=1)
        grad_y   = torch.stack([grad_y_r[:, 0, :, :], grad_y_g[:, 0, :, :], grad_y_b[:, 0, :, :]], dim=1)
        grad     = torch.stack([grad_x, grad_y], dim=0) 
        return grad

    def forward(self, input, target):
        input_grad  = self.get_gradients(input)
        target_grad = self.get_gradients(target)
        # return F.l1_loss(input, target, reduction=self.reduction) + 0.0*F.l1_loss(input_grad, target_grad, reduction=self.reduction)
        # return F.l1_loss(input, target, reduction=self.reduction) + 10*F.l1_loss(input_grad, target_grad, reduction=self.reduction)
        return F.l1_loss(input, target, reduction=self.reduction) + 5.0*F.l1_loss(input_grad, target_grad, reduction=self.reduction)
        # return F.l1_loss(input, target, reduction=self.reduction) + 1.0*F.mse_loss(input_grad, target_grad, reduction=self.reduction)
        # return F.l1_loss(input, target, reduction=self.reduction)


class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features[:35].eval().cuda()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        loss = F.mse_loss(x_vgg, y_vgg)
        return loss


class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):
        b,c,h,w = x.shape
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k


class L_colorYCbCr(nn.Module):
    def __init__(self):
        super(L_colorYCbCr, self).__init__()
        

    def forward(self, x ):
        b,c,h,w = x.shape
        x = ycbcr_to_rgb(x)
        mean_rgb  = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1) 
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k


class L_color_with_gt(nn.Module):
    def __init__(self):
        super(L_color_with_gt, self).__init__()

    def forward(self, x, gt ):
        b,c,h,w = x.shape
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        mean_rgb_gt = torch.mean(x,[2,3],keepdim=True)
        mr_gt,mg_gt, mb_gt = torch.split(mean_rgb_gt, 1, dim=1)
        Drg = torch.pow(mr-mr_gt,2)
        Drb = torch.pow(mr-mb_gt,2)
        Dgb = torch.pow(mb-mb_gt,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k


class L_color_with_gt2(nn.Module):
    def __init__(self):
        super(L_color_with_gt2, self).__init__()

    def forward(self, x, gt):
        b,c,h,w = x.shape
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        mean_rgb_gt = torch.mean(gt,[2,3],keepdim=True)
        mr_gt,mg_gt, mb_gt = torch.split(mean_rgb_gt, 1, dim=1)
        #r-g, g-b, b-r
        Drg = torch.pow(torch.abs(mr-mg)-torch.abs(mr_gt-mg_gt),2)
        Dgb = torch.pow(torch.abs(mg-mb)-torch.abs(mg_gt-mb_gt),2)
        Dbr = torch.pow(torch.abs(mb-mr)-torch.abs(mb_gt-mr_gt),2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Dgb,2) + torch.pow(Dbr,2),0.5)
        return k
			

class L_spa(nn.Module):
    def __init__(self,rank=None):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        if not rank==None:
            self.device = torch.device("cuda:"+str(rank))
        else:
            self.device = torch.device("cuda")
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).to(self.device).type(torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).to(self.device).type(torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).to(self.device).type(torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).to(self.device).type(torch.float32).unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
        
    def forward(self, org , enhance ):
        org  = torch.squeeze(org,dim=1)
        b,c,h,w = org.shape
        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)
        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	
        weight_diff =torch.max(torch.FloatTensor([1]).to(self.device).type(torch.float32) + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).to(self.device).type(torch.float32), torch.FloatTensor([0]).to(self.device).type(torch.float32)),torch.FloatTensor([0.5]).to(self.device).type(torch.float32))
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).to(self.device).type(torch.float32)) ,enhance_pool-org_pool)
        D_org_left = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)
        D_enhance_left = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)
        D_left = torch.pow(D_org_left - D_enhance_left,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)
        return E


class L_spa5(nn.Module):
    def __init__(self,rank=None):
        super(L_spa5, self).__init__()
        A = torch.FloatTensor([[0,0,0,0,0],[0,0,0,0,0],[-1,-1,2,0,0],[0,0,0,0,0],[0,0,0,0,0]])
        if not rank==None:
            self.device = torch.device("cuda:"+str(rank))
        else:
            self.device = torch.device("cuda")
        kernel_left     = A.to(self.device).type(torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_right    = torch.rot90(A,k=2).to(self.device).type(torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_up       = torch.rot90(A,k=-1).to(self.device).type(torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_down     = torch.rot90(A).to(self.device).type(torch.float32).unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up  = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool       = nn.AvgPool2d(4)
        
    def forward(self, org , enhance ):
        org  = torch.squeeze(org,dim=1)
        b,c,h,w = org.shape
        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)
        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	
        weight_diff =torch.max(torch.FloatTensor([1]).to(self.device).type(torch.float32) + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).to(self.device).type(torch.float32), torch.FloatTensor([0]).to(self.device).type(torch.float32)),torch.FloatTensor([0.5]).to(self.device).type(torch.float32))
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).to(self.device).type(torch.float32)) ,enhance_pool-org_pool)
        D_org_left = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)
        D_enhance_left = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)
        D_left = torch.pow(D_org_left - D_enhance_left,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)
        return E


class L_spa9(nn.Module):
    def __init__(self,rank=None):
        super(L_spa9, self).__init__()
        A = torch.FloatTensor([[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[-1,-1,-1,-1,4,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]])
        if not rank==None:
            self.device = torch.device("cuda:"+str(rank))
        else:
            self.device = torch.device("cuda")
        kernel_left     = A.to(self.device).type(torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_right    = torch.rot90(A,k=2).to(self.device).type(torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_up       = torch.rot90(A,k=-1).to(self.device).type(torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_down     = torch.rot90(A).to(self.device).type(torch.float32).unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up  = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool       = nn.AvgPool2d(4)
        
    def forward(self, org , enhance ):
        org  = torch.squeeze(org,dim=1)
        b,c,h,w = org.shape
        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)
        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	
        weight_diff =torch.max(torch.FloatTensor([1]).to(self.device).type(torch.float32) + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).to(self.device).type(torch.float32), torch.FloatTensor([0]).to(self.device).type(torch.float32)),torch.FloatTensor([0.5]).to(self.device).type(torch.float32))
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).to(self.device).type(torch.float32)) ,enhance_pool-org_pool)
        D_org_left = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)
        D_enhance_left = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)
        D_left = torch.pow(D_org_left - D_enhance_left,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)
        return E


# New spa loss SGZ : https://github.com/ShenZheng2000/Semantic-Guided-Low-Light-Image-Enhancement/blob/main/Myloss.py
class L_spa8(nn.Module):
    def __init__(self, patch_size=4, rank=None):
        super(L_spa8, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        # Build conv kernels
        if not rank==None:
            self.device = torch.device("cuda:"+str(rank))
        else:
            self.device = torch.device("cuda")
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).to(self.device).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).to(self.device).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).to(self.device).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).to(self.device).unsqueeze(0).unsqueeze(0)
        kernel_upleft = torch.FloatTensor( [[-1,0,0],[0,1,0],[0,0,0]]).to(self.device).unsqueeze(0).unsqueeze(0)
        kernel_upright = torch.FloatTensor( [[0,0,-1],[0,1,0],[0,0,0]]).to(self.device).unsqueeze(0).unsqueeze(0)
        kernel_loleft = torch.FloatTensor( [[0,0,0],[0,1,0],[-1,0,0]]).to(self.device).unsqueeze(0).unsqueeze(0)
        kernel_loright = torch.FloatTensor( [[0,0,0],[0,1,0],[0,0,-1]]).to(self.device).unsqueeze(0).unsqueeze(0)
        # convert to parameters
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.weight_upleft = nn.Parameter(data=kernel_upleft, requires_grad=False)
        self.weight_upright = nn.Parameter(data=kernel_upright, requires_grad=False)
        self.weight_loleft = nn.Parameter(data=kernel_loleft, requires_grad=False)
        self.weight_loright = nn.Parameter(data=kernel_loright, requires_grad=False)
        # pooling layer
        self.pool = nn.AvgPool2d(patch_size) # default is 4

    def forward(self, org, enhance):
        #b,c,h,w = org.shape
        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)
        org_pool =  self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)
        #weight_diff =torch.max(torch.FloatTensor([1]).to(device) + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).to(device),torch.FloatTensor([0]).to(device)),torch.FloatTensor([0.5]).to(device))
        #E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).to(device)) ,enhance_pool-org_pool)
        # Original output
        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)
        D_org_upleft = F.conv2d(org_pool , self.weight_upleft , padding=1)
        D_org_upright = F.conv2d(org_pool , self.weight_upright, padding=1)
        D_org_loleft = F.conv2d(org_pool , self.weight_loleft, padding=1)
        D_org_loright = F.conv2d(org_pool , self.weight_loright, padding=1)
        # Enhanced output
        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)
        D_enhance_upleft = F.conv2d(enhance_pool, self.weight_upleft, padding=1)
        D_enhance_upright = F.conv2d(enhance_pool, self.weight_upright, padding=1)
        D_enhance_loleft = F.conv2d(enhance_pool, self.weight_loleft, padding=1)
        D_enhance_loright = F.conv2d(enhance_pool, self.weight_loright, padding=1)
        # Difference
        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        D_upleft = torch.pow(D_org_upleft - D_enhance_upleft,2)
        D_upright = torch.pow(D_org_upright - D_enhance_upright,2)
        D_loleft = torch.pow(D_org_loleft - D_enhance_loleft,2)
        D_loright = torch.pow(D_org_loright - D_enhance_loright,2)
        # Total difference
        E = (D_left + D_right + D_up +D_down) + 0.5 * (D_upleft + D_upright + D_loleft + D_loright)
        # E = 25*(D_left + D_right + D_up +D_down)
        return E


class L_expYCbCr(nn.Module):
    def __init__(self,patch_size,mean_val):
        super(L_expYCbCr, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x ):
        b,c,h,w = x.shape
        x = ycbcr_to_rgb(x)
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)
        # d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).to(x.device),2))
        return d


class L_exp(nn.Module):
    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x ):
        b,c,h,w = x.shape
        # x = ycbcr_to_rgb(x)
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)
        # d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).to(x.device),2))
        return d


class L_TV1(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV1,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = (torch.abs((x[:,:,1:,:]-x[:,:,:h_x-1,:]))/count_h).sum()     #abs
        w_tv = (torch.abs((x[:,:,:,1:]-x[:,:,:,:w_x-1]))/count_w).sum()
        return self.TVLoss_weight*2*(h_tv+w_tv)/batch_size


class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        # h_tv = (torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2)/count_h).sum()
        # w_tv = (torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2)/count_w).sum()
        # return self.TVLoss_weight*2*(h_tv+w_tv)/batch_size
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size


class L_TVfactors(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TVfactors,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,X):
        loss = 0
        for x in X:
            batch_size = x.size()[0]
            h_x = x.size()[2]
            w_x = x.size()[3]
            count_h =  (x.size()[2]-1) * x.size()[3]
            count_w = x.size()[2] * (x.size()[3] - 1)
            # h_tv = (torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2)/count_h).sum()
            # w_tv = (torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2)/count_w).sum()
            # return self.TVLoss_weight*2*(h_tv+w_tv)/batch_size
            h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
            w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
            loss += self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
        
        return loss


class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)

    def forward(self, x ):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b,c,h,w = x.shape
        # x_de = x.cpu().detach().numpy()
        r,g,b = torch.split(x , 1, dim=1)
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r-mr
        Dg = g-mg
        Db = b-mb
        k =torch.pow( torch.pow(Dr,2) + torch.pow(Db,2) + torch.pow(Dg,2),0.5)
        k = torch.mean(k)
        return k


class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()
        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3


## ----------------------------------------------------------------------
## FROM https://github.com/vincentfung13/MINE/blob/main/network/layers.py
# def edge_aware_loss(img, disp, gmin=2.0, grad_ratio=0.1):
# def edge_aware_loss(img, disp, gmin=2.0, grad_ratio=0.01):
def edge_aware_loss(img, disp, gmin=4.0, grad_ratio=0.01):
    # Compute img grad and grad_max
    grad_img = torch.abs(spatial_gradient(img)).sum(1, keepdim=True).to(torch.float32)
    grad_img_x = grad_img[:, :, 0]
    grad_max_x = torch.amax(grad_img_x, dim=(1, 2, 3), keepdim=True)
    grad_img_y = grad_img[:, :, 1]
    grad_max_y = torch.amax(grad_img_y, dim=(1, 2, 3), keepdim=True)
    # Compute edge mask
    edge_mask_x = grad_img_x / (grad_max_x * grad_ratio)
    edge_mask_y = grad_img_y / (grad_max_y * grad_ratio)
    edge_mask_x = torch.where(edge_mask_x < 1, edge_mask_x, torch.ones_like(edge_mask_x).cuda())
    edge_mask_y = torch.where(edge_mask_y < 1, edge_mask_y, torch.ones_like(edge_mask_y).cuda())
    # Compute and normalize disp grad
    grad_disp = torch.abs(spatial_gradient(disp, normalized=False))
    grad_disp_x = F.instance_norm(grad_disp[:, :, 0])
    grad_disp_y = F.instance_norm(grad_disp[:, :, 1])
    # Compute loss
    grad_disp_x = grad_disp_x - gmin
    grad_disp_y = grad_disp_y - gmin
    loss_map_x = torch.where(grad_disp_x > 0.0, grad_disp_x, torch.zeros_like(grad_disp_x).cuda()) * (1.0 - edge_mask_x)
    loss_map_y = torch.where(grad_disp_y > 0.0, grad_disp_y, torch.zeros_like(grad_disp_y).cuda()) * (1.0 - edge_mask_y)
    return (loss_map_x + loss_map_y).mean()


def edge_aware_loss_v2(img, disp):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    mean_disp = disp.mean(2, True).mean(3, True)
    disp = disp / (mean_disp + 1e-9)
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()


class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
        self.sigma = 10

    def rgb2yCbCr(self, input_im):
        im_flat = input_im.contiguous().view(-1, 3).float()
        # mat = torch.Tensor([[0.257, -0.148, 0.439], [0.564, -0.291, -0.368], [0.098, 0.439, -0.071]]).cuda()
        # bias = torch.Tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]).cuda()
        mat = torch.Tensor([[0.257, -0.148, 0.439], [0.564, -0.291, -0.368], [0.098, 0.439, -0.071]]).to(input_im.device)
        bias = torch.Tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]).to(input_im.device)
        temp = im_flat.mm(mat) + bias
        out = temp.view(input_im.shape[0], 3, input_im.shape[2], input_im.shape[3])
        return out

    # output: output      input:input
    def forward(self, input, output):
        self.output = output
        # self.input = self.rgb2yCbCr(input)
        self.input  = input
        sigma_color = -1.0 / (2 * self.sigma * self.sigma)
        w1 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :] - self.input[:, :, :-1, :], 2), dim=1, keepdim=True) * sigma_color)
        w2 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :] - self.input[:, :, 1:, :], 2), dim=1, keepdim=True) * sigma_color)
        w3 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 1:] - self.input[:, :, :, :-1], 2), dim=1, keepdim=True) * sigma_color)
        w4 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-1] - self.input[:, :, :, 1:], 2), dim=1, keepdim=True) * sigma_color)
        w5 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-1] - self.input[:, :, 1:, 1:], 2), dim=1, keepdim=True) * sigma_color)
        w6 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 1:] - self.input[:, :, :-1, :-1], 2), dim=1, keepdim=True) * sigma_color)
        w7 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-1] - self.input[:, :, :-1, 1:], 2), dim=1, keepdim=True) * sigma_color)
        w8 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 1:] - self.input[:, :, 1:, :-1], 2), dim=1, keepdim=True) * sigma_color)
        w9 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :] - self.input[:, :, :-2, :], 2), dim=1, keepdim=True) * sigma_color)
        w10 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :] - self.input[:, :, 2:, :], 2), dim=1, keepdim=True) * sigma_color)
        w11 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 2:] - self.input[:, :, :, :-2], 2), dim=1, keepdim=True) * sigma_color)
        w12 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-2] - self.input[:, :, :, 2:], 2), dim=1, keepdim=True) * sigma_color)
        w13 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-1] - self.input[:, :, 2:, 1:], 2), dim=1, keepdim=True) * sigma_color)
        w14 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 1:] - self.input[:, :, :-2, :-1], 2), dim=1, keepdim=True) * sigma_color)
        w15 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-1] - self.input[:, :, :-2, 1:], 2), dim=1, keepdim=True) * sigma_color)
        w16 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 1:] - self.input[:, :, 2:, :-1], 2), dim=1, keepdim=True) * sigma_color)
        w17 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-2] - self.input[:, :, 1:, 2:], 2), dim=1, keepdim=True) * sigma_color)
        w18 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 2:] - self.input[:, :, :-1, :-2], 2), dim=1, keepdim=True) * sigma_color)
        w19 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-2] - self.input[:, :, :-1, 2:], 2), dim=1, keepdim=True) * sigma_color)
        w20 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 2:] - self.input[:, :, 1:, :-2], 2), dim=1, keepdim=True) * sigma_color)
        w21 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-2] - self.input[:, :, 2:, 2:], 2), dim=1, keepdim=True) * sigma_color)
        w22 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 2:] - self.input[:, :, :-2, :-2], 2), dim=1, keepdim=True) * sigma_color)
        w23 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-2] - self.input[:, :, :-2, 2:], 2), dim=1, keepdim=True) * sigma_color)
        w24 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 2:] - self.input[:, :, 2:, :-2], 2), dim=1, keepdim=True) * sigma_color)
        p = 1.0

        pixel_grad1 = w1 * torch.norm((self.output[:, :, 1:, :] - self.output[:, :, :-1, :]), p, dim=1, keepdim=True)
        pixel_grad2 = w2 * torch.norm((self.output[:, :, :-1, :] - self.output[:, :, 1:, :]), p, dim=1, keepdim=True)
        pixel_grad3 = w3 * torch.norm((self.output[:, :, :, 1:] - self.output[:, :, :, :-1]), p, dim=1, keepdim=True)
        pixel_grad4 = w4 * torch.norm((self.output[:, :, :, :-1] - self.output[:, :, :, 1:]), p, dim=1, keepdim=True)
        pixel_grad5 = w5 * torch.norm((self.output[:, :, :-1, :-1] - self.output[:, :, 1:, 1:]), p, dim=1, keepdim=True)
        pixel_grad6 = w6 * torch.norm((self.output[:, :, 1:, 1:] - self.output[:, :, :-1, :-1]), p, dim=1, keepdim=True)
        pixel_grad7 = w7 * torch.norm((self.output[:, :, 1:, :-1] - self.output[:, :, :-1, 1:]), p, dim=1, keepdim=True)
        pixel_grad8 = w8 * torch.norm((self.output[:, :, :-1, 1:] - self.output[:, :, 1:, :-1]), p, dim=1, keepdim=True)
        pixel_grad9 = w9 * torch.norm((self.output[:, :, 2:, :] - self.output[:, :, :-2, :]), p, dim=1, keepdim=True)
        pixel_grad10 = w10 * torch.norm((self.output[:, :, :-2, :] - self.output[:, :, 2:, :]), p, dim=1, keepdim=True)
        pixel_grad11 = w11 * torch.norm((self.output[:, :, :, 2:] - self.output[:, :, :, :-2]), p, dim=1, keepdim=True)
        pixel_grad12 = w12 * torch.norm((self.output[:, :, :, :-2] - self.output[:, :, :, 2:]), p, dim=1, keepdim=True)
        pixel_grad13 = w13 * torch.norm((self.output[:, :, :-2, :-1] - self.output[:, :, 2:, 1:]), p, dim=1, keepdim=True)
        pixel_grad14 = w14 * torch.norm((self.output[:, :, 2:, 1:] - self.output[:, :, :-2, :-1]), p, dim=1, keepdim=True)
        pixel_grad15 = w15 * torch.norm((self.output[:, :, 2:, :-1] - self.output[:, :, :-2, 1:]), p, dim=1, keepdim=True)
        pixel_grad16 = w16 * torch.norm((self.output[:, :, :-2, 1:] - self.output[:, :, 2:, :-1]), p, dim=1, keepdim=True)
        pixel_grad17 = w17 * torch.norm((self.output[:, :, :-1, :-2] - self.output[:, :, 1:, 2:]), p, dim=1, keepdim=True)
        pixel_grad18 = w18 * torch.norm((self.output[:, :, 1:, 2:] - self.output[:, :, :-1, :-2]), p, dim=1, keepdim=True)
        pixel_grad19 = w19 * torch.norm((self.output[:, :, 1:, :-2] - self.output[:, :, :-1, 2:]), p, dim=1, keepdim=True)
        pixel_grad20 = w20 * torch.norm((self.output[:, :, :-1, 2:] - self.output[:, :, 1:, :-2]), p, dim=1, keepdim=True)
        pixel_grad21 = w21 * torch.norm((self.output[:, :, :-2, :-2] - self.output[:, :, 2:, 2:]), p, dim=1, keepdim=True)
        pixel_grad22 = w22 * torch.norm((self.output[:, :, 2:, 2:] - self.output[:, :, :-2, :-2]), p, dim=1, keepdim=True)
        pixel_grad23 = w23 * torch.norm((self.output[:, :, 2:, :-2] - self.output[:, :, :-2, 2:]), p, dim=1, keepdim=True)
        pixel_grad24 = w24 * torch.norm((self.output[:, :, :-2, 2:] - self.output[:, :, 2:, :-2]), p, dim=1, keepdim=True)

        ReguTerm1 = torch.mean(pixel_grad1) \
                    + torch.mean(pixel_grad2) \
                    + torch.mean(pixel_grad3) \
                    + torch.mean(pixel_grad4) \
                    + torch.mean(pixel_grad5) \
                    + torch.mean(pixel_grad6) \
                    + torch.mean(pixel_grad7) \
                    + torch.mean(pixel_grad8) \
                    + torch.mean(pixel_grad9) \
                    + torch.mean(pixel_grad10) \
                    + torch.mean(pixel_grad11) \
                    + torch.mean(pixel_grad12) \
                    + torch.mean(pixel_grad13) \
                    + torch.mean(pixel_grad14) \
                    + torch.mean(pixel_grad15) \
                    + torch.mean(pixel_grad16) \
                    + torch.mean(pixel_grad17) \
                    + torch.mean(pixel_grad18) \
                    + torch.mean(pixel_grad19) \
                    + torch.mean(pixel_grad20) \
                    + torch.mean(pixel_grad21) \
                    + torch.mean(pixel_grad22) \
                    + torch.mean(pixel_grad23) \
                    + torch.mean(pixel_grad24)
        total_term = ReguTerm1
        return total_term


## ----------------------------------------------------------------------

# Denoising SMOOTHING from : https://github.com/fengzhang427/HEP/blob/main/models/loss.py

class Smooth_loss(nn.Module):
    def __init__(self):
        super(Smooth_loss, self).__init__()

    def forward(self, input_I, input_R):
        rgb_weights = torch.Tensor([0.2990, 0.5870, 0.1140]).to(input_R.device)
        input_gray = torch.tensordot(input_R, rgb_weights, dims=([1], [-1]))
        input_gray = torch.unsqueeze(input_gray, 1)
        return torch.mean(gradient(input_I, "x") * torch.exp(-10 * gradient(input_gray, "x")) +
                          gradient(input_I, "y") * torch.exp(-10 * gradient(input_gray, "y")))
  
 
def gradient(input_tensor, direction):
    weights = torch.tensor([[0., 0.],
                            [-1., 1.]]
                           ).to(input_tensor.device)
    weights_x = weights.view(1, 1, 2, 2).repeat(1, 1, 1, 1)
    weights_y = torch.transpose(weights_x, 2, 3)

    if direction == "x":
        weights = weights_x
    elif direction == "y":
        weights = weights_y
    grad_out = torch.abs(F.conv2d(input_tensor, weights, stride=1, padding=1))
    return grad_out


class IS_loss(nn.Module):
    def __init__(self):
        super(IS_loss, self).__init__()

    def forward(self, input_I, input_im):
        rgb_weights = torch.Tensor([0.2990, 0.5870, 0.1140]).to(input_im.device)
        input_gray = torch.tensordot(input_im, rgb_weights, dims=([1], [-1]))
        input_gray = torch.unsqueeze(input_gray, 1)
        low_gradient_x = gradient(input_I, "x")
        input_gradient_x = gradient(input_gray, "x")
        k = torch.full(input_gradient_x.shape, 0.01).to(input_im.device)
        x_loss = torch.abs(torch.div(low_gradient_x, torch.max(input_gradient_x, k)))
        low_gradient_y = gradient(input_I, "y")
        input_gradient_y = gradient(input_gray, "y")
        y_loss = torch.abs(torch.div(low_gradient_y, torch.max(input_gradient_y, k)))
        mut_loss = torch.mean(x_loss + y_loss)
        return mut_loss


# perceptual loss from HEP
class vgg_19(nn.Module):
    def __init__(self, index):
        super(vgg_19, self).__init__()
        vgg_model = torchvision.models.vgg19(pretrained=True)
        self.feature_ext = nn.Sequential(*list(vgg_model.features.children())[:index])

    def forward(self, x):
        if x.size(1) == 1:
            x = torch.cat((x, x, x), 1)
        out = self.feature_ext(x)
        return out


class Perceptual_loss(nn.Module):
    def __init__(self):
        super(Perceptual_loss, self).__init__()
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.vgg = vgg_19(20)
        self.vgg.eval()

    def vgg_preprocess(self, batch):
        tensor_type = type(batch.data)
        (r, g, b) = torch.chunk(batch, 3, dim=1)
        batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
        batch = batch * 255                 # * 0.5  [-1, 1] -> [0, 255]
        mean = tensor_type(batch.data.size()).to(batch.device)
        mean[:, 0, :, :] = 103.939
        mean[:, 1, :, :] = 116.779
        mean[:, 2, :, :] = 123.680
        batch = batch.sub(mean)  # subtract mean
        return batch

    def forward(self, img, target):
        img_vgg = self.vgg_preprocess(img)
        target_vgg = self.vgg_preprocess(target)
        img_fea = self.vgg(img_vgg)
        target_fea = self.vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)
