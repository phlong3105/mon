import os

import libs.FULL.utils.losses as MyLoss
import torch
import torch.linalg as la
import torch.nn as nn
# from kornia.color import rgb_to_hls, hls_to_rgb, rgb_to_ycbcr, ycbcr_to_rgb, rgb_to_grayscale, rgb_to_lab, lab_to_rgb, rgb_to_xyz, xyz_to_rgb
from kornia.filters import bilateral_blur
from torch.autograd import Variable

eps = torch.finfo(torch.float32).eps


########## MODULE 1
class Factorization(nn.Module):
    def __init__(self,config) -> None:
        super(Factorization,self).__init__()
        self.factors    = config.factors
        self.maxIt      = config.maxIt
        self.device     = config.device
        self.Etmean     = [[] for i in range(self.factors)]
        self.relu       = nn.ReLU(inplace=True)
        self.etaA       = config.etaA
        self.epoch      = 0
        self.freeze     = config.freeze
        self.p_resDir   = config.p_resDir
        self.mode       = config.mode
        self.dataMean   = config.dataMean
        self.f_initialize = True

        self.lmbda_A    = nn.ModuleList()
        self.lmbda_E    = nn.ModuleList()
        self.step       = nn.ModuleList()
        self.Xmean      = 0
        for i in range(self.factors):
            self.lmbda_A.append((nn.ParameterList([nn.Parameter(Variable(torch.tensor(0.0, dtype=torch.float32, device=self.device), requires_grad=True)) for t in range(self.maxIt)])))
            self.lmbda_E.append((nn.ParameterList([nn.Parameter(Variable(torch.tensor(0.0, dtype=torch.float32, device=self.device), requires_grad=True)) for t in range(self.maxIt)])))
            self.step.append((nn.ParameterList([nn.Parameter(Variable(torch.tensor(1.0, dtype=torch.float32, device=self.device), requires_grad=True)) for t in range(self.maxIt)])))
        self.lmbda_A_backup = [[torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=False) for t in range(self.maxIt)] for tt in range(self.factors)]
        self.step_backup    = [[torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=False) for t in range(self.maxIt)] for tt in range(self.factors)]
        self.lmbda_E_backup = [[torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=False) for t in range(self.maxIt)] for tt in range(self.factors)]

    def thresE(self, inputs, threshold):
        normed_inputs       = la.vector_norm(inputs, dim=1)
        out                 = torch.max(1-torch.div(threshold,(normed_inputs+eps)), torch.zeros([1,1], device=self.device)).unsqueeze(1).repeat(1,inputs.shape[1],1,1) * inputs
        return out

    def thresA(self, inputs, threshold):
        normed_inputs       = la.vector_norm(inputs, dim=1)
        normed_inputs_norm  = torch.sqrt(torch.sum(normed_inputs, dim=[1,2])+eps)
        out                 = torch.max(1-torch.div(threshold,(normed_inputs_norm+eps)), torch.zeros([1,1], device=self.device)).t().repeat(1,inputs.shape[1]).unsqueeze(2).unsqueeze(3) * inputs
        return out
    
    def CheckNegative(self, f):
        isNegative = False
        if self.mode=='train':
            for t in range(self.maxIt):
                if (self.lmbda_A[f][t].data < 0) or (self.lmbda_E[f][t].data < 0): 
                    isNegative = True
                    # print('Negative detected')
            if(isNegative):
                for t in range(self.maxIt):
                    self.lmbda_A[f][t].data     = self.lmbda_A_backup[f][t] 
                    self.lmbda_E[f][t].data     = self.lmbda_E_backup[f][t]
                    self.step[f][t].data        = self.step_backup[f][t]
            else:
                for t in range(self.maxIt):
                    self.lmbda_A_backup[f][t]    = self.lmbda_A[f][t].data
                    self.lmbda_E_backup[f][t]    = self.lmbda_E[f][t].data
                    self.step_backup[f][t]       = self.step[f][t].data
        return isNegative
    
    def InitializeThs(self, f, s, X_mean=None):
        eta_a       = self.etaA
        eta_b       = (f+1)/self.factors
        if s==0:
            if self.f_initialize: 
                self.lmbda_A[f][0].data  = torch.clone(eta_a*self.lmbda_A[f][0].data + (1-eta_a)*eta_b*X_mean)
                self.lmbda_E[f][0].data  = torch.clone(eta_a*self.lmbda_E[f][0].data + (1-eta_a)*(1-eta_b)*X_mean)
        if s>0 and self.f_initialize:           
            # THIS SHOULD HAPPEN ONLY FOR ONE TIME
            # print('INITIALIZING STAGE')
            self.lmbda_A[f][s].data      = torch.clone(self.lmbda_A[f][s-1].data)
            self.lmbda_E[f][s].data      = torch.clone(self.lmbda_E[f][s-1].data)
            self.step[f][s].data         = torch.clone(self.step[f][s-1].data)
            if s==self.maxIt-1: # last stage
                self.f_initialize = False
                # print('SWITCHING OFF INITIALIZATION !!!')

        if self.epoch>self.freeze:
            self.lmbda_A[f][s].requires_grad_ = False
            self.lmbda_E[f][s].requires_grad_ = False
            self.step[f][s].requires_grad_ = False
        return

    def factorize(self, X, f):
        eta_b       = (f+1)/self.factors
        X_2         = la.vector_norm(X, ord=2)
        X_mean      = torch.mean(torch.ravel(X))

        self.InitializeThs(f,0, X_mean)
        Eths        = self.lmbda_E[f][0]/self.step[f][0]
        E_t         = self.thresE(X, Eths)
        Aths        = self.lmbda_A[f][0]/self.step[f][0]
        A_t         = self.thresA(X - E_t, Aths)
        Y_t         = torch.div(X,X_2+eps)
        for t in range(1, self.maxIt):            
            self.InitializeThs(f,t)
            Eths        = self.lmbda_E[f][t]/self.step[f][t]
            E_t         = self.thresE(X - A_t - Y_t/self.step[f][t], Eths)
            Aths        = self.lmbda_A[f][t]/self.step[f][t]
            A_t         = self.thresA(X - E_t - Y_t/self.step[f][t], Aths)
            Y_t         = Y_t + self.step[f][t]*(E_t + A_t - X)

        E_t     = self.relu(E_t) 
        A_t     = self.relu(A_t)

        loss = torch.abs(torch.sum(E_t)/(torch.sum(X)+eps) - eta_b)      #1
        return E_t,loss
    
    def forward(self, input, epoch, imNum=None):
        self.epoch = epoch
        allE = torch.Tensor().to(input.device)
        loss = 0 
        
        if not os.path.exists(os.path.join(self.p_resDir, 'factors')):  os.makedirs(os.path.join(self.p_resDir, 'factors'))
        
        # FACTORIZATION 
        a    = input
        for i in range(self.factors):
            e,l = self.factorize(a,i)
            if self.mode=='train': 
                self.Etmean[i].append(torch.sum(e)/torch.sum(input))
                loss += l
            a = a - e
            if i>0: 
                e = torch.abs(e-allE[:,3*(i-1):3*i,:,:])
            allE = torch.cat([allE,e], dim=1)
        
        if self.mode=='train': self.Xmean = (self.Xmean+torch.mean(input))*0.5
        return allE,loss


########## MODULE 2
class Fusion(nn.Module):
    def __init__(self, config) -> None:
        super(Fusion,self).__init__()
        self.relu          = nn.ReLU(inplace=True)
        self.sigmoid       = nn.Sigmoid()
        self.factors       = config.factors
        self.p_resDir      = config.p_resDir
        self.device       = config.device
        self.mode          = config.mode
        n_filters          = 3
        #Encoder
        Hin                   = 3*(self.factors+1)
        Hout                  = 3*(self.factors+1) 
        # Hout                  = self.factors+1
        self.e_conv1        = nn.Conv2d(Hin, n_filters, 3,1,1,bias=True) 
        self.e_conv2        = nn.Conv2d(n_filters, n_filters, 3,1,1,bias=True) 
        self.e_conv3        = nn.Conv2d(n_filters, n_filters, 3,1,1,bias=True) 
        self.e_conv4        = nn.Conv2d(n_filters, n_filters, 3,1,1,bias=True) 

        self.d_conv5        = nn.Conv2d(n_filters*2, n_filters, 3,1,1,bias=True) 
        self.d_conv6        = nn.Conv2d(n_filters*2, n_filters, 3,1,1,bias=True) 
        self.d_conv7        = nn.Conv2d(n_filters*2, Hout, 3,1,1, bias=True) 
        self.encoder        = nn.ModuleList([self.e_conv1, self.e_conv2, self.e_conv3, self.e_conv4])
        self.decoder        = nn.ModuleList([self.d_conv5, self.d_conv6, self.d_conv7])
    
    def forward(self, S, imNum=None):
        s   = list(torch.split(S, 3, dim=1))
        w = [1, 1, 1, 1, 1, 1]
        # w = [1, 4, 4, 4, 4, 4]  #<-- GOLD lolsyn <---- <---- <----- <-----
        # w = [0.5, 8, 4, 2, 1, 1]    #<--GOLD qual
        for i in range(len(s)):
            s[i] = s[i]*w[i]
        S = torch.cat(s,dim=1)
        e1      = self.relu(self.e_conv1(S))
        e2      = self.relu(self.e_conv2(e1))
        e3      = self.relu(self.e_conv3(e2))
        e4      = self.relu(self.e_conv3(e3))
        d1      = self.relu(self.d_conv5(torch.cat([e3,e4],1)))
        d2      = self.relu(self.d_conv6(torch.cat([e2,d1],1)))
        O       = torch.tanh(self.d_conv7(torch.cat([e1,d2],1)))
        
        # SIMPLE Merge <---- GOLDEN
        r   = list(torch.split(O, 3, dim=1))
        x   = s[0]        
        for i in range(5):
            for j in range(self.factors+1):                                 
                x     = x + r[j]*(torch.pow(x,2)-x)
        
        # # BASELINE 1
        # r   = list(torch.split(O, 3, dim=1))
        # s   = list(torch.split(S, 3, dim=1))
        # x   = s[0]
        # w1 = []
        # for j in range(1,self.factors+1):
        #     # w1.append(torch.sum(s[j]).item())
        #     w1.append(torch.mean(s[j]).item())
        # w1 = np.array(w1)/sum(w1)
        # # w1 = 1-w1


        # y = rgb_to_lab(s[0])
        
        # for j in range(self.factors):
        #     # y = 0.5*(y + (s[j+1]))
        #     # y[:,0,:,:] = alpha*y[:,0,:,:] + (1-alpha)*torch.squeeze(gaussian_blur2d(torch.unsqueeze(rgb_to_lab(s[j+1])[:,0,:,:],dim=1), (25,25), (1,1)),dim=1)
        #     # y = alpha*y + (1-alpha)*gaussian_blur2d(rgb_to_lab(s[j+1]), (5,5), (1,1))
            
        #     filtSize = 4*j+1
        #     y = (1-w1[j])*y + w1[j]*gaussian_blur2d(rgb_to_lab(s[j+1]), (filtSize,filtSize), (1,1))
        #     ye   = lab_to_rgb(y)
        #     tmax  = torch.quantile(torch.ravel(ye),0.99,dim=0)
        #     tmin  = torch.quantile(torch.ravel(ye),0.01,dim=0)
        #     ye     = torch.clamp(ye,tmin,tmax)/tmax
        #     y = rgb_to_lab(ye)
        #     save_image(ye, os.path.join(self.p_resDir, 'baseline1' , imNum+'_'+str(j)+'.jpg'))
        # x = lab_to_rgb(y)
        #         
        return x


############### FULL ARCH
class RRNet(nn.Module):
    def __init__(self, config) -> None:
        super(RRNet,self).__init__()
        self.factNet        = Factorization(config)
        self.fuseNet        = Fusion(config)
        self.L_color        = MyLoss.L_color()
        self.L_exp          = MyLoss.L_exp(16,0.6) 
        self.L_TV           = MyLoss.L_TV()
        self.L              = dict.fromkeys(('L_color','L_exp','L_TV','L_fact'))
        self.wc             = config.wc
        self.we             = config.we
        self.wt             = config.wt
        self.wf             = config.wf
        self.freeze         = config.freeze
        self.factors        = config.factors
        self.maxIt          = config.maxIt
        self.f_denoise      = config.f_denoise
        self.mode           = config.mode
    
    def RRLoss(self, Xout, L_fact, epoch):
        if epoch>self.freeze:
            L_color             = self.wc*torch.mean(self.L_color(Xout))
            self.L['L_color']   = L_color.item()
            L_exp               = self.we*torch.mean(self.L_exp(Xout))
            self.L['L_exp']     = L_exp.item()
            L_TV                = self.wt*self.L_TV(Xout)
            self.L['L_TV']      = L_TV.item()
            L_fact              = 0 #self.wf*L_fact
            self.L['L_fact']    = 0 #L_fact.item()
            Floss               = L_color + L_exp + L_TV + L_fact
        else:
            self.L['L_color']   = 0
            self.L['L_exp']     = 0
            self.L['L_TV']      = 0
            L_fact              = self.wf*L_fact
            self.L['L_fact']    = L_fact.item()
            Floss               = L_fact
        return Floss
    
    def freezeFact(self,epoch):
        if epoch>self.freeze:
            for i in range(self.factors):
                for j in range(self.maxIt):
                    self.factNet.lmbda_A[i][j].requires_grad=False
                    self.factNet.lmbda_E[i][j].requires_grad=False
                    self.factNet.step[i][j].requires_grad=False
            for param in self.fuseNet.parameters(): param.requires_grad=True
        return
    
    def forward(self,Xin,epoch=0, imNum=None):
        loss = 0
        # Xfact,L_fact       = self.factNet(Xin, epoch, imNum)
        XfuseIn            = torch.cat([Xin,Xin,Xin,Xin,Xin,Xin],dim=1)
        Xfuse              = self.fuseNet(XfuseIn, imNum)
        L_fact              = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        if self.f_denoise:
            Xfuse              = bilateral_blur(Xfuse,(5,5), 0.5, (1,1))
            # Xfuse              = bilateral_blur(Xfuse,(5,5), 0.1, (0.1,0.1))
        if self.mode=='train':
            loss               += self.RRLoss(Xfuse, L_fact, epoch)
        return Xfuse,loss
