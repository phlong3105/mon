import numpy as np
from PIL import Image
import os, sys
from bm3d import bm3d_rgb
from experiment_funcs import get_experiment_noise
from scipy.io import savemat, loadmat

eps = np.finfo(float).eps

def normalizeMinMax(I):
    mn = np.min(np.matrix.flatten(I))
    mx = np.max(np.matrix.flatten(I))
    nI = (I-mn)/(mx-mn+eps)
    return nI


def denoiseCBM3D(p_Imat):
    M = loadmat(p_Imat)
    # lSim = M.get('lSim');
    efSim = M.get('efSim')
    # pMapSim = M.get('pMapSim');
    _,psd,_ = get_experiment_noise('g4',0.01,0,efSim.shape)
    
    # Idirect = bm3d_rgb(lSim,psd,'refilter','YCbCr')
    # Idirect = normalizeMinMax(Idirect)
    # print('Idirect created')
    Iqsef = bm3d_rgb(efSim,psd,'refilter','YCbCr')
    Iqsef = normalizeMinMax(Iqsef)
    # print('Iqsef created')    
    # Igrwf = bm3d_rgb(pMapSim,psd,'refilter','YCbCr')
    # Igrwf = normalizeMinMax(Igrwf)
    # print('Iqsef created')

    M = {'Iqsef':Iqsef}
    savemat(os.path.join(p_Imat),M) 
    # Idirect = Image.fromarray( np.uint8(Idirect*255) )
    # Idirect.save(os.path.join(p_Iin,Inum+'_Idirect.png'))
    # Iqsef = Image.fromarray( np.uint8(Iqsef*255) )
    # Iqsef.save(os.path.join(p_Iin,Inum+'_Iqsef.png'))
    # Igrwf = Image.fromarray( np.uint8(Igrwf*255) )
    # Igrwf.save(os.path.join(p_Iin,Inum+'_Igrwf.png'))
    
    print('Denoising COMPLETE \n')

p_Imat = sys.argv[1]
denoiseCBM3D(p_Imat)
