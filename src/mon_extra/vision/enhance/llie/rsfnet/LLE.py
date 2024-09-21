## SETUP #############################################
import cv2
import os, sys
from tqdm.auto import tqdm
import numpy as np
import quaternion as Q
from bm3d import bm3d_rgb
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import concurrent.futures

sys.path.append(os.path.join(os.getcwd(),'utils'))
from experiment_funcs import get_experiment_noise

eps = 1e-7
np.seterr(divide='ignore')
np.set_printoptions(precision=3)

## GLOBAL PARAMS #############################################
max_cpus = 8  # parallel processing jobs
simNum = 5    # number of factors
alpha = 0.1   # can be tweaked 
beta = 0.1
p_resDirRoot = os.path.join(os.getcwd(),'RESULTS')  # Ouput dir
if not os.path.exists(p_resDirRoot): os.makedirs(p_resDirRoot, exist_ok=True)
p_fileList = 'testFilelist.my'  # filepath to file containing absolute paths to images to be processed

## EXPOSURE FACTORIZATION #############################################
def qNorm(X):
    X_2 = np.sqrt(np.sum(np.square(np.absolute( np.ravel(X) ))))
    return X_2

def alm_lasso(X, lmbd, tol=1e-7, maxIter=1000):
    m = X.shape[0]
    n = 4
    A = Q.as_quat_array(np.zeros((m,n)))
    E = Q.as_quat_array(np.zeros((m,n)))
    X_inf = np.max(np.absolute(np.ravel(X,order='F')))
    X_2 = qNorm(X)
    mu = 1.25/(X_2+eps)     # 20.62
    # mu = 1.0/(X_2+eps)    # 20.60
    Y = X/(max(X_2, X_inf/lmbd)+eps)
    rho = 1.5
    for i in range(maxIter):
        # Update E
        # E = X-A+Y/mu
        E = X-A-Y/mu
        t = 1-(lmbd/mu)/np.absolute(E)
        t = [j if j>0 else 0 for j in t]
        E = t*E

        # Update A
        # v = X-E+Y/mu
        v = X-E-Y/mu
        v_2 = qNorm(v)
        A = np.max(1-(1/mu)/v_2, 0)*v

        R = X-A-E
        # Y = Y+mu*R
        Y = Y-mu*R
        mu = rho*mu

        R_2 = qNorm(R)
        if R_2/X_2 < tol:
            break
    return A,E

def Im2qVec(I,m,n):
    w = np.zeros([m*n,1])
    x = np.reshape(I[:,:,0], [m*n,1])
    y = np.reshape(I[:,:,1], [m*n,1])
    z = np.reshape(I[:,:,2], [m*n,1])
    qV = Q.as_quat_array(np.concatenate((w,x,y,z),axis=1))
    return qV

def qVec2Im(qV,m,n):
    qV = Q.as_float_array(qV)
    qVx = np.reshape(qV[:,1], [m,n])
    qVy = np.reshape(qV[:,2], [m,n])
    qVz = np.reshape(qV[:,3], [m,n])
    I = np.stack((qVx,qVy,qVz), axis=-1)
    I = np.float32(I)
    return I

def qFactorize(I,k):
    allA = []
    allE = []
    m,n,_ = I.shape
    wI = np.sum(np.ravel(I))
    qV = Im2qVec(I,m,n)
    i=0
    with tqdm(total=simNum, leave=False) as pbar:
        pbar.update(1)
        while i<simNum-1:
            A,E = alm_lasso(qV, k[i]/np.sqrt(m*n))
            wE = np.sum(np.ravel(qVec2Im(E,m,n)))/wI
            if wE>0.25:
                k[i:] = np.linspace(k[i]+beta,1,len(k)-i)
                continue
            allA.append(qVec2Im(A,m,n))
            allE.append(qVec2Im(E,m,n))
            qV = A
            i = i+1
            pbar.update(1)
    if np.sum(np.ravel(allA[-1]))>eps:
        allE.append(allA[-1])
    # print(f'Final k={k}')
    return allE

def groupLayers(I,allE,thresh=0.01):
    fullEnergy = sum(np.ravel(I))
    thresh= thresh*fullEnergy
    outE = procE(allE,thresh)
    return outE

def procE(inE,thresh):
    outE = []
    i=0
    f_lowE = True
    while i<len(inE):
        ImEnergy = sum(np.ravel(inE[i]))
        if ImEnergy == 0:
            i=i+1
            continue
        f_lowE = (ImEnergy<thresh)
        if len(inE[i])==0:
            inE.pop(i)
            i=i+1
            continue
        if f_lowE and (i<(len(inE)-2)) and len(inE[i+1])>0 :
            outE.append(inE[i]+inE[i+1])
            i=i+1
        elif f_lowE and (i==(len(inE)-1)):
            outE[-1] += inE[i]
        elif f_lowE and (len(inE)==2):
            outE.append(inE[0]+inE[1])
        else:
            outE.append(inE[i])
        i=i+1

    for i in outE:
        ImEnergy = sum(np.ravel(i))
        f_lowE = True if ImEnergy<thresh else False
        if f_lowE:
            outE = procE(outE,thresh)
    
    return outE
            
def prctileNorm(I,mx=99.9,mn=0.1):
    mxThr = np.percentile(np.ravel(I),mx)
    mnThr = np.percentile(np.ravel(I),mn)
    I[I>mxThr] = mxThr
    I[I<mnThr] = mnThr
    return I

def normalizeIm(I,mn=None,mx=None):
    imN = cv2.cvtColor(np.float32(I), cv2.COLOR_RGB2HSV)
    v = imN[:,:,2]
    if mx==None or mn==None:
        mx = max(np.ravel(v))
        mn = min(np.ravel(v))
    v = (v-mn)/(mx-mn + 1e-7)
    imN[:,:,2] = v
    imN = cv2.cvtColor(imN, cv2.COLOR_HSV2RGB)
    return imN

def normalizeMinMax(I):
    mn = np.min(np.ravel(I))
    mx = np.max(np.ravel(I))
    nI = (I-mn)/(mx-mn+eps)
    return nI

## EXPOSURE FUSION #############################################

def compute_weights(images, time_decay):
    (w_c, w_s, w_e) = (1, 1, 1)

    if time_decay is not None:
        tau = len(images)
        sigma2 = (tau**2)/(np.float32(time_decay)**2)
        t = np.array(range(tau-1, -1, -1))
        decay = np.exp(-((t)**2)/(2*sigma2))
    weights = []
    weights_sum = np.zeros(images[0].shape[:2], dtype=np.float32)
    i = 0

    for image in images:
        W = np.ones(image.shape[:2], dtype=np.float32)

        # contrast
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(image_gray, cv2.CV_32F)
        W_contrast = np.absolute(laplacian) ** w_c + 1
        W = np.multiply(W, W_contrast)

        # saturation
        W_saturation = image.std(axis=2, dtype=np.float32) ** w_s
        W = np.multiply(W, W_saturation)

        # well-exposedness
        sigma2 = 0.04    # = 0.2*0.2
        W_exposedness = np.prod(np.exp(-((image - 0.5)**2)/(2*sigma2)), axis=2, dtype=np.float32) ** w_e
        W = np.multiply(W, W_exposedness)

        if time_decay is not None:
            W *= decay[i]
            i += 1

        weights_sum += W
        weights.append(W)

    # normalization
    nonzero = weights_sum > 0
    for i in range(len(weights)):
        weights[i][nonzero] /= weights_sum[nonzero]
    return weights

def gaussian_kernel(size=3, sigma=0.25):
    return cv2.getGaussianKernel(ksize=size, sigma=sigma)

def image_reduce(image):
    kernel = gaussian_kernel()
    out_image = cv2.filter2D(image, -1, kernel)
    out_image = cv2.resize(out_image, None, fx=0.5, fy=0.5)
    return out_image

def image_expand(image, image2):
    kernel = gaussian_kernel()
    out_image = cv2.resize(image, None, fx=2, fy=2)
    m,n,_ = out_image.shape
    M,N,_ = image2.shape
    if not ((M==m) and (N==n)):
        out_image = cv2.resize(out_image,(N,M))
    out_image = cv2.filter2D(out_image, -1, kernel)
    return out_image

def gaussian_pyramid(img, depth):
    G = img.copy()
    gp = [G]
    for i in range(depth):
        G = image_reduce(G)
        gp.append(G)
    return gp

def laplacian_pyramid(img, depth):
    gp = gaussian_pyramid(img, depth)
    lp = [gp[depth-1]]
    for i in range(depth-1, 0, -1):
        GE = image_expand(gp[i], gp[i-1])
        L = cv2.subtract(gp[i-1], GE)
        lp = [L] + lp
    return lp

def pyramid_collapse(pyramid):
    depth = len(pyramid)
    collapsed = pyramid[depth-1]
    for i in range(depth-2, -1, -1):
        collapsed = cv2.add(image_expand(collapsed, pyramid[i]), pyramid[i])
    return collapsed

def exposure_fusion(images, depth=3, time_decay=None):
    if not isinstance(images, list) or len(images) < 2:
        print("Input has to be a list of at least two images")
        return None
    size = images[0].shape
    for i in range(len(images)):
        if not images[i].shape == size:
            print("Input images have to be of the same size")
            return None

    # compute weights
    weights = compute_weights(images, time_decay)

    # compute pyramids
    lps = []
    gps = []
    for (image, weight) in zip(images, weights):
        lps.append(laplacian_pyramid(image, depth))
        gps.append(gaussian_pyramid(weight, depth))

    # combine pyramids with weights
    LS = []
    for l in range(depth):
        ls = np.zeros(lps[0][l].shape, dtype=np.float32)
        for k in range(len(images)):
            lp = lps[k][l]
            gps_float = gps[k][l]
            gp = np.dstack((gps_float, gps_float, gps_float))
            lp_gp = cv2.multiply(lp, gp)
            ls = cv2.add(ls, lp_gp)
        LS.append(ls)

    # collapse pyramid
    fusion = pyramid_collapse(LS)
    return fusion

## DENOISING #############################################

def denoiseCBM3D(I):
    _,psd,_ = get_experiment_noise('g4', 0.005, 0, I.shape) # <--- GOLDEN
    # _,psd,_ = get_experiment_noise('g4', 0.0025, 0, I.shape) 
    # _,psd,_ = get_experiment_noise('g4', 0.001, 0, I.shape) 
    Iqsef = bm3d_rgb(I, psd, 'refilter','YCbCr')
    return Iqsef

## EXPOSURE STACK SIMULATION #############################################

def qSIM(p_Im):
    _,Iname = os.path.split(p_Im)
    Inum,_ = os.path.splitext(Iname)
    I = cv2.cvtColor(cv2.imread(p_Im), cv2.COLOR_BGR2RGB)/255.0
    # I = cv2.resize(I,(512,512))  #<---------------- for faster processing
    
    # Fatcorize
    k = np.linspace(2,1,simNum)
    allE = qFactorize(I,k)
    # allE = groupLayers(I,allE)   # to merge non-informative factors
    print(f'{Iname} Factorization Done.')

    # Normalize factors
    w1E = []
    En=[]
    for i,E in enumerate(allE):
        E = prctileNorm(E)
        E[E<0] = 0
        allE[i] = E
        En.append(normalizeIm(E))
        w1E.append(np.sum(np.ravel(E)))
    # wE = w1E/np.sum(w1E)
    # wI = np.mean(np.ravel(I))
   
    # Simulate exposure stack 
    # simEn = []
    # # f_overExp = False if (wI < 0.3) else True
    # f_overExp = False if (wI < 0.5) else True
    # if f_overExp:
    #     print(f'------------------- IMAGE OVEREXPOSED : {Iname} -------------------')
    #     plusminus = -1
    #     En.reverse()
    #     wE= np.flip(wE)
    # else:
    #     plusminus = 1
    # for i in range(len(En)+1):
    #     if i==0:
    #         simEn.append(I)
    #     else:
    #         t = np.zeros_like(I)
    #         I1 = cv2.cvtColor(np.float32(simEn[i-1]),cv2.COLOR_RGB2LAB)
    #         I2 = cv2.cvtColor(np.float32(En[i-1]),cv2.COLOR_RGB2LAB)
    #         t[:,:,0] = (1-alpha)*I1[:,:,0] + plusminus*alpha*I2[:,:,0]
    #         t[:,:,1] = (1-alpha)*I1[:,:,1] + plusminus*alpha*I2[:,:,1]
    #         t[:,:,2] = (1-alpha)*I1[:,:,2] + plusminus*alpha*I2[:,:,2]
    #         t = cv2.cvtColor(np.float32(t),cv2.COLOR_LAB2RGB)
    #         t = normalizeIm(t, 0,1-wE[i-1])  
    #         simEn.append(t)
    #     simEn[-1] = np.float32(simEn[-1])
    # print(f'{Iname} Simulation Done.')

    # Merge factors and enhance
    # depth = np.uint8(np.floor( np.log(np.min((I.shape[0],I.shape[0]))) / np.log(2) ))
    # if len(En)==1:
    #     Iqsef = I # DO NOTHINGgroupLayers
    # else:
    #     Iqsef = exposure_fusion(simEn, depth)
    # print(f'{Iname} Fusion Done.')

    # Iqsef = prctileNorm(Iqsef)
    # if not f_overExp:
    #     Iqsef = denoiseCBM3D(Iqsef)
    #     print(f'{Iname} Denoising Done.')
    # Iqsef = normalizeMinMax(Iqsef)
    # print(f'{Iname} FINAL wE = {wE} ')

    # save results
    p_resDir = os.path.join(p_resDirRoot,Inum)
    if not os.path.exists(p_resDir):
        os.makedirs(p_resDir)
    for i in range(len(allE)):
        p_res = os.path.join(p_resDir,Inum+'_E'+str(i)+'.jpg')
        cv2.imwrite(p_res, cv2.cvtColor(np.float32(allE[i]), cv2.COLOR_RGB2BGR)*255)
        p_res = os.path.join(p_resDir,Inum+'_En'+str(i)+'.jpg')
        cv2.imwrite(p_res, cv2.cvtColor(np.float32(En[i]), cv2.COLOR_RGB2BGR)*255)
        # p_res = os.path.join(p_resDir,Inum+'_simEn'+str(i)+'.jpg')
        # cv2.imwrite(p_res, cv2.cvtColor(np.float32(simEn[i]), cv2.COLOR_RGB2BGR)*255)
    # p_res = os.path.join(p_resDir, Inum+'_L.png')
    # cv2.imwrite(p_res, cv2.cvtColor(np.float32(Iqsef), cv2.COLOR_RGB2BGR)*255)
    # print(f'{Iname} Saving Done. ')

def qSIM_parallel(max_cpus, args):
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cpus) as executor:
        result = list(executor.map(qSIM, args))

################################################################################################
if __name__=='__main__':    
    with open('/home/saurabh/Desktop/Lolv1_testList.my') as f:
        L = f.readlines()
        LL = [l.split()[0] for l in L]    
        
        # parallel execution
        qSIM_parallel(max_cpus, LL)

        # singular execution
        # for LLL in tqdm(LL,colour='green'):
        #     qSIM(LLL)