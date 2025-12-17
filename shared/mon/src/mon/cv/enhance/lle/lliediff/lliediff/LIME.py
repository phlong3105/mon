import numpy as np
from scipy.fft import fft2, ifft2
from skimage import exposure


class LIME:
    # initiate parameters
    def __init__(self,
                 img,
                 iterations=1,
                 basic_step=30,
                 step=50,
                 alpha=0.15,
                 rho=1.1,
                 gamma=0.6,
                 strategy=2,
                 exact=True
                 ):
        self.iterations = iterations
        self.step = step
        self.iter = 0
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
        self.strategy = strategy
        self.exact = exact
        self.basic_step = basic_step
        self.initialize(img)

    # initiate Dx,Dy,DTD
    def initialize(self, img):
        self.low_light_img = img
        self.row = self.low_light_img.shape[0]
        self.col = self.low_light_img.shape[1]

        self.T_esti = np.max(self.low_light_img, axis=2)
        self.Dv = -np.eye(self.row) + np.eye(self.row, k=1)
        self.Dh = -np.eye(self.col) + np.eye(self.col, k=-1)

        dx = np.zeros((self.row, self.col))
        dy = np.zeros((self.row, self.col))
        dx[1, 0] = 1
        dx[1, 1] = -1
        dy[0, 1] = 1
        dy[1, 1] = -1
        dxf = fft2(dx)
        dyf = fft2(dy)
        self.DTD = np.conj(dxf) * dxf + np.conj(dyf) * dyf

        self.W = self.Strategy()

    # strategy 2
    def Strategy(self):
        if self.strategy == 2:
            self.Wv = 1 / (np.abs(self.Dv @ self.T_esti) + 1)
            self.Wh = 1 / (np.abs(self.T_esti @ self.Dh) + 1)
            return np.vstack((self.Wv, self.Wh))
        else:
            return np.ones((self.row * 2, self.col))

    # T subproblem
    def T_sub(self, G, Z, miu):
        X = G - Z / miu
        Xv = X[:self.row, :]
        Xh = X[self.row:, :]

        numerator = fft2(2 * self.T_esti + miu * (self.Dv @ Xv + Xh @ self.Dh))
        denominator = self.DTD * miu + 2
        T = np.real(ifft2(numerator / denominator))

        return exposure.rescale_intensity(T, (1e-3, 1), (0.001, 1))

    # G subproblem
    def G_sub(self, T, Z, miu, W):
        epsilon = self.alpha * W / miu
        temp = np.vstack((self.Dv @ T, T @ self.Dh)) + Z / miu
        return np.sign(temp) * np.maximum(np.abs(temp) - epsilon, 0)

    # Z subproblem
    def Z_sub(self, T, G, Z, miu):
        return Z + miu * (np.vstack((self.Dv @ T, T @ self.Dh)) - G)

    # miu subproblem
    def miu_sub(self, miu):
        return miu * self.rho

    def init_update(self):
        T = np.zeros((self.row, self.col))
        G = np.zeros((self.row * 2, self.col))
        Z = np.zeros((self.row * 2, self.col))
        miu = 1
        return T, G, Z, miu

    def update(self, G, Z, miu):
        T = self.T_sub(G, Z, miu)
        G = self.G_sub(T, Z, miu, self.W)
        Z = self.Z_sub(T, G, Z, miu)
        miu = self.miu_sub(miu)
        self.T = T
        Tc3 = np.repeat(T[..., None], 3, axis=-1)

        return Tc3, T, G, Z, miu

    def run(self, step=1):
        T, G, Z, miu = self.init_update()
        Tc3 = np.repeat(T[..., None], 3, axis=-1)

        # for i in range(0,self.iterations):
        # for i in range(0,self.basic_step):
        for i in range(0, step):
            Tc3, T, G, Z, miu = self.update(G, Z, miu)

        return Tc3