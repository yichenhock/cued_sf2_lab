"""coded by Yi Chen"""
# imports
import warnings
import inspect
import matplotlib.pyplot as plt
import numpy as np
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.dct import colxfm
from scipy.optimize import fsolve

from cued_sf2_lab.lbt import pot_ii, matrix_splitter, totalbits, dct_totalbits
from cued_sf2_lab.dct import dct_ii, regroup
from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.laplacian_pyramid import quantise

from .common import rms_err, calculate_bits

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class LBT():
    
    def __init__(self, X, N, quant_step=17):
        """Initialise the class"""
        self.quant_step = quant_step
        # define s, the degree of bi-orthogonality
        # self.s = np.sqrt(2)
        self.s = 1.3265

        self.X = X
        self.N = N

        self.Pf, self.Pr = pot_ii(N)
        self.CN = dct_ii(N)
        self.t = np.s_[N//2:-N//2]

        self.target_rms = rms_err(quantise(X, quant_step), X)

    def encode(self):
        Xp = self.prefilter()
        Y = self.dct(Xp, self.N)
        return Y

    def decode(self, Y):
        Z = self.inverse_dct(Y)
        Zp = self.postfilter(Z)
        return Zp

# -----------------------------------------------------------------------------

    def get_filters(self, N):
        self.C_N = dct_ii(N)
        self.Pf, self.Pr = pot_ii(N, self.s)

    def prefilter(self):
        Xp = np.copy(self.X)
        Xp[self.t,:] = colxfm(Xp[self.t,:], self.Pf)
        Xp[:,self.t] = colxfm(Xp[:,self.t].T, self.Pf).T
        return Xp

    def dct(self, Xp, N):
        Y = colxfm(colxfm(np.copy(Xp), self.CN).T, self.CN).T
        # Yr = regroup(np.copy(Y), N)/N
        return Y

    def inverse_dct(self, Y):
        Z = colxfm(colxfm(np.copy(Y).T, self.CN.T).T, self.CN.T)
        return Z

    def postfilter(self, Z):
        Zp = np.copy(Z)  #copy the non-transformed edges directly from Z
        Zp[:,self.t] = colxfm(Zp[:,self.t].T, self.Pr.T).T
        Zp[self.t,:] = colxfm(Zp[self.t,:], self.Pr.T)
        return Zp
    
    def dctbpp(self, Yr, N):
        entropies = np.zeros((N,N))
        # width of subimage: 
        w = int(np.shape(Yr)[0]/N) # = 32
        
        for i in range(N): 
            for j in range(N): 
                Ys = np.zeros((w,w)) # subimage
                for row in range(w): 
                    for col in range(w):
                        Ys[row,col]=Yr[i*w+row,j*w+col]
                entropies[i,j] = bpp(Ys) 

        bits = entropies * (w**2.0)
        total_bits = np.sum(bits)
        return total_bits

    def get_rms_error(self, X, step):
        Y = self.encode()
        Yq = quantise(Y, step)
        Zp = self.decode(Yq)
        return rms_err(Zp, X)

    def rms_diff(self, X, step):
        rms_diff = self.get_rms_error(X, step)
        return rms_diff - self.target_rms

    def get_optimum_step(self, X, N):
        self.__init__(X, N)

        step0 = [1, 5, 10, 15]
        for s0 in step0:
            try:
                result = fsolve(lambda step: self.rms_diff(X, step), s0, xtol=1e-4)
                return result[0]
            except RuntimeWarning:
                continue
        
        return False
    
    def enc_dec_quantise_rise(self, X, N, q_step, rise):
        # opt_step = self.get_optimum_step(X, N)
        self.__init__(X, N)
        Y = self.encode()
        Yq = quantise(Y, q_step, rise)
        Zp = self.decode(Yq)
        return Zp

    def get_cr_with_opt_step(self, X, N):
        opt_step = self.get_optimum_step(X, N)
        self.__init__(X, N)
        Y = self.encode()
        Yq = quantise(Y, opt_step)
        Yr = regroup(np.copy(Yq), N)/N
        cr = self.comp_ratio(Yr)
        return cr

    def comp_ratio(self, Yr):
        """Calculates the compression ratio of the LBT scheme"""
        bits_ref = calculate_bits(quantise(self.X, self.quant_step))
        bits_comp = self.dctbpp(Yr, 16)
        print(bits_ref, bits_comp)
        cr = bits_ref / bits_comp
        return cr