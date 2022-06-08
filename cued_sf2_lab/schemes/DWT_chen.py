"""coded by Yi Chen"""
# imports
import warnings
import inspect
import matplotlib.pyplot as plt
import numpy as np
import math

from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.dct import colxfm
from scipy.optimize import fsolve

from cued_sf2_lab.lbt import pot_ii, matrix_splitter, totalbits, dct_totalbits
from cued_sf2_lab.dct import dct_ii, regroup
from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.laplacian_pyramid import quantise

from cued_sf2_lab.dwt import dwt
from cued_sf2_lab.dwt import idwt

from .common import rms_err, calculate_bits

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class DWT():
    
    def __init__(self, X, n, quant_step=17):
        """Initialise the class"""
        self.quant_step = quant_step

        self.X = X
        self.n = n

        self.target_rms = rms_err(quantise(X, quant_step), X)
        self.bits_ref = calculate_bits(quantise(self.X, self.quant_step))

    def encode(self):
        # generate Y
        # self.__init__(X, n)
        Y = self.nlevdwt(np.copy(self.X),self.n)
        return Y

    def decode(self, Y):
        Z = self.nlevidwt(np.copy(Y), self.n)
        return Z

    def const_step_enc_dec(self):
        # generate Y
        Y = self.nlevdwt(self.X, self.n)
        # quantise and get Yq
        dwtstep = np.ones((3, self.n+1))*17 # constant step size 17

        Yq, dwtent, total_bits = self.quantdwt(Y, dwtstep)
        # reconstruct Z from Yq

        Z = self.nlevidwt(Yq, self.n)
        return Z

# -----------------------------------------------------------------------------

    def quantdwt(self, Y: np.ndarray, dwtstep: np.ndarray, rise = None):
        """
        Parameters:
            Y: the output of `dwt(X, n)`
            dwtstep: an array of shape `(3, n+1)`
        Returns:
            Yq: the quantized version of `Y`
            dwtent: an array of shape `(3, n+1)` containing the entropies
        """
        
        # dwtstep has the size of 3x(N+1)
        N = np.shape(dwtstep)[1]-1 # levels
        w = np.shape(Y)[0]//(2**(N-1)) # width of the 'Y' image of the smallest level

        dwtent = np.zeros_like(dwtstep)
        
        Yq = Y
        # top left quadrant
        Yq[0:w//2, 0:w//2] = quantise(Yq[0:w//2, 0:w//2], dwtstep[0,N], rise)

        dwtent[0,N] = bpp(Yq[0:w//2, 0:w//2])
        total_bits = bpp(Yq[0:w//2, 0:w//2])*np.shape(Yq[0:w//2, 0:w//2])[0]**2
        
        # quantise Y area by area
        for n in range(N): 
            for k in range(3): 
                if k == 0: # top right quadrant
                    r0 = 0
                    r1 = w//2
                    c0 = w//2
                    c1 = w
                elif k ==1: # bottom left quadrant
                    r0 = w//2
                    r1 = w
                    c0 = 0
                    c1 = w//2
                elif k == 2: # bottom right quadrant
                    r0 = w//2
                    r1 = w
                    c0 = w//2
                    c1 = w
                Yq[r0:r1, c0:c1] = quantise(Yq[r0:r1, c0:c1], dwtstep[k, N-n-1], rise)
                dwtent[k, N-n-1] = bpp(Yq[r0:r1, c0:c1])
                total_bits += dwtent[k, N-n-1]*np.shape(Yq[r0:r1, c0:c1])[0]**2
            w = w*2
            
        # quantise final low pass
        return Yq, dwtent, total_bits

    def nlevdwt(self, X, n):
        m = np.shape(X)[0]
        
        Y=dwt(X)

        for i in range(n-1): 
            m = m//2
            Y[:m,:m] = dwt(Y[:m,:m])
            
        return Y

    def nlevidwt(self, Y, n):
        m = np.shape(Y)[0]//2**(n-1)
        
        for i in range(n-1): 
            Y[:m,:m] = idwt(Y[:m,:m])
            m *= 2

        Z = idwt(Y)
        return Z
    
    # stuff to do with equal MSE
    def step_ratios_dwt_emse(self, X, N):
        
        dwtstep_ratios = np.zeros((3, N+1))
        sqrt_energies = np.zeros((3, N+1))
        
        # test image
        Xt = np.zeros_like(X)
        Yt = self.nlevdwt(Xt, N)
        
        mid = 256//2**(N+1)
        # set centre of subimage to 100
        Yt[mid, mid] = 100
        Xtr = self.nlevidwt(Yt, N)
        
        sqrt_energies[0,N] = math.sqrt(np.sum(Xtr**2.0))
        
        w = np.shape(X)[0]//(2**(N-1))
        for n in range(N): 
            for k in range(3): 
                if k == 0: # top right quadrant
                    r0 = 0
                    r1 = w//2
                    c0 = w//2
                    c1 = w
                elif k ==1: # bottom left quadrant
                    r0 = w//2
                    r1 = w
                    c0 = 0
                    c1 = w//2
                elif k == 2: # bottom right quadrant
                    r0 = w//2
                    r1 = w
                    c0 = w//2
                    c1 = w
                    
                # test image
                Xt = np.zeros_like(X)
                Yt = self.nlevdwt(Xt, n)
                # set centre of subimage to 100
                Yt[(r0+r1)//2, (c0+c1)//2] = 100
                Xtr = self.nlevidwt(Yt, N)
                sqrt_energies[k, N-n-1] = math.sqrt(np.sum(Xtr**2.0))
                
            w = w*2
        
        # use the sqrt_energies to get step ratio
        sqrt_energies = np.array(sqrt_energies)
        # let the ratio be over sqrt_energies[0,N]
        dwtstep_ratios = sqrt_energies[0,N]/sqrt_energies
        dwtstep_ratios[1:3,N] = 0
        return dwtstep_ratios

    def get_rms_error(self, dwtstep): 
        Y = self.nlevdwt(self.X,self.n) # generate Y
        Yq, dwtent, total_bits = self.quantdwt(Y, dwtstep) # quantise
        Z = self.nlevidwt(Yq, self.n) # reconstruct
        return np.std(self.X-Z)

    def rms_diff(self, dwtstep):
        return self.get_rms_error(dwtstep) - self.target_rms

    def get_optimum_step_ratio(self, X, n):
        self.__init__(X, n)

        step_ratios = self.step_ratios_dwt_emse(X, n)

        opt_step = None

        step0 = [1, 5, 10, 15]
        for s0 in step0:
            try:
                result = fsolve(lambda step: self.rms_diff(step*step_ratios), s0, xtol=1e-4)
                opt_step = result[0]
                break
            except RuntimeWarning:
                continue

        dwtstep = step_ratios*opt_step
        return dwtstep

    def enc_dec_quantise_rise(self, X, N, q_step, rise):
        self.__init__(X, N)
        
        # generate Y
        Y = self.nlevdwt(self.X, self.n)
        
        # quantise and get Yq
        step_ratios = self.step_ratios_dwt_emse(X, self.n)
        dwtstep = step_ratios*q_step
        Yq, dwtent, total_bits = self.quantdwt(Y, dwtstep, rise)

        Zp = self.nlevidwt(Yq, self.n)
        quant_error = rms_err(Zp, X)

        return total_bits, quant_error, Zp

    def get_cr_with_opt_step(self, X, N):
        dwtstep = self.get_optimum_step_ratio(X, N)
        self.__init__(X, N)

        # generate Y
        Y = self.encode()

        # quantise and get Yq
        Yq, dwtent, total_bits = self.quantdwt(Y, dwtstep)

        # reconstruct Z from Yq
        Z = self.decode(Yq)

        cr = self.bits_ref/total_bits

        return cr