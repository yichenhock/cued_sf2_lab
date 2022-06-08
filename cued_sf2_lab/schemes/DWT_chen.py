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

class DWT():
    
    def __init__(self, X, N, quant_step=17):
        """Initialise the class"""
        self.quant_step = quant_step

        self.X = X
        self.N = N

        # Le-Gall filters
        self.h1 = np.array([-1, 2, 6, 2, -1])/8
        self.h2 = np.array([-1, 2, -1])/4

        self.target_rms = self.rms_err(quantise(X, quant_step), X)

    def encode(self):

        return

    def decode(self, Y):
        
        return 

# -----------------------------------------------------------------------------

