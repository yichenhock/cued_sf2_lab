"""coded by Yi Chen

Contains common functions between the schemes
"""
import numpy as np
from cued_sf2_lab.laplacian_pyramid import bpp

def rms_err(Zp, X):
        """Calculates the RMS error between two images"""
        return np.std(Zp - X)

def calculate_bits(X): 
        """Calculates the number of bits in an image"""
        entropy = bpp(X)
        bits = np.shape(X)[0]*np.shape(X)[1]*entropy
        return bits
