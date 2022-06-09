from typing import Tuple, NamedTuple, Optional

import numpy as np
from cued_sf2_lab.laplacian_pyramid import quant1, quant2, quantise, bpp
from cued_sf2_lab.dct import dct_ii, colxfm, regroup
from cued_sf2_lab.bitword import bitword
from cued_sf2_lab.jpeg import diagscan, dwtgroup, huffdflt, huffgen, runampl, huffenc, huffdes
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.dwt import dwt
from cued_sf2_lab.dwt import idwt

import warnings


from cued_sf2_lab.schemes.DWT_chen import DWT

def nlevdwt(X, n):
    m = np.shape(X)[0]
    
    Y=dwt(X)

    for i in range(n-1): 
        m = m//2
        Y[:m,:m] = dwt(Y[:m,:m])
        
    return Y

def nlevidwt(Y, n):
    m = np.shape(Y)[0]//2**(n-1)
    
    for i in range(n-1): 
        Y[:m,:m] = idwt(Y[:m,:m])
        m *= 2

    Z = idwt(Y)
    return Z

def quant1_dwt(Y, dwtstep):
    # dwtstep has the size of 3x(N+1)
    N = np.shape(dwtstep)[1]-1 # levels
    w = np.shape(Y)[0]//(2**(N-1)) # width of the 'Y' image of the smallest level

    dwtent = np.zeros_like(dwtstep)
    
    Yq = Y
    # top left quadrant
    Yq[0:w//2, 0:w//2] = quant1(Yq[0:w//2, 0:w//2], dwtstep[0,N], rise1=dwtstep[0,N])

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
            Yq[r0:r1, c0:c1] = quant1(Yq[r0:r1, c0:c1], dwtstep[k, N-n-1], rise1=dwtstep[k, N-n-1])
            dwtent[k, N-n-1] = bpp(Yq[r0:r1, c0:c1])
            total_bits += dwtent[k, N-n-1]*np.shape(Yq[r0:r1, c0:c1])[0]**2
        w = w*2
        
    # quantise final low pass
    return Yq

def jpegenc_dwt(X: np.ndarray, n,
        opthuff: bool = False, dcbits: int = 8, log: bool = True
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Encodes the image in X to generate a variable length bit stream.

    Parameters:
        X: the input greyscale image
        qstep: the quantisation step to use in encoding
        N: the width of the DCT block (defaults to 8)
        M: the width of each block to be coded (defaults to N). Must be an
            integer multiple of N - if it is larger, individual blocks are
            regrouped.
        opthuff: if true, the Huffman table is optimised based on the data in X
        dcbits: the number of bits to use to encode the DC coefficients
            of the DCT.

    Returns:
        vlc: variable length output codes, where ``vlc[:,0]`` are the codes and
            ``vlc[:,1]`` the number of corresponding valid bits, so that
            ``sum(vlc[:,1])`` gives the total number of bits in the image
        hufftab: optional outputs containing the Huffman encoding
            used in compression when `opthuff` is ``True``.
    '''

    N = 2**n
    M = N

    # DWT on input image X.
    if log:
        print('Forward {} level DWT'.format(n))

    # Perform DWT
    Y = nlevdwt(np.copy(X), n)

    # Get optimum step ratios
    dwt_instance = DWT(X, n)
    dwtstep = dwt_instance.get_optimum_step_ratio(X, n)

    # Quantise 
    Yq = quant1_dwt(Y, dwtstep)

    # Regroup
    Yq = dwtgroup(np.copy(Yq), n)

    # # DCT on input image X.
    # if log:
    #     print('Forward {} x {} DCT'.format(N, N))
    # C8 = dct_ii(N)
    # Y = colxfm(colxfm(X, C8).T, C8).T

    # # Quantise to integers.
    # if log:
    #     print('Quantising to step size of {}'.format(qstep))
    # Yq = quant1(Y, qstep, qstep).astype('int')

    # Generate zig-zag scan of AC coefs.
    scan = diagscan(N)

    # On the first pass use default huffman tables.
    if log:
        print('Generating huffcode and ehuf using default tables')
    dhufftab = huffdflt(1)  # Default tables.
    huffcode, ehuf = huffgen(dhufftab)

    # Generate run/ampl values and code them into vlc(:,1:2).
    # Also generate a histogram of code symbols.
    if log:
        print('Coding rows')
    sy = Yq.shape
    huffhist = np.zeros(16 ** 2)
    vlc = []
    for r in range(0, sy[0], M):
        for c in range(0, sy[1], M):
            yq = Yq[r:r+M,c:c+M]
            # Possibly regroup
            if M > N:
                yq = regroup(yq, N)
            yqflat = yq.flatten('F')
            # Encode DC coefficient first
            dccoef = yqflat[0] + 2 ** (dcbits-1)
            if dccoef < 0 or dccoef > 2**(dcbits):
                warnings.warn('DC coefficients too large for desired number of bits')
                # raise ValueError(
                #     'DC coefficients too large for desired number of bits')
            vlc.append(np.array([[dccoef, dcbits]]))
            # Encode the other AC coefficients in scan order
            # huffenc() also updates huffhist.
            # print(yqflat[scan].astype('int')) # TODO prints
            ra1 = runampl(yqflat[scan].astype('int')) # TODO edited
            vlc.append(huffenc(huffhist, ra1, ehuf))
    # (0, 2) array makes this work even if `vlc == []`
    vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

    # Return here if the default tables are sufficient, otherwise repeat the
    # encoding process using the custom designed huffman tables.
    if not opthuff:
        if log:
            print('Bits for coded image = {}'.format(sum(vlc[:, 1])))
        return vlc, dhufftab, dwtstep

    # Design custom huffman tables.
    if log:
        print('Generating huffcode and ehuf using custom tables')
    dhufftab = huffdes(huffhist)
    huffcode, ehuf = huffgen(dhufftab)

    # Generate run/ampl values and code them into vlc(:,1:2).
    # Also generate a histogram of code symbols.
    if log:
        print('Coding rows (second pass)')
    huffhist = np.zeros(16 ** 2)
    vlc = []
    for r in range(0, sy[0], M):
        for c in range(0, sy[1], M):
            yq = Yq[r:r+M, c:c+M]

            # # Possibly regroup
            # if M > N:
            #     yq = regroup(yq, N)
            
            yqflat = yq.flatten('F')
            # Encode DC coefficient first
            dccoef = yqflat[0] + 2 ** (dcbits-1)
            vlc.append(np.array([[dccoef, dcbits]]))
            # Encode the other AC coefficients in scan order
            # huffenc() also updates huffhist.
            ra1 = runampl(yqflat[scan].astype('int'))
            vlc.append(huffenc(huffhist, ra1, ehuf))
    # (0, 2) array makes this work even if `vlc == []`
    vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

    if log:
        print('Bits for coded image = {}'.format(sum(vlc[:, 1])))
        print('Bits for huffman table = {}'.format(
            (16 + max(dhufftab.huffval.shape))*8))

    return vlc, dhufftab, dwtstep