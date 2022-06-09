from typing import Tuple, NamedTuple, Optional

import numpy as np
from cued_sf2_lab.laplacian_pyramid import quant1, quant2, quantise, bpp
from cued_sf2_lab.dct import dct_ii, colxfm, regroup
from cued_sf2_lab.bitword import bitword
from cued_sf2_lab.jpeg import diagscan, dwtgroup, huffdflt, huffgen, runampl, huffenc, huffdes, HuffmanTable
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.dwt import dwt
from cued_sf2_lab.dwt import idwt


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

def quant2_dwt(Y, dwtstep):
    # dwtstep has the size of 3x(N+1)
    N = np.shape(dwtstep)[1]-1 # levels
    w = np.shape(Y)[0]//(2**(N-1)) # width of the 'Y' image of the smallest level

    dwtent = np.zeros_like(dwtstep)
    
    Yq = Y
    # top left quadrant
    Yq[0:w//2, 0:w//2] = quant2(Yq[0:w//2, 0:w//2], dwtstep[0,N], rise1=dwtstep[0,N])

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
            Yq[r0:r1, c0:c1] = quant2(Yq[r0:r1, c0:c1], dwtstep[k, N-n-1], rise1=dwtstep[k, N-n-1])
            dwtent[k, N-n-1] = bpp(Yq[r0:r1, c0:c1])
            total_bits += dwtent[k, N-n-1]*np.shape(Yq[r0:r1, c0:c1])[0]**2
        w = w*2
        
    # quantise final low pass
    return Yq

def jpegdec_dwt(vlc: np.ndarray, dwtstep, n,
        hufftab: Optional[HuffmanTable] = None,
        dcbits: int = 8, W: int = 256, H: int = 256, log: bool = True
        ) -> np.ndarray:
    '''
    Decodes a (simplified) JPEG bit stream to an image

    Parameters:

        vlc: variable length output code from jpegenc
        qstep: quantisation step to use in decoding
        N: width of the DCT block (defaults to 8)
        M: width of each block to be coded (defaults to N). Must be an
            integer multiple of N - if it is larger, individual blocks are
            regrouped.
        hufftab: if supplied, these will be used in Huffman decoding
            of the data, otherwise default tables are used
        dcbits: the number of bits to use to decode the DC coefficients
            of the DCT
        W, H: the size of the image (defaults to 256 x 256)

    Returns:

        Z: the output greyscale image
    '''
    N = 2**n
    M = N

    opthuff = (hufftab is not None)
    
    # Set up standard scan sequence
    scan = diagscan(N)

    if opthuff:
        if len(hufftab.bits.shape) != 1:
            raise ValueError('bits.shape must be (len(bits),)')
        if log:
            print('Generating huffcode and ehuf using custom tables')
    else:
        if log:
            print('Generating huffcode and ehuf using default tables')
        hufftab = huffdflt(1)
    # Define starting addresses of each new code length in huffcode.
    # 0-based indexing instead of 1
    huffstart = np.cumsum(np.block([0, hufftab.bits[:15]]))
    # Set up huffman coding arrays.
    huffcode, ehuf = huffgen(hufftab)

    # Define array of powers of 2 from 1 to 2^16.
    k = 2 ** np.arange(17)

    # For each block in the image:

    # Decode the dc coef (a fixed-length word)
    # Look for any 15/0 code words.
    # Choose alternate code words to be decoded (excluding 15/0 ones).
    # and mark these with vector t until the next 0/0 EOB code is found.
    # Decode all the t huffman codes, and the t+1 amplitude codes.

    eob = ehuf[0]
    run16 = ehuf[15 * 16]
    i = 0
    Zq = np.zeros((H, W))

    if log:
        print('Decoding rows')
    for r in range(0, H, M):
        for c in range(0, W, M):
            yq = np.zeros(M**2)

            # Decode DC coef - assume no of bits is correctly given in vlc table.
            cf = 0
            if vlc[i, 1] != dcbits:
                raise ValueError(
                    'The bits for the DC coefficient does not agree with vlc table')
            yq[cf] = vlc[i, 0] - 2 ** (dcbits-1)
            i += 1

            # Loop for each non-zero AC coef.
            while np.any(vlc[i] != eob):
                run = 0

                # Decode any runs of 16 zeros first.
                while np.all(vlc[i] == run16):
                    run += 16
                    i += 1

                # Decode run and size (in bits) of AC coef.
                start = huffstart[vlc[i, 1].astype('int') - 1] # TODO 
                res = hufftab.huffval[start + vlc[i, 0].astype('int') - huffcode[start]] # TODO
                run += res // 16
                cf += run + 1
                si = res % 16
                i += 1

                # Decode amplitude of AC coef.
                if vlc[i, 1] != si:
                    raise ValueError(
                        'Problem with decoding .. you might be using the wrong hufftab table')
                ampl = vlc[i, 0]

                # Adjust ampl for negative coef (i.e. MSB = 0).
                thr = k[si - 1]
                yq[scan[cf-1]] = ampl - (ampl < thr) * (2 * thr - 1)

                i += 1

            # End-of-block detected, save block.
            i += 1

            yq = yq.reshape((M, M)).T

            # # Possibly regroup yq
            # if M > N:
            #     yq = regroup(yq, M//N)
            Zq[r:r+M, c:c+M] = yq
    # Un regroup
    Zq = dwtgroup(np.copy(Zq), -n)

    # Inverse quantise
    if log:
        print('Inverse quantising using quantdwt')
    Zi = quant2_dwt(Zq, dwtstep)

    # Inverse DWT
    if log:
        print('Inverse {} level DWT\n'.format(n))
    Z = nlevidwt(Zi, n)

    return Z