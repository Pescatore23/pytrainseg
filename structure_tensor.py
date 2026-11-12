# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:04:45 2026

structure tensor in dask
adapted from https://github.com/Skielex/structure-tensor/blob/master/structure_tensor/st3d.py

@author: ROFISCHE
"""

import dask.array as lib
import dask_image.ndfilters

def structure_tensor(volume, sigma, rho, truncate = 4):
    '''
    Parameters
    ----------
    volume : dask array
        3D dask array containing the grayscale volume for a time step
    sigma : float or int
        pre-filter sigma
    rho : float or int
        structure range = post-ftiler sigma

    Returns
    -------
    S : 5D dask array
        structure tensor for every pixel in volume

    '''
    
    # Computing derivatives (scipy implementation truncates filter at 4 sigma).
    Vx = dask_image.ndfilters.gaussian_filter(volume, sigma, order=(0, 0, 1), mode="nearest", truncate=truncate)  # type: ignore
    Vy = dask_image.ndfilters.gaussian_filter(volume, sigma, order=(0, 1, 0), mode="nearest", truncate=truncate)  # type: ignore
    Vz = dask_image.ndfilters.gaussian_filter(volume, sigma, order=(1, 0, 0), mode="nearest", truncate=truncate)  # type: ignore  
    
    S = [] # append to list instead of pre-allocate memory, dask takes care of thos
    
    # Integrating elements of structure tensor (scipy uses sequence of 1D).
    S.append(lib.multiply(Vx, Vx))
    S.append(lib.multiply(Vy, Vy))
    S.append(lib.multiply(Vz, Vz))
    S.append(lib.multiply(Vx, Vy))
    S.append(lib.multiply(Vx, Vz))
    S.append(lib.multiply(Vy, Vz))
    
    for i in range(6):
        S[i] = dask_image.ndfilters.gaussian_filter(S[i], rho, mode="nearest", truncate=truncate)
        
    S = lib.stack(S)
        
    return S

# def eig_special_3d(S):
def eig_special_3d_robert(S):   
#     #RF: the original code uses quite a few memory allocations incompitble with dask
#     # therfore the eigenvalue extraction is built directly from https://dl.acm.org/doi/epdf/10.1145/355578.366316
#     # testing shows that the built-in numpy (and dask methods) are faster for a single matrix, but full volume much much faster with QIM method
    
    #TODO: maybe only use half of the symmetric matrix as the original code does
    
    shp = S.shape
    S = S.reshape(6,-1)
    S0 = S[0]
    S1 = S[1]
    S2 = S[2]
    S3 = S[3]
    S4 = S[4]
    S5 = S[5]
    
    
    # 3m = trace(A)
    m = (S0+S1+S2)/3
    
    
    # 6p = "sum of squares of elements of (A-mI)
    Sq = lib.subtract(S[:3], m)
    s = lib.einsum("ij,ij->j", S[3:], S[3:])
    s *= 2
    p = lib.einsum("ij,ij->j", Sq, Sq)
    p += s
    p /= 6
    
    #2q = det(A - mI)
    S0q = Sq[0]
    S1q = Sq[1]
    S2q = Sq[2]
    
    det = S0q*S1q*S2q + S3*S5*S4 + S4*S3*S5 - S4*S1q*S4 - S3*S3*S2q - S0q*S5*S5
    q = det/2
    
    #phi
    phi = lib.atan(lib.sqrt(p**3-q**2)/q)/3
    
    #eigenvalues
    psqrt = lib.sqrt(p)
    cos = lib.cos(phi)
    sin = lib.sin(phi)
    
    v1 = m + 2*psqrt*cos
    v2 = m - lib.sqrt(p)*(cos + lib.sqrt(3)*sin)
    v3 = m - lib.sqrt(p)*(cos - lib.sqrt(3)*sin)
    
    val = lib.array([v1,v2,v3])
    valmax = val.max(axis=0)
    valmin = val.min(axis=0)
    
    valmax = valmax.reshape(valmax.shape[:-1] + shp[1:])
    valmin = valmin.reshape(valmin.shape[:-1] + shp[1:])
    
    return valmax, valmin