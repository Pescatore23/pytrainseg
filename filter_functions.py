# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:35:22 2022

@author: fische_r
"""

# necessary packages for feature stack
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import filters, feature, io
from skimage.morphology import disk,ball

# from sklearn.ensemble import RandomForestClassifier
import os
import imageio
import sys

import dask
import dask.array
from dask.distributed import Client, LocalCluster
# import cupy as cp
# import cucim
from itertools import combinations_with_replacement
import xarray as xr

# functions take chunked dask-array as input

# start-up cluster, TODO: option to connect to exisitng cluster
cluster = LocalCluster() 
client = Client(cluster)
print('Dashboard at '+cluster.dashboard_link)

default_feature_dict = {'Gaussian': True, 
               # 'Sobel': True,
               'Hessian': True,
               'Diff of Gaussians': True,
               'maximum': True,
               'minimum': True,
               'median': True,
               'extra_time_ranks': True,
              }

class image_filter:
    def __init__(self,
                 data_path = None,
                 sigmas = [0,2,4],
                 feature_dict = default_feature_dict,
                 mod_feat_dict = None,
                 chunksize = '20 MiB'
                 ):
        if mod_feat_dict is not None:
            for key in mod_feat_dict:
                feature_dict[key] = mod_feat_dict[key]
        
        self.data_path = data_path
        self.sigmas = sigmas        
        self.feature_dict = feature_dict
        # TODO: allow option of custom shaped chunks
        self.chunks = chunksize
        
        # not sure if this is clever, does dask understand that this data is reused?
        self.Gaussian_dict = {}
        self.Gradient_dict = {}
        self.calculated_features = {}
        
        
    def open_raw_data(self):
        data = xr.open_dataset(self.data_path, chunks = 'auto')
        da = dask.array.from_array(data.tomo).rechunk(chunks = self.chunks)
        da.name = 'original'
        self.data = da
    
    def load_raw_data(self):
        data = xr.load_dataset(self.data_path, chunks = 'auto')
        da = dask.array.from_array(data.tomo).rechunk(chunks = self.chunks)
        # da.name = 'original'
        self.data = da 
        
    def Gaussian_Blur_4D(self, sigma):
        deptharray = np.ones(self.data.ndim)+4*sigma
        deptharray = tuple(np.min([deptharray, self.data.shape], axis=0))
        G = self.data.map_overlap(filters.gaussian, depth=deptharray, boundary='nearest', sigma = sigma)
        # G.name = 'Gaussian_Blur_'+f'{sigma:.1f}'
        self.calculated_features.append(G)
        self.Gaussian_dict[sigma] = G
        
    def Gaussian_stack(self):
        flag = True
        for sigma in self.sigmas:
            if np.abs(sigma-0)<0.1:
                if flag:
                    flag = False
                    self.Gaussian_dict['original'] = self.data
            else:
                self.Gaussian_Blur_4D(self, sigma)
    
    def Gradients(self):
        for key in self.Gaussian_dict:
            G = self.Gaussian_dict[key]
            gradients = dask.array.gradient(G)
            axes = G.ndim
            for ax0 in range(axes):
                gradients[ax0].name = 'Gradient_sigma_'+key+'_'+str(ax0)
            self.Gradient_dict[key] = gradients
            
    def Hessian(self):
        # TODO: add
        for key in self.Gradient_dict.keys():
            axes = self.data.ndim
            gradients = self.Gradient_dict[key]
            H_elems = [dask.array.gradient(gradients[ax0], axis=ax1) for ax0, ax1 in combinations_with_replacement(axes, 2)]
            # elems = [(ax0,ax1) for ax0, ax1 in combinations_with_replacement(axes, 2)]
            
            # for (Helm, elm) in zip(H_elems,elems):
                # Helm.name = ''.join(['hessian_sigma_',key,'_',str(elm[0]),str(elm[1])])
            
            self.calculated_features = self.calculated_features+gradients+H_elems
    
    # TODO: include feature selection either in compute (better) or save
    def prepare(self):
        self.Gaussian_stack()
        self.Gradients()
        self.Hessian()
        
    def compute(self):
        self.prepare()
        for feat in self.calculated_features:
            feat.compute()
        
        
        




# def nd_gaussian(da, sig = 0):
#     if np.abs(sig-0)<0.1:
#         G = np.array(da)
#         fullname = 'original'
#     else:
#         deptharray = np.ones(da.ndim)+4*sig
#         deptharray = tuple(np.min([deptharray, da.shape], axis=0))
#         G = da.map_overlap(filters.gaussian, depth=deptharray, boundary='nearest', sigma = sig).compute()
#         # G = da.map_overlap(filters.gaussian, depth=4*sig+1, boundary='none', sigma = sig).compute()
#         fullname = ''.join(['gaussian_4D_',f'{sig:.1f}'])
#     return G, fullname

# def nd_gaussian_space(da, sig = 0):
#     if np.abs(sig-0)<0.1:
#         G = np.array(da)
#         fullname = 'original_space'
#         #TODO: remove sig = 0
#     else:
#         deptharray = np.ones(da.ndim)+4*sig
#         deptharray = tuple(np.min([deptharray, da.shape], axis=0))
#         G = da.map_overlap(filters.gaussian, depth=deptharray, boundary='nearest', sigma = (sig,sig,sig,0)).compute()
#         # G = da.map_overlap(filters.gaussian, depth=4*sig+1, boundary='none', sigma = sig).compute()
#         fullname = ''.join(['gaussian_space_',f'{sig:.1f}'])
#     return G, fullname

# def nd_gaussian_time(da, sig = 0):
#     if np.abs(sig-0)<0.1:
#         G = np.array(da)
#         fullname = 'original_time'
#         #TODO: remove sig = 0
#     else:
#         deptharray = np.ones(da.ndim)+4*sig
#         deptharray = tuple(np.min([deptharray, da.shape], axis=0))
#         G = da.map_overlap(filters.gaussian, depth=deptharray, boundary='nearest', sigma = (0,0,0,sig)).compute()
#         # G = da.map_overlap(filters.gaussian, depth=4*sig+1, boundary='none', sigma = sig).compute()
#         fullname = ''.join(['gaussian_time_',f'{sig:.1f}'])
#     return G, fullname

# #TODO create a class that makes the feature stacks and allows c.hosing suitable backends, e.g. dask, cupy, cucim etc.
# def nd_gaussian_stack(da, sigmas):
#     fullnames = []
#     gstack = np.zeros(list(da.shape) + [len(sigmas)])
#     for sig,i in zip(sigmas, range(len(sigmas))):
#         gstack[...,i], name = nd_gaussian(da, sig)
#         fullnames.append(name)
#     return gstack, fullnames

# def nd_gaussian_stack_space(da, sigmas):
#     fullnames = []
#     gstack = np.zeros(list(da.shape) + [len(sigmas)])
#     for sig,i in zip(sigmas, range(len(sigmas))):
#         gstack[...,i], name = nd_gaussian_space(da, sig)
#         fullnames.append(name)
#     return gstack, fullnames

# def nd_gaussian_stack_time(da, sigmas):
#     fullnames = []
#     tgstack = np.zeros(list(da.shape) + [len(sigmas)])
#     for sig,i in zip(sigmas, range(len(sigmas))):
#         tgstack[...,i], name = nd_gaussian_time(da, sig)
#         fullnames.append(name)
#     return tgstack, fullnames

# def nd_diff_of_gaussian(gstack, sigmas, mode='space'):
# #     #creates a stack of {size} (see below)
#     n = len(sigmas)
#     size = int(n*(n-1)/2)
#     dstack = np.zeros(list(da.shape) + [size])
#     fullnames = []
#     cc = 0
#     for i in range(1,n):
#         for j in range(i):
#             dstack[...,cc] = gstack[...,i] - gstack[...,j]
#             name = ''.join(['diff_of_gauss_',mode,'_',f'{sigmas[i]:.1f}','_',f'{sigmas[j]:.1f}'])
#             fullnames.append(name)
#             cc = cc + 1
#     return dstack, fullnames


# def ball_4d(sig):
#     bnd = np.zeros((sig*2+1,sig*2+1,sig*2+1,sig*2+1), dtype = bool)
#     bnd[sig,sig,sig,sig] = True
#     ecd = ndimage.distance_transform_edt(~bnd)
#     bnd = (ecd<sig+0.01).astype(int)
#     return bnd

# def nd_rank_like_filter(da, sigma, option):
#     """
#      input
#      da - chunked das array up to 4D
#      sigma - kernel size, scalar
#      option, str ('minimum', 'maximum', 'median')
#     """
#     if da.ndim == 2:
#         fp = disk(sigma)
#     if da.ndim == 3:
#         fp = ball(sigma)
#     if da.ndim == 4:
#         fp = ball_4d(sigma)
        
#     if option == 'minimum':
#         fun = ndimage.minimum_filter
#     elif option == 'maximum':
#         fun = ndimage.maximum_filter
#     elif option == 'median':
#         fun = ndimage.median_filter
#     else:
#         print(option+' not available')
#     deptharray = np.ones(da.ndim)+sigma
#     deptharray = tuple(np.min([deptharray, da.shape], axis=0))
#     M = da.map_overlap(fun, depth=deptharray, footprint=fp).compute()
#     # M = da.map_overlap(fun, depth=sigma+1, footprint=fp).compute()
#     fullname = ''.join([option,'_',f'{sigma:.1f}'])
#     return M, fullname

# def nd_rank_like_stack(da, sigmas, option):
#     fullnames = []
#     mstack = np.zeros(list(da.shape) + [len(sigmas)-1])
#     for sig,i in zip(sigmas[1:], range(len(sigmas)-1)):
#         mstack[...,i], name = nd_rank_like_filter(da, sig, option)
#         fullnames.append(name)
#     return mstack, fullnames   

# def nd_dynamic_rank_filter(da, sigma_3D, sigma_t, option):
#     """
#     same as rank filters above, but considering much larger range in time
#     TODO: add clever option with dynmaic model, maybe sigmoid-fit, oscillation, inverse-sigmoid, etc.
#     """
#     fp_3D = ball(sigma_3D)
#     fp_4D = np.zeros(list(fp_3D.shape)+[2*sigma_t], dtype=int)
#     fp_4D[fp_3D>0,:] = 1
    
#     if option == 'minimum':
#         fun = ndimage.minimum_filter
#     elif option == 'maximum':
#         fun = ndimage.maximum_filter
#     elif option == 'median':
#         fun = ndimage.median_filter
#     else:
#         print(option+' not available')
    
#     deptharray = np.ones(da.ndim)+sigma_3D
#     deptharray[-1] = sigma_t
#     deptharray = tuple(np.min([deptharray, da.shape], axis=0))
    
#     M = da.map_overlap(fun, depth=deptharray, footprint=fp_4D).compute()
#     fullname = ''.join(['dynamic_',option,'_',f'{sigma_3D:.1f}','_',f'{sigma_t:.1f}'])
#     return M, fullname

# def nd_dynamic_rank_like_stack(da, sigmas, option, sigma_t=20):
#     fullnames = []
#     mstack = np.zeros(list(da.shape) + [len(sigmas)])
#     for sig,i in zip(sigmas, range(len(sigmas))):
#         mstack[...,i], name = nd_dynamic_rank_filter(da, sig, sigma_t, option)
#         fullnames.append(name)
#     return mstack, fullnames  
        
    
# def diff_to_first_and_last(da):
#     fullnames = ['diff_first', 'diff_last']
#     stack = np.zeros(list(da.shape) + [2])
#     # stack[...,0] = da[...,:] - da[...,0][...,None]
#     # stack[...,1] = da[...,:] - da[...,-1][...,None]
#     # above does not work with lazy loaded xarray/dask arrays, probably the None-thing
#     # maybe somehow possible to avoid for-loop
#     for t in range(da.shape[-1]):
#         stack[:,:,:,t,0] = da[:,:,:,t] - da[:,:,:,0]
#         stack[:,:,:,t,1] = da[:,:,:,t] - da[:,:,:,-1]
    
#     return stack, fullnames
    
# def nd_Hessian_matrix(G):
#     """
#     copied from skimage.feature.hessian_matrix
#     just directly using Gaussian fitered arrays and dask
#     """
    
#     daG = dask.array.from_array(G)
#     gradients = dask.array.gradient(daG)
#     axes = range(G.ndim)
#     gradients = [gradients[ax0].compute() for ax0 in axes]
#     H_elems = [dask.array.gradient(gradients[ax0], axis=ax1).compute() for ax0, ax1 in combinations_with_replacement(axes, 2)]
#     elems = [(ax0,ax1) for ax0, ax1 in combinations_with_replacement(axes, 2)]
#     return H_elems, elems, gradients

# def nd_Hessian_stack(G, sigma):
#     H_elems, elems, gradients = nd_Hessian_matrix(G)
#     hstack = np.zeros(list(G.shape)+[len(elems)])
#     gradstack = np.zeros(list(G.shape)+[len(gradients)])
    
#     #TODO: this is slow, find some better numpy function
#     gradnames = []
#     for i in range(len(elems)):
#         hstack[...,i] = H_elems[i]
#     for i in range(len(gradients)):
#         gradnames.append(''.join(['gradient_',str(i),'_',f'{sigma:.1f}']))
#         gradstack[...,i] = gradients[i]
    
#     # print('got Hessian matrices, now doing the eigs')
#     # eigs = feature.hessian_matrix_eigvals(H_elems) 
#  # for now ignore the eigenvalues (too computationally expensive and H_elems already contains the image curvature  

#     fullnames = []
#     for i,j in elems:
#         fullname = ''.join(['hessian_',str(i),str(j),'_',f'{sigma:.1f}'])
#         fullnames.append(fullname) 
    
#     stack = np.concatenate([hstack,gradstack], axis=-1)
#     fullnames = fullnames+gradnames                         
        
#     return stack, fullnames

# def nd_Hessian_stacks(gstack, sigmas):
#     flag = True
#     fullnames = []
#     for (i, sigma) in zip(range(gstack.shape[-1]), sigmas):
#         a, b = nd_Hessian_stack(gstack[...,i], sigma)
#         asize = a.shape[-1]
#         if flag:
#             flag = False
#             hstacks = np.zeros(list(gstack[...,-1].shape)+[len(sigmas)*asize])
#         hstacks[...,i*asize:i*asize+asize] = a
#         fullnames = fullnames + b
#     return hstacks, fullnames
