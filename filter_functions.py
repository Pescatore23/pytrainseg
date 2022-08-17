# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:35:22 2022

@author: fische_r


class to create feature stack on 4D tomo data
transient dimension, e.g. time, should be 4th dimension of input data


TODO: add GPU support (CUDA, cupy, cucim)

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
from itertools import combinations_with_replacement, combinations
import xarray as xr

# functions take chunked dask-array as input

# start-up cluster, TODO: option to connect to exisitng cluster
# TODO: class/function to boot up cluster with custom options, e.g. workers/threads
# esp. use SSD for memory spilling
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

def ball_4d(sig):
    bnd = np.zeros((sig*2+1,sig*2+1,sig*2+1,sig*2+1), dtype = bool)
    bnd[sig,sig,sig,sig] = True
    ecd = ndimage.distance_transform_edt(~bnd)
    bnd = (ecd<sig+0.01).astype(int)
    return bnd

class image_filter:
    def __init__(self,
                 data_path = None,
                 outpath = None,
                 sigmas = [0,2,4],
                 feature_dict = default_feature_dict,
                 mod_feat_dict = None,
                 chunksize = '20 MiB',
                 outchunks = '300 MiB',
                 ranks = ['maximum', 'minimum', 'mean', 'median'],
                 sigma_t = 40
                 ):
        if mod_feat_dict is not None:
            for key in mod_feat_dict:
                feature_dict[key] = mod_feat_dict[key]
        
        self.data_path = data_path
        self.outpath = outpath,
        self.sigmas = sigmas        
        self.feature_dict = feature_dict
        # TODO: allow option of custom shaped chunks
        self.chunks = chunksize
        self.outchunks = outchunks
        
        # not sure if this is clever, does dask understand that this data is reused?
        self.Gaussian_4D_dict = {}
        self.Gaussian_space_dict = {}
        self.Gaussian_time_dict = {}
        self.Gradient_dict = {}
        self.calculated_features = []
        self.feature_names = []
        self.considered_ranks = ranks
        self.sigma_t = sigma_t
        
        
    def open_raw_data(self):
        data = xr.open_dataset(self.data_path, chunks = 'auto')
        da = dask.array.from_array(data.tomo).rechunk(chunks = self.chunks)
        # da.name = 'original'
        self.data = da
    
    def load_raw_data(self):
        data = xr.load_dataset(self.data_path, chunks = 'auto')
        da = dask.array.from_array(data.tomo).rechunk(chunks = self.chunks)
        # da.name = 'original'
        self.data = da 
        
    def Gaussian_Blur_4D(self, sigma):
        # TODO: check on boundary mode
        deptharray = np.ones(self.data.ndim)+4*sigma
        deptharray = tuple(np.min([deptharray, self.data.shape], axis=0))
        G = self.data.map_overlap(filters.gaussian, depth=deptharray, boundary='constant', sigma = sigma)
        self.feature_names.append('Gaussian_4D_Blur_'+f'{sigma:.1f}')
        self.calculated_features.append(G)
        self.Gaussian_4D_dict[sigma] = G
        
    def Gaussian_Blur_space(self, sigma):      
        deptharray = np.ones(self.data.ndim)+4*sigma
        deptharray[-1] = 0
        deptharray = tuple(np.min([deptharray, self.data.shape], axis=0))
        sigmas = np.ones(deptharray.shape)*sigma
        sigmas[-1] = 0
        G = self.data.map_overlap(filters.gaussian, depth=deptharray, boundary='constant', sigma = sigmas)
        self.feature_names.append('Gaussian_space_'+f'{sigma:.1f}')
        self.calculated_features.append(G)
        self.Gaussian_space_dict[sigma] = G

    def Gaussian_Blur_time(self, sigma):      
        deptharray = np.ones(self.data.ndim)+4*sigma
        deptharray[:-1] = 0
        deptharray = tuple(np.min([deptharray, self.data.shape], axis=0))
        sigmas = np.ones(deptharray.shape)*sigma
        sigmas[:-1] = 0
        G = self.data.map_overlap(filters.gaussian, depth=deptharray, boundary='constant', sigma = sigmas)
        self.feature_names.append('Gaussian_time_'+f'{sigma:.1f}')
        self.calculated_features.append(G)
        self.Gaussian_time_dict[sigma] = G
        
    def Gaussian_4D_stack(self):
        flag = True
        for sigma in self.sigmas:
            if np.abs(sigma-0)<0.1:
                if flag:
                    flag = False
                    self.Gaussian_4D_dict['original'] = self.data
                    self.feature_names.append('original')
            else:
                self.Gaussian_Blur_4D(sigma)
                
    def Gaussian_space_stack(self):
        flag = True
        for sigma in self.sigmas:
            if np.abs(sigma-0)<0.1:
                if flag:
                    flag = False
                    self.Gaussian_space_dict['original'] = self.data
            else:
                self.Gaussian_Blur_space(sigma)
                
    def Gaussian_time_stack(self):
        flag = True
        for sigma in self.sigmas:
            if np.abs(sigma-0)<0.1:
                if flag:
                    flag = False
                    self.Gaussian_time_dict['original'] = self.data
            else:
                self.Gaussian_Blur_time(sigma)
                
            
    def diff_Gaussian(self, mode):
        if mode == '4D':
            lookup_dict = self.Gaussian_4D_dict
        elif mode == 'space':
            lookup_dict = self.Gaussian_space_dict
        elif mode == 'time':
            lookup_dict = self.Gaussian_time_dict
            
        for comb in combinations(lookup_dict.keys(),2):
            DG = lookup_dict[comb[1]] - lookup_dict[comb[0]]
            name = ''.join(['diff_of_gauss_',mode,'_',comb[1],'_',comb[0]])
            self.calculated_features.append(DG)
            self.feature_names.append(name)
            
    def diff_to_first_and_last(self):
        options = self.Gaussian_space_dict.keys()
        for opt in options:
            DA = self.Gaussian_space_dict[opt]
            first = DA[...,0]
            last = DA[...,-1]
            DF = DA.copy()
            DL = DA.copy()
            for i in range(DA.shape[-1]):
                curr = DA[...,i]
                DF[...,i] = curr-first
                DL[...,i] = curr-last
            name1 = 'diff_to_first_'+opt
            name2 = 'diff_to_last_'+opt
            self.calculated_features.append(DF)
            self.calculated_features.append(DL)
            self.feature_names.append(name1)
            self.feature_names.append(name2)
            
            
    def Gradients(self):
        for key in self.Gaussian_dict:
            G = self.Gaussian_dict[key]
            gradients = dask.array.gradient(G)
            self.Gradient_dict[key] = gradients
            
    def Hessian(self):
        # TODO: add max of all dimensions
        for key in self.Gradient_dict.keys():
            axes = range(self.data.ndim)
            gradients = self.Gradient_dict[key]
            H_elems = [dask.array.gradient(gradients[ax0], axis=ax1) for ax0, ax1 in combinations_with_replacement(axes, 2)]
            
            gradnames = ['Gradient_sigma_'+key+'_'+str(ax0) for ax0 in range(axes)]
            elems = [(ax0,ax1) for ax0, ax1 in combinations_with_replacement(range(axes), 2)]
            hessnames = [''.join(['hessian_sigma_',key,'_',str(elm[0]),str(elm[1])]) for elm in elems ]
            
            self.feature_names = self.feature_names + gradnames + hessnames
            self.calculated_features = self.calculated_features+gradients+H_elems
    
    def rank_filter(self, option, sigma):
        # note: rank filters not yet available in CUCIM
        da = self.data
        if not np.abs(sigma-0)<1:
            if option == 'minimum':
                fun = filters.rank.minimum
            elif option == 'maximum':
                fun = filters.rank.maximum
            elif option == 'median':
                fun = filters.rank.median
            elif option == 'mean':
                fun = filters.rank.mean     

            fp = ball_4d(sigma)            
            deptharray = np.ones(da.ndim)+sigma
            deptharray = tuple(np.min([deptharray, da.shape], axis=0))
                     
            R = da.map_overlap(fun, depth=deptharray, footprint=fp)
            name = ''.join([option,'_',f'{sigma:.1f}'])
            self.calculated_features.append(R)
            self.feature_names.append(name)
    
    def dynamic_rank_filter(self, option, sigma):
        # TODO: add custom dynamic model, eg. sigmoid
        da = self.data
        if option == 'minimum':
            fun = filters.rank.minimum
        elif option == 'maximum':
            fun = filters.rank.maximum
        elif option == 'median':
            fun = filters.rank.median
        elif option == 'mean':
            fun = filters.rank.mean  
            
        fp_3D = ball(sigma)
        fp_4D = np.zeros(list(fp_3D.shape)+[2*self.sigma_t], dtype=int)
        fp_4D[fp_3D>0,:] = 1
        deptharray = np.ones(da.ndim)+sigma
        deptharray[-1] = self.sigma_t
        deptharray = tuple(np.min([deptharray, da.shape], axis=0))
        
        R = da.map_overlap(fun, depth=deptharray, footprint=fp_4D)
        name = ''.join([option,'_dynamic_',f'{sigma:.1f}'])
        self.calculated_features.append(R)
        self.feature_names.append(name)
                
            
    def rank_filter_stack(self):
        for option in self.considered_ranks:
            for sigma in self.sigmas:
                self.rank_filter(option, sigma)
                self.dynamic_rank_filter(option, sigma)
                
    
    # TODO: include feature selection either in compute (better) or save
    # TODO: maybe add purge function
    def prepare(self):
        self.Gaussian_4D_stack()
        # self.diff_Gaussian('4D')
        self.Gradients()
        self.Hessian()
        self.Gaussian_time_stack()
        self.diff_Gaussian('time')
        self.Gaussian_space_stack()
        self.diff_Gaussian('space')
        self.rank_filter_stack()
        self.prepared = True

        
    def compute(self):
        if not self.prepared:
            print('prepare first')
        else:
            for feat in self.calculated_features:
                # are multiplely used intermediate results persistent or recalculated, e.g. Gaussian Blur? what about spilling to disk?
                feat.compute()
            self.feature_stack = dask.array.stack(self.calculated_features, axis = 4)
            self.feature_stack.rechunk(self.outchunks)
        
    def store_xarray_nc(self, outpath = None):
        if outpath is None:
            outpath = self.outpath
        shp = self.feature_stack.shape
        # TODO: take coordinates from tomodata dataset
        self.result = xr.Dataset({'feature_stack': (['x','y','z','time', 'feature'], self.feature_stack)},
                                 coords = {'x': np.arange(shp[0]),
                                           'y': np.arange(shp[1]),
                                           'z': np.arange(shp[2]),
                                           'time': np.arange(shp[3]),
                                           'feature': self.feature_names}
                                 )
        self.result.to_netcdf4(outpath)
        
        
        




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
