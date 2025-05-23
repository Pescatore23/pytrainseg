# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:35:22 2022

@author: fische_r


class to create feature stack on 4D tomo data
transient dimension, e.g. time, should be 4th dimension of input data


TODO: add GPU support (CUDA, cupy, cucim)
TODO: store git commit sha

"""

# necessary packages for feature stack
import numpy as np
from scipy import ndimage
# from skimage import filters
import dask_image.ndfilters
from skimage.morphology import ball

import dask
import dask.array
# from dask.distributed import Client, LocalCluster

from itertools import combinations_with_replacement, combinations
import xarray as xr

# functions take chunked dask-array as input

# default_feature_dict = {'Gaussian': True, 
#                # 'Sobel': True,
#                'Hessian': True,
#                'Diff of Gaussians': True,
#                'maximum': True,
#                'minimum': True,
#                'median': True,
#                'extra_time_ranks': True,
#               }

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
                 sigmas = [0, 1, 3, 6],
                 sigma_for_ref = 2,
                 # feature_dict = default_feature_dict,
                 mod_feat_dict = None,
                 # chunksize = (64,64,64,1), #try to align chunks to extend far in time --> should be useful for most filters, esp. the dynamic rank filters
                # auto chunking of the feature stack appears to be more useful , --> potentially remove the rechunking
                 # outchunks = '300 MiB',
                 ranks = ['maximum', 'minimum', 'median'], #, 'mean'
                 sigma_t = 40,
                 sigma_0_derivatives = False
                 ):
        # if mod_feat_dict is not None:
        #     for key in mod_feat_dict:
        #         feature_dict[key] = mod_feat_dict[key]
        
        # this sigma is used for the diff of Gaussian time_stats
        if sigma_for_ref not in sigmas:
            sigmas.append(sigma_for_ref)
        
        self.sigma_for_ref = sigma_for_ref
        self.data_path = data_path
        self.outpath = outpath
        self.sigmas = sigmas        
        # self.feature_dict = feature_dict
        # TODO: allow option of custom shaped chunks
        # self.chunks = chunksize
        # self.outchunks = outchunks
        
        #wheter considering means for first and last time step
        self.take_means = True
        self.num_means = 7
        
        #wether to use the pixel coordinates as feature
        self.loc_features = False
        
        # wether to calculate image derivates for raw image (rather useless)
        self.sigma_0_derivatives = False
        
        # not sure if this is clever, does dask understand that this data is reused?
        self.Gaussian_4D_dict = {}
        self.Gaussian_space_dict = {}
        self.Gaussian_time_dict = {}
        self.Gradient_dict = {}
        self.calculated_features = []
        self.feature_names = []
        self.calculated_features_time_independent = []
        self.feature_names_time_independent = []
        self.considered_ranks = ranks
        self.sigma_t = sigma_t
        
        self.prepared = False
        self.computed = False
        
    
        
    def Gaussian_Blur_4D(self, sigma):
        # TODO: check on boundary mode
        # TODO: test dask-image Gaussian filter and use if it works
        # deptharray = np.ones(self.data.ndim)+4*sigma
        # deptharray = tuple(np.min([deptharray, self.data.shape], axis=0))
        # G = self.data.map_overlap(filters.gaussian, depth=deptharray, boundary='nearest', sigma = sigma)
        G = dask_image.ndfilters.gaussian_filter(self.data, mode='nearest', sigma = sigma)
                
        self.feature_names.append('Gaussian_4D_Blur_'+f'{sigma:.1f}')
        self.calculated_features.append(G)
        self.Gaussian_4D_dict[f'{sigma:.1f}'] = G
        
    def Gaussian_Blur_space(self, sigma):      
        # deptharray = np.ones(self.data.ndim)+4*sigma
        # deptharray[-1] = 0
        # deptharray = tuple(np.min([deptharray, self.data.shape], axis=0))
        sigmas = np.ones(self.data.ndim)*sigma
        sigmas[-1] = 0
        
        # G = self.data.map_overlap(filters.gaussian, depth=deptharray, boundary='nearest', sigma = sigmas)
        G = dask_image.ndfilters.gaussian_filter(self.data, mode='nearest', sigma = sigmas)
        
        self.feature_names.append('Gaussian_space_'+f'{sigma:.1f}')
        self.calculated_features.append(G)
        self.Gaussian_space_dict[f'{sigma:.1f}'] = G

    def Gaussian_Blur_time(self, sigma):      
        # deptharray = np.ones(self.data.ndim)+4*sigma
        # deptharray[:-1] = 0
        # deptharray = tuple(np.min([deptharray, self.data.shape], axis=0))
        
        sigmas = np.ones(self.data.ndim)*sigma
        sigmas[:-1] = 0
        # G = self.data.map_overlap(filters.gaussian, depth=deptharray, boundary='nearest', sigma = sigmas)
        G = dask_image.ndfilters.gaussian_filter(self.data, mode='nearest', sigma = sigmas)
        
        self.feature_names.append('Gaussian_time_'+f'{sigma:.1f}')
        self.calculated_features.append(G)
        self.Gaussian_time_dict[f'{sigma:.1f}'] = G
        
    def Gaussian_4D_stack(self):
        flag = True
        for sigma in self.sigmas:
            if np.abs(sigma-0)<0.1:
                if flag:
                    flag = False
                    # self.Gaussian_4D_dict['original'] = self.data
                    # self.calculated_features.append(self.data)
                    # self.feature_names.append('original')
                    sig = 0
                    self.Gaussian_Blur_4D(sig)
                    
            else:
                self.Gaussian_Blur_4D(sigma)
                
    def Gaussian_space_stack(self):
        flag = True
        for sigma in self.sigmas:
            if np.abs(sigma-0)<0.1:
                if flag:
                    flag = False
                    # self.Gaussian_space_dict['original'] = self.data
                    sig = 0
                    self.Gaussian_Blur_space(sig)
            else:
                self.Gaussian_Blur_space(sigma)
                
    def Gaussian_time_stack(self):
        flag = True
        for sigma in self.sigmas:
            if np.abs(sigma-0)<0.1:
                if flag:
                    flag = False
                    # self.Gaussian_time_dict['original'] = self.data
                    sig = 0
                    self.Gaussian_Blur_time(sig)
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
            G1 = lookup_dict[comb[1]]
            G0 = lookup_dict[comb[0]]
            # DG = lookup_dict[comb[1]] - lookup_dict[comb[0]]
            # DG = G1-G0
            DG = dask.array.subtract(G1,G0)
            name = ''.join(['diff_of_gauss_',mode,'_',comb[1],'_',comb[0]])
            self.calculated_features.append(DG)
            self.feature_names.append(name)

    def diff_to_first_and_last(self, take_mean, means):

        DA = self.data
        if take_mean:
            first = DA[...,:means].mean(axis=-1)
            last = DA[...,-means:].mean(axis=-1)
        else:
            first = DA[...,0]
            last = DA[...,-1]
        if type(first) is not np.ndarray:
            first = first.compute()
            last = last.compute()

        if type(first) is np.ndarray:
            firsts = dask.array.stack([first]*DA.shape[-1], axis=-1)
            lasts = dask.array.stack([last]*DA.shape[-1], axis=-1)
            firsts = firsts.rechunk(DA.chunksize)
            lasts = lasts.rechunk(DA.chunksize)
            DF = DA - firsts
            DL = DA - lasts
            self.calculated_features.append(DF)
            self.feature_names.append('diff_to_first_')
            self.calculated_features.append(DL)
            self.feature_names.append('diff_to_last_')
            
            self.feature_names_time_independent.append('first_')
            self.calculated_features_time_independent.append(first)
            self.feature_names_time_independent.append('last_')
            self.calculated_features_time_independent.append(last)
        else:
            print('Diff first and last is an unexplainable pain in the ass, solve this at one point')
            
    def time_stats(self):
        DA = self.data
        mean = DA.mean(axis=-1)
        minimum = DA.min(axis=-1)

        diff_min = DA - minimum[...,None]
        # TODO: consider diffs to gaussian filtered mimimum, and diffs of gaussians
        
        G = self.Gaussian_4D_dict[f'{self.sigma_for_ref:.1f}']
        Gmin = G.min(axis=-1)
        Gmindiff = G - Gmin[...,None]
        
        self.calculated_features.append(diff_min)
        self.feature_names.append('diff_to_min_')
        
        self.feature_names_time_independent.append('full_temp_mean_')
        self.calculated_features_time_independent.append(mean)
        self.feature_names_time_independent.append('full_temp_min_')
        self.calculated_features_time_independent.append(minimum)
        
        self.feature_names_time_independent.append(''.join(['full_temp_min_Gauss_',f'{self.sigma_for_ref:.1f}']))
        self.calculated_features_time_independent.append(Gmin)
        
        self.feature_names.append(''.join(['diff_temp_min_Gauss_',f'{self.sigma_for_ref:.1f}']))
        self.calculated_features.append(Gmindiff)

            
    def Gradients(self):
        for key in self.Gaussian_4D_dict:
            if key == '0.0' and not self.sigma_0_derivatives: continue
            G = self.Gaussian_4D_dict[key]
            gradients = dask.array.gradient(G)
            self.Gradient_dict[key] = gradients
            
    def Hessian(self):
        # TODO: add max of all dimensions
        for key in self.Gradient_dict.keys():
            if key == '0.0' and not self.sigma_0_derivatives: continue
            axes = range(self.data.ndim)
            gradients = self.Gradient_dict[key]
            H_elems = [dask.array.gradient(gradients[ax0], axis=ax1) for ax0, ax1 in combinations_with_replacement(axes, 2)]
            
            gradnames = ['Gradient_sigma_'+key+'_'+str(ax0) for ax0 in axes]
            elems = [(ax0,ax1) for ax0, ax1 in combinations_with_replacement(axes, 2)]
            hessnames = [''.join(['hessian_sigma_',key,'_',str(elm[0]),str(elm[1])]) for elm in elems ]
            
            self.feature_names = self.feature_names + gradnames + hessnames
            self.calculated_features = self.calculated_features+gradients+H_elems
    
    # def rank_filter(self, option, sigma):
    #     # note: rank filters not yet available in CUCIM and don't work with dask --> figure out why
    #     da = self.data
    #     if not np.abs(sigma-0)<1:
    #         if option == 'minimum':
    #             fun = filters.rank.minimum
    #         elif option == 'maximum':
    #             fun = filters.rank.maximum
    #         elif option == 'median':
    #             fun = filters.rank.median
    #         elif option == 'mean':
    #             fun = filters.rank.mean     

    #         fp = ball_4d(sigma)            
    #         deptharray = np.ones(da.ndim)+sigma
    #         deptharray = tuple(np.min([deptharray, da.shape], axis=0))
                     
    #         R = da.map_overlap(fun, depth=deptharray, footprint=fp)
    #         name = ''.join([option,'_',f'{sigma:.1f}'])
    #         self.calculated_features.append(R)
    #         self.feature_names.append(name)

    # def dynamic_rank_filter(self, option, sigma):
    #     # TODO: add custom dynamic model, eg. sigmoid
    #     da = self.data
    #     if option == 'minimum':
    #         fun = filters.rank.minimum
    #     elif option == 'maximum':
    #         fun = filters.rank.maximum
    #     elif option == 'median':
    #         fun = filters.rank.median
    #     elif option == 'mean':
    #         fun = filters.rank.mean  
            
    #     fp_3D = ball(sigma)
    #     fp_4D = np.zeros(list(fp_3D.shape)+[2*self.sigma_t], dtype=int)
    #     fp_4D[fp_3D>0,:] = 1
    #     deptharray = np.ones(da.ndim)+sigma
    #     deptharray[-1] = self.sigma_t
    #     deptharray = tuple(np.min([deptharray, da.shape], axis=0))
        
    #     R = da.map_overlap(fun, depth=deptharray, footprint=fp_4D)
    #     name = ''.join([option,'_dynamic_',f'{sigma:.1f}'])
    #     self.calculated_features.append(R)
    #     self.feature_names.append(name)
    
    def rank_like_filter(self, option, sigma):
        # note: rank filters not yet available in CUCIM
        da = self.data
        if not np.abs(sigma-0)<1:
            if option == 'minimum':
                fun = ndimage.minimum_filter
            elif option == 'maximum':
                fun = ndimage.maximum_filter
            elif option == 'median':
                fun = ndimage.median_filter
            # elif option == 'mean':
            #     fun = filters.rank.mean     

            fp = ball_4d(sigma)            
            deptharray = np.ones(da.ndim)+sigma
            deptharray = tuple(np.min([deptharray, da.shape], axis=0))
                     
            R = da.map_overlap(fun, depth=deptharray, footprint=fp)
            name = ''.join([option,'_',f'{sigma:.1f}'])
            self.calculated_features.append(R)
            self.feature_names.append(name)
    
    def dynamic_rank_like_filter(self, option, sigma):
        # TODO: add custom dynamic model, eg. sigmoid
        da = self.data
        if option == 'minimum':
            fun = ndimage.minimum_filter
        elif option == 'maximum':
            fun = ndimage.maximum_filter
        elif option == 'median':
            fun = ndimage.median_filter
        # elif option == 'mean':
        #     fun = filters.rank.mean  
            
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
                self.rank_like_filter(option, sigma)
                # self.dynamic_rank_like_filter(option, sigma)
                
    def pixel_coordinates(self):
        #create 3 arrays with the pixel coordinates
        da = self.data
        
        loc_x = dask.array.ones(da.shape[:-1])*dask.array.arange(da.shape[0])[:,None, None]
        self.feature_names_time_independent.append('loc_'+'x')
        self.calculated_features_time_independent.append(loc_x)
        
        loc_y = dask.array.ones(da.shape[:-1])*dask.array.arange(da.shape[1])[None,:, None]
        self.feature_names_time_independent.append('loc_'+'y')
        self.calculated_features_time_independent.append(loc_y)
        
        loc_z = dask.array.ones(da.shape[:-1])*dask.array.arange(da.shape[2])[None, None, :]
        self.feature_names_time_independent.append('loc_'+'z')
        self.calculated_features_time_independent.append(loc_z)

        
    
    # TODO: include feature selection either in compute (better) or save
    # TODO: maybe add purge function
    # TODO: maybe add iterative segmentation results, i.e. median filter of segmentation
    # TODO: don't create 4D features for time independent features (like coordinates, time stats) to save RAM
    def prepare(self):   
        self.Gaussian_4D_dict = {}
        self.Gaussian_space_dict = {}
        self.Gaussian_time_dict = {}
        self.Gradient_dict = {}
        self.calculated_features = []
        self.feature_names = []
        
        # self.diff_to_first_and_last(self.take_means, self.num_means) 
        self.Gaussian_4D_stack()
        self.diff_Gaussian('4D')
        self.Gradients()
        self.Hessian()
        self.Gaussian_time_stack()
        self.diff_Gaussian('time')
        self.Gaussian_space_stack()
        self.diff_Gaussian('space')
        # self.rank_filter_stack() #dask_image might provide rank-like filters soon you have to load the entire raw data set for the dynamic part of this filter --> not so good for many time steps
        self.time_stats() #does something similar like the dynamic rank filter, however only one pixel in space
        
        if self.loc_features:
            self.pixel_coordinates()
        #  #this feature is a double-edged sword, use with care!!
        
        self.prepared = True

    
    def stack_features(self):
        if not self.prepared:
            print('prepare first')
        else:
            self.feature_stack = dask.array.stack(self.calculated_features, axis = 4)
            self.feature_stack_time_independent = dask.array.stack(self.calculated_features_time_independent, axis=3)
            shp = self.feature_stack_time_independent.shape
            self.feature_stack_time_independent = self.feature_stack_time_independent.reshape(shp[0],shp[1],shp[2],1,shp[3])
            # TODO: rechunk?
    
    def compute(self):
        # self.feature_stack = self.feature_stack.compute()
        self.feature_stack = self.feature_stack.persist() #not sure, but apparently persist should be preferred
        self.computed = True
        
    def compute_time_independent_features(self):
        self.feature_stack_time_independent = self.feature_stack_time_independent.persist()
        
    def make_xarray_nc(self, outpath = None, store=False):
        if outpath is None:
            outpath = self.outpath
        shp = self.feature_stack.shape
        coords = {'x': np.arange(shp[0]), 'y': np.arange(shp[1]), 'z': np.arange(shp[2]), 'time': np.arange(shp[3]), 'time_0': [0],
                  'feature': self.feature_names,
                  'feature_time_independent': self.feature_names_time_independent}

        self.result = xr.Dataset({'feature_stack': (['x','y','z','time', 'feature'], self.feature_stack),
                                  'feature_stack_time_independent': (['x','y','z','time_0', 'feature_time_independent'], self.feature_stack_time_independent)},
                                     coords = coords
                                                 )     
            