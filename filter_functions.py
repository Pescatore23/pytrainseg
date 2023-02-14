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
from skimage import filters
from skimage.morphology import ball

import dask
import dask.array
# from dask.distributed import Client, LocalCluster

from itertools import combinations_with_replacement, combinations
import xarray as xr

# functions take chunked dask-array as input

# start-up cluster, TODO: option to connect to exisitng cluster
# TODO: class/function to boot up cluster with custom options, e.g. workers/threads
# esp. use SSD for memory spilling
# cluster = LocalCluster() 
# client = Client(cluster)
# print('Dashboard at '+cluster.dashboard_link)

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
                 sigmas = [0,2, 4],
                 feature_dict = default_feature_dict,
                 mod_feat_dict = None,
                 chunksize = '20 MiB', #try to align chunks to extend far in time --> should be useful for most filters, esp. the dynamic rank filters
                # auto chunking of the feature stack appears to be more useful , --> potentially remove the rechunking
                 outchunks = '300 MiB',
                 ranks = ['maximum', 'minimum', 'median'], #, 'mean'
                 sigma_t = 40
                 ):
        if mod_feat_dict is not None:
            for key in mod_feat_dict:
                feature_dict[key] = mod_feat_dict[key]
        
        self.data_path = data_path
        self.outpath = outpath
        self.sigmas = sigmas        
        self.feature_dict = feature_dict
        # TODO: allow option of custom shaped chunks
        self.chunks = chunksize
        self.outchunks = outchunks
        
        #wheter considering means for first and last time step
        self.take_means = True
        self.num_means = 7
        
        # not sure if this is clever, does dask understand that this data is reused?
        self.Gaussian_4D_dict = {}
        self.Gaussian_space_dict = {}
        self.Gaussian_time_dict = {}
        self.Gradient_dict = {}
        self.calculated_features = []
        self.feature_names = []
        self.considered_ranks = ranks
        self.sigma_t = sigma_t
        
        self.prepared = False
        self.computed = False
        
    
    # TODO: currently loads full dataset into memory, consider aligning desired chunk already for original dataset to avoid rechunking
    # .rechunk() causes problems downstream: "Assertion error" , WTF?!
    # if original data soe not fit in RAM, rechunk, store to disk and load again?
    def open_raw_data(self):
        data = xr.open_dataset(self.data_path)
        da = dask.array.from_array(data.tomo.data, chunks = self.chunks)
        
        self.original_dataset = data
        self.data = da
    
    def open_lazy_data(self, chunks=None):
        if chunks is None: 
            chunks = self.chunks
        data = xr.open_dataset(self.data_path, chunks = chunks)
        da = dask.array.from_array(data.tomo)
        # print('maybe re-introducing rechunking, but for large datasets auto might be ok')
        # print('smaller chunks might be better for slicewise training')
        # print('currently provided chunks are ignored')
        self.original_dataset = data#.rechunk(self.chunks)
        self.data = da
    
    def load_raw_data(self):
        data = xr.load_dataset(self.data_path)
        # da = dask.array.from_array(data.tomo).rechunk(chunks = self.chunks)
        
        da = dask.array.from_array(data.tomo.data, chunks = self.chunks)

        self.original_dataset = data
        self.data = da 
        
    def Gaussian_Blur_4D(self, sigma):
        # TODO: check on boundary mode
        deptharray = np.ones(self.data.ndim)+4*sigma
        deptharray = tuple(np.min([deptharray, self.data.shape], axis=0))
        G = self.data.map_overlap(filters.gaussian, depth=deptharray, boundary='nearest', sigma = sigma)
        self.feature_names.append('Gaussian_4D_Blur_'+f'{sigma:.1f}')
        self.calculated_features.append(G)
        self.Gaussian_4D_dict[f'{sigma:.1f}'] = G
        
    def Gaussian_Blur_space(self, sigma):      
        deptharray = np.ones(self.data.ndim)+4*sigma
        deptharray[-1] = 0
        sigmas = np.ones(deptharray.shape)*sigma
        deptharray = tuple(np.min([deptharray, self.data.shape], axis=0))
        
        sigmas[-1] = 0
        G = self.data.map_overlap(filters.gaussian, depth=deptharray, boundary='nearest', sigma = sigmas)
        self.feature_names.append('Gaussian_space_'+f'{sigma:.1f}')
        self.calculated_features.append(G)
        self.Gaussian_space_dict[f'{sigma:.1f}'] = G

    def Gaussian_Blur_time(self, sigma):      
        deptharray = np.ones(self.data.ndim)+4*sigma
        deptharray[:-1] = 0
        sigmas = np.ones(deptharray.shape)*sigma
        deptharray = tuple(np.min([deptharray, self.data.shape], axis=0))
        sigmas[:-1] = 0
        G = self.data.map_overlap(filters.gaussian, depth=deptharray, boundary='nearest', sigma = sigmas)
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
#         TODO: take temporal mean/median for first and last
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
        # ones = dask.array.ones(DA.shape, chunks=self.chunks)
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
            self.feature_names.append('first_')
            self.calculated_features.append(firsts)
            self.feature_names.append('last_')
            self.calculated_features.append(lasts)
        else:
            print('Diff first and last is an unexplainable pain in the ass, solve this at one point')
            
    def time_stats(self):
        DA = self.data
        mean = DA.mean(axis=-1)
        # median = dask.array.median(DA, axis = -1)
        std = DA.std(axis=-1)
        # skew = (mean - median)/std
        # minimum = DA.min(axis=-1)
        # maximum = DA.max(axis=-1)
        
        means = dask.array.stack([mean]*DA.shape[-1], axis=-1)
        stds = dask.array.stack([std]*DA.shape[-1], axis=-1)
        # medians = dask.array.stack([median]*DA.shape[-1], axis=-1)
        # skews = dask.array.stack([skew]*DA.shape[-1], axis=-1)
        # mins = dask.array.stack([minimum]*DA.shape[-1], axis=-1)
        # maxs = dask.array.stack([maximum]*DA.shape[-1], axis=-1)
        
        # maxmin = maxs - mins
        
        self.calculated_features.append(means)
        self.feature_names.append('full_temporal_mean_')
        self.calculated_features.append(stds)
        self.feature_names.append('full_temporal_std_')
        # self.calculated_features.append(medians)
        # self.feature_names.append('full_temporal_median_')
        # self.calculated_features.append(skews)
        # self.feature_names.append('full_temporal_skews_')
        # self.calculated_features.append(mins)
        # self.feature_names.append('full_temporal_mins_')
        # self.calculated_features.append(maxs)
        # self.feature_names.append('full_temporal_maxs_')
        # self.calculated_features.append(maxmin)
        # self.feature_names.append('full_temporal_maxmin_diff_')
            
    def Gradients(self):
        for key in self.Gaussian_4D_dict:
            G = self.Gaussian_4D_dict[key]
            gradients = dask.array.gradient(G)
            self.Gradient_dict[key] = gradients
            
    def Hessian(self):
        # TODO: add max of all dimensions
        for key in self.Gradient_dict.keys():
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
#         coords = dask.array.where(da)
       
#         for i in range(3):
#             loc = coords[i].compute().reshape(da.shape) #compute() to know shape of coords, maybe find another way
#             self.calculated_features.append(loc)
#             self.feature_names.append('loc_'+'xyz'[i])

        # the following looks less elegant, but seems more compatible with dask
        # TODO: check performance
        loc_x = dask.array.ones(da.shape)*dask.array.arange(da.shape[0])[:,None, None, None]
        self.calculated_features.append(loc_x)
        self.feature_names.append('loc_'+'x')
        
        loc_y = dask.array.ones(da.shape)*dask.array.arange(da.shape[1])[None,:, None, None]
        self.calculated_features.append(loc_y)
        self.feature_names.append('loc_'+'y')
        
        loc_z = dask.array.ones(da.shape)*dask.array.arange(da.shape[2])[None, None,:, None]
        self.calculated_features.append(loc_z)
        self.feature_names.append('loc_'+'z')

        
    
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
        
        self.diff_to_first_and_last(self.take_means, self.num_means) 
        self.Gaussian_4D_stack()
        self.diff_Gaussian('4D')
        self.Gradients()
        self.Hessian()
        self.Gaussian_time_stack()
        self.diff_Gaussian('time')
        self.Gaussian_space_stack()
        self.diff_Gaussian('space')
        self.pixel_coordinates()
        # self.rank_filter_stack() #you have to load the entire raw data set for the dynamic part of this filter --> not so good for many time steps
        self.time_stats() #does something similar like the dynamic rank filter, however only one pixel in space
        
        
        self.prepared = True

    
    def stack_features(self):
        if not self.prepared:
            print('prepare first')
        else:
            self.feature_stack = dask.array.stack(self.calculated_features, axis = 4)
            # TODO: rechunk?
    
    def compute(self):
        # self.feature_stack = self.feature_stack.compute()
        self.feature_stack = self.feature_stack.persist() #not sure, but apparently persist should be preferred
        self.computed = True
        
    def make_xarray_nc(self, outpath = None, store=False):
        if outpath is None:
            outpath = self.outpath
        shp = self.feature_stack.shape
        coords = {'x': np.arange(shp[0]), 'y': np.arange(shp[1]), 'z': np.arange(shp[2]), 'time': np.arange(shp[3]), 'feature': self.feature_names}
        if store:
            if self.computed:
                if not type(self.feature_stack) is np.ndarray:
                    self.feature_stack.rechunk(self.outchunks)
                
                #TODO avoid this explcit conversion. however seems necessary ?...
                # if type(self.feature_stack) is not np.ndarray: 
                    # self.feature_stack = self.feature_stack.compute()
                    
                self.result = xr.Dataset({'feature_stack': (['x','y','z','time', 'feature'], self.feature_stack)},
                         coords = coords
                         )
                self.result.to_netcdf(outpath)
            else:
                print('maybe you have to compute the stack first ... ?!')
                      
        else:
            self.result = xr.Dataset({'feature_stack': (['x','y','z','time', 'feature'], self.feature_stack)},
                                     coords = coords
                                                 )     
            