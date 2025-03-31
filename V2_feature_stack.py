# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

TODO: allow general deselection. does it it need a rerun of "prepare"?
TODO: allow deselection in derived features (gradients, hessian)

"""

import numpy as np
from scipy import ndimage
import dask_image.ndfilters

import dask
import dask.array

from itertools import combinations_with_replacement, combinations
import xarray as xr

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
                 mod_feat_dict = None,
                 sigma_t = 40,
                 sigma_0_derivatives = False,
                 take_means = True,
                 num_means = 7,
                 ignored_features = None
                 ):
        if sigma_for_ref not in sigmas:
            sigmas.append(sigma_for_ref)
            
        self.sigma_for_ref = sigma_for_ref
        self.sigmas = sigmas   
        
        #wheter considering means for first and last time step
        self.take_means = take_means
        self.num_means = num_means
        
        #wether to use the pixel coordinates as feature (not recommended, therefore no variable, can be set after initialization)
        self.loc_features = False
        # wether to calculate image derivates for raw image (rather useless because of high noise, set as above)
        self.sigma_0_derivatives = False
        
        # set up dicts and lists to feed dask graph
        # not sure if this is clever, does dask understand that this data is reused?
        self.Gaussian_4D_dict = {}
        self.Gaussian_space_dict = {}
        self.Gaussian_time_dict = {}
        self.Gradient_dict = {}
        self.calculated_features = []
        self.feature_names = []
        self.calculated_features_time_independent = []
        self.feature_names_time_independent = []
        self.sigma_t = sigma_t
        
        self.prepared = False
        self.computed = False
        self.verbose = False
    
## features
    def Gaussian_Blur_4D(self, sigma):
        G = dask_image.ndfilters.gaussian_filter(self.data, mode='nearest', sigma = sigma)
                
        self.feature_names.append('Gaussian_4D_Blur_'+f'{sigma:.1f}')
        self.calculated_features.append(G)
        self.Gaussian_4D_dict[f'{sigma:.1f}'] = G
        
    def Gaussian_Blur_space(self, sigma):      
        sigmas = np.ones(self.data.ndim)*sigma
        sigmas[-1] = 0   # potenital option: weak time sigma
        G = dask_image.ndfilters.gaussian_filter(self.data, mode='nearest', sigma = sigmas)
        
        self.feature_names.append('Gaussian_space_'+f'{sigma:.1f}')
        self.calculated_features.append(G)
        self.Gaussian_space_dict[f'{sigma:.1f}'] = G
        
    def Gaussian_Blur_time(self, sigma):      
        sigmas = np.ones(self.data.ndim)*sigma
        sigmas[:-1] = 0 # potenital option: weak space sigma
        G = dask_image.ndfilters.gaussian_filter(self.data, mode='nearest', sigma = sigmas)
        
        self.feature_names.append('Gaussian_time_'+f'{sigma:.1f}')
        self.calculated_features.append(G)
        self.Gaussian_time_dict[f'{sigma:.1f}'] = G
        
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
        if type(first) is not np.ndarray:  # TODO can this compute be avoided?
            first = first.compute()
            last = last.compute()

        if type(first) is np.ndarray: #TODO: this has to be possible differently, but I remember the pain without faking up a 4D array and explcit calculation
            # maybe now better in new dask ??? TODO: priority to test this
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
        # median sounds good on paper, but is an expensive calculation with little benefit over mean
    
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
        
# stack featrues
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
                
    def prepare(self):   
         
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
            self.stack_has_been_reduced = False
            self.feature_stack = dask.array.stack(self.calculated_features, axis = 4)
            self.feature_stack_time_independent = dask.array.stack(self.calculated_features_time_independent, axis=3)
            shp = self.feature_stack_time_independent.shape
            self.feature_stack_time_independent = self.feature_stack_time_independent.reshape(shp[0],shp[1],shp[2],1,shp[3])
            
        if self.ignored_features is not None:
            ids_time = np.ones(len(self.feature_names), dtype=bool)
            ids_independent = np.ones(len(self.feature_names_time_independent), dtype=bool)
            for i in range(len(ids_time)):
                if self.feature_names[i] in self.ignored_features:
                    ids_time[i] = False      
            for i in range(len(ids_independent)):
                if self.feature_names_time_independent[i] in self.ignored_features:
                    ids_independent[i] = False
            self.reduce_feature_stack(ids_time, ids_independent, verbose=self.verbose)
            
            
    def reduce_feature_stack(self, ids_time, ids_independent):
        # what about adding features? --> not implemented, better idea to start from the start and use saved label images
        # in jupyter notebook to add a step to reduce feature stack
        # TODO: option selection in GUI would be nice, maybe jupyter widget is a straightforward option
        """

        Parameters
        ----------
        ids_time : boolean array
            same length as self.calculated_features True for features to use
        ids_independent : boolean array
            same length as self.calculated_features True for features to use
        verbose : boolean, optional
            Print overview of used features. The default is False.

        Returns
        -------
        2 lazy dask arrays only using selected features
        2 lists of feature names

        """
        
        self.reduced_stack = self.feature_stack[...,ids_time]
        self.reduced_stack_time_independent = self.feature_stack[...,ids_independent]
        self.feature_names_reduced  = self.feature_names[ids_time]
        self.feature_names_reduced_time_independent = self.feature_names_time_independent[ids_independent]
        self.feature_selection = ids_time
        self.feature_selection_time_idependent = ids_independent
        self.stack_has_been_reduced = True
        if self.verbose:
            print('Considered dynamic features')
            for name in self.feature_names_reduced:
                print(name)
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('Considered static features')
            for name in self.feature_names_reduced_time_independent:
                print(name)
                
    def make_xarray(self, use_reduced=True):
        
        if use_reduced and self.stack_has_been_reduced:
            shp = self.reduced_stack.shape
            feature_names = self.feature_names_reduced
            feature_names_time_independent = self.feature_names_reduced_time_independent
            feature_stack = self.reduced_stack
            feature_stack_time_independent = self.feature_stack_reduced_time_independent
            print('using reduced feature stack')
        else:
            shp = self.feature_stack.shape
            feature_names = self.feature_names
            feature_names_time_independent = self.feature_names_time_independent
            feature_stack = self.feature_stack
            feature_stack_time_independent = self.feature_stack_time_independent
            print('using full feature stack because')
            if not use_reduced:
                print('- reduced stack not selected')
            if use_reduced and not self.stack_has_been_reduced:
                print('- reduced stack not calculated')
        
        
        coords = {'x': np.arange(shp[0]), 'y': np.arange(shp[1]), 'z': np.arange(shp[2]), 'time': np.arange(shp[3]), 'time_0': [0],
                  'feature': feature_names,
                  'feature_time_independent': feature_names_time_independent}

        self.feature_xarray = xr.Dataset({'feature_stack': (['x','y','z','time', 'feature'], feature_stack),
                                  'feature_stack_time_independent': (['x','y','z','time_0', 'feature_time_independent'], feature_stack_time_independent)},
                                     coords = coords
                                                 )     

        
                
        