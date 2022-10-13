"""
TODO: stor git_sha
"""


import xarray as xr
import pickle
import os
import numpy as np


class segmentation:
    def __init__(self,
                 feature_path = None,
                 classifier_path = None,
                 training_path = None
                ):
#         TODO: get these paths from training class
        self.feature_path = feature_path
        self.clf_path = classifier_path
        self.training_path = training_path
        
    def import_classifier(self, clf):
        self.clf = clf
        
    def load_classifier(self):
        self.clf = pickle.load(open(self.clf_path, 'rb'))
        
    def open_feature_data(self):
        self.feat_data = xr.open_dataset(self.feature_path)
        self.feature_names = self.feat_data['feature'].data
        self.lazy = False
        
    def import_feature_data(self, data):
        self.feat_data = data
        self.feature_names = self.feat_data['feature'].data
        self.lazy = False

    def import_lazy_feature_data(self, data):
        self.feat_data = data
        self.feature_names = self.feat_data['feature'].data
        self.lazy = True
    
    def classify_all(self):
#         TODO: streamline classifier and feature calculation. maybe integrate both within dask
#               especially if original and segmented dataset don't fit in RAM
        feat_stack = self.feat_data['feature_stack']
        num_feat = feat_stack.shape[-1]
        clf = self.clf
        if not self.lazy:
            print('classifying ...')
            # result = clf.predict(feat_stack.reshape(-1,num_feat))
            result = clf.predict(feat_stack.data.reshape(-1,num_feat))
        else:
            print('calculate feature stack and then classify. might take a while ... ')
            result = clf.predict(feat_stack.data.reshape(-1,num_feat))
        result = result.reshape(feat_stack[...,0].shape).astype(np.uint8)
        self.segmented_data = result
    
    def store_segmented_data(self):
        path = os.path.join(self.training_path, 'segmented.nc')
        
        #TODO: propagate labels from raw data
        #TODO: if self.segmented_data is a dask array, rechunk for saving
        shp = self.segmented_data.shape
        data = xr.Dataset({'segmented': (['x','y','z','time'], self.segmented_data)},
                                       coords = {'x': np.arange(shp[0]),
                                       'y': np.arange(shp[1]),
                                       'z': np.arange(shp[2]),
                                       'time': np.arange(shp[3]),
                                       'feature': self.feature_names}
                             )
        data.to_netcdf(path)