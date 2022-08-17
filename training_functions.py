# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 16:04:34 2022

to be loaded in Jupyter


@author: fische_r
"""

#reload after kernel reset
import xarray as xr
import os
from skimage import io, exposure
import matplotlib.pyplot as plt
import numpy as np

#the classifier
from sklearn.ensemble import RandomForestClassifier
#stuff for painting on the image
from ipywidgets import Image
from ipywidgets import ColorPicker, IntSlider, link, AppLayout, HBox
from ipycanvas import  hold_canvas,  MultiCanvas #RoughCanvas,Canvas,

default_classifier = RandomForestClassifier(n_estimators = 300, n_jobs=-1, random_state = 42, max_features=None) 

def extract_training_data(truth, feat_stack):
    #pixelwise training data
    phase1 = truth==1
    phase2 = truth==2
    phase3 = truth==4   
    X1 = feat_stack[phase1]
    y1 = np.zeros(X1.shape[0])
    X2 = feat_stack[phase2]
    y2 = np.ones(X2.shape[0])
    X3 = feat_stack[phase3]
    y3 = 2*np.ones(X3.shape[0])

    y = np.concatenate([y1,y2,y3])
    X = np.concatenate([X1,X2,X3])
    
    return X,y

def classify(X,y,im, feat_stack, clf = default_classifier):
   # TODO: allow choice and manipulation of ML method 
   clf.fit(X, y)
   num_feat = feat_stack.shape[-1]
   ypred = clf.predict(feat_stack.reshape(-1,num_feat))
   result = ypred.reshape(im.shape).astype(np.uint8)
   return result, clf

def training_function(im, truth, feat_stack, training_dict, slice_name):
    flag = False
    slices = list(training_dict.keys())
    if slice_name in slices: 
        slices.remove(slice_name)
    if len(slices)>0:
        flag = True
        Xall = training_dict[slices[0]][0]
        yall = training_dict[slices[0]][1]
        for i in range(1,len(slices)):
            Xall = np.concatenate([Xall, training_dict[slices[i]][0]])
            yall = np.concatenate([yall, training_dict[slices[i]][1]])
            
    X,y = extract_training_data(truth, feat_stack)
    
    print('training and classifying')
    
    if flag:
        Xt = np.concatenate([Xall,X])
        yt = np.concatenate([yall,y])
        Xall = None
        yall = None
    else:
        Xt = X
        yt = y  
    result, clf = classify(Xt, yt, im, feat_stack)
    
    # store training data of current slice in dict
    training_dict[slice_name] = (X,y)
    return result, clf, training_dict

def adjust_image_contrast(im, low, high):
    # careful, rescales in every case to 255
    im = exposure.rescale_intensity(im, (low,high))*255
    return im
    
    
def plot_im_histogram(im):
    hist = np.histogram(im, bins=100)
    plt.plot(hist[1][1:],hist[0])

class train_segmentation:
    def __init__(self,
                 feature_path = None,
                 training_path = None
                 ):
        self.feature_path = feature_path,
        self.training_path = training_path
        self.label_path = os.path.join(training_path, 'label_images')
        
        if not os.path.exists(self.label_path):
            os.mkdir(self.label_path)
            
    def open_feature_data(self):
        self.feat_data = xr.open_dataset(self.feature_path)
        self.feature_names = self.feat_data['feature'].data


    def suggest_training_set(self):
        dimensions = list(self.feat_data.coords.keys())[:-1]

        test_dims = np.random.choice(dimensions, 2, replace=False)
        p1 = np.random.choice(range(len(self.feat_data[test_dims[0]])))
        p2 = np.random.choice(range(len(self.feat_data[test_dims[1]])))
        
        print('You could try ',test_dims[0],'=',p1,' and ',test_dims[1],'=',p2)
        
    def load_training_set(self, c1, p1, c2, p2):
        
        data = self.feat_data['feature_stack']
        
        # this has be possible in a more elegant way!
        if c1 == 'x':
            stage1 = data.sel(x=p1)
        elif c1 == 'y':
            stage1 = data.sel(y=p1)
        elif c1 == 'z':
            stage1 = data.sel(z=p1)
        elif c1 == 'time':
            print('time cannot be first coordinate')
        
        if not c1=='time':
            if c2 == 'x':
                im = stage1.sel(feature = 'original', x = p2).data
                feat_stack = stage1.sel(x = p2).data
                imfirst = None
            elif c2 == 'y':
                im = stage1.sel(feature = 'original', y = p2).data
                feat_stack = stage1.sel(y = p2).data
                imfirst = None
            elif c2 == 'z':
                im = stage1.sel(feature = 'original', z = p2).data
                feat_stack = stage1.sel(z = p2).data
                imfirst = None
            elif c2 == 'time':
                im = stage1.sel(feature = 'original', time = p2).data
                feat_stack = stage1.sel(time = p2).data
                imfirst = stage1.sel(feature = 'original', time = 0).data
            
            im8 = im-im.min()
            im8 = im8/im8.max()*255
            
            if imfirst is not None:
                diff = im-imfirst
                diff = diff/diff.max()*255
                self.current_diff_im = diff
            else:
                self.current_diff_im = None
            
            slice_name = ''.join([c1,str(p1),c2,str(p2)])
            truthpath = os.path.join(self.trainingpath, ''.join(['label_image_',slice_name,'.tif']))
            
            resultim = np.zeros(im.shape, dtype=np.uint8)
            if os.path.exists(truthpath):
                truth = io.imread(truthpath)
                print('existing label set loaded')
            else:
                truth = resultim.copy()
            
            self.current_coordinates = [c1,p1,c2,p2]
            self.current_im = im
            self.current_im8 = im8
            self.current_feat_stack = feat_stack
            self.current_first_im = imfirst
            self.current_truth = truth
            self.current_result = resultim
    
    def interface(self):
        # TODO: add ipycanvas stuff
        
            
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    