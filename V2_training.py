# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 14:31:06 2025

@author: ROFISCHE
"""

import os
from skimage import io, exposure
import matplotlib.pyplot as plt
import numpy as np
import dask 
import pickle
# from dask.distributed import wait

#the classifier
from sklearn.ensemble import RandomForestClassifier
from dask.distributed import Client, LocalCluster

default_classifier = RandomForestClassifier(n_estimators = 300, n_jobs=-1, random_state = 42, max_features=None) 

def reboot_client(client, dashboard_address=':35000', memory_limit = '400GB', n_workers=2):
    client.shutdown()
    cluster = LocalCluster(dashboard_address=dashboard_address, memory_limit = memory_limit, n_workers=n_workers)
    client = Client(cluster)
    return client


def extract_training_data(truth, feat_stack, ids = None):
    #pixelwise training data
    phase1 = truth==1
    phase2 = truth==2
    phase3 = truth==4   
    phase4 = truth==3 #3 and 4 are flipped for lagacy reasons and existing training data
    X1 = feat_stack[phase1]
    y1 = np.zeros(X1.shape[0])
    X2 = feat_stack[phase2]
    y2 = np.ones(X2.shape[0])
    X3 = feat_stack[phase3]
    y3 = 2*np.ones(X3.shape[0])
    X4 = feat_stack[phase4]
    y4 = 3*np.ones(X4.shape[0])

    y = np.concatenate([y1,y2,y3,y4])
    X = np.concatenate([X1,X2,X3,X4])
    
    if ids is not None:
        X = X[:,ids]   # ids is a binary list where True entries indicate the features to be considered, use in conjuction with feature name list. TODO: keep coordinated with adaptive feature creation
    return X,y

def extract_coords(labelname):
    parts = labelname.split('_')
    c1 = parts[2]
    p1 = int(parts[3])
    c2 = parts[4]
    p2 = int(parts[5])
    return c1, p1, c2, p2

def classify(X,y,feat_stack, clf):
   # TODO: allow choice and manipulation of ML method, just feat a different clf
   clf.fit(X, y)
   shp = feat_stack.shape
   num_feat = shp[-1]
   ypred = clf.predict(feat_stack.reshape(-1,num_feat))
   result = ypred.reshape(shp[:-1]).astype(np.uint8)
   return result, clf

def training_function(truth, feat_stack, training_dict, slice_name, clf):
    flag = False
    slices = list(training_dict.keys())
    if slice_name in slices: 
        slices.remove(slice_name)
    if len(slices)>0:
        flag = True
        Xall = training_dict[slices[0]][0]
        yall = training_dict[slices[0]][1]
        for i in range(1,len(slices)): #why was there 1, in range ? because first initiates the Xall, np.stack could be an alternative way
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
    result, clf = classify(Xt, yt, feat_stack, clf)
    
    # store training data of current slice in dict
    training_dict[slice_name] = (X,y)
    return result, clf, training_dict

class training:
    def __init__(self,
                 training_path = None,
                 clf_method = default_classifier):
        if training_path is None:
            print('no training path given, re-init with setting training_path')
        self.label_path = os.path.join(training_path, 'label_images')
        self.training_dict = {}
        self.clf_method = clf_method
        if not os.path.exists(self.label_path):
            os.mkdir(self.label_path)        
        existing_sets = os.listdir(self.label_path)
        existing_sets.sort()
        if len(existing_sets)>0:
            print('There are existing training sets, run .train() if you want to use them:')
            for training_set in existing_sets:
                print(training_set)
            
    def suggest_training_set(self):
        dimensions = list(self.feat_data.coords.keys())[:3]
        timesteps = self.feat_data.time.data
        test_dim = np.random.choice(dimensions)
        p1 = np.random.choice(range(len(self.feat_data[test_dim])))
        ts = np.random.choice(timesteps)
        print('You could try ',test_dim,'=',str(p1),' at time step ',str(ts))
        
    def load_training_set(self, c1, p1, c2, p2):
        
        # select correct feature stack for slice
        feat_data = self.feat_data
        if c1 == 'x':
            feat_stack = feat_data['feature_stack'].sel(x = p1, time = p2)
            feat_stack_t_idp = feat_data['feature_stack_time_independent'].sel(x = p1, time_0 = 0)
        if c1 == 'y':
            feat_stack = feat_data['feature_stack'].sel(y = p1, time = p2)#.data
            feat_stack_t_idp = feat_data['feature_stack_time_independent'].sel(y = p1, time_0 = 0)
        if c1 == 'z':
            feat_stack = feat_data['feature_stack'].sel(z = p1, time = p2)#.data
            feat_stack_t_idp = feat_data['feature_stack_time_independent'].sel(z = p1, time_0 = 0)
            
        self.current_coordinates = [c1,p1,c2,p2]
        self.current_feat_stack = feat_stack
        self.feat_stack_t_idp = feat_stack_t_idp
        self.current_result = np.zeros((feat_stack.shape[0],feat_stack.shape[1]), dtype=np.uint8)
        
        # select the raw image as default and calculate it
        im = feat_stack[:,:,0]
        if type(im) is not np.ndarray:
            fut = self.client.scatter(im)
            fut = fut.result()
            fut = fut.compute()
            im = fut.data
            self.client.restart(wait_for_workers=False)
            if not len(self.client.cluster.workers)>1:
                    self.client = reboot_client(self.client, memory_limit=self.memlim, n_workers=self.n_workers)
        
        self.current_im = im
        
        # make a scaled 8bit image for display on the canvas
        im8 = im-im.min()
        im8 = im8/im8.max()*255
        self.current_im8 = im8
        
        # check if a correpsonding label exists and load it
        slice_name = ''.join([c1,'_',str(p1),'_',c2,'_',str(p2),'_'])
        truthpath = os.path.join(self.label_path, ''.join(['label_image_',slice_name,'.tif']))
        resultim = np.zeros(im.shape, dtype=np.uint8)
        if os.path.exists(truthpath):
            truth = io.imread(truthpath)
            print('existing label set loaded')
        else:
            truth = resultim.copy()
        self.current_truth = truth
        self.current_truthpath = truthpath
        self.current_slice_name = slice_name
        
    def train_slice(self):
        training_dict = self.training_dict
        slice_name = self.current_slice_name
        feat_stack = self.current_feat_stack_full
        truth = self.current_truth
        
        if type(feat_stack) is not np.ndarray:
            print('feat_stack is not a numpy array! check why')
            
        resultim, clf, training_dict = training_function(truth, feat_stack, training_dict, slice_name, self.clf_method)
        self.training_dict = training_dict #this necessary ? yes!
        self.clf = clf
        self.current_result = resultim
        
    def training_set_per_image(self, label_name, trainingpath, feat_data, lazy = False):
        c1, p1, c2, p2 = extract_coords(label_name)
        # print(label_name)
        # print(c1, p1, c2, p2)
        truth = io.imread(os.path.join(trainingpath, label_name))
        if np.any(truth>0):
            
            # temporary workaround, make general
            feat_data = self.feat_data
            if c1 == 'x':
                feat_stack = feat_data['feature_stack'].sel(x = p1, time = p2)
                feat_stack_t_idp = feat_data['feature_stack_time_independent'].sel(x = p1, time_0 = 0)
            if c1 == 'y':
                feat_stack = feat_data['feature_stack'].sel(y = p1, time = p2)#.data
                feat_stack_t_idp = feat_data['feature_stack_time_independent'].sel(y = p1, time_0 = 0)
            if c1 == 'z':
                feat_stack = feat_data['feature_stack'].sel(z = p1, time = p2)#.data
                feat_stack_t_idp = feat_data['feature_stack_time_independent'].sel(z = p1, time_0 = 0)

            if type(feat_stack) is not np.ndarray:
                    fut = self.client.scatter(feat_stack)
                    fut = fut.result()
                    fut = fut.compute()
                    feat_stack = fut.data
                    try:
                        self.client.restart()
                    except:
#            if not len(self.client.cluster.workers)>1:
     	                self.client = reboot_client(self.client, memory_limit=self.memlim, n_workers=self.n_workers)
                    # TODO client reboot if workers can't return
            if type(feat_stack_t_idp) is not np.ndarray:
                    fut = self.client.scatter(feat_stack_t_idp)
                    fut = fut.result()
                    fut = fut.compute()
                    feat_stack_t_idp = fut.data
                    try:
                        self.client.restart()
#            if not len(self.client.cluster.workers)>1:
                    except:
                        self.client = reboot_client(self.client, memory_limit=self.memlim, n_workers=self.n_workers)
                    
            feat_stack = np.concatenate([feat_stack, feat_stack_t_idp], axis = 2)
            
            X, y = extract_training_data(truth, feat_stack)
            return X,y
        
        else:
            return 'no labels', 'y', False
            print('label image is empty')
        
    def train(self, clear_dict= False, redo=False, first_set=0):
        path = self.label_path
        feat_data = self.feat_data 
        if clear_dict:
            self.training_dict = {}
        labelnames = os.listdir(path)
        if len(labelnames)>0:
            print('training with existing label images')
            flag = True
            for label_name in labelnames[first_set:]:
                if label_name in self.training_dict.keys() and not redo: 
                    print(label_name+' already done')
                    continue
                print(label_name)
                X, y = self.training_set_per_image(label_name, path, feat_data)
                self.training_dict[label_name] = X,y
                if flag:
                    Xall = X
                    yall = y
                    flag = False
                else:
                    Xall = np.concatenate([Xall,X])
                    yall = np.concatenate([yall,y])
            if flag:
                print('no label image actually contained labels, no classifier trained')
            else:
                clf =  self.clf_method
                clf.fit(Xall, yall)
                self.clf = clf  
        else:
            print('no label images found, start creating some')


class display:
    def adjust_image_contrast(im, low, high):
        # careful, rescales in every case to 255
        im = exposure.rescale_intensity(im, (low,high))*255
        return im  
    
    def plot_im_histogram(im):
        hist = np.histogram(im, bins=100)
        plt.plot(hist[1][1:],hist[0])
        
