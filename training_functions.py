# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 16:04:34 2022

to be loaded in Jupyter

TODO: properly include feature de-selection by reforming feature dict into xarray dataset
@author: fische_r
"""
# import xarray as xr
import os
from skimage import io, exposure
import matplotlib.pyplot as plt
import numpy as np
import dask 
import pickle
# from dask.distributed import wait

#the classifier
from sklearn.ensemble import RandomForestClassifier

default_classifier = RandomForestClassifier(n_estimators = 300, n_jobs=-1, random_state = 42, max_features=None) 

def extract_training_data(truth, feat_stack, ids = None):
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
    
    if ids is not None:
        X = X[:,ids]
    return X,y

def classify(X,y,im, feat_stack, clf):
   # TODO: allow choice and manipulation of ML method 
   clf.fit(X, y)
   num_feat = feat_stack.shape[-1]
   ypred = clf.predict(feat_stack.reshape(-1,num_feat))
   result = ypred.reshape(im.shape).astype(np.uint8)
   return result, clf

def training_function(im, truth, feat_stack, training_dict, slice_name, clf):
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
    result, clf = classify(Xt, yt, im, feat_stack, clf)
    
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
    
def extract_coords(labelname):
    parts = labelname.split('_')
    c1 = parts[2]
    p1 = int(parts[3])
    c2 = parts[4]
    p2 = int(parts[5])
    return c1, p1, c2, p2

def training_set_per_image(label_name, trainingpath, feat_data, client, lazy = False):
    c1, p1, c2, p2 = extract_coords(label_name)
    # print(label_name)
    # print(c1, p1, c2, p2)
    truth = io.imread(os.path.join(trainingpath, label_name))
    if np.any(truth>0):
        
        # temporary workaround, make general
        if c1 == 'x' and c2 == 'time':
            feat_stack = feat_data['feature_stack'].sel(x = p1, time = p2)
            feat_stack_t_idp = feat_data['feature_stack_time_independent'].sel(x = p1, time_0 = 0)
        elif c1 == 'x' and c2 == 'y':
            feat_stack = feat_data['feature_stack'].sel(x = p1, y = p2).data
            feat_stack_t_idp = feat_data['feature_stack_time_independent'].sel(x = p1, y = p2)
        elif c1 == 'x' and c2 == 'z':
            feat_stack = feat_data['feature_stack'].sel(x = p1, z = p2).data
            feat_stack_t_idp = feat_data['feature_stack_time_independent'].sel(x = p1, z = p2)
        elif c1 == 'y' and c2 == 'z':
            feat_stack = feat_data['feature_stack'].sel(y = p1, z = p2).data
            feat_stack_t_idp = feat_data['feature_stack_time_independent'].sel(y = p1, z = p2)
        elif c1 == 'y' and c2 == 'time':
            feat_stack = feat_data['feature_stack'].sel(y = p1, time = p2).data
            feat_stack_t_idp = feat_data['feature_stack_time_independent'].sel(y = p1, time_0 = 0)
        elif c1 == 'z' and c2 == 'time':
            feat_stack = feat_data['feature_stack'].sel(z = p1, time = p2).data
            feat_stack_t_idp = feat_data['feature_stack_time_independent'].sel(z = p1, time_0 = 0)
        else:
            print('coordinates not found')
        # if lazy:
        #     print('Need to actually calculate the features for each slice, seems inefficient')
        # #   not sure how efficient this is
        # #   multiple training slices might be faster with the chunks
        # #   probably getting the feature stack at least as persist is better
        #     feat_stack = feat_stack.compute()
        # else:
        if type(feat_stack) is not np.ndarray:
                fut = client.scatter(feat_stack)
                feat_stack = fut.result().compute().data
                del fut
        if type(feat_stack_t_idp) is not np.ndarray:
                fut2 = client.scatter(feat_stack_t_idp)
                feat_stack_t_idp = fut2.result().compute().data
                del fut2
                        
                
        feat_stack = np.concatenate([feat_stack, feat_stack_t_idp], axis = 2)
        
        X, y = extract_training_data(truth, feat_stack)
        return X,y, True
    
    else:
        return 'no labels', 'y', False
        print('label image is empty')

class train_segmentation:
    def __init__(self,
                 feature_path = None,
                 training_path = None,
                 clf_method = default_classifier
                 ):
        self.feature_path = feature_path
        self.training_path = training_path
        self.label_path = os.path.join(training_path, 'label_images')
        self.training_dict = {}
        self.clf_method = clf_method
                
        if not os.path.exists(self.label_path):
            os.mkdir(self.label_path)
            
        self.lazy = False #maybe this can be more elegant without flag
            
    # def open_feature_data(self):
    #     self.feat_data = xr.open_dataset(self.feature_path)
    #     self.feature_names = self.feat_data['feature'].data
    
    # def import_feature_data(self, data):
    #     self.feat_data = data
    #     self.feature_names = self.feat_data['feature'].data
    #     self.lazy = False
    
    def import_lazy_feature_data(self, data, rawdata, lazy = True):
        self.raw_data = rawdata
        self.feat_data = data
        # self.feat_data_tme_idp = data['feature_stack_time_independent']
        self.feature_names = self.feat_data['feature'].data
        self.feature_names_time_independent = self.feat_data['feature_time_independent'].data
        
        # self.combined_feature_names = list(self.feature_names) + list(self.feature_names_time_independent) #no idea why this is not working
        self.lazy = lazy

    def suggest_training_set(self):
        dimensions = list(self.feat_data.coords.keys())[:-1]

        test_dims = np.random.choice(dimensions, 2, replace=False)
        p1 = np.random.choice(range(len(self.feat_data[test_dims[0]])))
        p2 = np.random.choice(range(len(self.feat_data[test_dims[1]])))
        
        print('You could try ',test_dims[0],'=',str(p1),' and ',test_dims[1],'=',str(p2))
        print('However, please sort it like the original '+''.join(dimensions))
        
    def load_training_set(self, c1, p1, c2, p2):
        
        # data = self.feat_data['feature_stack']
        rawdata = self.raw_data['tomo']
        
        
        # this has to be possible in a more elegant way!
        if c1 == 'x':
            stage1 = rawdata.sel(x=p1)
            # stage1feat = data.sel(x=p1)
        elif c1 == 'y':
            stage1 = rawdata.sel(y=p1)
            # stage1feat = data.sel(y=p1)
        elif c1 == 'z':
            stage1 = rawdata.sel(z=p1)
            # stage1feat = data.sel(z=p1)
        elif c1 == 'time':
            print('time cannot be first coordinate')
        
        if not c1=='time':
            if c2 == 'x':
                im = stage1.sel( x = p2).data #feature = 'original',
                # feat_stack = stage1feat.sel(x = p2).data
                imfirst = None
            elif c2 == 'y':
                im = stage1.sel( y = p2).data
                # feat_stack = stage1feat.sel(y = p2).data
                imfirst = None
            elif c2 == 'z':
                im = stage1.sel( z = p2).data
                # feat_stack = stage1feat.sel(z = p2).data
                imfirst = None
            elif c2 == 'time':
                im = stage1.sel(time = p2).data
                # feat_stack = stage1feat.sel(time = p2).data
                imfirst = stage1.sel(time = 0).data
            
#             if self.lazy:
# #                 get the reference images directly as numpy array
#                 # im = im.compute()
#                 # if imfirst is not None:
#                     # imfirst = imfirst.compute()
#                 #already start calculating the feature stack
#                 # feat_stack.persist()
#                 self.current_computed = False
#             else:
#                 self.current_computed = True
                
            if type(im) is not np.ndarray:
                fut = self.client.scatter(im)
                im = fut.result().compute().data
                del fut
            if imfirst is not None and type(imfirst) is not np.ndarray:
                fut2 = self.client.scatter(imfirst)
                imfirst = fut2.result().compute().data
                del fut2
                
            im8 = im-im.min()
            im8 = im8/im8.max()*255
            
            if imfirst is not None:
                diff = im-imfirst
                # diff = diff/diff.max()*255
                self.current_diff_im = diff
            else:
                self.current_diff_im = None
            
            slice_name = ''.join([c1,'_',str(p1),'_',c2,'_',str(p2),'_'])
            truthpath = os.path.join(self.label_path, ''.join(['label_image_',slice_name,'.tif']))
            
            resultim = np.zeros(im.shape, dtype=np.uint8)
            if os.path.exists(truthpath):
                truth = io.imread(truthpath)
                print('existing label set loaded')
            else:
                truth = resultim.copy()
            
            self.current_coordinates = [c1,p1,c2,p2]
            self.current_im = im
            self.current_im8 = im8
            # self.current_feat_stack = feat_stack
            self.current_first_im = imfirst
            self.current_truth = truth
            self.current_result = resultim
            self.current_truthpath = truthpath
            self.current_slice_name = slice_name
            # currently not getting the feature data while loading slice, probably good idea
    
    
    def get_slice_feat_stack(self):
        
        feat_data = self.feat_data
        [c1,p1,c2,p2] = self.current_coordinates
        
        
        if c1 == 'x' and c2 == 'time':
            feat_stack = feat_data['feature_stack'].sel(x = p1, time = p2)
            feat_stack_t_idp = feat_data['feature_stack_time_independent'].sel(x = p1, time_0 = 0)
        elif c1 == 'x' and c2 == 'y':
            feat_stack = feat_data['feature_stack'].sel(x = p1, y = p2).data
            feat_stack_t_idp = feat_data['feature_stack_time_independent'].sel(x = p1, y = p2)
        elif c1 == 'x' and c2 == 'z':
            feat_stack = feat_data['feature_stack'].sel(x = p1, z = p2).data
            feat_stack_t_idp = feat_data['feature_stack_time_independent'].sel(x = p1, z = p2)
        elif c1 == 'y' and c2 == 'z':
            feat_stack = feat_data['feature_stack'].sel(y = p1, z = p2).data
            feat_stack_t_idp = feat_data['feature_stack_time_independent'].sel(y = p1, z = p2)
        elif c1 == 'y' and c2 == 'time':
            feat_stack = feat_data['feature_stack'].sel(y = p1, time = p2).data
            feat_stack_t_idp = feat_data['feature_stack_time_independent'].sel(y = p1, time_0 = 0)
        elif c1 == 'z' and c2 == 'time':
            feat_stack = feat_data['feature_stack'].sel(z = p1, time = p2).data
            feat_stack_t_idp = feat_data['feature_stack_time_independent'].sel(z = p1, time_0 = 0)
            
                   
        self.current_feat_stack = dask.array.concatenate([feat_stack, feat_stack_t_idp], axis = 2)
        
        if type(self.current_feat_stack) is not np.ndarray:
            self.current_computed = False
    
    def train_slice(self):
        #fetch variables
        im = self.current_im
        truth = self.current_truth
        training_dict = self.training_dict
        slice_name = self.current_slice_name
        
        
        feat_stack = self.current_feat_stack
               
        
        
        #re-consider these lines
        if self.lazy and not self.current_computed and type(feat_stack) is not np.ndarray:
            print('now actually calculating the features')
            # feat_stack = feat_stack.persist() #compute() persist may prevent an memory blow up https://stackoverflow.com/questions/73770527/dask-compute-uses-twice-the-expected-memory
            # wait(feat_stack) #if you use persist(), you have to wait for the calculation to finish before passing the feat stack to sklearn
            fut = self.client.scatter(feat_stack)
            feat_stack = fut.result()
            
            
            self.current_computed = True
        if type(feat_stack) is not np.ndarray:
            print('feat_stack is not a numpy array! check why')
            # feat_stack = feat_stack.compute()      
        
        self.current_feat_stack = feat_stack
        #train
        # print('training ...')
        # TODO: do I have to wait for the persist() to finish or does this work automatically ??
        resultim, clf, training_dict = training_function(im, truth, feat_stack, training_dict, slice_name, self.clf_method)

        # update variables
        
        self.current_result = resultim
        self.training_dict = training_dict #this necessary ? yes!
        self.clf = clf
        
    def plot_importance(self, figsize=(16,9)):
        plt.figure(figsize=figsize)
        plt.stem(self.feature_names, self.clf.feature_importances_,'x')
        plt.xticks(rotation=90)
        plt.ylabel('importance') 
        
    def pickle_training_dict(self):
        pickle.dump(self.training_dict, open(os.path.join(self.training_path, 'training_dict.p'),'wb'))
        
    def pickle_classifier(self):
        pickle.dump(self.clf, open(os.path.join(self.training_path, 'classifier.p'),'wb'))
    
    def train(self, clear_dict= False, redo=False):
        path = self.label_path
        feat_data = self.feat_data #probably requires computed feature data, added the flag below
        if clear_dict:
            self.training_dict = {}
        labelnames = os.listdir(path)
        if len(labelnames)>0:
            print('training with existing label images')
            flag = True
            for label_name in labelnames:
                if label_name in self.training_dict.keys() and not redo: 
                    print(label_name+' already done')
                    continue
                print(label_name)
                X, y, labelflag = training_set_per_image(label_name, path, feat_data, self.client, self.lazy)
                if labelflag:
                    self.training_dict[label_name] = X,y
                    if flag:
                        Xall = X
                        yall = y
                        flag = False
                    else:
                        Xall = np.concatenate([Xall,X])
                        yall = np.concatenate([yall,y])
            if flag:
                print('no label image actually contained labels')
            else:
                clf =  self.clf_method
                clf.fit(Xall, yall)
                self.clf = clf
                
        else:
            print('no label images found, start creating some')
        
    # def train_with_existing_label_set(self):
        #variant to above attempting to avoid redundant calculations, however, there is probably nromally not that much to gain
        # path = self.label_path
        # feat_data = self.feat_data #
        # training_dict = {}
        # labelnames = os.listdir(path)
        # TODO
    
    # def train_parallel(self):
    # #come up with a way to train() in parallel
    # # maybe with dask.delayed
    # # avoid redundant calculations, however, there is probably nromally not that much to gain
    #     path = self.label_path
    #     feat_data = self.feat_data
    #     training_dict = {}
    #     labelnames = os.listdir(path)
    #     XX = []
    #     yy = []
    #     for label_name in labelnames:
    #         X, y = dask.delayed(training_set_per_image)(label_name, path, feat_data.persist(), self.lazy)
    #         training_dict[label_name] = X,y
    #         XX.append(X)
    #         yy.append(y)
    #     Xall = np.concatenate(XX)
    #     yall = np.concatenate(yy)
    #     clf =  self.clf_method
    #     clf.fit(Xall, yall)
    #     self.clf = clf
    #     self.training_dict = training_dict