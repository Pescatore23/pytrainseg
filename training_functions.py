# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 16:04:34 2022

to be loaded in Jupyter


@author: fische_r
"""
import xarray as xr
import os
from skimage import io, exposure
import matplotlib.pyplot as plt
import numpy as np
import pickle

#the classifier
from sklearn.ensemble import RandomForestClassifier

#stuff for painting on the image
# from ipywidgets import Image
# from ipywidgets import ColorPicker, IntSlider, link, AppLayout, HBox
# from ipycanvas import  hold_canvas,  MultiCanvas #RoughCanvas,Canvas,

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

def training_set_per_image(label_name, trainingpath, feat_data, lazy = False):
    c1, p1, c2, p2 = extract_coords(label_name)
    # print(label_name)
    # print(c1, p1, c2, p2)
    truth = io.imread(os.path.join(trainingpath, label_name))
    
    # temporary workaround, make general
    if c1 == 'x' and c2 == 'time':
        feat_stack = feat_data['feature_stack'].sel(x = p1, time = p2).data
    elif c1 == 'x' and c2 == 'y':
        feat_stack = feat_data['feature_stack'].sel(x = p1, y = p2).data
    elif c1 == 'x' and c2 == 'z':
        feat_stack = feat_data['feature_stack'].sel(x = p1, z = p2).data
    elif c1 == 'y' and c2 == 'z':
        feat_stack = feat_data['feature_stack'].sel(y = p1, z = p2).data
    elif c1 == 'y' and c2 == 'time':
        feat_stack = feat_data['feature_stack'].sel(y = p1, time = p2).data
    elif c1 == 'z' and c2 == 'time':
        feat_stack = feat_data['feature_stack'].sel(z = p1, time = p2).data
    else:
        print('coordinates not found')
    if lazy:
        print('Need to actually calculate the features for each slice, seems inefficient')
    #   not sure how efficient this is
    #   multiple training slices might be faster with the chunks
    #   probably getting the feature stack at least as persist is better
        feat_stack = feat_stack.compute()
    else:
        if type(feat_stack) is not np.ndarray:
            feat_stack = feat_stack.compute()
    
    X, y = extract_training_data(truth, feat_stack)
    return X,y

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
            
    def open_feature_data(self):
        self.feat_data = xr.open_dataset(self.feature_path)
        self.feature_names = self.feat_data['feature'].data
    
    def import_feature_data(self, data):
        self.feat_data = data
        self.feature_names = self.feat_data['feature'].data
        self.lazy = False
    
    def import_lazy_feature_data(self, data, lazy = True):
        self.feat_data = data
        self.feature_names = self.feat_data['feature'].data
        self.lazy = lazy

    def suggest_training_set(self):
        dimensions = list(self.feat_data.coords.keys())[:-1]

        test_dims = np.random.choice(dimensions, 2, replace=False)
        p1 = np.random.choice(range(len(self.feat_data[test_dims[0]])))
        p2 = np.random.choice(range(len(self.feat_data[test_dims[1]])))
        
        print('You could try ',test_dims[0],'=',str(p1),' and ',test_dims[1],'=',str(p2))
        print('However, please sort it like the original '+''.join(dimensions))
        
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
            
            if self.lazy:
#                 get the reference images directly as numpy array
                im = im.compute()
                if imfirst is not None:
                    imfirst = imfirst.compute()
                #already start calculating the feature stack
                feat_stack.persist()
                self.current_computed = False
            else:
                self.current_computed = True
                
            if type(im) is not np.ndarray:
                im = im.compute()
            if imfirst is not None and type(imfirst) is not np.ndarray:
                imfirst = imfirst.compute()

            im8 = im-im.min()
            im8 = im8/im8.max()*255
            
            if imfirst is not None:
                diff = im-imfirst
                diff = diff/diff.max()*255
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
            self.current_feat_stack = feat_stack
            self.current_first_im = imfirst
            self.current_truth = truth
            self.current_result = resultim
            self.current_truthpath = truthpath
            self.current_slice_name = slice_name
            #TODO: maybe keep lazy computed feature stacks of older slices somewhere and purge if using up too much RAM
    
#     def interface(self, alpha=0.15):
#         # TODO: add ipycanvas stuff
#         im8 = self.current_im8
#         resultim = self.current_result
        
#         width = im8.shape[1]
#         height = im8.shape[0]

#         Mcanvas = MultiCanvas(4, width=width, height=height)
#         background = Mcanvas[0]
#         resultdisplay = Mcanvas[2]
#         truthdisplay = Mcanvas[1]
#         canvas = Mcanvas[3]
#         canvas.sync_image_data = True

#         drawing = False
#         position = None
#         shape = []

#         def on_mouse_down(x, y):
#             global drawing
#             global position
#             global shape

#             drawing = True
#             position = (x, y)
#             shape = [position]

#         def on_mouse_move(x, y):
#             global drawing
#             global position
#             global shape

#             if not drawing:
#                 return

#             with hold_canvas():
#                 canvas.stroke_line(position[0], position[1], x, y)

#                 position = (x, y)

#             shape.append(position)

#         def on_mouse_up(x, y):
#             global drawing
#             global positiondu
#             global shape

#             drawing = False

#             with hold_canvas():
#                 canvas.stroke_line(position[0], position[1], x, y)
#                 canvas.fill_polygon(shape)

#             shape = []

#         image_data = np.stack((im8, im8, im8), axis=2)
#         # image_data = np.stack((diff*2, diff*2, diff*2), axis=2)
#         background.put_image_data(image_data, 0, 0)

#         resultdisplay.global_alpha = alpha
        
#         if np.any(resultim>0):
#             result_data = np.stack((255*(resultim==0), 255*(resultim==1), 255*(resultim==2)), axis=2)
#         else:
#             result_data = np.stack((0*resultim, 0*resultim, 0*resultim), axis=2)
#         resultdisplay.put_image_data(result_data, 0, 0)

#         canvas.on_mouse_down(on_mouse_down)
#         canvas.on_mouse_move(on_mouse_move)
#         canvas.on_mouse_up(on_mouse_up)

#         # canvas.stroke_style = "#749cb8"
#         # canvas.global_alpha = 0.75

#         picker = ColorPicker(description="Color:", value="#ff0000")
#         slidealpha = IntSlider(description="Result overlay", value=0.15)

#         link((picker, "value"), (canvas, "stroke_style"))
#         link((picker, "value"), (canvas, "fill_style"))
#         # link((slidealpha, "value"), (resultdisplay, "global_alpha"))

#         return HBox((Mcanvas, picker, slidealpha))
#         #print('paint image with #ff0000 for air, #00ff00 for water and #0000ff for fiber')
        
#     def fetch_labels(self):
#         label_set = canvas.get_image_data()

#         self.current_truth[label_set[:,:,0]>0] = 1
#         self.current_truth[label_set[:,:,1]>0] = 2
#         self.current_truth[label_set[:,:,2]>0] = 4

#         imageio.imsave(self.current_truthpath, self.current_truth)
        
    def train_slice(self):
        #fetch variables
        im = self.current_im
        truth = self.current_truth
        training_dict = self.training_dict
        slice_name = self.current_slice_name
        feat_stack = self.current_feat_stack
        
        #re-consider these lines
        if self.lazy and not self.current_computed:
            print('now actually calculating the features')
            # self.current_feat_stack.rechunk('auto') #why rechunk 'auto' ?! if anything should be something small fot massive parallel
            feat_stack = feat_stack.compute() 
            self.current_computed = True
        if type(feat_stack) is not np.ndarray:
            feat_stack = feat_stack.compute()      

        #train
        # print('training ...')
        resultim, clf, training_dict = training_function(im, truth, feat_stack, training_dict, slice_name, self.clf_method)

        # update variables
        self.current_result = resultim
        self.training_dict = training_dict #this necessary ?
        self.clf = clf
        
    def plot_importance(self, figsize=(16,9)):
        plt.figure(figsize=figsize)
        plt.stem(self.feature_names, self.clf.feature_importances_,'x')
        plt.xticks(rotation=90)
        plt.ylabel('importance') 
        
    def pickle_classifier(self):
        pickle.dump(self.clf, open(os.path.join(self.training_path, 'classifier.p'),'wb'))
    
    def train(self):
        path = self.label_path
        feat_data = self.feat_data #probably requires computed feature data, added the flag below
        training_dict = {}
        labelnames = os.listdir(path)
        flag = True
        for label_name in labelnames:
            X, y = training_set_per_image(label_name, path, feat_data, self.lazy)
            training_dict[label_name] = X,y
            if flag:
                Xall = X
                yall = y
                flag = False
            else:
                Xall = np.concatenate([Xall,X])
                yall = np.concatenate([yall,y])

        clf =  self.clf_method
        clf.fit(Xall, yall)
        self.clf = clf
        self.training_dict = training_dict