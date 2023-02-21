# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 14:05:31 2023

@author: fische_r
"""

from ipywidgets import Image
from ipywidgets import ColorPicker, IntSlider, link, AppLayout, HBox
from ipycanvas import  hold_canvas,  MultiCanvas #RoughCanvas,Canvas,
import numpy as np
import matplotlib.pyplot as plt
import imageio

# TODO: maybe make a class

def on_mouse_down(x, y):
    global drawing
    global position
    global shape
    drawing = True
    position = (x, y)
    shape = [position]

def on_mouse_move(x, y):
    global drawing
    global position
    global shape
    if not drawing:
        return
    with hold_canvas():
        canvas.stroke_line(position[0], position[1], x, y)
        position = (x, y)
    shape.append(position)

def on_mouse_up(x, y):
    global drawing
    global positiondu
    global shape
    drawing = False
    with hold_canvas():
        canvas.stroke_line(position[0], position[1], x, y)
        canvas.fill_polygon(shape)
    shape = []
    
def display_feature(i, TS):
    print('selected '+TS.feature_names[i])
    im = TS.current_feat_stack[:,:,i]
    im8 = im-im.min()
    im8 = im8/im8.max()*255
    return im8

# alpha = 0.1

# zoom1 = (0,400)
# zoom2 = (500,1400)

zoom1 = (0, -1)
zoom2 = (0, -1)


def training_canvas(im8, TS, alpha=0.1, zoom1=zoom1, zoom2=zoom2):
    # im8 = TS.current_im8
    #trick: use gaussian_time_4_0 to label static phases ()
    # im8 = display_feature(-2, TS)
    print('original shape: ',im8.shape)
    im8_display = im8.copy()[zoom1[0]:zoom1[1], zoom2[0]:zoom2[1]]
    print('diyplay shape : ',im8_display.shape,' at: ', (zoom1[0], zoom2[0]))
    
    resultim = TS.current_result.copy()
    
    resultim_display = resultim[zoom1[0]:zoom1[1], zoom2[0]:zoom2[1]]
    
    
    width = im8_display.shape[1]
    height = im8_display.shape[0]
    Mcanvas = MultiCanvas(4, width=width, height=height)
    background = Mcanvas[0]
    resultdisplay = Mcanvas[2]
    truthdisplay = Mcanvas[1]
    canvas = Mcanvas[3]
    canvas.sync_image_data = True
    drawing = False
    position = None
    shape = []
    image_data = np.stack((im8_display, im8_display, im8_display), axis=2)
    background.put_image_data(image_data, 0, 0)
    slidealpha = IntSlider(description="Result overlay", value=0.15)
    resultdisplay.global_alpha = alpha #slidealpha.value
    if np.any(resultim>0):
        result_data = np.stack((255*(resultim_display==0), 255*(resultim_display==1), 255*(resultim_display==2)), axis=2)
    else:
        result_data = np.stack((0*resultim, 0*resultim, 0*resultim), axis=2)
    resultdisplay.put_image_data(result_data, 0, 0)
    canvas.on_mouse_down(on_mouse_down)
    canvas.on_mouse_move(on_mouse_move)
    canvas.on_mouse_up(on_mouse_up)
    picker = ColorPicker(description="Color:", value="#ff0000") #red
    # picker = ColorPicker(description="Color:", value="#0000ff") #blue
    # picker = ColorPicker(description="Color:", value="#00ff00") #green
    
    link((picker, "value"), (canvas, "stroke_style"))
    link((picker, "value"), (canvas, "fill_style"))
    link((slidealpha, "value"), (resultdisplay, "global_alpha"))
    
    HBox((Mcanvas,picker))
# HBox((Mcanvas,)) #picker


def inspect(TS):
    fig, axes = plt.subplots(1,6, figsize=(20,10))
    axes[0].imshow(TS.current_result, 'gray')
    axes[1].imshow(TS.current_im8, 'gray')
    
    # TS.current_diff_im = TS.current_im-TS.current_first_im
    # TS.current_diff_im = TS.current_diff_im/TS.current_diff_im.max()*255
    axes[2].imshow(-TS.current_diff_im)#,vmin=6e4)
    # axes[3].imshow(im8old, 'gray')
    axes[3].imshow(TS.current_first_im, 'gray')
    axes[4].imshow(TS.current_truth)
    if TS.current_computed:
        axes[5].imshow(TS.current_feat_stack[:,:,-2])
    else:
        axes[5].imshow(TS.current_result, 'gray')
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        
def update_training_set(TS):
    label_set = canvas.get_image_data()

    TS.current_truth[zoom1[0]:zoom1[1], zoom2[0]:zoom2[1]][label_set[:,:,0]>0] = 1
    TS.current_truth[zoom1[0]:zoom1[1], zoom2[0]:zoom2[1]][label_set[:,:,1]>0] = 2
    TS.current_truth[zoom1[0]:zoom1[1], zoom2[0]:zoom2[1]][label_set[:,:,2]>0] = 4
    
    imageio.imsave(TS.current_truthpath, TS.current_truth)