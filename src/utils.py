import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use("ggplot")
# %matplotlib inline

from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split


#Helper Functions
def callbacks():
    callbacks = [
        EarlyStopping(monitor='loss',patience=10, verbose=1),
        ReduceLROnPlateau(monitor='loss',factor=0.1, patience=3, min_lr=0.00000001, verbose=1),
        ModelCheckpoint('saved_models/model3.h5', monitor='loss',verbose=1, save_best_only=True, save_weights_only=False)
    ]

def hex_to_rgb(value):
     value = value.lstrip('#')
 lv = len(value)
 return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def get_CDLRGB(cdlvalue):
 # print("get cdl:", cdlvalue)
 for x in range(len(colormap)):
   if(int(colormap[x][1])==cdlvalue):
    hex = colormap[x][0]
    rgb = hex_to_rgb(hex)
    return rgb

def view_mapped(temp, plot=True):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0,256):
        r[temp==l]=label_colours[l,0]
        g[temp==l]=label_colours[l,1]
        b[temp==l]=label_colours[l,2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = (r/255.0)#[:,:,0]
    rgb[:,:,1] = (g/255.0)#[:,:,1]
    rgb[:,:,2] = (b/255.0)#[:,:,2]
    if plot:
        plt.imshow(rgb)
        plt.show()
