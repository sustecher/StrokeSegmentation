#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 17:18:38 2017

@author: Inom Mirzaev
"""

from __future__ import division, print_function
from collections import defaultdict
import os, pickle, sys
import shutil
from functools import partial

import cv2
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from scipy.misc import imresize
from skimage.transform import resize
from skimage.exposure import equalize_adapthist, equalize_hist
import pandas as pd
import random
import SimpleITK as sitk
import scipy.io as scio

#from model_Unet import *
#from model_Xnet import *
#from model_UnetPP import *
from model_Unet_pading import *
img_gen = ImageDataGenerator()

from metrics import *
from data import *
from loss import get_loss, dice

data_path = 'H:\\ATLAStoke\\PP_data\\'
code_path = 'H:\\ATLAStoke\\code\\'
current_fold=0
FD=5
nb_epoch=50
batch_size=8
AUG=0

Model_name='Unet_Whole_M3'
#Model_name='Unet_SingleChannel'
#Model_name='Unet_Whole'

def augmentation(x_0, x_1, y):
    theta = (np.random.uniform(-15, 15) * np.pi) / 180.
    shear = np.random.uniform(-.1, .1)
    zx, zy = np.random.uniform(.9, 1.1, 2)
    transform_parameters = {'theta': theta,
                                'zx': zx,
                                'zy': zy,
                                'shear': shear}
    x_0 = img_gen.apply_transform(x_0,transform_parameters)
    x_1 = img_gen.apply_transform(x_1, transform_parameters)
    y = img_gen.apply_transform(y, transform_parameters)
    return x_0, x_1, y

def load_data(plane,current_fold):
    X_train = np.load(code_path+'NPY_data\\'+plane+'_X_FD'+str(current_fold)+'.npy')
    y_train = np.load(code_path+'NPY_data\\'+plane+'_y_FD'+str(current_fold)+'.npy')
    M_train = np.load(code_path+'NPY_data\\'+plane+'_M_FD'+str(current_fold)+'.npy')

    return X_train, y_train, M_train

def get_model(img_rows, img_cols,current_fold):
    model = UNet((img_rows, img_cols, 2))
    #model = UNet((img_rows, img_cols, 1))
    model.summary()
    ver = Model_name+'_FD%s_EP%s.csv' % (current_fold, nb_epoch)
    csv_logger = CSVLogger(code_path + 'logs\\' + ver)
    model.compile(optimizer=Adam(lr=2e-4), loss=get_loss, metrics=[dice])
    return model, csv_logger

def train_start(regenerate=False,plane='Trans',current_fold=0):
    if regenerate:
        data_to_array_with_MASK(data_path,code_path,current_fold,FD)
    
    X_train, y_train, M_train = load_data(plane,current_fold)
    print(np.shape(X_train))
    #X_train = X_train*M_train
    img_rows = X_train.shape[1]
    img_cols = X_train.shape[2]

    if AUG==1:
        print('Data augmentation was used')
        augtimes=1
        X_NoAug = X_train
        M_NoAug = M_train
        y_NoAug = y_train
        # Repeat data to X times
        while augtimes < 2:
            X_aug = np.zeros(X_NoAug.shape, dtype=np.float32)
            M_aug = np.zeros(X_NoAug.shape, dtype=np.float32)
            y_aug = np.zeros(y_NoAug.shape, dtype=np.float32)
            for i in range(X_NoAug.shape[0]):
                X_aug[i,:,:],M_aug[i,:,:],y_aug[i,:,:] = augmentation(X_NoAug[i,:,:],M_NoAug[i,:,:],y_NoAug[i,:,:])
                #print(np.shape(X_aug))
            X_train = np.concatenate((X_train, X_aug), axis=0)
            M_train = np.concatenate((M_train, M_aug), axis=0)
            y_train = np.concatenate((y_train, y_aug), axis=0)
            augtimes+=1
            print(augtimes)
    else:
        print('Data augmentation was NOT used')
        X_train=X_train
        M_train=M_train
        y_train=y_train
    
    X_train = np.concatenate([X_train,M_train],axis=3)
    
    print('Validation was used')
    print(np.shape(X_train))
    sample_num=len(X_train)
    list_all = set(range(sample_num))
    train_list = random.sample(list_all,int(sample_num*0.8))
    val_list = list( list_all - set(train_list))
 
    X_train_train = X_train[train_list,::]
    y_train_train = y_train[train_list, ::]
    X_train_val = X_train[val_list,::]
    y_train_val = y_train[val_list, ::]

    model,csv_logger = get_model(img_rows, img_cols,current_fold)
    model_checkpoint = ModelCheckpoint(code_path + '/models/'+Model_name+plane+'_FD' + str(current_fold) + '.h5',
                                        monitor='val_loss', save_best_only=True)

    history = model.fit(X_train_train, y_train_train,
                        validation_data=(X_train_val, y_train_val),
                        batch_size=batch_size,
                        epochs=nb_epoch,
                        verbose=1,
                        shuffle=True,
                        callbacks=[model_checkpoint, csv_logger])
    


if __name__=='__main__':

    import time

    start = time.time()

    #train_start(regenerate=False,current_fold=0)
    #train_start(regenerate=False,plane='Trans',current_fold=0)
    #train_start(regenerate=False,plane='Coron',current_fold=0)
    train_start(regenerate=False,plane='Sagit',current_fold=0)

    #train_start(regenerate=True,plane='Trans',current_fold=1)
    #train_start(regenerate=False,plane='Coron',current_fold=1)
    train_start(regenerate=False,plane='Sagit',current_fold=1)

    #train_start(regenerate=False,plane='Trans',current_fold=2)
    #train_start(regenerate=False,plane='Coron',current_fold=2)
    train_start(regenerate=False,plane='Sagit',current_fold=2)

    #train_start(regenerate=True,plane='Trans',current_fold=3)
    #train_start(regenerate=False,plane='Coron',current_fold=3)
    train_start(regenerate=False,plane='Sagit',current_fold=3)

    #train_start(regenerate=False,plane='Trans',current_fold=4)
    #train_start(regenerate=False,plane='Coron',current_fold=4)
    train_start(regenerate=False,plane='Sagit',current_fold=4)


    end = time.time()

    print('Elapsed time:', round((end-start)/60, 2 ) )
