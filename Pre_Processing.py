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


import numpy as np
from scipy.misc import imresize
from skimage.transform import resize
from skimage.exposure import equalize_adapthist, equalize_hist
import pandas as pd
import random
import SimpleITK as sitk
import scipy.io as scio
from skimage import measure

root_path='H://ATLAStoke//raw_data//'
T2_list=['031948','031950','031952','031954','031955','031957','031959','031961','031972']

def Coundt_file_number(input_path):
    FileNum=len([lists for lists in os.listdir(input_path)])
    return FileNum

def PreProcessing():
    #for siteidx in range(9):
        #site_path=root_path+'Site'+str(siteidx+1)
        #files = os.listdir(site_path)
        #print(files)
        #for file in files:
        #    input_path=site_path+'//'+file+'//t01//'

    for siteidx in range(1):
        site_path=root_path+'Site'+str(8)
        for file in T2_list:
            input_path=site_path+'//'+file+'//t02//'
            print(input_path)
            FileNum=Coundt_file_number(input_path)
            img_itk = sitk.ReadImage(input_path+file+'_t1w_deface_stx.nii.gz')
            img_array = sitk.GetArrayFromImage(img_itk)
            
            gs_itk = sitk.ReadImage(input_path+file+'_LesionSmooth_stx.nii.gz')
            gs_array = sitk.GetArrayFromImage(gs_itk)
            gs_array[gs_array>0.5]=1
            if FileNum>2:
                Extra_lesion=FileNum-2
                for i in range(Extra_lesion):
                    Extra_itk = sitk.ReadImage(input_path+file+'_LesionSmooth_'+str(i+1)+'_stx.nii.gz')
                    Extra_array = sitk.GetArrayFromImage(Extra_itk)
                    gs_array[Extra_array>0.5]=1

            itkimage = sitk.GetImageFromArray(img_array, isVector=False)
            itkimage.SetSpacing(img_itk.GetSpacing())
            itkimage.SetOrigin(img_itk.GetOrigin())
            sitk.WriteImage(itkimage, 'H:/ATLAStoke/PP_data/Image/'+ str(file) + 't02_IMGAG.nii', True)     

            itkimage = sitk.GetImageFromArray(gs_array, isVector=False)
            itkimage.SetSpacing(img_itk.GetSpacing())
            itkimage.SetOrigin(img_itk.GetOrigin())
            sitk.WriteImage(itkimage, 'H:/ATLAStoke/PP_data/Label/'+ str(file) + 't02_LABEL.nii', True)     

if __name__=='__main__':
    PreProcessing() 
  