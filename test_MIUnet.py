#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 2017
@author: Inom Mirzaev
"""

from __future__ import division, print_function
import numpy as np
import os,  sys
import keras
from keras.optimizers import Adam
import SimpleITK as sitk
from skimage.measure import find_contours

import xlwt
import scipy.io as scio
#from model_Unet import *
#from model_Unet_pading import *
from model_deeplabv3 import *

from metrics import *
from data import *
from loss import get_loss, dice

data_path = 'F:\\Stroke_7Label\\PP_data\\'
code_path = 'F:\\Stroke_7Label\\'
FD=5
nb_epoch=200
batch_size=8
AUG=0

#Model_name='Unet_Whole_M7'
#Model_name='DeeplabV3P_MI'
#Model_name='UNet_MI_Affine'
Model_name='DeeplabV3P_MI_Affine'

def get_model(img_rows, img_cols,plane,current_fold):
    #model =  UNet((img_rows, img_cols, 2))
    model = Deeplabv3(input_shape=(img_rows, img_cols, 2),classes=1, backbone='xception', weights=None,activation='sigmoid')
    model.load_weights(code_path+'\\models\\'+ Model_name+plane+'_FD' + str(current_fold) + '.h5')
    model.compile(optimizer=Adam(), loss=get_loss, metrics=[dice])
    return model
# im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')im = np.array(Image.fromarray(im).resize((h, int(w * aspect_ratio))))

def check_predictions_Trans(Len2,Len3,current_fold):
    model_Trans = get_model(Len2, Len3,'Trans',current_fold)


    vol_scores_Trans = []

    row_num=0

    imlb_list = os.listdir(data_path+'Image')
    imlb_list.sort()
    #print(imlb_list)
    num_patient_train=len(imlb_list)
    #num_patient_train=3
    #print(num_patient_train)
    
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('mysheet', cell_overwrite_ok=True)
    sheet.write(row_num, 0, 'CaseName')
    sheet.write(row_num, 1, 'Trans')
   
    for l in range(num_patient_train):
        if l % FD ==current_fold:
            file_im=imlb_list[l]
            idx=file_im[0:file_im.find('_')]
            print(idx)
            image_path =data_path+'Image\\'+file_im
            label_path =data_path+'Label\\'+idx+'_LABEL.nii'
            #Mask_path =data_path+'Mask\\'+idx+'_REG.nii'
            Mask_path =data_path+'AffinedResults\\'+idx+'_IMGAG-label.nii'


            image_itk = sitk.ReadImage(image_path)
            image_array  = sitk.GetArrayFromImage(image_itk)
            Mask_array = itk_to_array(Mask_path)

            pred_Pading_Trans = np.zeros(image_array.shape)
           
            ## Trans
            Crop_image=image_array[:,5:229,3:195]
            Crop_Mask=Mask_array[:,5:229,3:195]
            Crop_image = preprocess(Crop_image)
            Crop_Mask = preprocess(Crop_Mask)
            
            input_array = np.concatenate([Crop_image,Crop_Mask],axis=3)
            #input_array = Crop_image
            pred_array = model_Trans.predict(input_array, verbose=1,batch_size=4)
            pred_Pading_Trans[:,5:229,3:195] = pred_array[:,:,:,0]

           
            # quan
            pred_Pading_Trans[pred_Pading_Trans>=0.5]=1
            pred_Pading_Trans[pred_Pading_Trans<0.5]=0
           
            label_itk = sitk.ReadImage(label_path)
            label_array = sitk.GetArrayFromImage(label_itk)
            y_true=label_array
            VDSC_Trans=numpy_dice( y_true , pred_Pading_Trans , axis=None)

            print(VDSC_Trans)
            vol_scores_Trans.append(VDSC_Trans)
          

            row_num+=1
            sheet.write(row_num, 0, idx)
            sheet.write(row_num, 1, VDSC_Trans)
     
            #y_pred_up.astype(int)

            itkimage = sitk.GetImageFromArray(pred_Pading_Trans, isVector=False)
            itkimage.SetSpacing(image_itk.GetSpacing())
            itkimage.SetOrigin(image_itk.GetOrigin())
            sitk.WriteImage(itkimage, 'F:\\Stroke_7Label\\SEG_Results_MIDeeplabAffine\\'+idx+'.nii', True)
        else:
            continue
        
    vol_scores_Trans = np.array(vol_scores_Trans)
 

    print('Mean volumetric DSC:', vol_scores_Trans.mean())

    row_num+=2
    sheet.write(row_num, 0, 'Mean')
    sheet.write(row_num, 1, vol_scores_Trans.mean())

    book.save(code_path+'Statisis\\'+Model_name+'_FD_'+str(current_fold)+'.xls')

if __name__=='__main__':
    check_predictions_Trans(224,192,0)   
    check_predictions_Trans(224,192,1)
    check_predictions_Trans(224,192,2)
    check_predictions_Trans(224,192,3)
    check_predictions_Trans(224,192,4)
 

