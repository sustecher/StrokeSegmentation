#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 2017
@author: Inom Mirzaev
"""

from __future__ import division, print_function

import cv2
import numpy as np
import os,  sys
from keras.optimizers import Adam
import SimpleITK as sitk
from skimage.transform import resize
from skimage.measure import find_contours

import xlwt
import scipy.io as scio
#from model_Unet import *
from model_Unet_pading import *

from metrics import *
from data import *
from loss import get_loss, dice

Mean_Stroke=46.26
STD_Stroke=21.42
data_path = 'H:\\ATLAStoke\\PP_data\\'
code_path = 'H:\\ATLAStoke\\code\\'
FD=5
nb_epoch=50
batch_size=8
AUG=0

#Model_name='Unet_MASK3'
#Model_name='Unet_SingleChannel'
#Model_name='Unet_Whole'
Model_name='Unet_Whole_M3'

def get_model(img_rows, img_cols,plane,current_fold):
    model =  UNet((img_rows, img_cols, 2))
    model.load_weights(code_path+'\\models\\'+ Model_name+plane+'_FD' + str(current_fold) + '.h5')
    model.compile(optimizer=Adam(), loss=get_loss, metrics=[dice])
    return model

def check_predictions_DSC_25D(Len1,Len2,Len3,current_fold):
    model_Trans = get_model(Len2, Len3,'Trans',current_fold)
    model_Sagit = get_model(Len1, Len3,'Sagit',current_fold)
    model_Coron = get_model(Len2, Len1,'Coron',current_fold)

    vol_scores_Trans = []
    vol_scores_Sagit = []
    vol_scores_Coron = []
    vol_scores_Mean =[]
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
    sheet.write(row_num, 2, 'Sagit')
    sheet.write(row_num, 3, 'Coron')
    sheet.write(row_num, 4, 'Mean')
    for l in range(num_patient_train):
        if l % FD ==current_fold:
            file_im=imlb_list[l]
            idx=file_im[0:file_im.find('_')]
            print(idx)
            image_path =data_path+'Image\\'+file_im
            label_path =data_path+'Label\\'+idx+'_LABEL.nii'
            Mask_path =data_path+'Mask\\'+idx+'_REG.nii'

            image_itk = sitk.ReadImage(image_path)
            image_array  = sitk.GetArrayFromImage(image_itk)
            Mask_array = itk_to_array(Mask_path)

            pred_Pading_Trans = np.zeros(image_array.shape)
            pred_Pading_Sagit = np.zeros(image_array.shape)
            pred_Pading_Coron = np.zeros(image_array.shape)
            pred_Pading_Mean = np.zeros(image_array.shape)
            ## Trans
            Crop_image=image_array
            Crop_Mask=Mask_array
            Crop_image = preprocess(Crop_image)
            Crop_Mask = preprocess(Crop_Mask)
            input_array = np.concatenate([Crop_image,Crop_Mask],axis=3)
            #input_array = Crop_image
            pred_array = model_Trans.predict(input_array, verbose=1,batch_size=32)
            pred_Pading_Trans = pred_array[:,:,:,0]

            ## Sagit
            Crop_image=image_array
            Crop_Mask=Mask_array
            Crop_image=np.transpose(Crop_image,[1,0,2])
            Crop_Mask=np.transpose(Crop_Mask,[1,0,2])
            Crop_image = preprocess(Crop_image)
            Crop_Mask = preprocess(Crop_Mask)
            input_array = np.concatenate([Crop_image,Crop_Mask],axis=3)
            #input_array = Crop_image
            pred_array = model_Sagit.predict(input_array, verbose=1,batch_size=32)
            temp=pred_array[:,:,:,0]
            temp=np.transpose(temp,[1,0,2])
            pred_Pading_Sagit= temp

            ## Coron
            Crop_image=image_array
            Crop_Mask=Mask_array
            Crop_image=np.transpose(Crop_image,[2,1,0])
            Crop_Mask=np.transpose(Crop_Mask,[2,1,0])
            Crop_image = preprocess(Crop_image)
            Crop_Mask = preprocess(Crop_Mask)            
            input_array = np.concatenate([Crop_image,Crop_Mask],axis=3)
            #input_array = Crop_image
            pred_array = model_Coron.predict(input_array, verbose=1,batch_size=32)
            temp=pred_array[:,:,:,0]
            temp=np.transpose(temp,[2,1,0])
            pred_Pading_Coron = temp

            # quan
            pred_Pading_Trans[pred_Pading_Trans>=0.5]=1
            pred_Pading_Trans[pred_Pading_Trans<0.5]=0
            pred_Pading_Sagit[pred_Pading_Sagit>=0.5]=1
            pred_Pading_Sagit[pred_Pading_Sagit<0.5]=0
            pred_Pading_Coron[pred_Pading_Coron>=0.5]=1
            pred_Pading_Coron[pred_Pading_Coron<0.5]=0

            pred_Pading_Mean[pred_Pading_Trans+pred_Pading_Sagit+pred_Pading_Coron>1.5]=1
           
            label_itk = sitk.ReadImage(label_path)
            label_array = sitk.GetArrayFromImage(label_itk)
            y_true=label_array
            VDSC_Trans=numpy_dice( y_true , pred_Pading_Trans , axis=None)
            VDSC_Sagit=numpy_dice( y_true , pred_Pading_Sagit , axis=None)
            VDSC_Coron=numpy_dice( y_true , pred_Pading_Coron , axis=None)
            VDSC_Mean=numpy_dice( y_true , pred_Pading_Mean, axis=None)

            print(VDSC_Trans,VDSC_Sagit,VDSC_Coron,VDSC_Mean)
            vol_scores_Trans.append(VDSC_Trans)
            vol_scores_Sagit.append(VDSC_Sagit)
            vol_scores_Coron.append(VDSC_Coron)
            vol_scores_Mean.append(VDSC_Mean)

            row_num+=1
            sheet.write(row_num, 0, idx)
            sheet.write(row_num, 1, VDSC_Trans)
            sheet.write(row_num, 2, VDSC_Sagit)
            sheet.write(row_num, 3, VDSC_Coron)
            sheet.write(row_num, 4, VDSC_Mean)         
            #y_pred_up.astype(int)
            itkimage = sitk.GetImageFromArray(pred_Pading_Trans, isVector=False)
            itkimage.SetSpacing(image_itk.GetSpacing())
            itkimage.SetOrigin(image_itk.GetOrigin())
            sitk.WriteImage(itkimage, 'H:\\ATLAStoke\\SEG_results\\'+ Model_name +'_Trans\\'+idx+'.nii', True)

            itkimage = sitk.GetImageFromArray(pred_Pading_Sagit, isVector=False)
            itkimage.SetSpacing(image_itk.GetSpacing())
            itkimage.SetOrigin(image_itk.GetOrigin())
            sitk.WriteImage(itkimage, 'H:\\ATLAStoke\\SEG_results\\'+ Model_name +'_Sagit\\'+idx+'.nii', True)

            itkimage = sitk.GetImageFromArray(pred_Pading_Coron, isVector=False)
            itkimage.SetSpacing(image_itk.GetSpacing())
            itkimage.SetOrigin(image_itk.GetOrigin())
            sitk.WriteImage(itkimage, 'H:\\ATLAStoke\\SEG_results\\'+ Model_name +'_Coron\\'+idx+'.nii', True)
        else:
            continue
        
    vol_scores_Trans = np.array(vol_scores_Trans)
    vol_scores_Sagit = np.array(vol_scores_Sagit)
    vol_scores_Coron = np.array(vol_scores_Coron)
    vol_scores_Mean = np.array(vol_scores_Mean)

    print('Mean volumetric DSC:', vol_scores_Trans.mean(),vol_scores_Sagit.mean(),vol_scores_Coron.mean(),vol_scores_Mean.mean())

    row_num+=2
    sheet.write(row_num, 0, 'Mean')
    sheet.write(row_num, 1, vol_scores_Trans.mean())
    sheet.write(row_num, 2, vol_scores_Sagit.mean())
    sheet.write(row_num, 3, vol_scores_Coron.mean())
    sheet.write(row_num, 4, vol_scores_Mean.mean())

    book.save(code_path+'Statisis\\'+Model_name+'_FD_'+str(current_fold)+'25D.xls')


if __name__=='__main__':
   check_predictions_DSC_25D(189,233,197,0)
   check_predictions_DSC_25D(189,233,197,1)
   check_predictions_DSC_25D(189,233,197,2)
   check_predictions_DSC_25D(189,233,197,3)
   check_predictions_DSC_25D(189,233,197,4)
 

