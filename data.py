'''
data generator
ATLAS dataset has been transformed into .h5 format
'''

import numpy as np
import os
import cv2
import SimpleITK as sitk
import scipy.io as scio
from metrics import *
from skimage.exposure import equalize_adapthist, equalize_hist

# 189,233,197,
# [], 192[21:213],160[19:179]
def SBS_equalize_adapthist(imgs):

    new_imgs = np.zeros([len(imgs), imgs.shape[1], imgs.shape[2]])
    for mm, img in enumerate(imgs):
        img = equalize_adapthist( img, clip_limit=0.05 )
        new_imgs[mm] = img

    return new_imgs

Mean_Stroke=46.26
STD_Stroke=21.42

def data_to_array(data_path,code_path,current_fold,FD,Mean_Stroke,STD_Stroke):
    images_z = []
    labels_z= []
    imlb_list = os.listdir(data_path+'Image')
    imlb_list.sort()
    num_patient_train=len(imlb_list)
    #num_patient_train=5
    print(num_patient_train)
    for l in range(num_patient_train):
        if l % FD ==current_fold:
            continue
        else:
            file_im=imlb_list[l]
            #print(file_im.find('_'))
            idx=file_im[0:file_im.find('_')]
            print(idx)
            image_path =data_path+'Image\\'+file_im
            label_path =data_path+'Label\\'+idx+'_LABEL.nii'

            label_itk = sitk.ReadImage(label_path)
            label_array = sitk.GetArrayFromImage(label_itk)
            #print(label_itk.GetSpacing())
        
            label_array[label_array>0.5]=1
            image_array = itk_to_array(image_path)

            # 
            #image_array= (image_array - Mean_Stroke) / float(STD_Stroke)
            Min_z,Max_z,Min_x,Max_x,Min_y,Max_y=Find_ranges(label_array,0,0)
        
            Crop_image=image_array[Min_z:Max_z,21:213,19:179]
            Crop_label=label_array[Min_z:Max_z,21:213,19:179]
            print(np.shape(Crop_label))
            #Crop_image=Crop_image/50 - 1
            #Crop_image=SBS_equalize_adapthist(Crop_image)
            images_z.append(Crop_image)
            labels_z.append(Crop_label)
        
    images_z = np.concatenate(images_z, axis=0).reshape(-1, 192, 160, 1)
    labels_z = np.concatenate(labels_z, axis=0).reshape(-1, 192, 160, 1)
    print(np.shape(labels_z))

    #scio.savemat('H:\\ATLAStoke\\code_matlab\\X_z_FD'+str(current_fold)+'.mat', {'images':images_z})
    #scio.savemat('H:\\ATLAStoke\\code_matlab\\y_z_FD'+str(current_fold)+'.mat', {'masks':labels_z})
    
    np.save(code_path+'NPY_data\\X_FD'+str(current_fold)+'.npy', images_z)
    np.save(code_path+'NPY_data\\y_FD'+str(current_fold)+'.npy', labels_z)
'''
#Mask
def data_to_array_with_MASK(data_path,code_path,current_fold,FD):
    images_z = []
    labels_z= []
    masks_z= []
    images_y = []
    labels_y= []
    masks_y= []
    images_x = []
    labels_x= []
    masks_x= []

    imlb_list = os.listdir(data_path+'Image')
    imlb_list.sort()
    num_patient_train=len(imlb_list)
   # num_patient_train=5
    print(num_patient_train)
    for l in range(num_patient_train):
        if l % FD ==current_fold:
            continue
        else:
            file_im=imlb_list[l]
            print(file_im)
            idx=file_im[0:file_im.find('_')]
            print(idx)
            image_path =data_path+'Image\\'+file_im
            label_path =data_path+'Label\\'+idx+'_LABEL.nii'
            Mask_path =data_path+'Mask\\'+idx+'_REG.nii'

            label_itk = sitk.ReadImage(label_path)
            label_array = sitk.GetArrayFromImage(label_itk)
        
            label_array[label_array>0.5]=1
            image_array = itk_to_array(image_path)
            Mask_array = itk_to_array(Mask_path)

            # 
            #print(Mask_array.shape) 189, 233,197 # S A S 
            Min_z,Max_z,Min_x,Max_x,Min_y,Max_y=Find_ranges(label_array,0,0)
        
            Crop_image=image_array[Min_z:Max_z,21:213,19:179]
            Crop_label=label_array[Min_z:Max_z,21:213,19:179]
            Crop_Mask=Mask_array[Min_z:Max_z,21:213,19:179]
            images_z.append(Crop_image)
            labels_z.append(Crop_label)
            masks_z.append(Crop_Mask)

            Crop_image=image_array[7:167,Min_x:Max_x,19:179]
            Crop_label=label_array[7:167,Min_x:Max_x,19:179]
            Crop_Mask=Mask_array[7:167,Min_x:Max_x,19:179]
            Crop_image=np.transpose(Crop_image,[1,0,2])
            Crop_label=np.transpose(Crop_label,[1,0,2])
            Crop_Mask=np.transpose(Crop_Mask,[1,0,2])
            images_x.append(Crop_image)
            labels_x.append(Crop_label)
            masks_x.append(Crop_Mask)
            print(Crop_image.shape)

            Crop_image=image_array[7:167,21:213,Min_y:Max_y]
            Crop_label=label_array[7:167,21:213,Min_y:Max_y]
            Crop_Mask=Mask_array[7:167,21:213,Min_y:Max_y]
            Crop_image=np.transpose(Crop_image,[2,1,0])
            Crop_label=np.transpose(Crop_label,[2,1,0])
            Crop_Mask=np.transpose(Crop_Mask,[2,1,0])
            images_y.append(Crop_image)
            labels_y.append(Crop_label)
            masks_y.append(Crop_Mask)
   
    images_z = np.concatenate(images_z, axis=0).reshape(-1, 192, 160, 1)
    labels_z = np.concatenate(labels_z, axis=0).reshape(-1, 192, 160, 1)
    masks_z = np.concatenate(masks_z, axis=0).reshape(-1, 192, 160, 1)

    images_x = np.concatenate(images_x, axis=0).reshape(-1, 160, 160, 1)
    labels_x = np.concatenate(labels_x, axis=0).reshape(-1, 160, 160, 1)
    masks_x = np.concatenate(masks_x, axis=0).reshape(-1, 160, 160, 1)

    images_y = np.concatenate(images_y, axis=0).reshape(-1, 192, 160, 1)
    labels_y = np.concatenate(labels_y, axis=0).reshape(-1, 192, 160, 1)
    masks_y = np.concatenate(masks_y, axis=0).reshape(-1, 192, 160, 1)
    print(np.shape(labels_z))

    #scio.savemat('H:\\ATLAStoke\\code_matlab\\Trans_X_FD'+str(current_fold)+'.mat', {'images':images_z})
    #scio.savemat('H:\\ATLAStoke\\code_matlab\\Trans_y_FD'+str(current_fold)+'.mat', {'labels':labels_z})
    #scio.savemat('H:\\ATLAStoke\\code_matlab\\Trans_M_FD'+str(current_fold)+'.mat', {'masks':masks_z})
    np.save(code_path+'NPY_data\\Trans_X_FD'+str(current_fold)+'.npy', images_z)
    np.save(code_path+'NPY_data\\Trans_y_FD'+str(current_fold)+'.npy', labels_z)
    np.save(code_path+'NPY_data\\Trans_M_FD'+str(current_fold)+'.npy', masks_z)

    #scio.savemat('H:\\ATLAStoke\\code_matlab\\Sagit_X_FD'+str(current_fold)+'.mat', {'images':images_x})
    #scio.savemat('H:\\ATLAStoke\\code_matlab\\Sagit_y_FD'+str(current_fold)+'.mat', {'labels':labels_x})
    #scio.savemat('H:\\ATLAStoke\\code_matlab\\Sagit_M_FD'+str(current_fold)+'.mat', {'masks':masks_x})
    np.save(code_path+'NPY_data\\Sagit_X_FD'+str(current_fold)+'.npy', images_x)
    np.save(code_path+'NPY_data\\Sagit_y_FD'+str(current_fold)+'.npy', labels_x)
    np.save(code_path+'NPY_data\\Sagit_M_FD'+str(current_fold)+'.npy', masks_x)

    #scio.savemat('H:\\ATLAStoke\\code_matlab\\Coron_X_FD'+str(current_fold)+'.mat', {'images':images_y})
    #scio.savemat('H:\\ATLAStoke\\code_matlab\\Coron_y_FD'+str(current_fold)+'.mat', {'labels':labels_y})
    #scio.savemat('H:\\ATLAStoke\\code_matlab\\Coron_M_FD'+str(current_fold)+'.mat', {'masks':masks_y})
    np.save(code_path+'NPY_data\\Coron_X_FD'+str(current_fold)+'.npy', images_y)
    np.save(code_path+'NPY_data\\Coron_y_FD'+str(current_fold)+'.npy', labels_y)
    np.save(code_path+'NPY_data\\Coron_M_FD'+str(current_fold)+'.npy', masks_y)
'''
#FD=5
#Mask

def Find_lesion(mask,axis=0):
    if axis==0:
        max_axis0=np.zeros([mask.shape[0],1])
        for i in range(mask.shape[0]):
            #print(np.max(mask[i,:,:]))
            max_axis0[i]=np.max(mask[i,:,:])
        #print(max_axis0.shape)
        arr=np.nonzero(max_axis0)
        #print(arr)
    elif axis==1:
        max_axis1=np.zeros([mask.shape[1],1])
        for i in range(mask.shape[1]):
            #print(np.max(mask[:,i,:]))
            max_axis1[i]=np.max(mask[:,i,:])
        arr=np.nonzero(max_axis1)
    elif axis==2:
        max_axis2=np.zeros([mask.shape[2],1])
        for i in range(mask.shape[2]):
            #print(np.max(mask[:,:,i]))
            max_axis2[i]=np.max(mask[:,:,i])
        #print(max_axis0.shape)
        arr=np.nonzero(max_axis2)
        #print(arr)
    return arr[0]

def data_to_array_with_MASK(data_path,code_path,current_fold,FD):
    images_z = []
    labels_z= []
    masks_z= []
    images_y = []
    labels_y= []
    masks_y= []
    images_x = []
    labels_x= []
    masks_x= []

    imlb_list = os.listdir(data_path+'Image')
    imlb_list.sort()
    num_patient_train=len(imlb_list)
    #num_patient_train=5
    print(num_patient_train)
    for l in range(num_patient_train):
        if l % FD ==current_fold:
            continue
        else:
            file_im=imlb_list[l]
            print(file_im)
            idx=file_im[0:file_im.find('_')]
            print(idx)
            image_path =data_path+'Image\\'+file_im
            label_path =data_path+'Label\\'+idx+'_LABEL.nii'
            Mask_path =data_path+'Mask\\'+idx+'_REG.nii'

            label_itk = sitk.ReadImage(label_path)
            label_array = sitk.GetArrayFromImage(label_itk)
        
            label_array[label_array>0.5]=1
            image_array = itk_to_array(image_path)
            Mask_array = itk_to_array(Mask_path)
            '''
            arr0=Find_lesion(label_array,0)
            arr1=Find_lesion(label_array,1)
            arr2=Find_lesion(label_array,2)
        
            Crop_image=image_array[arr0,:,:]
            Crop_label=label_array[arr0,:,:]
            Crop_Mask=Mask_array[arr0,:,:]
            images_z.append(Crop_image)
            labels_z.append(Crop_label)
            masks_z.append(Crop_Mask)
            #print(Crop_image.shape)

            Crop_image=image_array[:,arr1,:]
            Crop_label=label_array[:,arr1,:]
            Crop_Mask=Mask_array[:,arr1,:]
            Crop_image=np.transpose(Crop_image,[1,0,2])
            Crop_label=np.transpose(Crop_label,[1,0,2])
            Crop_Mask=np.transpose(Crop_Mask,[1,0,2])
            images_x.append(Crop_image)
            labels_x.append(Crop_label)
            masks_x.append(Crop_Mask)

            Crop_image=image_array[:,:,arr2]
            Crop_label=label_array[:,:,arr2]
            Crop_Mask=Mask_array[:,:,arr2]
            Crop_image=np.transpose(Crop_image,[2,1,0])
            Crop_label=np.transpose(Crop_label,[2,1,0])
            Crop_Mask=np.transpose(Crop_Mask,[2,1,0])
            images_y.append(Crop_image)
            labels_y.append(Crop_label)
            masks_y.append(Crop_Mask)
            #print(Crop_image.shape)
            '''
            
            Min_z,Max_z,Min_x,Max_x,Min_y,Max_y=Find_ranges(label_array ,0,0)
            Crop_image=image_array[Min_z:Max_z,:,:]
            Crop_label=label_array[Min_z:Max_z,:,:]
            Crop_Mask=Mask_array[Min_z:Max_z,:,:]
            images_z.append(Crop_image)
            labels_z.append(Crop_label)
            masks_z.append(Crop_Mask)
            #print(Crop_image.shape)

            Crop_image=image_array[:,Min_x:Max_x,:]
            Crop_label=label_array[:,Min_x:Max_x,:]
            Crop_Mask=Mask_array[:,Min_x:Max_x,:]
            Crop_image=np.transpose(Crop_image,[1,0,2])
            Crop_label=np.transpose(Crop_label,[1,0,2])
            Crop_Mask=np.transpose(Crop_Mask,[1,0,2])
            images_x.append(Crop_image)
            labels_x.append(Crop_label)
            masks_x.append(Crop_Mask)

            Crop_image=image_array[:,:,Min_y:Max_y]
            Crop_label=label_array[:,:,Min_y:Max_y]
            Crop_Mask=Mask_array[:,:,Min_y:Max_y]
            Crop_image=np.transpose(Crop_image,[2,1,0])
            Crop_label=np.transpose(Crop_label,[2,1,0])
            Crop_Mask=np.transpose(Crop_Mask,[2,1,0])
            images_y.append(Crop_image)
            labels_y.append(Crop_label)
            masks_y.append(Crop_Mask)

   
    images_z = np.concatenate(images_z, axis=0).reshape(-1, 233,197, 1)
    labels_z = np.concatenate(labels_z, axis=0).reshape(-1, 233,197, 1)
    masks_z = np.concatenate(masks_z, axis=0).reshape(-1, 233, 197, 1)

    #images_x = np.concatenate(images_x, axis=0).reshape(-1, 189,197, 1)
    #labels_x = np.concatenate(labels_x, axis=0).reshape(-1, 189,197, 1)
    #masks_x = np.concatenate(masks_x, axis=0).reshape(-1, 189,197, 1)

    images_y = np.concatenate(images_y, axis=0).reshape(-1, 233, 189, 1)
    labels_y = np.concatenate(labels_y, axis=0).reshape(-1, 233, 189, 1)
    masks_y = np.concatenate(masks_y, axis=0).reshape(-1,233, 189, 1)
    print(np.shape(labels_z))

    #scio.savemat('H:\\ATLAStoke\\code_matlab\\Trans_X_FD'+str(current_fold)+'.mat', {'images':images_z})
    #scio.savemat('H:\\ATLAStoke\\code_matlab\\Trans_y_FD'+str(current_fold)+'.mat', {'labels':labels_z})
    #scio.savemat('H:\\ATLAStoke\\code_matlab\\Trans_M_FD'+str(current_fold)+'.mat', {'masks':masks_z})
    np.save(code_path+'NPY_data\\Trans_X_FD'+str(current_fold)+'.npy', images_z)
    np.save(code_path+'NPY_data\\Trans_y_FD'+str(current_fold)+'.npy', labels_z)
    np.save(code_path+'NPY_data\\Trans_M_FD'+str(current_fold)+'.npy', masks_z)

    #scio.savemat('H:\\ATLAStoke\\code_matlab\\Sagit_X_FD'+str(current_fold)+'.mat', {'images':images_x})
    #scio.savemat('H:\\ATLAStoke\\code_matlab\\Sagit_y_FD'+str(current_fold)+'.mat', {'labels':labels_x})
    #scio.savemat('H:\\ATLAStoke\\code_matlab\\Sagit_M_FD'+str(current_fold)+'.mat', {'masks':masks_x})
    #np.save(code_path+'NPY_data\\Sagit_X_FD'+str(current_fold)+'.npy', images_x)
    #np.save(code_path+'NPY_data\\Sagit_y_FD'+str(current_fold)+'.npy', labels_x)
    #np.save(code_path+'NPY_data\\Sagit_M_FD'+str(current_fold)+'.npy', masks_x)

    #scio.savemat('H:\\ATLAStoke\\code_matlab\\Coron_X_FD'+str(current_fold)+'.mat', {'images':images_y})
    #scio.savemat('H:\\ATLAStoke\\code_matlab\\Coron_y_FD'+str(current_fold)+'.mat', {'labels':labels_y})
    #scio.savemat('H:\\ATLAStoke\\code_matlab\\Coron_M_FD'+str(current_fold)+'.mat', {'masks':masks_y})
    np.save(code_path+'NPY_data\\Coron_X_FD'+str(current_fold)+'.npy', images_y)
    np.save(code_path+'NPY_data\\Coron_y_FD'+str(current_fold)+'.npy', labels_y)
    np.save(code_path+'NPY_data\\Coron_M_FD'+str(current_fold)+'.npy', masks_y)

#current_fold=0
#data_path = 'H:\\ATLAStoke\\PP_data\\'
#code_path = 'H:\\ATLAStoke\\code\\'
#data_to_array_with_MASK(data_path,code_path,current_fold,FD)