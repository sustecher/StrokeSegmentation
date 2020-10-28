from __future__ import division, print_function

from keras import backend as K
K.set_image_data_format('channels_last')
import numpy as np
from scipy.ndimage import morphology

import SimpleITK as sitk
from skimage.exposure import equalize_adapthist, equalize_hist
#import cv2
import math
import random
from skimage import measure
from scipy import ndimage

smooth=1.0

def Avoid_out_of_range(Center_0,size_0,leng_0):
    if Center_0-size_0/2<0:
        Center_0=size_0/2+1

    if Center_0+size_0/2>leng_0:
        Center_0=leng_0-size_0/2-1
    new_center=int(Center_0)
    return new_center

def smooth_images(imgs, t_step=0.125, n_iter=5):
    """
    Curvature driven image denoising.
    In my experience helps significantly with segmentation.
    """
    for mm in range(len(imgs)):
        img = sitk.GetImageFromArray(imgs[mm])
        img = sitk.CurvatureFlow(image1=img,
                                        timeStep=t_step,
                                        numberOfIterations=n_iter)

        imgs[mm] = sitk.GetArrayFromImage(img)
    return imgs

def preprocess(imgs):
    """add one more axis as tf require"""
    imgs = imgs[..., np.newaxis]
    return imgs

def preprocess_front(imgs):
    imgs = imgs[np.newaxis, ...]
    return imgs

def img_resize(imgs, img_rows, img_cols, equalize=True):

    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        if equalize:
            img = equalize_adapthist( img, clip_limit=0.05 )

        new_imgs[mm] = cv2.resize( img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST )

    return new_imgs

def imgs_adapthist(imgs):
    new_imgs = np.zeros([len(imgs),imgs.shape[1],imgs.shape[2]])
    for mm, img in enumerate(imgs):
        img = equalize_adapthist( img, clip_limit=0.05 )
        new_imgs[mm] = img

    return new_imgs

def Find_rangeZ(GS_3D):
    max_Z = np.zeros([GS_3D.shape[0]])
    for i in range(GS_3D.shape[0]):
        max_Z[i]=np.max(GS_3D[i,:,:])
        #print(max_Z[i])
    arr = np.nonzero(max_Z)
    start_idx = min(arr[0])
    end_idx = max(arr[0])+1
    return start_idx, end_idx

def itk_to_array(image_filename):
    image_itk = sitk.ReadImage(image_filename)
    image_array = sitk.GetArrayFromImage(image_itk)
    return image_array

def Label_connection(prob_):
    area_list = []
    binary = np.zeros(prob_.shape,dtype=np.uint8)
    binary[prob_>0.5]=1
    #binary = prob_ > 0.5
    label_prob_ = measure.label(binary)
    region_label_prob_ = measure.regionprops(label_prob_)
    for region in region_label_prob_:
        area_list.append(region.area)
    idx_max = np.argmax(area_list)
    binary[label_prob_ != idx_max + 1] = 0
    temp_prob_ = binary
    #temp_prob_.astype(np.uint8)
    #plt.figure('Orginal Prob'), plt.imshow(temp_prob_, cmap='gray')
    #plt.show()
    return temp_prob_
'''
def dice_coef(y_true, y_pred, smooth=1.0):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
'''

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

#focal loss
def focal_loss(y_true, y_pred, gamma=2.0):
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    eps = K.epsilon()
    y_pred = K.clip(y_pred, eps, 1. - eps)
    loss = -K.sum(K.pow(1. - y_pred, gamma) * y_true * K.log(y_pred), axis=-1)
    return loss

def numpy_dice(y_true, y_pred, axis=None, smooth=1.0):

    intersection = y_true*y_pred

    return ( 2. * intersection.sum(axis=axis) +smooth)/ (np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis) +smooth )


def rel_abs_vol_diff(y_true, y_pred):

    return np.abs( (y_pred.sum()/y_true.sum() - 1)*100)


def get_boundary(data, img_dim=2, shift = -1):
    data  = data>0
    edge = np.zeros_like(data)
    for nn in range(img_dim):
        edge += ~(data ^ np.roll(~data,shift=shift,axis=nn))
    return edge.astype(int)



def surface_dist(input1, input2, sampling=1, connectivity=1):
    input1 = np.squeeze(input1)
    input2 = np.squeeze(input2)

    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))


    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = input_1 - morphology.binary_erosion(input_1, conn)
    Sprime = input_2 - morphology.binary_erosion(input_2, conn)


    dta = morphology.distance_transform_edt(~S,sampling)
    dtb = morphology.distance_transform_edt(~Sprime,sampling)

    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])

    return sds

def Find_ranges(mask,margin_xy,margin_z):
    #print(np.shape(mask))
    arr=np.nonzero(mask)
    Min_z=int(min(arr[0])-margin_z)
    Max_z=int(max(arr[0])+margin_z)

    Min_x=int(min(arr[1])-margin_xy)
    Max_x=int(max(arr[1])+margin_xy)
    Min_y=int(min(arr[2])-margin_xy)
    Max_y=int(max(arr[2])+margin_xy)
    
    Min_z=max(0,Min_z)
    Max_z=min(np.shape(mask)[0],Max_z)

    return Min_z,Max_z,Min_x,Max_x,Min_y,Max_y