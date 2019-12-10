from keras.models import Model
from keras.initializers import RandomNormal, VarianceScaling
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import  merge, UpSampling2D, Dropout, Cropping2D, BatchNormalization
import os
import time
import scipy
import numpy as np
import tensorflow as tf
import keras
from keras.models import Model
from keras import metrics
from keras.layers import Lambda, Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, BatchNormalization, Activation, Dropout, Add
from keras.layers import DepthwiseConv2D
#SE
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
#
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from scipy.misc import imresize
import warnings
from keras.initializers import RandomNormal, VarianceScaling
import numpy as np
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.advanced_activations import PReLU
import numpy as np

smooth=1.

def dice_coef_for_training(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef_for_training(y_true, y_pred)


def conv_bn_relu(nd, k=3, inputs=None):
    conv = Conv2D(nd, k, padding='same', kernel_initializer='he_normal')(inputs)  # , kernel_initializer='he_normal'
    #conv = spatial_se(conv)
    bn = BatchNormalization()(conv)
    relu = Activation('relu')(bn)
    #relu= channel_se(relu)
    return relu

 # Baseline U-net

def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)

def UNet(img_shape):
    inputs = Input(shape=img_shape)

    concat_axis = -1
    filters = 3

    conv1 = conv_bn_relu(64, filters, inputs)
    conv1 = conv_bn_relu(64, 3, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_bn_relu(128, 3, pool1)
    conv2 = conv_bn_relu(128, 3, conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_bn_relu(256, 3, pool2)
    conv3 = conv_bn_relu(256, 3, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_bn_relu(512, 3, pool3)
    conv4 = conv_bn_relu(512, 3, conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_bn_relu(1024, 3, pool4)
    conv5 = conv_bn_relu(512, 3, conv5)

    up1_C5 = UpSampling2D(size=(2, 2))(conv5)
    ch, cw = get_crop_shape(conv4,up1_C5)
    crop_conv4= Cropping2D(cropping=(ch,cw))(conv4)
    up1 = concatenate([up1_C5, crop_conv4], axis=concat_axis)
    conv6 = conv_bn_relu(512, 3, up1)
    conv6 = conv_bn_relu(256, 3, conv6)

    up2_C6 = UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv3,up2_C6)
    crop_conv3= Cropping2D(cropping=(ch,cw))(conv3)
    up2= concatenate([up2_C6, crop_conv3], axis=concat_axis)
    conv7 = conv_bn_relu(256, 3, up2)
    conv7 = conv_bn_relu(128, 3, conv7)

    up3_C7 = UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv2,up3_C7)
    crop_conv2= Cropping2D(cropping=(ch,cw))(conv2)
    up3 = concatenate([up3_C7, crop_conv2], axis=concat_axis)
    conv8 = conv_bn_relu(128, 3, up3)
    conv8 = conv_bn_relu(64, 3, conv8)

    up4_C8 = UpSampling2D(size=(2, 2))(conv8)
    ch, cw = get_crop_shape(conv1,up4_C8)
    crop_conv1= Cropping2D(cropping=(ch,cw))(conv1)
    up5 = concatenate([up4_C8, crop_conv1], axis=concat_axis)
    conv9 = conv_bn_relu(64, 3, up5)
    conv9 = conv_bn_relu(32, 3, conv9)
    
    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = ZeroPadding2D(padding=(ch, cw))(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9)  # , kernel_initializer='he_normal'
    
    model = Model(inputs=inputs, outputs=conv10)
    #model.compile(optimizer=Adam(lr=(2e-4)), loss=dice_coef_loss)

    return model




if  __name__=='__main__':

    model = UNet((192, 160,1))
    model.summary()
