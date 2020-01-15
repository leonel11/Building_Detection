# Скрипт с реализованной архитектурой сети DLinkNet
import numpy as np
from keras.models import Model, Sequential
from keras.models import Input
from keras.layers import concatenate, BatchNormalization, Activation, Add
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras import backend as K


np.random.seed(1) # для повторной воспроизводимости результатов

IMG_CHANNELS = 3
IMG_ROWS = 512
IMG_COLS = 512
CLASSES = 2

K.set_image_dim_ordering('tf') # установка порядка измерений для картинок как в бэкэнде Tensorflow


def res_block(input_tensor, n_filters):
    branch = Conv2D(filters=n_filters, kernel_size=3, padding='same', activation='relu')(input_tensor)
    branch = Conv2D(filters=n_filters, kernel_size=3, padding='same', activation='relu')(branch)
    return Add()([branch, input_tensor])


def dilation_block(input_tensor, n_filters):
    branch1 = Conv2D(filters=n_filters, kernel_size=3, padding='same', activation='relu')(input_tensor)
    branch1 = Conv2D(filters=n_filters, kernel_size=3, padding='same', activation='relu', dilation_rate=(1, 1))(branch1)
    branch1 = Conv2D(filters=n_filters, kernel_size=3, padding='same', activation='relu', dilation_rate=(2, 2))(branch1)
    branch1 = Conv2D(filters=n_filters, kernel_size=3, padding='same', activation='relu', dilation_rate=(4, 4))(branch1)
    branch1 = Conv2D(filters=n_filters, kernel_size=3, padding='same', activation='relu', dilation_rate=(8, 8))(branch1)
    branch2 = Conv2D(filters=n_filters, kernel_size=3, padding='same', activation='relu')(input_tensor)
    branch2 = Conv2D(filters=n_filters, kernel_size=3, padding='same', activation='relu', dilation_rate=(1, 1))(branch2)
    branch2 = Conv2D(filters=n_filters, kernel_size=3, padding='same', activation='relu', dilation_rate=(2, 2))(branch2)
    branch2 = Conv2D(filters=n_filters, kernel_size=3, padding='same', activation='relu', dilation_rate=(4, 4))(branch2)
    branch3 = Conv2D(filters=n_filters, kernel_size=3, padding='same', activation='relu')(input_tensor)
    branch3 = Conv2D(filters=n_filters, kernel_size=3, padding='same', activation='relu', dilation_rate=(1, 1))(branch3)
    branch3 = Conv2D(filters=n_filters, kernel_size=3, padding='same', activation='relu', dilation_rate=(2, 2))(branch3)
    branch4 = Conv2D(filters=n_filters, kernel_size=3, padding='same', activation='relu')(input_tensor)
    branch4 = Conv2D(filters=n_filters, kernel_size=3, padding='same', activation='relu', dilation_rate=(1, 1))(branch4)
    branch5 = Conv2D(filters=n_filters, kernel_size=3, padding='same', activation='relu')(input_tensor)
    return Add()([branch1, branch2, branch3, branch4, branch5])


def up_block(input_tensor, m, n):
    x = Conv2D(m, kernel_size=1, padding='same', activation='relu')(input_tensor)
    x = Conv2DTranspose(m, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(x)
    x = Conv2D(n, kernel_size=1, padding='same', activation='relu')(x)
    return x


def get_model():
    inputs = Input(shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS))
    conv1 = Conv2D(32, kernel_size=7, strides=(2, 2), padding='same', activation='relu')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    res2 = res_block(pool1, n_filters=32)
    res2 = res_block(res2, n_filters=32)
    res2 = res_block(res2, n_filters=32)
    conv2 = Conv2D(64, kernel_size=3, padding='same', activation='relu')(res2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    res3 = res_block(pool2, n_filters=64)
    res3 = res_block(res3, n_filters=64)
    res3 = res_block(res3, n_filters=64)
    res3 = res_block(res3, n_filters=64)
    conv3 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(res3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    res4 = res_block(pool3, n_filters=128)
    res4 = res_block(res4, n_filters=128)
    res4 = res_block(res4, n_filters=128)
    res4 = res_block(res4, n_filters=128)
    res4 = res_block(res4, n_filters=128)
    res4 = res_block(res4, n_filters=128)
    conv4 = Conv2D(256, kernel_size=3, padding='same', activation='relu')(res4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    res5 = res_block(pool4, n_filters=256)
    res5 = res_block(res5, n_filters=256)
    res5 = res_block(res5, n_filters=256)
    dil5 = dilation_block(res5, n_filters=256)
    up6 = up_block(dil5, m=256, n=128)
    add6 = Add()([up6, res4])
    up7 = up_block(add6, m=128, n=64)
    add7 = Add()([up7, res3])
    up8 = up_block(add7, m=64, n=32)
    add8 = Add()([up8, res2])
    up9 = up_block(add8, m=32, n=32)
    transp10 = Conv2DTranspose(16, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(up9)
    conv10 = Conv2D(1, kernel_size=3, padding='same', activation='relu', dilation_rate=(1, 1))(transp10)
    outputs = Activation('softmax')(conv10)
    model = Model(name='DLinkNet', inputs=inputs, outputs=outputs)
    return model