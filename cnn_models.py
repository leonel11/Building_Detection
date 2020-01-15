# Скрипт с архитектурами сверточных нейронных сетей
import numpy as np
from keras.models import Model, Sequential
from keras.models import Input
from keras.layers import concatenate, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras import backend as K
import linknet
import dlinknet


np.random.seed(1) # для повторной воспроизводимости результатов

IMG_CHANNELS = 3
IMG_ROWS = 512
IMG_COLS = 512
CLASSES = 2

K.set_image_dim_ordering('tf') # установка порядка измерений для картинок как в бэкэнде Tensorflow


def UNet():
    inputs = Input(shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS))
    conv1 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(inputs)
    conv1 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(pool1)
    conv2 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, kernel_size=3, padding='same', activation='relu')(pool2)
    conv3 = Conv2D(64, kernel_size=3, padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(pool3)
    conv4 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(256, kernel_size=3, padding='same', activation='relu')(pool4)
    conv5 = Conv2D(256, kernel_size=3, padding='same', activation='relu')(conv5)
    up6 = UpSampling2D(size=(2, 2))(conv5)
    merge6 = concatenate([up6, conv4], axis=3)
    conv6 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(merge6)
    conv6 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(conv6)
    up7 = UpSampling2D(size=(2, 2))(conv6)
    merge7 = concatenate([up7, conv3], axis=3)
    conv7 = Conv2D(64, kernel_size=3, padding='same', activation='relu')(merge7)
    conv7 = Conv2D(64, kernel_size=3, padding='same', activation='relu')(conv7)
    up8 = UpSampling2D(size=(2, 2))(conv7)
    merge8 = concatenate([up8, conv2], axis=3)
    conv8 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(merge8)
    conv8 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(conv8)
    up9 = UpSampling2D(size=(2, 2))(conv8)
    merge9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(merge9)
    conv9 = Conv2D(16,kernel_size=3, padding='same', activation='relu')(conv9)
    outputs = Conv2D(1, kernel_size=1, activation='sigmoid')(conv9)
    model = Model(name='UNet', inputs=inputs, outputs=outputs)
    return model


def DeeperUNet():
    inputs = Input(shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS))
    conv1 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    conv1 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, kernel_size=3, padding='same', activation='relu')(pool1)
    conv2 = Conv2D(64, kernel_size=3, padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(pool2)
    conv3 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, kernel_size=3, padding='same', activation='relu')(pool3)
    conv4 = Conv2D(256, kernel_size=3, padding='same', activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(512, kernel_size=3, padding='same', activation='relu')(pool4)
    conv5 = Conv2D(512, kernel_size=3, padding='same', activation='relu')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    conv6 = Conv2D(1024, kernel_size=3, padding='same', activation='relu')(pool5)
    conv6 = Conv2D(1024, kernel_size=3, padding='same', activation='relu')(conv6)
    up7 = UpSampling2D(size=(2, 2))(conv6)
    merge7 = concatenate([up7, conv5], axis=3)
    conv7 = Conv2D(512, kernel_size=3, padding='same', activation='relu')(merge7)
    conv7 = Conv2D(512, kernel_size=3, padding='same', activation='relu')(conv7)
    up8 = UpSampling2D(size=(2, 2))(conv7)
    merge8 = concatenate([up8, conv4], axis=3)
    conv8 = Conv2D(256, kernel_size=3, padding='same', activation='relu')(merge8)
    conv8 = Conv2D(256, kernel_size=3, padding='same', activation='relu')(conv8)
    up9 = UpSampling2D(size=(2, 2))(conv8)
    merge9 = concatenate([up9, conv3], axis=3)
    conv9 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(merge9)
    conv9 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(conv9)
    up10 = UpSampling2D(size=(2, 2))(conv9)
    merge10 = concatenate([up10, conv2], axis=3)
    conv10 = Conv2D(64, kernel_size=3, padding='same', activation='relu')(merge10)
    conv10 = Conv2D(64, kernel_size=3, padding='same', activation='relu')(conv10)
    up11 = UpSampling2D(size=(2, 2))(conv10)
    merge11 = concatenate([up11, conv1], axis=3)
    conv11 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(merge11)
    conv11 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(conv11)
    outputs = Conv2D(1, kernel_size=1, activation='sigmoid')(conv11)
    model = Model(name='DeeperUNet', inputs=inputs, outputs=outputs)
    return model


def SegNetBasic():
    encoding_layers = [
        Conv2D(16, kernel_size=3, input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(16, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(32, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(32, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(64, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(128, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(128, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(),
    ]
    autoencoder = Sequential()
    autoencoder.encoding_layers = encoding_layers
    for l in autoencoder.encoding_layers:
        autoencoder.add(l)
    decoding_layers = [
        UpSampling2D(),
        Conv2D(128, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        UpSampling2D(),
        Conv2D(128, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        UpSampling2D(),
        Conv2D(64, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(32, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        UpSampling2D(),
        Conv2D(32, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(16, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        UpSampling2D(),
        Conv2D(16, kernel_size=3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(1, kernel_size=1, padding='valid', activation='sigmoid'),
        BatchNormalization(),
    ]
    autoencoder.decoding_layers = decoding_layers
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)
    autoencoder.add(Activation('softmax'))
    return autoencoder


def TernausNet():
    inputs = Input(shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS))
    conv1 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(inputs)
    conv1 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(pool1)
    conv2 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(pool2)
    conv3 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(conv3)
    conv3 = Conv2D(128, kernel_size=1, padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(pool3)
    conv4 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(conv4)
    conv4 = Conv2D(128, kernel_size=1, padding='same', activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(pool4)
    conv5 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(conv5)
    conv5 = Conv2D(128, kernel_size=1, padding='same', activation='relu')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    conv6 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(pool5)
    transp6 = Conv2DTranspose(64, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(conv6)
    merge7 = concatenate([transp6, conv5], axis=3)
    conv7 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(merge7)
    transp7 = Conv2DTranspose(64, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(conv7)
    merge8 = concatenate([transp7, conv4], axis=3)
    conv8 = Conv2D(128, kernel_size=3, padding='same', activation='relu')(merge8)
    transp8 = Conv2DTranspose(32, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(conv8)
    merge9 = concatenate([transp8, conv3], axis=3)
    conv9 = Conv2D(64, kernel_size=3, padding='same', activation='relu')(merge9)
    transp9 = Conv2DTranspose(16, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(conv9)
    merge10 = concatenate([transp9, conv2], axis=3)
    conv10 = Conv2D(32, kernel_size=3, padding='same', activation='relu')(merge10)
    transp10 = Conv2DTranspose(16, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(conv10)
    merge11 = concatenate([transp10, conv1], axis=3)
    conv11 = Conv2D(16, kernel_size=3, padding='same', activation='relu')(merge11)
    conv11 = Conv2D(1, kernel_size=3, padding='same', activation='relu')(conv11)
    outputs = Activation('softmax')(conv11)
    model = Model(name='TernausNet', inputs=inputs, outputs=outputs)
    return model


def LinkNet():
    return linknet.get_model()


def DLinkNet():
    return dlinknet.get_model()
