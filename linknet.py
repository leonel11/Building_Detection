# Скрипт с реализованной архитектурой сети LinkNet
import numpy as np
from keras.models import Model
from keras.models import Input
from keras.layers.convolutional import Conv2D, UpSampling2D, MaxPooling2D
from keras.layers import BatchNormalization, Activation, concatenate
from keras.regularizers import l2
from keras import backend as K


np.random.seed(1) # для повторной воспроизводимости результатов

IMG_CHANNELS = 3
IMG_ROWS = 512
IMG_COLS = 512
CLASSES = 2

K.set_image_dim_ordering('tf') # установка порядка измерений для картинок как в бэкэнде Tensorflow


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]
    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[3], kernel_size=1, strides=(stride_width, stride_height),
                          padding="valid", kernel_initializer="he_normal", kernel_regularizer=l2(0.0001))(input)
    return concatenate([shortcut, residual], axis=3)


def encoder_block(inp, filters):
    x = BatchNormalization()(inp)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=3, strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=3, padding="same")(x)
    added_1 = _shortcut(inp, x)
    x = BatchNormalization()(added_1)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=3, padding="same")(x)
    added_2 = _shortcut(added_1, x)
    return added_2


def decoder_block(inp, m, n):
    x = BatchNormalization()(inp)
    x = Activation('relu')(x)
    x = Conv2D(filters=int(m/4), kernel_size=1)(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=int(m/4), kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=n, kernel_size=1)(x)
    return x


def get_model():
    inputs = Input(shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS))
    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(filters=32, kernel_size=7, strides=(2, 2), padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    encoder_1 = encoder_block(inp=x, filters=32)
    encoder_2 = encoder_block(inp=encoder_1, filters=64)
    encoder_3 = encoder_block(inp=encoder_2, filters=128)
    encoder_4 = encoder_block(inp=encoder_3, filters=256)
    decoder_4 = decoder_block(inp=encoder_4, m=256, n=128)
    decoder_3_in = concatenate([decoder_4, encoder_3], axis=3)
    decoder_3_in = Activation('relu')(decoder_3_in)
    decoder_3 = decoder_block(inp=decoder_3_in, m=128, n=64)
    decoder_2_in = concatenate([decoder_3, encoder_2], axis=3)
    decoder_2_in = Activation('relu')(decoder_2_in)
    decoder_2 = decoder_block(inp=decoder_2_in, m=64, n=32)
    decoder_1_in = concatenate([decoder_2, encoder_1], axis=3)
    decoder_1_in = Activation('relu')(decoder_1_in)
    decoder_1 = decoder_block(inp=decoder_1_in, m=32, n=32)
    x = UpSampling2D(size=(2, 2))(decoder_1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=16, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=16, kernel_size=3, padding="same")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1, kernel_size=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    model = Model(name='LinkNet', inputs=inputs, outputs=x)
    return model