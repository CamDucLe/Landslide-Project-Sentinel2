import os
import tensorflow as tf

# from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D, SeparableConv2D, UpSampling2D, UpSampling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, MultiHeadAttention
from tensorflow.keras.layers import BatchNormalization, Add, Average, Concatenate, LeakyReLU, Softmax
from tensorflow.keras.layers import Reshape, Multiply, Permute, Lambda
from tensorflow.keras.activations import sigmoid,softmax

from losses import *


def AttentionBlock(x):
    w = x.shape[-3]
    h = x.shape[-2]
    c = x.shape[-1]
    
    se = GlobalAveragePooling2D()(x)            # c
    se = Dense(int(c/2), activation='selu')(se) #c/2
    se = Dense(c, activation='sigmoid')(se)     # c
    se = Reshape((1,1,c))(se)                   # 1x1xc
    se = x * se                                 #

    return se

def DownBlock(x, filters):
    x1 = Conv2D(filters, kernel_size=4, padding='same', activation='selu', kernel_initializer="he_normal")(x)
    x2 = Conv2DTranspose(filters, kernel_size=4, padding='same', activation='selu', kernel_initializer="he_normal")(x)
    x3 = SeparableConv2D(filters, kernel_size=4, padding='same', activation='selu', kernel_initializer="he_normal")(x)
    x4 = Conv2D(filters, kernel_size=4, dilation_rate=4, padding='same', activation='selu', kernel_initializer="he_normal")(x)

    x  = Concatenate(axis=-1)([x1,x2,x3,x4])
    x  = AttentionBlock(x)

    x_down = Conv2D(4*filters, strides=2, kernel_size=2, padding='same', activation='selu', kernel_initializer="he_normal")(x)

    return x_down, x


def UpBlock(x, y, filters):
    x = Conv2DTranspose(filters=filters, strides=2, kernel_size=2, padding='same', activation='selu', kernel_initializer="he_normal")(x)
    x = Concatenate(axis=-1)([x,y])
    x = AttentionBlock(x)

    x = Conv2D(filters=filters, kernel_size=4, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    return x


def getCUNet(input_shape=(128,128,21), num_classes=2, dropout_ratio=0.2):
    filter_d  = [50,70,90,110]
    filter_u  = [440,360,280,200]

    input   = Input(shape=input_shape)

    # encode
    x, y128 = DownBlock(input, filter_d[0]) # 64x64x160 and 128x128x160
    x, y64  = DownBlock(x,     filter_d[1]) # 32x32x240 and 64x64x240
    x, y32  = DownBlock(x,     filter_d[2]) # 16x16x320 and 32x32x320
    x, y16  = DownBlock(x,     filter_d[3]) # 8x8x400   and 16x16x400

    # decode
    x = UpBlock(x, y16,  filter_u[0])  # 16x16x320
    x = UpBlock(x, y32,  filter_u[1])  # 32x32x240
    x = UpBlock(x, y64,  filter_u[2])  # 64x64x160
    x = UpBlock(x, y128, filter_u[3])  # 128x128x80

    x = Dropout(dropout_ratio)(x)

    x64  = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', kernel_initializer="he_normal", activation='relu')(x) # 64x64x64
    x128 = x # 128x128x64
    x256 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', kernel_initializer="he_normal", activation='relu')(x)  # 256x256x64

    out1 = Conv2D(num_classes, 1, activation='softmax')(x256) # 256x256x2
    out2 = Conv2D(num_classes, 1, activation='softmax')(x128) # 128x128x2
    out3 = Conv2D(num_classes, 1, activation='softmax')(x64)  # 64x64x2

    return Model(input, [out1,out2,out3], name='CUNet')


## create full model
def getModel(pre_trained_model=None):
    network = getCUNet(input_shape=(128,128,23))
    if pre_trained_model is not None:
        model.set_weights(pre_trained_model.get_weights())
    network.trainable = True
    opt = tf.keras.optimizers.Adam(learning_rate=4e-4)
    network.compile(loss=[FocalLoss(), IOULoss()], optimizer=opt, metrics=[])

    return network


