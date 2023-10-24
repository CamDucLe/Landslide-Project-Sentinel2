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
    ## channel
    tl1 = tf.math.reduce_mean(x, axis=-1)                                  # WxH
    tl1 = MultiHeadAttention(num_heads=4, key_dim=12)(tl1, tl1)            # WxH
    tl1 = sigmoid(tl1)            # WxH
    tl1 = Reshape((w,h,1))(tl1)   # WxHx1
    tl1 = Multiply()([x, tl1])                    # WxHxC * WxHx1 -> WxHxC
    ## width
    tl2 = tf.math.reduce_mean(x, axis=-2)                                   # WxC
    tl2 = MultiHeadAttention(num_heads=4, key_dim=12)(tl2, tl2)             # WxC
    tl2 = sigmoid(tl2)            # WxCx1
    tl2 = Reshape((w,1,c))(tl2)   # Wx1xC
    tl2 = Multiply()([x, tl2])                    # HxWxC * Wx1xC -> WxHxC
    ## height
    tl3 = tf.math.reduce_mean(x, axis=-3)                                    # HxC
    tl3 = MultiHeadAttention(num_heads=4, key_dim=12)(tl3, tl3)              # HxC
    tl3 = sigmoid(tl3)            # HxC
    tl3 = Reshape((1,h,c))(tl3)   # 1xHxC
    tl3 = Multiply()([x, tl3])                    # WxHxC * 1xHxC -> WxHxC
    ## average
    t = Average()([tl1,tl2,tl3]) # WxHxC

    return t


def DownBlock(x, filters):
    x = Conv2D(filters=filters, kernel_size=3, strides=2, padding='same', kernel_initializer="he_normal")(x)
    x = AttentionBlock(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    return x


def UpBlock(x, filters):
    x = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same', kernel_initializer="he_normal")(x)  # WxHxF
    x = AttentionBlock(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    return x


def getWaveNet(input_shape, num_classes=2, dropout_ratio=0.2):
    num_block = 5

    input = Input(shape=input_shape)

    x = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='relu', kernel_initializer="he_normal")(input)

    lst = []
    for i in range(num_block):
        x_init = x
        x = UpBlock(x_init, filters=64)
        x = DownBlock(x, filters=64)
        x = DownBlock(x, filters=64)
        x = UpBlock(x, filters=64)
        x = Concatenate()([x_init,x])


    x = Dropout(dropout_ratio)(x)

    x64  = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', kernel_initializer="he_normal", activation='relu')(x) # 64x64x64
    x128 = x # 128x128x64
    x256 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', kernel_initializer="he_normal", activation='relu')(x)  # 256x256x64

    out1 = Conv2D(num_classes, 1, activation='softmax')(x256) # 256x256x2
    out2 = Conv2D(num_classes, 1, activation='softmax')(x128) # 128x128x2
    out3 = Conv2D(num_classes, 1, activation='softmax')(x64)  # 64x64x2

    return Model(input, [out1,out2,out3], name='WaveNet')


## create full model
def getModel(pre_trained_model=None):
    network = getWaveNet(input_shape=(128,128,23))
    if pre_trained_model is not None:
        model.set_weights(pre_trained_model.get_weights())
    network.trainable = True
    opt = tf.keras.optimizers.Adam(learning_rate=4e-4)
    network.compile(loss=[FocalLoss(), IOULoss()], optimizer=opt, metrics=[])

    return network


