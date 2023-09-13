import os
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV3Small, EfficientNetB0
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
from tensorflow.keras.layers import Reshape, multiply, Permute, Lambda
from tensorflow.keras.activations import sigmoid,softmax

from losses import *

def convolution_block(x, num_filters=96, kernel_size=3, dilation_rate=1):
    x = Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape

    x = AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1)
    out_pool = UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1  = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6  = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])

    output = convolution_block(x, kernel_size=1)

    return output

def getMNet(input_shape=(128,128,23), num_classes=2, dropout_ratio=0.2):
    image_size = 128

    backbone = EfficientNetB0(include_top=False, weights=None, input_tensor=Input(shape=input_shape)) # 4x4x576 = 4096 


    input_a = backbone.get_layer("block5a_expand_activation").output
    input_a = Conv2DTranspose(filters=96, kernel_size=3, strides=2, padding='same', kernel_initializer="he_normal", activation='selu')(input_a)  
    input_a = Conv2DTranspose(filters=96, kernel_size=3, strides=2, padding='same', kernel_initializer="he_normal", activation='selu')(input_a)  
    input_a = DilatedSpatialPyramidPooling(input_a)

    input_b = backbone.get_layer("block4a_expand_activation").output
    input_b = Conv2DTranspose(filters=96, kernel_size=3, strides=2, padding='same', kernel_initializer="he_normal", activation='selu')(input_b)  
    input_b = DilatedSpatialPyramidPooling(input_b)

    input_c = backbone.get_layer("block3a_expand_activation").output
    input_c = DilatedSpatialPyramidPooling(input_c)

    x = Concatenate(axis=-1)([input_a, input_b, input_c])

    x = convolution_block(x)
    x = convolution_block(x)
    x = UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)

    x = Dropout(dropout_ratio)(x)

    x64  = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', kernel_initializer="he_normal", activation='relu')(x) # 64x64x64
    x128 = x # 128x128x16
    x256 = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', kernel_initializer="he_normal", activation='relu')(x)  # 256x256x64

    out1 = Conv2D(num_classes, 1, activation='softmax')(x256) # 256x256x2
    out2 = Conv2D(num_classes, 1, activation='softmax')(x128) # 128x128x2
    out3 = Conv2D(num_classes, 1, activation='softmax')(x64)  # 64x64x2

    return Model(backbone.input, [out1,out2,out3], name='MNet')


## create full model
def getModel(pre_trained_model=None):
    network = getMNet(input_shape=(128,128,23))
    if pre_trained_model is not None:
        model.set_weights(pre_trained_model.get_weights())
    network.trainable = True
    opt = tf.keras.optimizers.Adam(learning_rate=4e-4)
    network.compile(loss=[FocalLoss(), IOULoss()], optimizer=opt, metrics=[])

    return network

