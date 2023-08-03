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
from tensorflow.keras.layers import Reshape, multiply, Permute, Lambda
from tensorflow.keras.activations import sigmoid,softmax

from losses import *


def downBlock(x, filters):
    x1 = Conv2D(filters=filters, kernel_size=3, strides=2, padding='same', kernel_initializer="he_normal")(x)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU()(x1)

    x2 = SeparableConv2D(filters=filters, kernel_size=3, strides=2, padding='same', kernel_initializer="he_normal")(x)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU()(x2)

    x  = x1 + x2
    
    return x


def upBlock(x, filters):
    x1 = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same', kernel_initializer="he_normal")(x)  # WxHxF
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU()(x1)

    x2 = UpSampling2D()(x)
    x2 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', kernel_initializer="he_normal")(x2)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU()(x2)

    x  = x1 + x2
    
    return x


def getWaveNet(input_shape=(128,128,21), num_classes=2, dropout_ratio=0.2):
    num_block = 6

    input = Input(shape=input_shape)
    
    x = Conv2D(filters=128, kernel_size=1, strides=1, padding='same', activation='swish', kernel_initializer="he_normal")(input)

    for i in range(num_block):
        x_init = x
        x = upBlock(x_init, filters=256)
        x = downBlock(x, filters=128)
        x = downBlock(x, filters=64)
        x = upBlock(x, filters=128)

        
        x = x_init + x

    x = Dropout(dropout_ratio)(x)

    output = Conv2D(num_classes, 1, activation='softmax', kernel_initializer="he_normal")(x) # 128x128x2

    return Model(input, output, name='WaveNet')


## create full model
def getModel(pre_trained_model=None):
    network = getWaveNet(input_shape=(128,128,21))
    if pre_trained_model is not None:
        model.set_weights(pre_trained_model.get_weights())
    network.trainable = True
    opt = tf.keras.optimizers.Adam(learning_rate=4e-4)
    network.compile(loss=[FocalLoss(), IOULoss()], optimizer=opt, metrics=[])
    
    return network

