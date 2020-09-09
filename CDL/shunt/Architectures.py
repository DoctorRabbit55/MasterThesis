from keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, Add
from keras import Model
from keras import regularizers

from CDL.models.MobileNet_v3 import _se_block

import numpy as np

def createArch1(input_shape, output_shape, num_stride_layers, use_se):

    input_net = Input(shape=input_shape)
    x = input_net

    x = Conv2D(192, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    if num_stride_layers > 0:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    else:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    if use_se: x = _se_block(x, filters=192, se_ratio=0.25, prefix="shunt_1_")
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    x = Conv2D(64, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = Conv2D(192, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    if num_stride_layers > 1:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    else:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    if use_se: x = _se_block(x, filters=192, se_ratio=0.25, prefix="shunt_2_")
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    x = Conv2D(output_shape[-1], kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)

    model = Model(inputs=input_net, outputs=x, name='shunt')

    return model

def createArch4(input_shape, output_shape, num_stride_layers, use_se):

    input_net = Input(shape=input_shape)
    x = input_net

    x = Conv2D(128, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    if num_stride_layers > 1:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    else:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    if use_se: x = _se_block(x, filters=128, se_ratio=0.25, prefix="shunt_")
    x = Conv2D(output_shape[-1], kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)

    model = Model(inputs=input_net, outputs=x, name='shunt')

    return model

def createArch5(input_shape, output_shape, num_stride_layers, use_se):

    input_net = Input(shape=input_shape)
    x = input_net

    x = Conv2D(192, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    if num_stride_layers > 0:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    else:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    if use_se: x = _se_block(x, filters=192, se_ratio=0.25, prefix="shunt_0_")
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    x = Conv2D(64, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = Conv2D(192, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    if num_stride_layers > 1:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    else:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    if use_se: x = _se_block(x, filters=192, se_ratio=0.25, prefix="shunt_1_")
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    x = Conv2D(64, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = Conv2D(192, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    if num_stride_layers > 2:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    else:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    if use_se: x = _se_block(x, filters=192, se_ratio=0.125, prefix="shunt_2_")
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    x = Conv2D(output_shape[-1], kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)


    model = Model(inputs=input_net, outputs=x, name='shunt')

    return model

def createShunt(input_shape, output_shape, arch, use_se=False):

    assert(arch in [1,4,5])
    assert(np.log2(input_shape[1] / output_shape[1]) == int(np.log2(input_shape[1] / output_shape[1])))

    model_shunt = None
    num_stride_layers = np.log2(input_shape[1] / output_shape[1])

    # list of maximum strides for each architecture
    max_stride_list = {1:2, 4:1, 5:3}

    if max_stride_list[arch] < num_stride_layers:
        raise Exception("Chosen shunt architecture does not support {} many stride layers. Only {} are supported.".format(num_stride_layers, max_stride_list[arch]))

    if arch == 1:
        model_shunt = createArch1(input_shape, output_shape, num_stride_layers, use_se)
    if arch == 4:
        model_shunt = createArch4(input_shape, output_shape, num_stride_layers, use_se)
    if arch == 5:
        model_shunt = createArch5(input_shape, output_shape, num_stride_layers, use_se)

    return model_shunt


if __name__ == '__main__':

    # calc MAccs of different architectures

    pass