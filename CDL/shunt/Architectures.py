from keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, Add
from keras import Model
from keras import regularizers

import numpy as np

def createArch1(input_shape, output_shape, num_stride_layers):

    input_net = Input(shape=input_shape)
    x = input_net

    x = Conv2D(192, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    if num_stride_layers > 1:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    else:
        x = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
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
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    x = Conv2D(output_shape[-1], kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)

    model = Model(inputs=input_net, outputs=x, name='shunt')

    return model

def createArch4(input_shape, output_shape, num_stride_layers):

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
    x = Conv2D(output_shape[-1], kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)

    model = Model(inputs=input_net, outputs=x, name='shunt')

    return model

def createShunt(input_shape, output_shape, arch=1):

    assert(arch > -1 and arch < 10)
    assert(np.sqrt(input_shape[0] / output_shape[0] == int(np.sqrt(input_shape[0] / output_shape[0])))

    model_shunt = None
    num_stride_layers = np.sqrt(input_shape[0] / output_shape[0])

    # list of maximum strides for each architecture
    max_stride_list = {1:2, 4:1}

    if max_stride_list[arch] < num_stride_layers:
        raise Exception("Chosen shunt architecture does not support {} many stride layers. Only {} are supported.".format(num_stride_layers, max_stride_list[arch]))

    if arch == 1:
        model_shunt = createArch1(input_shape, output_shape, num_stride_layers)
    if arch == 4:
        model_shunt = createArch4(input_shape, output_shape, num_stride_layers)

    return model_shunt


if __name__ == '__main__':

    # calc MAccs of different architectures

    pass