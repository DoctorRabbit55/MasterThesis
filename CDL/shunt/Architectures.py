from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add
from keras import Model
from keras import regularizers


def createArch1(input_shape, output_shape):

    input_net = Input(shape=input_shape)
    x = input_net

    kernel_size = (3,3)

    x = Conv2D(192, kernel_size=kernel_size, strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    x = Conv2D(192, kernel_size=kernel_size, strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    x = Conv2D(64, kernel_size=kernel_size, strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = Conv2D(192, kernel_size=kernel_size, strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    x = Conv2D(192, kernel_size=kernel_size, strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    x = ReLU(6.)(x)
    x = Conv2D(output_shape[-1], kernel_size=kernel_size, strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)

    model = Model(inputs=input_net, outputs=x, name='shunt')

    return model

def createArch4(input_shape, output_shape):

    input_net = Input(shape=input_shape)
    x = input_net

    kernel_size = (3,3)

    #x = Conv2D(input_shape[-1], kernel_size=kernel_size, strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    #x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    #x = ReLU(6.)(x)
    #x = Conv2D(input_shape[-1], kernel_size=kernel_size, strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    #x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)
    #x = ReLU(6.)(x)
    x = Conv2D(output_shape[-1], kernel_size=kernel_size, strides=(1,1), padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999)(x)

    model = Model(inputs=input_net, outputs=x, name='shunt')

    return model

def createShunt(input_shape, output_shape, arch=1):

    assert(arch > 0 and arch < 6)

    model_shunt = None

    if arch == 1:
        model_shunt = createArch1(input_shape, output_shape)
    if arch == 4:
        model_shunt = createArch4(input_shape, output_shape)

    return model_shunt


if __name__ == '__main__':

    # calc MAccs of different architectures

    pass