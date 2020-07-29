from keras.datasets import cifar10
from keras.utils import to_categorical


def load_and_preprocess_CIFAR10():
        
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_test = x_test.astype('float32')  #argmax_ch
    x_train = x_train.astype('float32')  #argmax_ch

    def ch_wise_normalization(X_type, ch):
        mean_ch = X_type[:, :, :, ch].mean()
        std_ch = X_type[:, :, :, ch].std()
        X_type[:, :, :, ch] = (X_type[:, :, :, ch] - mean_ch) / std_ch
        return X_type[:, :, :, ch]

    x_test[:, :, :, 0]  = ch_wise_normalization(x_test, 0)
    x_test[:, :, :, 1]  = ch_wise_normalization(x_test, 1)
    x_test[:, :, :, 2]  = ch_wise_normalization(x_test, 2)
    x_train[:, :, :, 0]  = ch_wise_normalization(x_train, 0)
    x_train[:, :, :, 1]  = ch_wise_normalization(x_train, 1)
    x_train[:, :, :, 2]  = ch_wise_normalization(x_train, 2)

    y_test  = to_categorical(y_test, 10)
    y_train = to_categorical(y_train, 10)

    return (x_train, y_train), (x_test, y_test)