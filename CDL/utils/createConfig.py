import configparser
import logging

if __name__ == '__main__':


    config = configparser.ConfigParser()

    # GENERAL
    config['GENERAL'] = {'logging level': logging.INFO,
                         'save models': True,
                         'calc knowledge quotient': True}

    # DATASET
    config['DATASET'] = {'name': 'CIFAR10',
                         'subset fraction': 1.0,
                         'number classes': 10}

    # MODEL
    config['MODEL'] = {'from file': True,
                       'filepath': "C:/Users/bha/Documents/CDL/Tensorflow_2.2/python_code/mobilenet_v2-like5__2018-08-26-16-42-57_instance.h5",
                       'pretrained': True,
                       'weightspath': "C:/Users/bha/Documents/CDL/Tensorflow_2.2/python_code/mobilenet_v2-like5__2018-08-26-16-42-57_weights.h5",
                       'pretrained on ImageNet': False,
                       'scale to ImageNet': False}

    # TRAINING
    config['TRAINING_ORIGINAL_MODEL'] = {'batchsize': 32,
                                         'epochs': 10,
                                         'learning rate': 1e-1}

    config['TRAINING_SHUNT_MODEL'] = {'batchsize': 32,
                                      'epochs': 10,
                                      'learning rate': 1e-2}

    # SHUNT
    config['SHUNT'] = {'location': '7,12',
                       'arch': 1,
                       'input shape': '8,8,64',
                       'output shape': '8,8,96',
                       'load featuremaps': True,
                       'featuremapspath': "C:/Users/bha/Documents/CDL/Tensorflow_2.2/python_code/saved/feature_maps/pretrained_CIFAR/",
                       'save featuremaps': True}


    with open('main.cfg', 'w') as configfile:
        config.write(configfile)