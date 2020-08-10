import configparser
import logging

if __name__ == '__main__':


    config = configparser.ConfigParser()

    # GENERAL
    config['GENERAL'] = {'logging level': logging.INFO,
                         'calc knowledge quotient': False,
                         'train original model': False,
                         'train shunt model': False,
                         'train final model': False}

    # DATASET
    config['DATASET'] = {'name': 'CIFAR10'}

    # MODEL
    config['MODEL'] = {'type': 'MobileNetV2',
                       'from file': False,
                       'filepath': "",
                       'pretrained': False,
                       'weightspath': "",
                       'scale up input': False,
                       'change stride layers': 2}

    # TRAINING
    config['TRAINING_ORIGINAL_MODEL'] = {'batchsize': 16,
                                         'epochs': 100,
                                         'learning rate': 0.1}

    config['TRAINING_SHUNT_MODEL'] = {'batchsize': 64,
                                      'epochs': 100,
                                      'learning rate': 0.2}

    config['TRAINING_FINAL_MODEL'] = {'batchsize': 16,
                                      'epochs': 100,
                                      'learning rate': 1e-3}

    # SHUNT
    config['SHUNT'] = {'location': '7,12',
                       'arch': 1,
                       'input shape': '8,8,64',
                       'output shape': '8,8,96',
                       'from file': False,
                       'filepath': "",
                       'pretrained': False,
                       'weightspath': "",
                       'load featuremaps': True,
                       'featuremapspath': "",
                       'save featuremaps': True}


    # FINAL MODEL
    config['FINAL_MODEL'] = {'pretrained': False,
                             'weightspath': ""}

    with open('classification.cfg', 'w') as configfile:
        config.write(configfile)