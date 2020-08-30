import configparser
import logging

if __name__ == '__main__':


    config = configparser.ConfigParser(allow_no_value=True)
    config.optionxform=str

    # GENERAL
    config['GENERAL'] = {'logging level': logging.INFO,
                         'calc knowledge quotient': False,
                         'train original model': False,
                         'train shunt model': False,
                         'test shunt model': False,
                         'train final model': False,
                         'test fine-tune strategies': False,
                         'test latency': False}

    # DATASET
    config['DATASET'] = {'name': 'CIFAR10',
                         '# names: CIFAR10': None}

    # MODEL
    config['MODEL'] = {'type': 'MobileNetV2',
                       '# types: MobileNetV2, MobileNetV3Small, MobileNetV3Large': None,
                       'from file': False,
                       'filepath': "",
                       'pretrained': False,
                       'weightspath': "",
                       'scale up input': False,
                       'change stride layers': 2}

    # TRAINING
    config['TRAINING_ORIGINAL_MODEL'] = {'batchsize': 16,
                                         'epochs first cycle': 100,
                                         'learning rate first cycle': 1e-1,
                                         'epochs second cycle': 100,
                                         'learning rate second cycle': 1e-3}

    config['TRAINING_SHUNT_MODEL'] = {'batchsize': 64,
                                      'epochs first cycle': 100,
                                      'learning rate first cycle': 2e-1,
                                      'epochs second cycle': 100,
                                      'learning rate second cycle': 1e-3}

    config['TRAINING_FINAL_MODEL'] = {'finetune strategy': 'unfreeze_all',
                                      '# strategies: unfreeze_all, unfreeze_shunt, unfreeze_per_epoch_starting_top, unfreeze_per_epoch_starting_shunt': None,
                                      'batchsize': 16,
                                      'epochs first cycle': 100,
                                      'learning rate first cycle': 1e-3,
                                      'epochs second cycle': 100,
                                      'learning rate second cycle': 1e-5}

    # SHUNT
    config['SHUNT'] = {'location': '62,114',
                       'arch': 1,
                       'from file': False,
                       'filepath': "",
                       'pretrained': False,
                       'weightspath': "",
                       'featuremapspath': ""}


    # FINAL MODEL
    config['FINAL_MODEL'] = {'pretrained': False,
                             'weightspath': ""}

    with open('classification.cfg', 'w') as configfile:
        config.write(configfile)