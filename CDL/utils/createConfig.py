import configparser
import logging
import sys

if __name__ == '__main__':

    if len(sys.argv) < 2:
        raise Exception("Please specify your desired config: -c|-s")

    if sys.argv[1] == '-c':

        config = configparser.ConfigParser(allow_no_value=True)
        config.optionxform=str

        # GENERAL
        config['GENERAL'] = {'train_original_model': False,
                            'calc_knowledge_quotient': False,
                            'train_shunt_model': False,
                            'test_shunt_model': False,
                            'train_final_model': False,
                            'test_fine-tune_strategies': False,
                            'test_latency': False}

        # DATASET
        config['DATASET'] = {'name': 'CIFAR10',
                             'path': '',
                             '# names: CIFAR10': None}

        # MODEL
        config['MODEL'] = {'type': 'MobileNetV2',
                        '# types: MobileNetV2, MobileNetV3Small, MobileNetV3Large': None,
                        'from_file': False,
                        'filepath': "",
                        'pretrained': False,
                        'weightspath': "",
                        'input_image_size': 32,
                        'change_stride_layers': 2}

        # SHUNT
        config['SHUNT'] = {'location': '62,114',
                        'arch': 1,
                        'from file': False,
                        'filepath': "",
                        'pretrained': False,
                        'weightspath': "",
                        'featuremapspath': ""}

        # TRAINING
        config['TRAINING_ORIGINAL_MODEL'] = {'batchsize': 16,
                                            'epochs_first_cycle': 150,
                                            'learning_rate_first_cycle': 1e-1,
                                            'epochs_second_cycle': 150,
                                            'learning_rate_second_cycle': 1e-3}

        config['TRAINING_SHUNT_MODEL'] = {'batchsize': 64,
                                        'epochs_first_cycle': 50,
                                        'learning_rate_first_cycle': 1e-1,
                                        'epochs_second_cycle': 500,
                                        'learning_rate_second_cycle': 1e-3}

        config['TRAINING_FINAL_MODEL'] = {'finetune_strategy': 'unfreeze_all',
                                        '# strategies: unfreeze_all, unfreeze_shunt, unfreeze_per_epoch_starting_top, unfreeze_per_epoch_starting_shunt': None,
                                        'batchsize': 8,
                                        'epochs_first_cycle': 50,
                                        'learning_rate_first_cycle': 1e-3,
                                        'epochs_second_cycle': 50,
                                        'learning_rate_second_cycle': 1e-5}

        # FINAL MODEL
        config['FINAL_MODEL'] = {'pretrained': False,
                                'weightspath': ""}

        with open('classification.cfg', 'w') as configfile:
            config.write(configfile)

    elif sys.argv[1] == '-s':
        
        config = configparser.ConfigParser(allow_no_value=True)
        config.optionxform=str

        # GENERAL
        config['GENERAL'] = {'train_original_model': False,
                            'calc_knowledge_quotient': False,
                            'train_shunt_model': False,
                            'test_shunt_model': False,
                            'train_final_model': False,
                            'test_fine-tune_strategies': False,
                            'test_latency': False}

        # DATASET
        config['DATASET'] = {'name': 'VOC2012',
                             '# names: VOC2012': None,
                             'path': 'C:/Users/bha/Documents/CDL/Tensorflow_2.2/python_code/saved/datasets/pascal_voc_seg/pascal_voc_seg/VOCdevkit/VOC2012_refactored'}

        # MODEL
        config['MODEL'] = {'backbone': 'MobileNetV2',
                        '# types: MobileNetV2, MobileNetV3': None,
                        'from_file': False,
                        'filepath': "",
                        'pretrained': False,
                        'weightspath': "",
                        '# write "imagenet" for loading keras pretrained model': None,
                        'scale_up_input': False,
                        'change stride layers': 2}

        # SHUNT
        config['SHUNT'] = {'location': '62,114',
                        'arch': 1,
                        'from_file': False,
                        'filepath': "",
                        'pretrained': False,
                        'weightspath': "",
                        'featuremapspath': ""}

        # TRAINING
        config['TRAINING_ORIGINAL_MODEL'] = {'batchsize': 16,
                                            'epochs_first_cycle': 150,
                                            'learning_rate_first_cycle': 1e-1,
                                            'epochs_second_cycle': 150,
                                            'learning_rate_second_cycle': 1e-3}

        config['TRAINING_SHUNT_MODEL'] = {'batchsize': 64,
                                        'epochs_first_cycle': 50,
                                        'learning_rate_first_cycle': 1e-1,
                                        'epochs_second_cycle': 500,
                                        'learning_rate_second_cycle': 1e-3}

        config['TRAINING_FINAL_MODEL'] = {'finetune_strategy': 'unfreeze_all',
                                        '# strategies: unfreeze_all, unfreeze_shunt, unfreeze_per_epoch_starting_top, unfreeze_per_epoch_starting_shunt': None,
                                        'batchsize': 8,
                                        'epochs_first_cycle': 50,
                                        'learning_rate_first_cycle': 1e-3,
                                        'epochs_second_cycle': 50,
                                        'learning_rate_second_cycle': 1e-5}

        # FINAL MODEL
        config['FINAL_MODEL'] = {'pretrained': False,
                                'weightspath': ""}

        with open('segmentation.cfg', 'w') as configfile:
            config.write(configfile)

    else:
        raise Exception("Please specify your desired config: -c|-s")

    