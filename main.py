import sys
import os
import configparser
import logging
import time
from shutil import copyfile

from pathlib import Path
import numpy as np

from CDL.models.MobileNet_v2 import create_mobilenet_v2
from CDL.models.MobileNet_v3 import create_mobilenet_v3
from CDL.shunt import Architectures
from CDL.shunt.create_shunt_trainings_model import create_shunt_trainings_model
from CDL.utils.calculateFLOPS import calculateFLOPs_model, calculateFLOPs_blocks
from CDL.utils.dataset_utils import *
from CDL.utils.get_knowledge_quotients import get_knowledge_quotients
from CDL.utils.generic_utils import *
from CDL.utils.keras_utils import extract_feature_maps, modify_model, identify_residual_layer_indexes
from CDL.utils.custom_callbacks import UnfreezeLayersCallback, LearningRateSchedulerCallback
from CDL.utils.custom_generators import Imagenet_generator, Imagenet_train_shunt_generator

import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import keras
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


from matplotlib import pyplot as plt

from sklearn.metrics import classification_report

def mean_squared_diff(y_true, y_pred):
    #y_pred = ops.convert_to_tensor_v2(y_pred)
    return keras.backend.sum(math_ops.square(y_pred), axis=-1)


if __name__ == '__main__':

    # READ CONFIG
    config_path = Path(sys.path[0], "config", "classification.cfg")
    config = configparser.ConfigParser()
    config.read(config_path)

    # USER PARAMS

    for i in range(1,len(sys.argv)):
        arg = sys.argv[i]
        group = arg.split('#')[0]
        param = arg.split('#')[1]
        value = arg.split('#')[2]
        config[group][param] = value

    modes = {}
    modes['calc_knowledge_quotients'] = config['GENERAL'].getboolean('calc_knowledge_quotients')
    modes['train_original_model'] = config['GENERAL'].getboolean('train_original_model')
    modes['train_final_model'] = config['GENERAL'].getboolean('train_final_model')
    modes['train_shunt_model'] = config['GENERAL'].getboolean('train_shunt_model')
    modes['test_shunt_model'] = config['GENERAL'].getboolean('test_shunt_model')
    modes['test_fine-tune_strategies'] = config['GENERAL'].getboolean('test_fine-tune_strategies')
    modes['test_latency'] = config['GENERAL'].getboolean('test_latency')
    loglevel = 20

    dataset_name = config['DATASET']['name']
    dataset_path = config['DATASET']['path']

    model_type = config['MODEL']['type']
    number_change_stride_layers = config['MODEL'].getint('change_stride_layers')
    load_model_from_file = config['MODEL'].getboolean('from_file')
    if (load_model_from_file):
        model_file_path = config['MODEL']['filepath']
    else:
        scale_to_imagenet = config['MODEL'].getboolean('scale_to_imagenet')
    input_image_size = config['MODEL'].getint('input_image_size')
    pretrained = config['MODEL'].getboolean('pretrained')
    weights_file_path = config['MODEL']['weightspath']
    
    training_original_model = config['TRAINING_ORIGINAL_MODEL']
    training_shunt_model = config['TRAINING_SHUNT_MODEL']
    training_final_model = config['TRAINING_FINAL_MODEL']

    shunt_params = {}
    shunt_params['arch'] = config['SHUNT'].getint('arch')
    shunt_params['use_se'] = config['SHUNT'].getboolean('use_squeeze_and_excite')
    shunt_params['locations'] = tuple(map(int, config['SHUNT']['location'].split(',')))
    shunt_params['from_file'] = config['SHUNT'].getboolean('from file')
    shunt_params['filepath'] = config['SHUNT']['filepath']
    shunt_params['pretrained'] = config['SHUNT'].getboolean('pretrained')
    shunt_params['weightspath'] = config['SHUNT']['weightspath']
    shunt_params['featuremapspath'] = config['SHUNT']['featuremapspath']

    final_model_params = {}
    final_model_params['pretrained'] = config['FINAL_MODEL'].getboolean('pretrained')
    final_model_params['weightspath'] = config['FINAL_MODEL']['weightspath']

    # init logging
    folder_name_logging = Path(sys.path[0], "log", time.strftime("%Y%m%d"), time.strftime("%H_%M_%S"))
    Path(folder_name_logging).mkdir(parents=True, exist_ok=True)
    log_file_name = Path(folder_name_logging, "output.log")
    logging.basicConfig(filename=log_file_name, level=loglevel , format='%(message)s')
    logger = logging.getLogger(__name__)

    # prepare data
    x_train = y_train = x_test = y_test = None
    datagen_train = datagen_val = None
    flow_from_directory = False
    input_shape = None

    with open( Path(folder_name_logging, "config.cfg"), 'w') as configfile:
        config.write(configfile)

    if dataset_name == 'CIFAR10':

        (x_train, y_train), (x_test, y_test) = load_and_preprocess_CIFAR10()
        input_shape = (32,32,3)
        num_classes = 10
        len_train_data = 50000
        len_val_data = 10000
        feature_maps_fit_in_ram = True

        datagen_train = ImageDataGenerator(
            featurewise_center=False, 
            featurewise_std_normalization=False, 
            rotation_range=0.0,
            width_shift_range=0.2, 
            height_shift_range=0.2, 
            vertical_flip=False,
            horizontal_flip=True)
        
        datagen_val = ImageDataGenerator(
            featurewise_center=False, 
            featurewise_std_normalization=False, 
            rotation_range=0.0,
            width_shift_range=0.0, 
            height_shift_range=0.0, 
            vertical_flip=False,
            horizontal_flip=False)
        

        print('CIFAR10 was loaded successfully!')


    if dataset_name == 'imagenet':

        dataset_val_image_path = Path(dataset_path, "val", "images")
        dataset_train_image_path = Path(dataset_path, "train")
        dataset_ground_truth_file_path = Path(dataset_path, "val", "val.txt")

        num_classes = 1000
        input_shape = (224,224,3)
        len_train_data = 1281167
        len_val_data = 50000

        feature_maps_fit_in_ram = False

        datagen_val = Imagenet_generator(dataset_val_image_path, dataset_ground_truth_file_path, shuffle=False)

        datagen_train = ImageDataGenerator(
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=0.0,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

        print('Imagenet was loaded successfully!')

    # load/create model
    model_original = None

    if model_type == 'MobileNetV2':
        if load_model_from_file:
            model_original = keras.models.load_model(model_file_path)
        elif weights_file_path == 'imagenet':
            model_original = create_mobilenet_v2(is_pretrained=True, num_classes=num_classes, input_shape=input_shape, mobilenet_shape=(input_image_size,input_image_size,3), num_change_strides=number_change_stride_layers)
        else:
            model_original = create_mobilenet_v2(is_pretrained=False, num_classes=num_classes, input_shape=input_shape, mobilenet_shape=(input_image_size,input_image_size,3), num_change_strides=number_change_stride_layers)


    if 'MobileNetV3' in model_type:

        is_small = True
        if model_type[11:] == 'Large':
            is_small = False

        if load_model_from_file:
            model_original = keras.models.load_model(model_file_path)
        elif weights_file_path == 'imagenet':
            model_original = create_mobilenet_v3(is_pretrained=True, num_classes=num_classes, is_small=is_small, input_shape=input_shape, mobilenet_shape=(input_image_size,input_image_size,3), num_change_strides=number_change_stride_layers)           
        else:
            model_original = create_mobilenet_v3(is_pretrained=False, num_classes=num_classes, is_small=is_small, input_shape=input_shape, mobilenet_shape=(input_image_size,input_image_size,3), num_change_strides=number_change_stride_layers)

    if pretrained:
        model_original.load_weights(weights_file_path)
        print('Weights loaded successfully!')


    batch_size_original = training_original_model.getint('batchsize')
    epochs_first_cycle_original = training_original_model.getint('epochs_first_cycle')
    epochs_second_cycle_original = training_original_model.getint('epochs_second_cycle')
    epochs_original = epochs_first_cycle_original + epochs_second_cycle_original
    learning_rate_first_cycle_original = training_original_model.getfloat('learning_rate_first_cycle')
    learning_rate_second_cycle_original = training_original_model.getfloat('learning_rate_second_cycle')

    model_original.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=learning_rate_first_cycle_original, momentum=0.9, decay=0.0), metrics=[keras.metrics.categorical_crossentropy, 'accuracy'])

    logging.info('')
    logging.info('#######################################################################################################')
    logging.info('########################################### ORIGINAL MODEL ############################################')
    logging.info('#######################################################################################################')
    logging.info('')
    model_original.summary(print_fn=logger.info, line_length=150)
    print('{} created successfully!'.format(model_type))

    flops_original = calculateFLOPs_model(model_original)

    callback_checkpoint = keras.callbacks.ModelCheckpoint(str(Path(folder_name_logging, "original_model_weights.h5")), save_best_only=True, monitor='val_accuracy', mode='max', save_weights_only=True)
    callback_learning_rate = LearningRateSchedulerCallback(epochs_first_cycle=epochs_first_cycle_original, learning_rate_second_cycle=learning_rate_second_cycle_original)

    if modes['train_original_model']:
        print('Train original model:')
        if dataset_name == 'imagenet':
            history_original = model_original.fit(datagen_train.flow_from_directory(dataset_train_image_path, shuffle=True, target_size=(224,224), batch_size=batch_size_original), epochs=epochs_original, validation_data=(x_test, y_test), verbose=1, callbacks=[callback_checkpoint, callback_learning_rate])
        elif dataset_name == 'CIFAR10':
            history_original = model_original.fit(datagen_train.flow(x_train, y_train, batch_size=batch_size_original), epochs=epochs_original, validation_data=(x_test, y_test), verbose=1, callbacks=[callback_checkpoint, callback_learning_rate])

        model_original.load_weights(str(Path(folder_name_logging, "original_model_weights.h5")))
        #save_history_plot(history_original, "original", folder_name_logging)

    # test original model
    '''
    print('Test original model')
    if dataset_name == 'imagenet':
        val_loss_original, val_entropy_original, val_acc_original = model_original.evaluate(datagen_val, verbose=1, use_multiprocessing=True, workers=32, max_queue_size=64)
    elif dataset_name == 'CIFAR10':
        val_loss_original, val_entropy_original, val_acc_original = model_original.evaluate(x_test, y_test, verbose=1)
    print('Loss: {:.5f}'.format(val_loss_original))
    print('Entropy: {:.5f}'.format(val_entropy_original))
    print('Accuracy: {:.4f}'.format(val_acc_original))
    '''
    if modes['calc_knowledge_quotients']:
        if dataset_name == 'imagenet':
            know_quot = get_knowledge_quotients(model=model_original, datagen=datagen_val, val_acc_model=val_acc_original)
        elif dataset_name == 'CIFAR10':
            know_quot = get_knowledge_quotients(model=model_original, datagen=(x_test, y_test), val_acc_model=val_acc_original)

        logging.info('')
        logging.info('################# RESULT ###################')
        logging.info('')
        logging.info('Original model: loss: {:.5f}, acc: {:.5f}'.format(val_loss_original, val_acc_original))
        logging.info('')
        for (residual_idx, end_idx, value) in know_quot:
            logging.info("Block starts with: {}, location: {}".format(model_original.get_layer(index=residual_idx+1).name, residual_idx+1))
            logging.info("Block ends with: {}, location: {}".format(model_original.get_layer(index=end_idx).name, end_idx))   
            logging.info("Block knowledge quotient: {}\n".format(value)) 
        exit()


    logging.info('')
    logging.info('#######################################################################################################')
    logging.info('############################################ SHUNT MODEL ##############################################')
    logging.info('#######################################################################################################')
    logging.info('')

    loc1 = shunt_params['locations'][0]
    loc2 = shunt_params['locations'][1]
    
    if shunt_params['from_file']:
        model_shunt = keras.models.load_model(shunt_params['filepath'])
        print('Shunt model loaded successfully!')
    else:
        input_shape_shunt = model_original.get_layer(index=loc1).input_shape[1:]
        output_shape_shunt = model_original.get_layer(index=loc2).output_shape[1:]
        model_shunt = Architectures.createShunt(input_shape_shunt, output_shape_shunt, arch=shunt_params['arch'], use_se=shunt_params['use_se'])
    
    model_shunt.summary(print_fn=logger.info, line_length=150)

    keras.models.save_model(model_shunt, Path(folder_name_logging, "shunt_model.h5"))
    logging.info('')
    logging.info('Shunt model saved to {}'.format(folder_name_logging))
    
    batch_size_shunt = training_shunt_model.getint('batchsize')
    epochs_first_cycle_shunt = training_shunt_model.getint('epochs_first_cycle')
    epochs_second_cycle_shunt = training_shunt_model.getint('epochs_second_cycle')
    epochs_shunt = epochs_first_cycle_shunt + epochs_second_cycle_shunt
    learning_rate_first_cycle_shunt = training_shunt_model.getfloat('learning_rate_first_cycle')
    learning_rate_second_cycle_shunt = training_shunt_model.getfloat('learning_rate_second_cycle')

    # if feature maps do not fit in memory, feature map extracting model has to be used
    #if dataset_name == 'imagenet':
    model_training_shunt = create_shunt_trainings_model(model_original, model_shunt, (loc1, loc2))
    model_training_shunt.compile(loss=mean_squared_diff, optimizer=keras.optimizers.Adam(learning_rate=learning_rate_first_cycle_shunt, decay=0.0), metrics=None)

    if shunt_params['pretrained']:
        if dataset_name == 'imagenet':
            model_training_shunt.load_weights(shunt_params['weightspath'])
            print('Shunt weights loaded successfully!')
        elif dataset_name == 'CIFAR10':
            model_shunt.load_weights(shunt_params['weightspath'])
            print('Shunt weights loaded successfully!')

    flops_shunt = calculateFLOPs_model(model_shunt)

    model_shunt.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(learning_rate=learning_rate_first_cycle_shunt, decay=0.0), metrics=[keras.metrics.MeanSquaredError()])

    callback_checkpoint = keras.callbacks.ModelCheckpoint(str(Path(folder_name_logging, "shunt_model_weights.h5")), save_best_only=True, monitor='val_loss', mode='min', save_weights_only=True)
    callback_learning_rate = LearningRateSchedulerCallback(epochs_first_cycle=epochs_first_cycle_shunt, learning_rate_second_cycle=learning_rate_second_cycle_shunt)

    # Feature maps

    if modes['test_shunt_model'] or modes['train_shunt_model']:

        fm1_train = fm2_train = fm1_test = fm2_test = None
        
        if dataset_name == 'imagenet':

            if modes['train_shunt_model']:
                print('Train shunt model:')
                train_dummy_data = [None] * len_train_data
                datagen_val_dummy = Imagenet_train_shunt_generator(dataset_val_image_path, dataset_ground_truth_file_path, shuffle=False)

                history_shunt = model_training_shunt.fit(zip(datagen_train.flow_from_directory(dataset_train_image_path, class_mode=None, shuffle=True, target_size=(224,224), batch_size=batch_size_shunt), train_dummy_data), epochs=epochs_shunt, steps_per_epoch=len_train_data//batch_size_shunt, validation_data=datagen_val_dummy, verbose=1, callbacks=[callback_checkpoint, callback_learning_rate],
                                                         use_multiprocessing=True, workers=32, max_queue_size=64)

                model_training_shunt.load_weights(str(Path(folder_name_logging, "shunt_model_weights.h5")))

            if modes['test_shunt_model']:
                print('Test shunt model')
                datagen_val_dummy = Imagenet_train_shunt_generator(dataset_val_image_path, dataset_ground_truth_file_path, shuffle=False)
                val_loss_shunt, val_acc_shunt, = model_training_shunt.evaluate(datagen_val_dummy, verbose=1)
                print('Loss: {:.5f}'.format(val_loss_shunt))
                print('Accuracy: {:.5f}'.format(val_acc_shunt))

        elif dataset_name == 'CIFAR10':

            if modes['train_shunt_model']:
                print('Train shunt model:')
                train_dummy_data = [None] * len_train_data
                val_dummy_data = [None] * len_val_data

                history_shunt = model_training_shunt.fit(x_train, train_dummy_data, batch_size=batch_size_shunt, epochs=epochs_shunt, validation_data=(x_test, val_dummy_data), verbose=1, callbacks=[callback_checkpoint, callback_learning_rate],
                                                         use_multiprocessing=False, workers=1, max_queue_size=64)

                model_training_shunt.load_weights(str(Path(folder_name_logging, "shunt_model_weights.h5")))

            if modes['test_shunt_model']:
                print('Test shunt model')
                val_dummy_data = np.zeros((len_val_data,))
                val_loss_shunt, val_acc_shunt, = model_training_shunt.evaluate(x_test, val_dummy_data, verbose=1)
                print('Loss: {:.5f}'.format(val_loss_shunt))
                print('Accuracy: {:.5f}'.format(val_acc_shunt))
            '''
            if os.path.isfile(Path(shunt_params['featuremapspath'], "fm1_train_{}_{}.npy".format(loc1, loc2))):
                fm1_train = np.load(Path(shunt_params['featuremapspath'], "fm1_train_{}_{}.npy".format(loc1, loc2)))
                fm2_train = np.load(Path(shunt_params['featuremapspath'], "fm2_train_{}_{}.npy".format(loc1, loc2)))
                fm1_test = np.load(Path(shunt_params['featuremapspath'], "fm1_test_{}_{}.npy".format(loc1, loc2)))
                fm2_test = np.load(Path(shunt_params['featuremapspath'], "fm2_test_{}_{}.npy".format(loc1, loc2)))
                print('Feature maps loaded successfully!')
            else:              
                print('Feature maps extracting started:')
                (fm1_train, fm2_train)  = extract_feature_maps(model_original, x_train, [loc1-1, loc2]) # -1 since we need the input of the layer
                (fm1_test, fm2_test) = extract_feature_maps(model_original, x_test, [loc1-1, loc2]) # -1 since we need the input of the layer

                np.save(Path(shunt_params['featuremapspath'], "fm1_train_{}_{}".format(loc1, loc2)), fm1_train)
                np.save(Path(shunt_params['featuremapspath'], "fm2_train_{}_{}".format(loc1, loc2)), fm2_train)
                np.save(Path(shunt_params['featuremapspath'], "fm1_test_{}_{}".format(loc1, loc2)), fm1_test)
                np.save(Path(shunt_params['featuremapspath'], "fm2_test_{}_{}".format(loc1, loc2)), fm2_test)

                logging.info('')
                logging.info('Featuremaps saved to {}'.format(shunt_params['featuremapspath']))

            data_train = (fm1_train, fm2_train)
            data_val = (fm1_test, fm2_test)

            if modes['train_shunt_model']:
                print('Train shunt model:')
                history_shunt = model_shunt.fit(data_train[0], y=data_train[1], batch_size=batch_size_shunt, epochs=epochs_shunt, validation_data=data_val, verbose=1, callbacks=[callback_checkpoint, callback_learning_rate])
                #save_history_plot(history_shunt, "shunt", folder_name_logging)
                model_shunt.load_weights(str(Path(folder_name_logging, "shunt_model_weights.h5")))

            if modes['test_shunt_model']:
                print('Test shunt model')
                val_loss_shunt, val_acc_shunt, = model_shunt.evaluate(fm1_test, fm2_test, verbose=1)
                print('Loss: {:.5f}'.format(val_loss_shunt))
                print('Accuracy: {:.5f}'.format(val_acc_shunt))

            '''

        fm1_test = fm1_train = fm2_test = fm2_train = None


    model_final = modify_model(model_original, layer_indexes_to_delete=range(loc1, loc2+1), shunt_to_insert=model_shunt) # +1 needed because of the way range works
    
    keras.models.save_model(model_final, Path(folder_name_logging, "final_model.h5"))
    logging.info('')
    logging.info('Final model saved to {}'.format(folder_name_logging))

    flops_final = calculateFLOPs_model(model_final)

    # log FLOPs
    logging.info('')
    logging.info('#######################################################################################################')
    logging.info('################################################ FLOPS ################################################')
    logging.info('#######################################################################################################')
    logging.info('')
    logging.info('Original model: {}'.format(flops_original))
    logging.info('Shunt model: {}'.format(flops_shunt))
    logging.info('Final model: {}'.format(flops_final))

    reduction = 100*(flops_original['total']-flops_final['total']) / flops_original['total']
    logging.info('')
    logging.info('FLOPs got reduced by {:.2f}%!'.format(reduction))

    batch_size_final = training_final_model.getint('batchsize')
    epochs_first_cycle_final = training_final_model.getint('epochs_first_cycle')
    epochs_second_cycle_final = training_final_model.getint('epochs_second_cycle')
    epochs_final = epochs_first_cycle_final + epochs_second_cycle_final
    learning_rate_first_cycle_final = training_final_model.getfloat('learning_rate_first_cycle')
    learning_rate_second_cycle_final = training_final_model.getfloat('learning_rate_second_cycle')
    
    logging.info('')
    logging.info('#######################################################################################################')
    logging.info('############################################ FINAL MODEL ##############################################')
    logging.info('#######################################################################################################')
    logging.info('')
    model_final.summary(print_fn=logger.info, line_length=150)

    callback_checkpoint = keras.callbacks.ModelCheckpoint(str(Path(folder_name_logging, "final_model_weights.h5")), save_best_only=True, monitor='val_accuracy', mode='max', save_weights_only=True)
    callback_learning_rate = LearningRateSchedulerCallback(epochs_first_cycle=epochs_first_cycle_final, learning_rate_second_cycle=learning_rate_second_cycle_final)
    callbacks = [callback_checkpoint]

    model_final.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=learning_rate_first_cycle_final, momentum=0.9, decay=0.0, nesterov=False), metrics=[keras.metrics.categorical_crossentropy, 'accuracy'])

    print('Test shunt inserted model')
    if dataset_name == 'imagenet':
        val_loss_inserted, val_entropy_inserted, val_acc_inserted = model_final.evaluate(datagen_val, verbose=1)
    elif dataset_name == 'CIFAR10':
        val_loss_inserted, val_entropy_inserted, val_acc_inserted = model_final.evaluate(x_test, y_test, verbose=1)
 
    print('Loss: {:.5f}'.format(val_loss_inserted))
    print('Entropy: {:.5f}'.format(val_entropy_inserted))
    print('Accuracy: {:.4f}'.format(val_acc_inserted))

    if final_model_params['pretrained']:
        model_final.load_weights(final_model_params['weightspath'])
        print('Weights for final model loaded successfully!')
        print('Test shunt inserted model with loaded weights')
        if dataset_name == 'imagenet':
            val_loss_inserted, val_entropy_inserted, val_acc_inserted = model_final.evaluate(datagen_val, verbose=1)
        elif dataset_name == 'CIFAR10':
            val_loss_inserted, val_entropy_inserted, val_acc_inserted = model_final.evaluate(x_test, y_test, verbose=1)        
        print('Loss: {:.5f}'.format(val_loss_inserted))
        print('Entropy: {:.5f}'.format(val_entropy_inserted))
        print('Accuracy: {:.4f}'.format(val_acc_inserted))
        
    if modes['test_fine-tune_strategies']:

        strategies = [ 'unfreeze_after_shunt', 'unfreeze_from_shunt', 'unfreeze_all']

        logging.info('')
        logging.info('#######################################################################################################')
        logging.info('############################################# FINE-TUNING #############################################')
        logging.info('#######################################################################################################')
        logging.info('')

        old_weights = model_final.get_weights()

        for strategy in strategies:
            
            model_final.set_weights(old_weights)

            # unfreeze layers
            for i, layer in enumerate(model_final.layers):
                if strategy == 'unfreeze_all':
                    pass
                elif strategy == 'unfreeze_from_shunt':
                    if i < loc1:
                        layer.trainable = False
                elif strategy == 'unfreeze_after_shunt':
                    if i < loc1 + len(model_shunt.layers) - 1:
                        layer.trainable = False

            model_final.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=learning_rate_first_cycle_final, momentum=0.9, decay=0.0, nesterov=False), metrics=[keras.metrics.categorical_crossentropy, 'accuracy'])

            callback_checkpoint = keras.callbacks.ModelCheckpoint(str(Path(folder_name_logging, "final_model_{}_weights.h5".format(strategy))), save_best_only=True, monitor='val_accuracy', mode='max', save_weights_only=True)
            callbacks = [callback_checkpoint, callback_learning_rate]

            print('Train final model with strategy {}:'.format(strategy))
            train_start = time.process_time()
            if dataset_name == 'imagenet':
                history_final = model_final.fit(datagen_train.flow_from_directory(dataset_train_image_path, shuffle=True, target_size=(224,224), batch_size=batch_size_final), epochs=epochs_final, validation_data=(x_test, y_test), verbose=1, callbacks=callbacks)
            elif dataset_name == 'CIFAR10':
               history_final = model_final.fit(datagen_train.flow(x_train, y_train, batch_size=batch_size_final), epochs=epochs_final, validation_data=(x_test, y_test), verbose=1, callbacks=callbacks)
            train_stop = time.process_time()
            model_final.load_weights(str(Path(folder_name_logging, "final_model_{}_weights.h5".format(strategy))))
            #save_history_plot(history_final, "final_{}".format(strategy), folder_name_logging)

            print('Test final model with strategy {}:'.format(strategy))
            if dataset_name == 'imagenet':
                val_loss_finetuned, val_entropy_finetuned, val_acc_finetuned = model_final.evaluate(datagen_val, verbose=1)
            elif dataset_name == 'CIFAR10':
                val_loss_finetuned, val_entropy_finetuned, val_acc_finetuned = model_final.evaluate(x_test, y_test, verbose=1)
            print('Loss: {:.5f}'.format(val_loss_finetuned))
            print('Entropy: {:.5f}'.format(val_entropy_inserted))
            print('Accuracy: {:.4f}'.format(val_acc_finetuned))

            logging.info('')
            logging.info('{}: loss: {:.5f}, acc: {:.5f}, time: {:.1f} min'.format(strategy, val_loss_finetuned, val_acc_finetuned, (train_stop-train_start)/60))

    else:
        
        
        if training_final_model['finetune_strategy'] == 'feature_maps':
            pass
            '''
            residual_layer_dic, _ = identify_residual_layer_indexes(model_final)
            #residual_layer_dic = {}
            residual_layer_dic[len(model_final.layers)-1] = len(model_final.layers)-2
            fine_tune_locations = residual_layer_dic.keys()

            for location in fine_tune_locations:

                if location <= loc1+len(model_shunt.layers):
                    continue


                model_reduced = modify_model(model_final, layer_indexes_to_delete=range(location+1,len(model_final.layers)))

                fm1_train = fm2_train = fm1_test = fm2_test = None
        
                if os.path.isfile(Path(shunt_params['featuremapspath'], "ft1_train_{}_{}.npy".format(residual_layer_dic[location], location))):
                
                    ft1_train = np.load(Path(shunt_params['featuremapspath'], "ft1_train_{}_{}.npy".format(residual_layer_dic[location], location)))
                    ft2_train = np.load(Path(shunt_params['featuremapspath'], "ft2_train_{}_{}.npy".format(residual_layer_dic[location], location)))
                    ft1_test = np.load(Path(shunt_params['featuremapspath'], "ft1_test_{}_{}.npy".format(residual_layer_dic[location], location)))
                    ft2_test = np.load(Path(shunt_params['featuremapspath'], "ft2_test_{}_{}.npy".format(residual_layer_dic[location], location)))
                    print('Feature maps loaded successfully!')

                else:
                    
                    print('Feature maps extracting started:')

                    loc1_original_model = (loc2-loc1)-len(model_shunt.layers) + residual_layer_dic[location]
                    loc2_original_model = (loc2-loc1)-len(model_shunt.layers) + location

                    (_, ft2_train) = extract_feature_maps(model_original, x_train, [loc1_original_model+2, loc2_original_model+2]) # -1 since we need the input of the layer
                    (_, ft2_test) = extract_feature_maps(model_original, x_test, [loc1_original_model+2, loc2_original_model+2]) # -1 since we need the input of the layer
                    (ft1_train, _) = extract_feature_maps(model_final, x_train, [residual_layer_dic[location], location]) # -1 since we need the input of the layer
                    (ft1_test, _) = extract_feature_maps(model_final, x_test, [residual_layer_dic[location], location]) # -1 since we need the input of the layer


                    #np.save(Path(shunt_params['featuremapspath'], "ft1_train_{}_{}".format(loc1+len(model_shunt.layers), location)), ft1_train)
                    #np.save(Path(shunt_params['featuremapspath'], "ft2_train_{}_{}".format(loc1+len(model_shunt.layers), location)), ft2_train)
                    #np.save(Path(shunt_params['featuremapspath'], "ft1_test_{}_{}".format(loc1+len(model_shunt.layers), location)), ft1_test)
                    #np.save(Path(shunt_params['featuremapspath'], "ft2_test_{}_{}".format(loc1+len(model_shunt.layers), location)), ft2_test)

                    logging.info('')
                    logging.info('Featuremaps saved to {}'.format(shunt_params['featuremapspath']))

                model_reduced = modify_model(model_reduced, layer_indexes_to_delete=range(0,residual_layer_dic[location]+1))
                model_reduced.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(lr=learning_rate_first_cycle_final, decay=0.0), metrics=[keras.metrics.MeanSquaredError()])

                print('Train reduced model:')
                history_final = model_reduced.fit(x=ft1_train, y=ft2_train, batch_size=batch_size_final, epochs=epochs_final, validation_data=(ft1_test, ft2_test), verbose=1, callbacks=[callback_checkpoint, callback_learning_rate])
                #save_history_plot(history_shunt, "final_{}".format(location), folder_name_logging)

                model_final = modify_model(model_final, layer_indexes_to_delete=range(residual_layer_dic[location]+1, location+1), shunt_to_insert=model_reduced)
                model_final.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=learning_rate_first_cycle_final, momentum=0.9, decay=0.0, nesterov=False), metrics=[keras.metrics.categorical_crossentropy, 'accuracy'])
                #print(model_final.summary())

                print('Test_final_model')
                val_loss_finetuned, val_entropy_finetuned, val_acc_finetuned = model_final.evaluate(x_test, y_test, verbose=1)
                print('Loss: {}'.format(val_loss_finetuned))
                print('Entropy: {:.5f}'.format(val_entropy_finetuned))
                print('Accuracy: {}'.format(val_acc_finetuned))
            '''
        else:

            if training_final_model['finetune_strategy'] == 'unfreeze_shunt':
                callbacks.append(callback_learning_rate)
                for i, layer in enumerate(model_final.layers):
                    if i < loc1 - 1 or i > loc1 + len(model_shunt.layers):
                        layer.trainable = False

            if training_final_model['finetune_strategy'] == 'unfreeze_after_shunt':
                callbacks.append(callback_learning_rate)
                for i, layer in enumerate(model_final.layers):
                    if i < loc1 + len(model_shunt.layers) - 1:
                        model_final.layers[i].trainable = False


            if training_final_model['finetune_strategy'] == 'unfreeze_per_epoch_starting_top':
                callback_unfreeze = UnfreezeLayersCallback(epochs=epochs_final, epochs_per_unfreeze=2, learning_rate=learning_rate_first_cycle_final, unfreeze_to_index=loc1+len(model_shunt.layers), start_at=len(model_final.layers), direction=-1)
                callbacks.append(callback_unfreeze)
                for i, layer in enumerate(model_final.layers):
                    layer.trainable = False

            if training_final_model['finetune_strategy'] == 'unfreeze_all':
                callbacks.append(callback_learning_rate)

            if training_final_model['finetune_strategy'] == 'unfreeze_per_epoch_starting_shunt':
                callback_unfreeze = UnfreezeLayersCallback(epochs=epochs_final, epochs_per_unfreeze=2, learning_rate=learning_rate_first_cycle_final, unfreeze_to_index=0, start_at=loc1+len(model_shunt.layers)-2, direction=1)
                # TODO: test this
                callbacks.append(callback_unfreeze)
                for i, layer in enumerate(model_final.layers):
                    layer.trainable = False

            model_final.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=learning_rate_first_cycle_final, momentum=0.9, decay=0.0, nesterov=False), metrics=[keras.metrics.categorical_crossentropy, 'accuracy'])

            if  modes['train_final_model']:
                print('Train final model:')
                if dataset_name == 'imagenet':
                    history_final = model_final.fit(datagen_train.flow_from_directory(dataset_train_image_path, shuffle=True, target_size=(224,224), batch_size=batch_size_final), epochs=epochs_final, validation_data=(x_test, y_test), verbose=1, callbacks=callbacks)
                elif dataset_name == 'CIFAR10':
                    history_final = model_final.fit(datagen_train.flow(x_train, y_train, batch_size=batch_size_final), epochs=epochs_final, validation_data=(x_test, y_test), verbose=1, callbacks=callbacks)
                #save_history_plot(history_final, "final", folder_name_logging)

                model_final.load_weights(str(Path(folder_name_logging, "final_model_weights.h5")))

                print('Test_final_model')
                if dataset_name == 'imagenet':
                    val_loss_finetuned, val_entropy_finetuned, val_acc_finetuned = model_final.evaluate(datagen_val, verbose=1)
                elif dataset_name == 'CIFAR10':
                    val_loss_finetuned, val_entropy_finetuned, val_acc_finetuned = model_final.evaluate(x_test, y_test, verbose=1)
                print('Loss: {}'.format(val_loss_finetuned))
                print('Entropy: {:.5f}'.format(val_entropy_finetuned))
                print('Accuracy: {}'.format(val_acc_finetuned))

                model_final.load_weights(str(Path(folder_name_logging, "final_model_weights.h5")))
        
        logging.info('')
        logging.info('Final model weights saved to {}'.format(folder_name_logging))

        logging.info('')
        logging.info('#######################################################################################################')
        logging.info('############################################## ACCURACY ###############################################')
        logging.info('#######################################################################################################')
        logging.info('')
        logging.info('Original model: loss: {:.5f}, acc: {:.5f}'.format(val_loss_original, val_acc_original))
        if modes['test_shunt_model']:
            logging.info('Shunt model: loss: {:.5f}, acc: {:.5f}'.format(val_loss_shunt, val_acc_shunt))
        logging.info('Inserted model: loss: {:.5f}, acc: {:.5f}'.format(val_loss_inserted, val_acc_inserted))
        if  modes['train_final_model']: logging.info('Finetuned model: loss: {:.5f}, acc: {:.5f}'.format(val_loss_finetuned, val_acc_finetuned))

    # latency test

    if modes['test_latency']:

        original_list = []
        final_list = []

        # warmup
        if dataset_name == 'imagenet':
            model_original.predict(datagen_val, verbose=1, batch_size=1, steps=10000)
            model_final.predict(datagen_val, verbose=1, batch_size=1, steps=10000)
        elif dataset_name == 'CIFAR10':
            model_original.predict(x_test, verbose=1, batch_size=1)
            model_final.predict(x_test, verbose=1, batch_size=1)

        for i in range(5):

            start_original = time.process_time()
            if dataset_name == 'imagenet':
                model_original.predict(datagen_val, verbose=1, batch_size=1, steps=10000)
            elif dataset_name == 'CIFAR10':
                model_original.predict(x_test, verbose=1, batch_size=1)
            end_original = time.process_time()

            start_final = time.process_time()
            if dataset_name == 'imagenet':
                model_final.predict(datagen_val, verbose=1, batch_size=1, steps=10000)
            elif dataset_name == 'CIFAR10':
                model_final.predict(x_test, verbose=1, batch_size=1)           
            end_final = time.process_time()

            time_original = (end_original-start_original)/len(x_test)
            time_final = (end_final-start_final)/len(x_test)

            original_list.append(time_original)
            final_list.append(time_final)
    
        for i in range(5):

            start_final = time.process_time()
            if dataset_name == 'imagenet':
                model_final.predict(datagen_val, verbose=1, batch_size=1, steps=10000)
            elif dataset_name == 'CIFAR10':
                model_final.predict(x_test, verbose=1, batch_size=1)           
            end_final = time.process_time()


            start_original = time.process_time()
            if dataset_name == 'imagenet':
                model_original.predict(datagen_val, verbose=1, batch_size=1, steps=10000)
            elif dataset_name == 'CIFAR10':
                model_original.predict(x_test, verbose=1, batch_size=1)
            end_original = time.process_time()

            time_original = (end_original-start_original)/len(x_test)
            time_final = (end_final-start_final)/len(x_test)

            original_list.append(time_original)
            final_list.append(time_final)
        
        time_original = np.mean(np.asarray(original_list))
        time_final = np.mean(np.asarray(final_list))

        logging.info('')
        logging.info('#######################################################################################################')
        logging.info('############################################## LATENCY ################################################')
        logging.info('#######################################################################################################')
        logging.info('')
        logging.info('Original model: time: {:.5f}'.format(time_original))
        logging.info('Final model: time: {:.5f}'.format(time_final))
        logging.info('Speedup: {:.2f}%'.format((time_original-time_final)/time_original * 100))
