import sys
import configparser
import logging
import time

from pathlib import Path
import numpy as np

from CDL.models.MobileNet_v2 import MobileNetV2_extended
from CDL.shunt import ExtractFeatureMaps
from CDL.shunt import Architectures
from CDL.utils.calculateFLOPS import calculateFLOPs_model, calculateFLOPs_blocks

import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
#from keras_applications.mobilenet_v3 import MobileNetV3Small
import keras

from matplotlib import pyplot as plt

if __name__ == '__main__':

    tf.compat.v1.disable_eager_execution()

    # READ CONFIG

    config = configparser.ConfigParser()
    config.read(Path(sys.path[0], "config", "main.cfg"))

    modes = {}
    modes['calc knowledge quotient'] = config['GENERAL'].getboolean('calc knowledge quotient')
    modes['train original model'] = config['GENERAL'].getboolean('train original model')
    modes['train final model'] = config['GENERAL'].getboolean('train final model')
    loglevel = config['GENERAL'].getint('logging level')
    save_models = config['GENERAL'].getboolean('save models')

    dataset_name = config['DATASET']['name']
    subset_fraction = config['DATASET']['subset fraction']
    num_classes = config['DATASET'].getint('number classes')
    
    load_model_from_file = config['MODEL'].getboolean('from file')
    if (load_model_from_file):
        model_file_path = config['MODEL']['filepath']
    else:
        pretrained_on_imagenet = config['MODEL'].getboolean('pretrained on imagenet')
        scale_to_imagenet = config['MODEL'].getboolean('scale to imagenet')
    pretrained = config['MODEL'].getboolean('pretrained')
    if pretrained:
        weights_file_path = config['MODEL']['weightspath']
    
    training_original_model = config['TRAINING_ORIGINAL_MODEL']
    training_shunt_model = config['TRAINING_SHUNT_MODEL']
    training_final_model = config['TRAINING_FINAL_MODEL']

    shunt_params = {}
    shunt_params['arch'] = config['SHUNT'].getint('arch')
    shunt_params['input shape'] = tuple(map(int, config['SHUNT']['input shape'].split(',')))
    shunt_params['output shape'] = tuple(map(int, config['SHUNT']['output shape'].split(',')))
    shunt_params['locations'] = tuple(map(int, config['SHUNT']['location'].split(',')))
    shunt_params['load featuremaps'] = config['SHUNT'].getboolean('load featuremaps')
    if shunt_params['load featuremaps']: shunt_params['featuremapspath'] = config['SHUNT']['featuremapspath']
    shunt_params['save featuremaps'] = config['SHUNT'].getboolean('save featuremaps')

    # init logging
    folder_name_logging = Path(sys.path[0], "log", time.strftime("%Y%m%d"), time.strftime("%H_%M_%S"))
    Path(folder_name_logging).mkdir(parents=True, exist_ok=True)
    log_file_name = Path(folder_name_logging, "output.log")
    logging.basicConfig(filename=log_file_name, level=loglevel , format='%(message)s')
    logger = logging.getLogger(__name__)

    # prepare data

    x_train = y_train = x_test = y_test = None
    datagen = None
    input_shape = None

    if dataset_name == 'CIFAR10':

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        input_shape = (32,32,3)

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

        datagen = ImageDataGenerator(
            featurewise_center=False, 
            featurewise_std_normalization=False, 
            rotation_range=0.0,
            width_shift_range=0.2, 
            height_shift_range=0.2, 
            vertical_flip=False,
            horizontal_flip=True)
        datagen.fit(x_train)

        print('CIFAR10 was loaded successfully!')

    # load/create model
    model_extended = None

    if load_model_from_file:
        model_tmp = keras.models.load_model(model_file_path)
        model_extended = MobileNetV2_extended(model_tmp.input, model_tmp.output)
    elif pretrained_on_imagenet:
        model_extended = MobileNetV2_extended.create(is_pretrained=True, num_classes=num_classes)
    else:
        if scale_to_imagenet:
            model_extended = MobileNetV2_extended.create(is_pretrained=False, num_classes=10, input_shape=input_shape, mobilenet_shape=(224,224,3))
        else:
            model_extended = MobileNetV2_extended.create(is_pretrained=False, num_classes=10, input_shape=input_shape, mobilenet_shape=(32,32,3))

    if pretrained:
        model_extended.load_weights(weights_file_path)

    epochs_original = int(training_original_model['epochs'])
    batch_size_original = int(training_original_model['batchsize'])
    learning_rate_original = float(training_original_model['learning rate'])

    model_extended.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=learning_rate_original, momentum=0.9, decay=learning_rate_original/epochs_original), metrics=['accuracy'])

    logging.info('')
    logging.info('#######################################################################################################')
    logging.info('########################################### ORIGINAL MODEL ############################################')
    logging.info('#######################################################################################################')
    logging.info('')
    model_extended.summary(print_fn=logger.info)

    flops_original = calculateFLOPs_model(model_extended)

    # train model if weights are not loaded

    if modes['train original model']:
        print('Train original model:')
        model_extended.fit(datagen.flow(x_train, y_train, batch_size=batch_size_original), steps_per_epoch=len(x_train) / batch_size_original, epochs=epochs_original, validation_data=(x_test, y_test), verbose=1)
        model_extended.save_weights(str(Path(folder_name_logging, "original_model_weights.h5")))
        


    # test original model
    print('Test original model')
    val_loss_original, val_acc_original = model_extended.evaluate(x_test, y_test, verbose=1)
    
    if modes['calc knowledge quotient']:
        know_quot = model_extended.getKnowledgeQuotients(data=(x_test, y_test))
        logging.info('')
        logging.info('################# RESULT ###################')
        logging.info('')
        logging.info('Original model: loss: {:.5f}, acc: {:.5f}'.format(val_loss_original, val_acc_original))
        logging.info('')
        logging.info(know_quot)

        exit()

    loc1 = shunt_params['locations'][0]
    loc2 = shunt_params['locations'][1]

    # Feature maps

    fm1_train = fm2_train = fm1_test = fm2_test = None

    if shunt_params['load featuremaps']:
    
        fm1_train = np.load(Path(shunt_params['featuremapspath'], "fm1_train_{}_{}.npy".format(loc1, loc2)))
        fm2_train = np.load(Path(shunt_params['featuremapspath'], "fm2_train_{}_{}.npy".format(loc1, loc2)))
        fm1_test = np.load(Path(shunt_params['featuremapspath'], "fm1_test_{}_{}.npy".format(loc1, loc2)))
        fm2_test = np.load(Path(shunt_params['featuremapspath'], "fm2_test_{}_{}.npy".format(loc1, loc2)))
        print('Loaded feature maps successfully!')

    else:
        
        print('Feature maps extracting started:')
        (fm1_train, fm2_train) = ExtractFeatureMaps.getFeatureMaps(model_extended, (loc1,loc2), x_train)
        (fm1_test, fm2_test) = ExtractFeatureMaps.getFeatureMaps(model_extended, (loc1,loc2), x_test)

        if shunt_params['save featuremaps']:
            np.save(Path(folder_name_logging, "fm1_train_{}_{}".format(loc1, loc2)), fm1_train)
            np.save(Path(folder_name_logging, "fm2_train_{}_{}".format(loc1, loc2)), fm2_train)
            np.save(Path(folder_name_logging, "fm1_test_{}_{}".format(loc1, loc2)), fm1_test)
            np.save(Path(folder_name_logging, "fm2_test_{}_{}".format(loc1, loc2)), fm2_test)

            logging.info('')
            logging.info('Featuremaps saved to {}'.format(folder_name_logging))
    

    flops_dropped_blocks = calculateFLOPs_blocks(model_extended, range(loc1,loc2+1))

    model_shunt = Architectures.createShunt(shunt_params['input shape'],shunt_params['output shape'], arch=shunt_params['arch'])
    
    logging.info('')
    logging.info('#######################################################################################################')
    logging.info('############################################ SHUNT MODEL ##############################################')
    logging.info('#######################################################################################################')
    logging.info('')
    model_shunt.summary(print_fn=logger.info)

    if save_models:
        keras.models.save_model(model_shunt, Path(folder_name_logging, "shunt_model.h5"))
        logging.info('')
        logging.info('Shunt model saved to {}'.format(folder_name_logging))
    
    flops_shunt = calculateFLOPs_model(model_shunt)

    epochs_shunt = int(training_shunt_model['epochs'])
    batch_size_shunt = int(training_shunt_model['batchsize'])
    learning_rate_shunt = float(training_shunt_model['learning rate'])

    model_shunt.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(learning_rate=learning_rate_shunt, decay=learning_rate_shunt/epochs_shunt), metrics=['accuracy'])
    print('Train shunt model:')
    model_shunt.fit(x=fm1_train, y=fm2_train, batch_size=batch_size_shunt, epochs=epochs_shunt, validation_data=(fm1_test, fm2_test), verbose=1)
    print('Test shunt model')
    val_loss_shunt, val_acc_shunt = model_shunt.evaluate(fm1_test, fm2_test, verbose=1)


    model_final = model_extended.insertShunt(model_shunt, range(loc1, loc2+1))

    if save_models:
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
    logging.info('Dropped blocks: {}'.format(flops_dropped_blocks))
    logging.info('Shunt model: {}'.format(flops_shunt))
    logging.info('Final model: {}'.format(flops_final))

    reduction = 100*(flops_original['total']-flops_final['total']) / flops_original['total']
    logging.info('')
    logging.info('Model got reduced by {:.2f}%!'.format(reduction))

    epochs_final = int(training_final_model['epochs'])
    batch_size_final = int(training_final_model['batchsize'])
    learning_rate_final = float(training_final_model['learning rate'])
    model_final.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=learning_rate_final, momentum=0.9, decay=learning_rate_final/epochs_final, nesterov=False), metrics=['accuracy'])
    
    logging.info('')
    logging.info('#######################################################################################################')
    logging.info('############################################ FINAL MODEL ##############################################')
    logging.info('#######################################################################################################')
    logging.info('')
    model_final.summary(print_fn=logger.info)

    print('Test shunt inserted model')
    val_loss_inserted, val_acc_inserted = model_final.evaluate(x_test, y_test, verbose=1)

    if  modes['train final model']:
        print('Train final model:')
        model_final.fit(datagen.flow(x_train, y_train, batch_size=batch_size_final), steps_per_epoch=len(x_train) / batch_size_final, epochs=epochs_final, validation_data=(x_test, y_test), verbose=1)
        print('Test final model')
        val_loss_finetuned, val_acc_finetuned = model_final.evaluate(x_test, y_test, verbose=1)

        if save_models:
            model_final.save_weights(str(Path(folder_name_logging, "final_model_weights.h5")))
            logging.info('')
            logging.info('Final model weights saved to {}'.format(folder_name_logging))

    logging.info('')
    logging.info('#######################################################################################################')
    logging.info('############################################## ACCURACY ###############################################')
    logging.info('#######################################################################################################')
    logging.info('')
    logging.info('Original model: loss: {:.5f}, acc: {:.5f}'.format(val_loss_original, val_acc_original))
    logging.info('Shunt model: loss: {:.5f}, acc: {:.5f}'.format(val_loss_shunt, val_acc_shunt))
    logging.info('Inserted model: loss: {:.5f}, acc: {:.5f}'.format(val_loss_inserted, val_acc_inserted))
    if  modes['train final model']: logging.info('Finetuned model: loss: {:.5f}, acc: {:.5f}'.format(val_loss_finetuned, val_acc_finetuned))
