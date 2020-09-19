import sys
import configparser
import logging
import time
from shutil import copyfile

from pathlib import Path
import numpy as np

from CDL.models.deeplabv3p import Deeplabv3
from CDL.shunt import Architectures
from CDL.shunt.create_shunt_trainings_model import create_shunt_trainings_model
from CDL.utils.calculateFLOPS import calculateFLOPs_model, calculateFLOPs_blocks
from CDL.utils.dataset_utils import *
from CDL.utils.get_knowledge_quotients import get_knowledge_quotients
from CDL.utils.generic_utils import *
from CDL.utils.keras_utils import extract_feature_maps, modify_model, mean_iou, mean_squared_diff
from CDL.utils.custom_callbacks import UnfreezeLayersCallback

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from CDL.utils.custom_generators import VOC2012_generator
from CDL.utils.custom_callbacks import LearningRateSchedulerCallback

import tensorflow.keras as keras
import tensorflow.keras.backend as K

from matplotlib import pyplot as plt

if __name__ == '__main__':

    # READ CONFIG
    config_path = Path(sys.path[0], "config", "segmentation.cfg")
    config = configparser.ConfigParser()
    config.read(config_path)

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

    backbone_type = config['MODEL']['backbone']
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
    datagen = None
    input_shape = None

    copyfile(config_path, Path(folder_name_logging, "config.cfg"))

    file_names_train = file_names_val = x_dir = y_dir = None

    if dataset_name == 'VOC2012':

        input_shape = (512,512,3)
        num_classes = 21

        x_dir = Path(dataset_path, "JPEGImages")
        y_dir = Path(dataset_path, "SegmentationClassRaw")
        file_names_val = []
        file_names_train = []

        with open(Path(dataset_path, "ImageSets/Segmentation/val.txt")) as f:
            for l in f:
                file_names_val.append(l[:-1])

        with open(Path(dataset_path, "ImageSets/Segmentation/trainval.txt")) as f:
            for l in f:
                file_names_train.append(l[:-1])

        datagen_val = VOC2012_generator(x_dir=x_dir, y_dir=y_dir, file_names=file_names_val, batch_size=1)
        datagen_train = VOC2012_generator(x_dir=x_dir, y_dir=y_dir, file_names=file_names_train, batch_size=1)

        print('VOC2012 was loaded successfully!')


    # load/create model
    model_original = None

    if backbone_type == 'MobileNetV2':
        if load_model_from_file:
            model_original = keras.models.load_model(model_file_path)
        elif weights_file_path == 'pascal_voc':
            model_original = Deeplabv3(backbone='MobileNetV2', weights='pascal_voc', classes=num_classes)
        else:
            model_original = Deeplabv3(weights=None, input_shape=input_shape, classes=21, backbone='MobileNetV2', OS=8)


    if backbone_type == 'MobileNetV3':

        if load_model_from_file:
            model_original = keras.models.load_model(model_file_path)
        elif weights_file_path == 'cityscapes':
            model_original = Deeplabv3(is_pretrained=True, num_classes=num_classes)
        else:
            model_original = Deeplabv3(weights=None, input_shape=input_shape, classes=21, backbone='MobileNetV3')

    if pretrained:
        model_original = model_original.load_weights(weights_file_path)
        print('Weights loaded successfully!')


    batch_size_original = training_original_model.getint('batchsize')
    epochs_first_cycle_original = training_original_model.getint('epochs_first_cycle')
    epochs_second_cycle_original = training_original_model.getint('epochs_second_cycle')
    epochs_original = epochs_first_cycle_original + epochs_second_cycle_original
    learning_rate_first_cycle_original = training_original_model.getfloat('learning_rate_first_cycle')
    learning_rate_second_cycle_original = training_original_model.getfloat('learning_rate_second_cycle')

    model_original.compile(loss=['categorical_crossentropy'], optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0), metrics=[mean_iou])

    logging.info('')
    logging.info('#######################################################################################################')
    logging.info('########################################### ORIGINAL MODEL ############################################')
    logging.info('#######################################################################################################')
    logging.info('')
    model_original.summary(print_fn=logger.info, line_length=150)
    print('DeeplabV3 with {} backbone created successfully!'.format(backbone_type))

    flops_original = calculateFLOPs_model(model_original)

    callback_checkpoint = keras.callbacks.ModelCheckpoint(str(Path(folder_name_logging, "original_model_weights.h5")), save_best_only=True, save_weights_only=True)
    callback_learning_rate = LearningRateSchedulerCallback(epochs_first_cycle=epochs_first_cycle_original, learning_rate_second_cycle=learning_rate_second_cycle_original)

    if modes['train_original_model']:
        print('Train original model:')
        history_original = model_original.fit(datagen_train, epochs=epochs_original, validation_data=datagen_val, verbose=1, callbacks=[callback_checkpoint, callback_learning_rate])
        model_original.load_weights(str(Path(folder_name_logging, "original_model_weights.h5")))
        save_history_plot(history_original, "original", folder_name_logging, ['loss', 'mean_iou'])

    # test original model
    #print('Test original model')
    #val_loss_original, val_iou_original = model_original.evaluate(datagen_val, verbose=1)
    #print('Loss: {:.5f}'.format(val_loss_original))
    #print('mIOU: {:.4f}'.format(val_iou_original))

    if modes['calc_knowledge_quotients']:
        know_quot = get_knowledge_quotients(model=model_original, datagen=datagen_val, val_acc_model=val_iou_original, metric=mean_iou, is_deeplab=True)
        logging.info('')
        logging.info('################# RESULT ###################')
        logging.info('')
        logging.info('Original model: loss: {:.5f}, mIOU: {:.5f}'.format(val_loss_original, val_iou_original))
        logging.info('')
        for (residual_idx, end_idx, value) in know_quot:
            logging.info("Block starts with: {}, location: {}".format(model_original.get_layer(index=residual_idx+1).name, residual_idx+1))
            logging.info("Block ends with: {}, location: {}".format(model_original.get_layer(index=end_idx).name, end_idx))   
            logging.info("Block knowledge quotient: {}\n".format(value)) 
        exit()


    loc1 = shunt_params['locations'][0]
    loc2 = shunt_params['locations'][1]
    
    if shunt_params['from_file']:
        model_shunt = keras.models.load_model(shunt_params['filepath'])
        print('Shunt model loaded successfully!')
    else:

        input_shape_shunt = model_original.get_layer(index=loc1).input_shape[1:]
        output_shape_shunt = model_original.get_layer(index=loc2).output_shape[1:]

        model_shunt = Architectures.createShunt(input_shape_shunt, output_shape_shunt, arch=shunt_params['arch'])
    
    logging.info('')
    logging.info('#######################################################################################################')
    logging.info('############################################ SHUNT MODEL ##############################################')
    logging.info('#######################################################################################################')
    logging.info('')
    model_shunt.summary(print_fn=logger.info, line_length=150)

    keras.models.save_model(model_shunt, Path(folder_name_logging, "shunt_model.h5"))
    logging.info('')
    logging.info('Shunt model saved to {}'.format(folder_name_logging))
    
    if shunt_params['pretrained']:
        model_shunt.load_weights(shunt_params['weightspath'])
        print('Shunt weights loaded successfully!')

    flops_shunt = calculateFLOPs_model(model_shunt)

    batch_size_shunt = training_shunt_model.getint('batchsize')
    epochs_first_cycle_shunt = training_shunt_model.getint('epochs_first_cycle')
    epochs_second_cycle_shunt = training_shunt_model.getint('epochs_second_cycle')
    epochs_shunt = epochs_first_cycle_shunt + epochs_second_cycle_shunt
    learning_rate_first_cycle_shunt = training_shunt_model.getfloat('learning_rate_first_cycle')
    learning_rate_second_cycle_shunt = training_shunt_model.getfloat('learning_rate_second_cycle')

    model_training_shunt = create_shunt_trainings_model(model_original, model_shunt, (loc1, loc2))
    model_training_shunt.compile(loss=mean_squared_diff, optimizer=keras.optimizers.Adam(learning_rate=learning_rate_first_cycle_shunt, decay=0.0))
    model_training_shunt.add_loss(mean_squared_diff(None, model_training_shunt.outputs[0]))

    if shunt_params['pretrained']:
        model_training_shunt.load_weights(shunt_params['weightspath'])
        print('Shunt weights loaded successfully!')

    model_shunt.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(learning_rate=learning_rate_first_cycle_shunt, decay=0.0), metrics=[keras.metrics.MeanSquaredError()])

    callback_checkpoint = keras.callbacks.ModelCheckpoint(str(Path(folder_name_logging, "shunt_model_weights.h5")), save_best_only=True, save_weights_only=True)
    callback_learning_rate = LearningRateSchedulerCallback(epochs_first_cycle=epochs_first_cycle_shunt, learning_rate_second_cycle=learning_rate_second_cycle_shunt)

    # Feature maps

    if modes['test_shunt_model'] or modes['train_shunt_model']:

        if modes['train_shunt_model']:
            print('Train_shunt_model:')

            datagen_val_dummy = VOC2012_generator(x_dir=x_dir, y_dir=y_dir, file_names=file_names_val, batch_size=batch_size_shunt, include_labels=False)
            datagen_train_dummy = VOC2012_generator(x_dir=x_dir, y_dir=y_dir, file_names=file_names_train, batch_size=batch_size_shunt, include_labels=False)

            history_shunt = model_training_shunt.fit(datagen_train_dummy, epochs=epochs_shunt, validation_data=datagen_val_dummy, verbose=1, callbacks=[callback_checkpoint, callback_learning_rate],
                                                        use_multiprocessing=False, workers=32, max_queue_size=64)
            save_history_plot(history_shunt, "shunt", folder_name_logging, ['loss'])
            model_training_shunt.load_weights(str(Path(folder_name_logging, "shunt_model_weights.h5")))

        if modes['test_shunt_model']:
            print('Test shunt model')
            datagen_val_dummy = VOC2012_generator(x_dir=x_dir, y_dir=y_dir, file_names=file_names_val, batch_size=batch_size_shunt, include_labels=False)
            val_loss_shunt = model_training_shunt.evaluate(datagen_val_dummy, verbose=1)
            print('Loss: {:.5f}'.format(val_loss_shunt))


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
    epochs_first_cycle_final = training_final_model.getint('epochs first cycle')
    epochs_second_cycle_final = training_final_model.getint('epochs second cycle')
    epochs_final = epochs_first_cycle_final + epochs_second_cycle_final
    learning_rate_first_cycle_final = training_final_model.getfloat('learning rate first cycle')
    learning_rate_second_cycle_final = training_final_model.getfloat('learning rate second cycle')
    
    logging.info('')
    logging.info('#######################################################################################################')
    logging.info('############################################ FINAL MODEL ##############################################')
    logging.info('#######################################################################################################')
    logging.info('')
    model_final.summary(print_fn=logger.info, line_length=150)

    callback_checkpoint = keras.callbacks.ModelCheckpoint(str(Path(folder_name_logging, "final_model_weights.h5")), save_best_only=True, save_weights_only=True)
    callback_learning_rate = LearningRateSchedulerCallback(epochs_first_cycle=epochs_first_cycle_final, learning_rate_second_cycle=learning_rate_second_cycle_final)
    callbacks = [callback_checkpoint]

    model_final.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=learning_rate_first_cycle_final, momentum=0.9, decay=0.0, nesterov=False), metrics=[keras.metrics.categorical_crossentropy, 'accuracy'])

    print('Test shunt inserted model')
    val_loss_inserted, val_entropy_inserted, val_acc_inserted = model_final.evaluate(x_test, y_test, verbose=1)
    print('Loss: {:.5f}'.format(val_loss_inserted))
    print('Entropy: {:.5f}'.format(val_entropy_inserted))
    print('Accuracy: {:.4f}'.format(val_acc_inserted))

    if final_model_params['pretrained']:
        model_final.load_weights(final_model_params['weightspath'])
        print('Weights for final model loaded successfully!')
        print('Test shunt inserted model with loaded weights')
        val_loss_inserted, val_entropy_inserted, val_acc_inserted = model_final.evaluate(x_test, y_test, verbose=1)
        print('Loss: {:.5f}'.format(val_loss_inserted))
        print('Entropy: {:.5f}'.format(val_entropy_inserted))
        print('Accuracy: {:.4f}'.format(val_acc_inserted))
    if modes['test fine-tune strategies']:

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

            callback_checkpoint = keras.callbacks.ModelCheckpoint(str(Path(folder_name_logging, "final_model_{}_weights.h5".format(strategy))), save_best_only=False, save_weights_only=True)
            callbacks = [callback_checkpoint]

            print('Train final model with strategy {}:'.format(strategy))
            train_start = time.process_time()
            history_final = model_final.fit(datagen.flow(x_train, y_train, batch_size=batch_size_final), epochs=epochs_final, validation_data=(x_test, y_test), verbose=1, callbacks=callbacks)
            train_stop = time.process_time()
            save_history_plot(history_final, "final_{}".format(strategy), folder_name_logging)

            print('Test final model with strategy {}:'.format(strategy))
            val_loss_finetuned, val_acc_finetuned = model_final.evaluate(x_test, y_test, verbose=1)
            print('Loss: {}'.format(val_loss_finetuned))
            print('Accuracy: {}'.format(val_acc_finetuned))

            logging.info('')
            logging.info('{}: loss: {:.5f}, acc: {:.5f}, time: {:.1f} min'.format(strategy, val_loss_finetuned, val_acc_finetuned, (train_stop-train_start)/60))

    else:

        if training_final_model['finetune strategy'] == 'unfreeze_shunt':
            callbacks.append(callback_learning_rate)
            for i, layer in enumerate(model_final.layers):
                if i < loc1 - 1 or i > loc1 + len(model_shunt.layers):
                    layer.trainable = False

        if training_final_model['finetune strategy'] == 'unfreeze_after_shunt':
            callbacks.append(callback_learning_rate)
            for i, layer in enumerate(model_final.layers):
                if i < loc1 + len(model_shunt.layers) - 1:
                    model_final.layers[i].trainable = False


        if training_final_model['finetune strategy'] == 'unfreeze_per_epoch_starting_top':
            callback_unfreeze = UnfreezeLayersCallback(epochs=epochs_final, epochs_per_unfreeze=2, learning_rate=learning_rate_first_cycle_final, unfreeze_to_index=loc1+len(model_shunt.layers), start_at=len(model_final.layers), direction=-1)
            callbacks.append(callback_unfreeze)
            for i, layer in enumerate(model_final.layers):
                layer.trainable = False

        if training_final_model['finetune strategy'] == 'unfreeze_per_epoch_starting_shunt':
            callback_unfreeze = UnfreezeLayersCallback(epochs=epochs_final, epochs_per_unfreeze=2, learning_rate=learning_rate_first_cycle_final, unfreeze_to_index=0, start_at=loc1+len(model_shunt.layers)-2, direction=1)
            # TODO: test this
            callbacks.append(callback_unfreeze)
            for i, layer in enumerate(model_final.layers):
                layer.trainable = False

        if training_final_model['finetune strategy'] == 'unfreeze_all':
            callbacks.append(callback_learning_rate)
            for i, layer in enumerate(model_final.layers):
                if i < loc1 - 1 or i > loc1 + len(model_shunt.layers):
                    layer.trainable = False

        model_final.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=learning_rate_first_cycle_final, momentum=0.9, decay=0.0, nesterov=False), metrics=['accuracy'])

        if  modes['train final model']:
            print('Train final model:')
            history_final = model_final.fit(datagen.flow(x_train, y_train, batch_size=batch_size_final), epochs=epochs_final, validation_data=(x_test, y_test), verbose=1, callbacks=callbacks)
            save_history_plot(history_final, "final", folder_name_logging, ['loss', 'mean_iou'])

            print('Test final model')
            val_loss_finetuned, val_acc_finetuned = model_final.evaluate(x_test, y_test, verbose=1)
            print('Loss: {}'.format(val_loss_finetuned))
            print('Accuracy: {}'.format(val_acc_finetuned))

        model_final.save_weights(str(Path(folder_name_logging, "final_model_weights.h5")))
        logging.info('')
        logging.info('Final model weights saved to {}'.format(folder_name_logging))

        logging.info('')
        logging.info('#######################################################################################################')
        logging.info('############################################## ACCURACY ###############################################')
        logging.info('#######################################################################################################')
        logging.info('')
        logging.info('Original model: loss: {:.5f}, acc: {:.5f}'.format(val_loss_original, val_acc_original))
        if modes['test shunt model']:
            logging.info('Shunt model: loss: {:.5f}, acc: {:.5f}'.format(val_loss_shunt, val_acc_shunt))
        logging.info('Inserted model: loss: {:.5f}, acc: {:.5f}'.format(val_loss_inserted, val_acc_inserted))
        if  modes['train final model']: logging.info('Finetuned model: loss: {:.5f}, acc: {:.5f}'.format(val_loss_finetuned, val_acc_finetuned))


    # latency test

    if modes['test latency']:

        original_list = []
        final_list = []

        for i in range(5):

            start_original = time.process_time()
            model_original.predict(x_test, verbose=1, batch_size=1)
            end_original = time.process_time()

            start_final = time.process_time()
            model_final.predict(x_test, verbose=1, batch_size=1)
            end_final = time.process_time()

            time_original = (end_original-start_original)/len(x_test)
            time_final = (end_final-start_final)/len(x_test)

            original_list.append(time_original)
            final_list.append(final_list)

        for i in range(5):

            start_final = time.process_time()
            model_final.predict(x_test, verbose=1, batch_size=1)
            end_final = time.process_time()


            start_original = time.process_time()
            model_original.predict(x_test, verbose=1, batch_size=1)
            end_original = time.process_time()

            time_original = (end_original-start_original)/len(x_test)
            time_final = (end_final-start_final)/len(x_test)

            original_list.append(time_original)
            final_list.append(final_list)

        time_original = np.mean(original_list)
        time_final = np.mean(final_list)

        logging.info('')
        logging.info('#######################################################################################################')
        logging.info('############################################## LATENCY ################################################')
        logging.info('#######################################################################################################')
        logging.info('')
        logging.info('Original model: time: {:.5f}'.format(time_original))
        logging.info('Final model: time: {:.5f}'.format(time_final))
        logging.info('Speedup: {:.2f}%'.format((time_original-time_final)/time_original * 100))