from tensorflow.keras.optimizers import SGD
import tensorflow.keras as keras

from .keras_utils import *

def get_knowledge_quotients(model, datagen, val_acc_model, metric=keras.metrics.categorical_accuracy, is_deeplab=False):

    know_quot = []
    add_input_dic, _ = identify_residual_layer_indexes(model)

    for add_layer_index in add_input_dic.keys():

        start_layer_index = add_input_dic[add_layer_index] + 1

        model_reduced = modify_model(model, range(start_layer_index, add_layer_index+1), is_deeplab=is_deeplab)

        model_reduced.compile(loss='categorical_crossentropy', optimizer=SGD(lr=2e-2, momentum=0.9, decay=0.0, nesterov=False), metrics=[metric])

        if isinstance(datagen, tuple):
            val_loss, val_acc = model_reduced.evaluate(datagen[0], datagen[1], verbose=1)
        else:
            val_loss, val_acc = model_reduced.evaluate(datagen, verbose=1)
        print('Test loss for block {}: {:.5f}'.format(add_input_dic[add_layer_index], val_loss))
        print('Test accuracy for block {}: {:.5f}'.format(add_input_dic[add_layer_index], val_acc))

        know_quot.append((add_input_dic[add_layer_index], add_layer_index, val_acc / val_acc_model))

    return know_quot

def get_knowledge_quotient(model, datagen, val_acc_model, locations):

    add_input_dic, _ = identify_residual_layer_indexes(model)

    indexes_to_delete = []

    for add_layer_index in add_input_dic.keys():
        if add_layer_index in range(locations[0], locations[1]+1):
            start_layer_index = add_input_dic[add_layer_index] + 1
            indexes_to_delete = indexes_to_delete + list(range(start_layer_index, add_layer_index+1))
    print(indexes_to_delete)

    model_reduced = modify_model(model, indexes_to_delete)
    print(model_reduced.summary())
    model_reduced.compile(loss='categorical_crossentropy', optimizer=SGD(lr=2e-2, momentum=0.9, decay=0.0, nesterov=False), metrics=['accuracy'])

    if isinstance(datagen, tuple):
        _, val_acc = model_reduced.evaluate(datagen[0], datagen[1], verbose=1)
    else:
        _, val_acc = model_reduced.evaluate(datagen, verbose=1)

    return val_acc/val_acc_model
