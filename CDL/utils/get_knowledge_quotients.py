from keras.optimizers import SGD

from .keras_utils import *

def get_knowledge_quotients(model, datagen, val_acc_model):

    know_quot = []
    add_input_dic, _ = identify_residual_layer_indexes(model)

    for add_layer_index in add_input_dic.keys():

        start_layer_index = add_input_dic[add_layer_index] + 1

        model_reduced = modify_model(model, range(start_layer_index, add_layer_index+1))

        model_reduced.compile(loss='categorical_crossentropy', optimizer=SGD(lr=2e-2, momentum=0.9, decay=0.0, nesterov=False), metrics=['accuracy'])

        val_loss, val_acc = model_reduced.evaluate(datagen, verbose=1)
        print('Test loss for block {}: {:.5f}'.format(add_input_dic[add_layer_index], val_loss))
        print('Test accuracy for block {}: {:.5f}'.format(add_input_dic[add_layer_index], val_acc))

        know_quot.append((add_input_dic[add_layer_index], add_layer_index, val_acc / val_acc_model))

    return know_quot