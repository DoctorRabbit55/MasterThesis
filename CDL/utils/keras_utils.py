import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Add, Multiply, Input, Activation, Concatenate
from tensorflow.keras.layers import deserialize as layer_from_config
from keras.utils.generic_utils import get_custom_objects

import unittest
import numpy as np
from pathlib import Path

from tensorflow.keras.layers import ReLU, Lambda
from tensorflow.keras.applications import MobileNetV2
#from keras_applications.mobilenet_v3 import MobileNetV3Small

def create_mean_squared_diff_loss(factor=1):
    def mean_squared_diff(y_true, y_pred):
        return factor*K.mean(K.square(y_pred))

    return mean_squared_diff

def mean_iou(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    pred_pixels = K.argmax(y_pred, axis=-1)
    true_pixels = K.argmax(y_true, axis=-1)
    for i in range(1, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(true_pixels, i)
        pred_labels = K.equal(pred_pixels, i)
        inter = tf.cast(true_labels & pred_labels, dtype=tf.int32)
        union = tf.cast(true_labels | pred_labels, dtype=tf.int32)
        legal_batches = K.sum(tf.cast(true_labels, dtype=tf.int32), axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(ious[legal_batches]))

    iou = tf.stack(iou)
    legal_labels = ~tf.math.is_nan(iou)
    iou = iou[legal_labels]

    return K.mean(iou)

class HardSwish(Activation):

    def __init_(self, activation, **kwargs):
        super(HardSwish, self).__init__(activation, **kwargs)
        self.__name__ = 'hard_swish'

def hard_swish(x):
    return Multiply()([Activation(keras.activations.hard_sigmoid)(x), x])

def get_index_of_layer(model, layer):
    for i in range(len(model.layers)):
        if layer.name == model.layers[i].name:
            return i

def get_index_by_name(model, name):
    for i in range(len(model.layers)):
        if name == model.layers[i].name:
            return i

def get_first_layer_by_index(model, layers):
    smallest_index = len(model.layers)
    for layer in layers:
        index = get_index_of_layer(model, layer)
        if index < smallest_index:
            smallest_index = index
    return smallest_index

def identify_residual_layer_indexes(model):

    layers = model.layers
    add_incoming_index_dic = {}
    mult_incoming_index_dic = {}

    for i in range(len(layers)):

        layer = layers[i]

        if isinstance(layer, Add):
            input_layers = layer._inbound_nodes[-1].inbound_layers
            incoming_index = get_first_layer_by_index(model, input_layers)
            add_incoming_index_dic[i] = incoming_index

        if isinstance(layer, Multiply):
            input_layers = layer._inbound_nodes[-1].inbound_layers
            incoming_index = get_first_layer_by_index(model, input_layers)
            mult_incoming_index_dic[i] = incoming_index

    return add_incoming_index_dic, mult_incoming_index_dic

def modify_model(model, layer_indexes_to_delete=[], layer_indexes_to_output=[], shunt_to_insert=None, is_deeplab=False, layer_name_prefix=""):
    from CDL.models.deeplabv3p import deeplab_head

    get_custom_objects().update({'hard_swish': hard_swish})
    add_input_index_dic, mult_input_index_dic = identify_residual_layer_indexes(model)
    add_input_tensors = {}
    mult_input_tensors = {}

    outputs = []

    got_shunt_inserted = False
    shunt_output = None

    if 0 in layer_indexes_to_delete:
        input_net = Input(model.layers[layer_indexes_to_delete[-1]-1].output_shape[1:])
    else:
        input_net = Input(model.input_shape[1:])
    
    x = input_net  

    # if input of residual layers gets deleted, we must remap them
    for i in range(len(layer_indexes_to_delete)-1, -1, -1):

        layer_index_to_delete = layer_indexes_to_delete[i]

        for add_index, input_index in add_input_index_dic.items():
            if layer_index_to_delete == input_index:
                if not shunt_to_insert:
                    add_input_index_dic[add_index] = input_index-1
                else:
                    add_input_index_dic[add_index] = -1 # use shunt output
        for mult_index, input_index in mult_input_index_dic.items():
            if layer_index_to_delete == input_index:
                mult_input_index_dic[mult_index] = input_index-1

    for i in range(1,len(model.layers)):

        layer = model.layers[i]
        config = layer.get_config()
        config['name'] = layer_name_prefix + config['name']

        # there is a bug in layer_from_config, where custom Activation are not passed correctly

        next_layer = layer_from_config({'class_name': layer.__class__.__name__, 'config': config})
        should_delete = False
        for layer_index_to_delete in layer_indexes_to_delete:
            if i == layer_index_to_delete:
                should_delete = True
                break

        #print(i, "   ", layer.name)

        if should_delete:
            if shunt_to_insert and not got_shunt_inserted:
                add_input_index_shunt_dic, mult_input_index_shunt_dic = identify_residual_layer_indexes(shunt_to_insert)
                add_input_shunt_tensors = {}
                mult_input_shunt_tensors = {}
                for j, shunt_layer in enumerate(shunt_to_insert.layers[1:]):
                    j = j + 1
                    if isinstance(shunt_layer, Multiply):
                        second_input_index = mult_input_index_shunt_dic[j]
                        x = shunt_layer([x, mult_input_shunt_tensors[second_input_index]])
                    elif isinstance(shunt_layer, Add):
                        second_input_index = add_input_index_shunt_dic[j]
                        x = shunt_layer([x, add_input_shunt_tensors[second_input_index]])
                    else:
                        x = shunt_layer(x)
                    if j in add_input_index_shunt_dic.values():
                        add_input_shunt_tensors[j] = x
                    if j in mult_input_index_shunt_dic.values():
                        mult_input_shunt_tensors[j] = x

                shunt_output = x
                got_shunt_inserted = True
            continue

        if layer.name == 'start_deeplab':
            break

        if isinstance(next_layer, Multiply):
            second_input_index = mult_input_index_dic[i]
            x = next_layer([x, mult_input_tensors[second_input_index]])
        elif isinstance(next_layer, Add):
            second_input_index = add_input_index_dic[i]
            if second_input_index == -1: # use shunt or input layer
                if shunt_to_insert:
                    x = next_layer([x, shunt_output])
                else:
                    x = next_layer([x, input_net])
            else:
                x = next_layer([x, add_input_tensors[second_input_index]])
        else:
            x = next_layer(x)

        if i in add_input_index_dic.values():
            add_input_tensors[i] = x
        if i in mult_input_index_dic.values():
            mult_input_tensors[i] = x
        if i in layer_indexes_to_output:
            #print(layer.name)
            outputs.append(x)

    if is_deeplab:
        x = deeplab_head(x, model.input, 21)

    outputs.append(x)
    assert(len(outputs) == len(layer_indexes_to_output)+1)


    model_reduced = keras.models.Model(inputs=input_net, outputs=outputs, name=model.name)
    #print(model_reduced.summary())
    for j in range(1,len(model_reduced.layers)):

        layer = model_reduced.layers[j]
        if isinstance(layer, ReLU) or isinstance(layer, Lambda) or isinstance(layer, Activation):
            continue

        # skip shunt
        if shunt_to_insert:
            if j in range(layer_indexes_to_delete[0], layer_indexes_to_delete[0]+len(shunt_to_insert.layers)-1):
                weights = shunt_to_insert.get_layer(name=layer.name).get_weights()
                if len(weights) > 0:
                    model_reduced.layers[j].set_weights(weights)
                continue

        weights = model.get_layer(name=layer.name[len(layer_name_prefix):]).get_weights()
        if len(weights) > 0:
            #print(layer.name)
            model_reduced.layers[j].set_weights(weights)

    return model_reduced


def extract_feature_maps(model, x_data, locations, x_data_path=None, data_count=None):

    from CDL.utils.custom_generators import Imagenet_generator
    model = modify_model(model, layer_indexes_to_output=locations)

    if isinstance(x_data, np.ndarray):
        predictions = model.predict(x_data, verbose=1)
    elif isinstance(x_data, Imagenet_generator):
        if data_count:
            predictions = model.predict(x_data, verbose=1, steps=data_count//32)
        else:
            predictions = model.predict(x_data, verbose=1)
    else:
        if data_count:
            print("batch: ", x_data.batch_index)
            predictions = model.predict(x_data, verbose=1, steps=data_count//32)
            print("batch: ", x_data.batch_index)
        else:
            predictions = model.predict(x_data, verbose=1)

    return predictions[:-1]

def add_regularization(model, regularizer=keras.regularizers.l2(4e-5)):

    if not isinstance(regularizer, keras.regularizers.Regularizer):
      print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
      return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
              setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    weights_tmp = model.get_weights()

    # load the model from the config
    model = keras.models.model_from_json(model_json)
    
    # Reload the model weights
    model.set_weights(weights_tmp)
    return model

class TestIdentifyResidualLayerIndexesMethod(unittest.TestCase):

    def test_indexes_MobileNetV2(self):
        mobilenetv2 = MobileNetV2(input_shape=(224,224,3))

        sol_add = {27:18, 45:36, 54:45, 72:63, 81:72, 90:81, 107:98, 116:107, 134:125, 143:134}

        dic_add, dic_multiply = identify_residual_layer_indexes(mobilenetv2)

        self.assertEqual(sol_add, dic_add)
        self.assertEqual({}, dic_multiply)

    def test_index_MobileNetV3(self):
        mobilenetv3 = MobileNetV3Small(weights='imagenet', include_top=True, input_shape=(224,224,3), backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)

        sol_add = {35:26, 67:51, 83:67, 114:98, 146:130, 162:146}
        sol_mult = {15:8, 49:42, 64:57, 80:73, 96:89, 111:104, 128:121, 143:136, 159:152}

        dic_add, dic_multiply = identify_residual_layer_indexes(mobilenetv3)

        self.assertEqual(sol_add, dic_add)
        self.assertEqual(sol_mult, dic_multiply)

if __name__ == '__main__':

    model = MobileNetV2(weights='imagenet', include_top=True, input_shape=(224,224,3))
    
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0), metrics=['accuracy'])
    print(model.losses)
    model = add_regularization(model)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0), metrics=['accuracy'])
    print(model.losses)

    exit()

    for i in range(len(model.layers)):
        print(i)
        print(model.layers[i].name)        

    unittest.main()
