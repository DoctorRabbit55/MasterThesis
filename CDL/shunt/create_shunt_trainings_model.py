import tensorflow as tf
import tf.keras as keras
from tf.keras.utils.generic_utils import get_custom_objects
from tf.keras.layers import deserialize as layer_from_config
from tf.keras.layers import Input, Add, Multiply, Subtract, Flatten, Lambda
import tf.keras.backend as K

import numpy as np

from CDL.shunt.Architectures import createShunt
from CDL.utils.keras_utils import HardSwish, hard_swish, identify_residual_layer_indexes

def create_shunt_trainings_model(model, model_shunt, shunt_locations):

    get_custom_objects().update({'hard_swish': hard_swish})

    add_input_index_dic, mult_input_index_dic = identify_residual_layer_indexes(model)
    add_input_tensors = {}
    mult_input_tensors = {}

    input_net = Input(model.input_shape[1:])    
    x = input_net  

    shunt_input = None

    for i in range(1,shunt_locations[1]+1):

        layer = model.layers[i]
        config = layer.get_config()

        # there is a bug in layer_from_config, where custom Activation are not passed correctly
        next_layer = layer_from_config({'class_name': layer.__class__.__name__, 'config': config})

        if isinstance(next_layer, Multiply):
            second_input_index = mult_input_index_dic[i]
            x = next_layer([x, mult_input_tensors[second_input_index]])
        elif isinstance(next_layer, Add):
            second_input_index = add_input_index_dic[i]
            x = next_layer([x, add_input_tensors[second_input_index]])
        else:
            x = next_layer(x)

        if i in add_input_index_dic.values():
            add_input_tensors[i] = x
        if i in mult_input_index_dic.values():
            mult_input_tensors[i] = x
        if i == shunt_locations[0]-1: # input of shunt
            print('input shunt:', layer.name)
            shunt_input = x

    output_original_model = x
    #output_original_model = Flatten()(output_original_model)
    #output_original_model = K.l2_normalize(output_original_model,axis=1)

    x = model_shunt(shunt_input)

    #x = Flatten()(x)
    #x = K.l2_normalize(x,axis=1)
    x = Subtract()([x, output_original_model])
    #x = Multiply()([x, x])
    #x = keras.backend.sum(x, axis=1)
    #x = keras.backend.sqrt(x)
    #x = Lambda(lambda x: x * 1/(model_shunt.output_shape[1]*model_shunt.output_shape[2]))(x)

    model_training = keras.models.Model(inputs=input_net, outputs=[x], name='shunt_training')

    for j in range(1,len(model_training.layers)):

        layer = model_training.layers[j]
        try:
            weights = model.get_layer(name=layer.name).get_weights()
        except:
            continue
        layer.trainable = False
        if len(weights) > 0:
            model_training.layers[j].set_weights(weights)



    return model_training