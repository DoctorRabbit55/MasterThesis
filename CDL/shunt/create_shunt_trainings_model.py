import tensorflow as tf
import tensorflow.keras as keras
from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.layers import deserialize as layer_from_config
from tensorflow.keras.layers import Input, Add, Multiply, Subtract, Flatten, Lambda
import tensorflow.keras.backend as K

import numpy as np

from CDL.shunt.Architectures import createShunt
from CDL.utils.keras_utils import HardSwish, hard_swish, identify_residual_layer_indexes

def create_shunt_trainings_model(model, model_shunt, shunt_locations):

    shunt_input = model.layers[shunt_locations[0]-1].output
    output_original_model = model.layers[shunt_locations[1]].output
    output_original_model = Flatten()(output_original_model)
    #output_original_model = K.l2_normalize(output_original_model,axis=1)

    x = model_shunt(shunt_input)

    x = Flatten()(x)
    #x = K.l2_normalize(x,axis=1)
    x = Subtract()([x, output_original_model])
    #x = Multiply()([x, x])
    #x = keras.backend.sum(x, axis=1)
    #x = keras.backend.sqrt(x)
    #x = Lambda(lambda x: x * 1/(model_shunt.output_shape[1]*model_shunt.output_shape[2]))(x)

    model_training = keras.models.Model(inputs=model.input, outputs=[x], name='shunt_training')
    for layer in model_training.layers: layer.trainable = False
    model_training.get_layer(name=model_shunt.name).trainable = True

    return model_training