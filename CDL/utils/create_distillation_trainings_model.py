import tensorflow as tf
import tensorflow.keras as keras
from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.layers import deserialize as layer_from_config
from tensorflow.keras.layers import Input, Add, Multiply, Subtract, Flatten, Lambda, Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

import numpy as np

from CDL.shunt.Architectures import createShunt
from CDL.utils.keras_utils import HardSwish, hard_swish, identify_residual_layer_indexes

def create_dark_knowledge_model(model_student, model_teacher, temperature=3):
    
    #assert isinstance(model_teacher.layers[-1], keras.layers.Activation), 'Teacher model: Last layer must be a softmax layer, but it is {}!'.format(str(model_teacher.layers[-1]))
    #assert isinstance(model_student.layers[-1], keras.layers.Activation), 'Student model: Last layer must be a softmax layer, but it is {}!'.format(str(model_teacher.layers[-1]))

    # teacher network
    softmax_input = model_teacher.layers[-2].output
    softmax_input = Lambda(lambda x: x / temperature, name='Temperature_teacher')(softmax_input)
    prediction_with_temperature = Activation('softmax', name='Softened_softmax_teacher')(softmax_input)
    model_teacher = Model(model_teacher.input, prediction_with_temperature, name='Teacher')
    model_teacher.trainable = False

    # student network
    softmax_input = model_student.layers[-2].output
    softend_softmax_input = Lambda(lambda x: x / temperature, name='Temperature_student')(softmax_input)
    prediction_with_temperature = Activation('softmax', name='Softened_softmax_student')(softend_softmax_input)
    prediction_without_temperature = Activation('softmax', name='softmax_student')(softmax_input)
    model_student = Model(model_student.input, [prediction_without_temperature, prediction_with_temperature], name='Student')

    input_net = Input(shape=model_student.input_shape[1:])
    dark_knowledge_loss = Subtract(name='dark_knowledge_without_temperature')([model_teacher(input_net), model_student(input_net)[1]])
    dark_knowledge_loss = Lambda(lambda x: x * temperature * temperature, name='dark_knowledge')(dark_knowledge_loss)
    model_dark_knowledge = Model(input_net, [model_student(input_net)[0], dark_knowledge_loss], name='Final_model_dark_knowledge')

    print(model_dark_knowledge.summary())
    return model_dark_knowledge


def add_attention_transfer(model_student, model_teacher, shunt_locations):

    input_teacher = model_student.layers[shunt_locations[1]].output
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