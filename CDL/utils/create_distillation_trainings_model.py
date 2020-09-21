import tensorflow as tf
import tensorflow.keras as keras
from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.layers import deserialize as layer_from_config
from tensorflow.keras.layers import Input, Add, Multiply, Subtract, Flatten, Lambda, Activation, Conv2D
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


def create_attention_transfer_model(model_student, model_teacher, shunt_locations, index_offset, max_number_transfers=3):

    # count how many COV2D layers there are
    conv_index_list = []
    for i, layer in enumerate(model_teacher.layers[shunt_locations[1]:]):
        if isinstance(layer, Conv2D):
            conv_index_list.append(shunt_locations[1]+i)

    number_conv = len(conv_index_list)
    if max_number_transfers >= number_conv:
        transfer_indices_teacher = conv_index_list
    else: # too many conv found
        indices = list(map(int, list(np.linspace(0, number_conv-1, max_number_transfers))))
        print(number_conv, indices)
        transfer_indices_teacher = list(np.asarray(conv_index_list)[indices])

    outputs_teacher = []
    outputs_student = []
    for index in transfer_indices_teacher:
        outputs_teacher.append(model_teacher.layers[index].output)
        outputs_student.append(model_student.layers[index+index_offset].output)

    attention_losses = []
    for i in range(len(outputs_student)):
        loss = Subtract()([outputs_teacher[i], outputs_student[i]])
        loss = Flatten()(loss)
        loss = K.l2_normalize(loss,axis=1)
        attention_losses.append(loss)

    input_net = Input(shape=model_student.input_shape[1:])
    model_at = Model(input_net, [model_student(input_net)] + attention_losses, 'attention_transfer')

    print(model_at.summary())

    return model_at