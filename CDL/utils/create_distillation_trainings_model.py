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

    input_net = model_student.input
    outputs_student = model_student(input_net)
    outputs_teacher = model_teacher(input_net)
    dark_knowledge_loss = Subtract(name='dark_knowledge_without_temperature')([outputs_teacher, outputs_student[1]])
    dark_knowledge_loss = Lambda(lambda x: x * temperature * temperature, name='dark_knowledge')(dark_knowledge_loss)
    model_dark_knowledge = Model(input_net, [outputs_student[0], dark_knowledge_loss], name='Final_model_dark_knowledge')

    print(model_dark_knowledge.summary())
    return model_dark_knowledge


def create_attention_transfer_model(model_student, model_teacher, shunt_locations, index_offset, max_number_transfers=3):

    # count how many COV2D layers there are
    conv_index_list = []
    for i, layer in enumerate(model_teacher.layers[shunt_locations[1]:-3]):
        if isinstance(layer, Conv2D):
            conv_index_list.append(shunt_locations[1]+i)

    number_conv = len(conv_index_list)
    if max_number_transfers >= number_conv:
        transfer_indices_teacher = conv_index_list
    else: # too many conv found
        indices = list(map(int, list(np.linspace(0, number_conv-1, max_number_transfers))))
        transfer_indices_teacher = list(np.asarray(conv_index_list)[indices])

    outputs_teacher = []
    outputs_student = []
    for index in transfer_indices_teacher:
        print('Chose {} layer for attention transfer!'.format(model_teacher.layers[index].name))
        outputs_teacher.append(model_teacher.layers[index].output)
        outputs_student.append(model_student.layers[index+index_offset].output)

    model_student_with_outputs = Model(model_student.input, [model_student.output] + outputs_student, name='Student')
    model_teacher_with_outputs = Model(model_teacher.input, outputs_teacher, name='Teacher')
    model_teacher_with_outputs.trainable = False

    input_net = model_student.input

    outputs_teacher = model_teacher_with_outputs(input_net)
    outputs_student = model_student_with_outputs(input_net)

    attention_losses = []
    for i in range(len(outputs_teacher)):
        loss = Subtract()([outputs_teacher[i], outputs_student[i+1]])
        loss = Flatten(name='a_t_{}'.format(i))(loss)
        #loss = K.l2_normalize(loss,axis=1)
        attention_losses.append(loss)

    model_at = Model(input_net, [outputs_student[0]] + attention_losses, name='attention_transfer')

    print(model_at.summary())

    return model_at


def create_knowledge_distillation_model(model_student, model_teacher, add_dark_knowledge=False, temperature=3, add_attention_transfer=False, shunt_locations=None, index_offset=None, max_number_transfers=None):

    outputs_teacher = []
    outputs_student = []

    if add_attention_transfer:
        # count how many COV2D layers there are
        conv_index_list = []
        for i, layer in enumerate(model_teacher.layers[shunt_locations[1]:-3]):
            if isinstance(layer, Conv2D):
                conv_index_list.append(shunt_locations[1]+i)

        number_conv = len(conv_index_list)

        if not max_number_transfers: # auto mode
            max_number_transfers = np.max(2, np.ceil(number_conv / 3))

        if max_number_transfers >= number_conv:
            transfer_indices_teacher = conv_index_list
        else: # too many conv found
            indices = list(map(int, list(np.linspace(0, number_conv-1, max_number_transfers))))
            transfer_indices_teacher = list(np.asarray(conv_index_list)[indices])

        for index in transfer_indices_teacher:
            assert model_teacher.layers[index].name == model_student.layers[index+index_offset].name[6:], 'Index offset of shunt inserted model seems to be wrong! Layer names: teacher: {}, student: {}'.format(model_teacher.layers[index].name,model_student.layers[index+index_offset].name[6:])
            print('Chose {} layer for attention transfer!'.format(model_teacher.layers[index].name))
            outputs_teacher.append(model_teacher.layers[index].output)
            outputs_student.append(model_student.layers[index+index_offset].output)

    if add_dark_knowledge:
        # teacher network
        softmax_input = model_teacher.layers[-2].output
        softmax_input = Lambda(lambda x: x / temperature, name='Temperature_teacher')(softmax_input)
        prediction_with_temperature = Activation('softmax', name='Softened_softmax_teacher')(softmax_input)
        outputs_teacher.append(prediction_with_temperature)

        # student network
        softmax_input = model_student.layers[-2].output
        softend_softmax_input = Lambda(lambda x: x / temperature, name='Temperature_student')(softmax_input)
        prediction_with_temperature = Activation('softmax', name='Softened_softmax_student')(softend_softmax_input)
        outputs_student.append(prediction_with_temperature)    
    
    model_student_with_outputs = Model(model_student.input, [model_student.output] + outputs_student, name='Student')
    model_teacher_with_outputs = Model(model_teacher.input, outputs_teacher, name='Teacher')
    model_teacher_with_outputs.trainable = False

    input_net = model_student.input

    outputs_teacher = model_teacher_with_outputs(input_net)
    outputs_student = model_student_with_outputs(input_net)

    if not isinstance(outputs_teacher, list):   # outputs has to be a list
        outputs_teacher = [outputs_teacher]

    losses = []

    if add_attention_transfer:  # add layers for attention transfer loss
        for i in range(len(outputs_teacher)-1): 
            loss = Subtract()([outputs_teacher[i], outputs_student[i+1]])
            loss = Flatten(name='a_t_{}'.format(i))(loss)
            #loss = K.l2_normalize(loss,axis=1)
            losses.append(loss)

    if add_dark_knowledge:   # add dark knowledge loss
        dark_knowledge_loss = Subtract(name='dark_knowledge_without_temperature')([outputs_teacher[-1], outputs_student[-1]])
        dark_knowledge_loss = Lambda(lambda x: x * temperature * temperature, name='dark_knowledge')(dark_knowledge_loss)
        losses.append(dark_knowledge_loss)

    model_distillation = Model(input_net, [outputs_student[0]] + losses, name='knowledge_distillation')

    print(model_distillation.summary())

    return model_distillation