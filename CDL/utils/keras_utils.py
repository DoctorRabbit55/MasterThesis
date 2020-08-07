import keras
from keras.layers import Add, Multiply, Input, Activation
from keras.layers import deserialize as layer_from_config

import unittest
from keras.applications import MobileNetV2
#from keras_applications.mobilenet_v3 import MobileNetV3Small

def hard_swish(x):
    return Multiply()([Activation(keras.activations.hard_sigmoid)(x), x])

def get_index_of_layer(model, layer):
    for i in range(len(model.layers)):
        if layer.name == model.layers[i].name:
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
            input_layers = layer._inbound_nodes[0].inbound_layers
            incoming_index = get_first_layer_by_index(model, input_layers)
            add_incoming_index_dic[i] = incoming_index

        if isinstance(layer, Multiply):
            input_layers = layer._inbound_nodes[0].inbound_layers
            incoming_index = get_first_layer_by_index(model, input_layers)
            mult_incoming_index_dic[i] = incoming_index
        
    return add_incoming_index_dic, mult_incoming_index_dic

def modify_model(model, layer_indexes_to_delete=[], layer_indexes_to_output=[], shunt_to_insert=None):

    add_input_index_dic, mult_input_index_dic = identify_residual_layer_indexes(model)
    add_input_tensors = {}
    mult_input_tensors = {}

    outputs = []
    
    got_shunt_inserted = False
    shunt_output = None

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

        # there is a bug in layer_from_config, where custom Activation are not passed correctly
        if 'activation' in config:
            if type(config['activation']) is not str:
                next_layer = Activation(hard_swish, name=layer.name)
            else:
                next_layer = layer_from_config({'class_name': layer.__class__.__name__, 'config': config})  
        else:
            next_layer = layer_from_config({'class_name': layer.__class__.__name__, 'config': config})
        should_delete = False
        for layer_index_to_delete in layer_indexes_to_delete:
            if i == layer_index_to_delete:
                should_delete = True
                break

        if should_delete:
            if shunt_to_insert and not got_shunt_inserted:
                for shunt_layer in shunt_to_insert.layers[1:]:
                    x = shunt_layer(x)
                    shunt_output = x
                got_shunt_inserted = True
            continue

        if isinstance(next_layer, Multiply):
            second_input_index = mult_input_index_dic[i]
            x = next_layer([x, mult_input_tensors[second_input_index]])
        elif isinstance(next_layer, Add):
            second_input_index = add_input_index_dic[i]
            if second_input_index == -1: # use shunt
                x = next_layer([x, shunt_output])
            else:
                x = next_layer([x, add_input_tensors[second_input_index]])
        else:
            x = next_layer(x)

        if i in add_input_index_dic.values():
            add_input_tensors[i] = x
        if i in mult_input_index_dic.values():
            mult_input_tensors[i] = x

        if i in layer_indexes_to_output:
            outputs.append(x)

    outputs.append(x)
    assert(len(outputs) == len(layer_indexes_to_output)+1)
    model_reduced = keras.models.Model(inputs=input_net, outputs=outputs, name=model.name)

    for j in range(1,len(model_reduced.layers)):

        layer = model_reduced.layers[j]

        # skip shunt
        if shunt_to_insert:
            if j in range(layer_indexes_to_delete[0], layer_indexes_to_delete[0]+len(shunt_to_insert.layers)-1):
                weights = shunt_to_insert.get_layer(name=layer.name).get_weights()
                if len(weights) > 0:
                    model_reduced.layers[j].set_weights(weights)
                continue

        weights = model.get_layer(name=layer.name).get_weights()
        if len(weights) > 0:
            model_reduced.layers[j].set_weights(weights)

    return model_reduced

def extract_feature_maps(model, x_data, locations):

    model = modify_model(model, layer_indexes_to_output=locations)

    predictions = model.predict(x_data, verbose=1)

    return predictions[:-1]


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
    
    for i in range(len(model.layers)):
        print(i)
        print(model.layers[i].name)

    unittest.main()