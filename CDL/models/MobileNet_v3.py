import tensorflow as tf
import keras
from keras_applications.mobilenet_v3 import MobileNetV3Small
from keras_applications.mobilenet_v3 import MobileNetV3Large
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Input, UpSampling2D, GlobalAveragePooling2D, Dense, Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, InputLayer, Add, Multiply, Reshape, Activation, Dropout, Flatten, Softmax
from keras import Model
from keras.layers import deserialize as layer_from_config
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

import queue

def hard_swish(x):
    return Multiply()([Activation(keras.activations.hard_sigmoid)(x), x])

class MobileNetV3_extended(Model):

    def __init__(self, inputs, outputs):
        super().__init__(inputs=inputs, outputs=outputs, name='MobileNetV3')

    @classmethod
    def create(self, input_shape=(32,32,3), num_classes=10, is_pretrained=False, mobilenet_shape=(224,224,3), is_small=True):

        assert(input_shape[0] == input_shape[1])
        assert(224 % input_shape[0] == 0)

        get_custom_objects().update({'hard_swish': Activation(hard_swish)})

        mobilenet = None
        self.is_small = is_small

        if is_pretrained:   

            if is_small:
                mobilenet = MobileNetV3Small(weights='imagenet', include_top=False, classes=num_classes, input_shape=mobilenet_shape, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
            else:
                mobilenet = MobileNetV3Large(weights='imagenet', include_top=False, classes=num_classes, input_shape=mobilenet_shape, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)

            for layer in mobilenet.layers:
                layer.trainable = False

            input_net = Input(input_shape)
            scale_factor = 224 // input_shape[0]
            if scale_factor > 1:
                x = UpSampling2D((scale_factor, scale_factor))(input_net)
            else:
                x = input_net

            block_num = 1
            output_residual = queue.Queue(2)
            multiply_input = queue.Queue(2)

            # add first 4 layers which are not part of blocks
            for layer in mobilenet.layers[1:5]:
                config = layer.get_config()
                next_layer = layer_from_config({'class_name': layer.__class__.__name__, 'config': config})
                x = next_layer(x)

            for layer in mobilenet.layers[5:]:

                config = layer.get_config()
                next_layer = layer_from_config({'class_name': layer.__class__.__name__, 'config': config})

                # look if next block starts
                if next_layer.name.endswith('/expand'):
                    block_num += 1 

                # add block name in front of layer name
                next_layer._name = 'block_' + str(block_num) + '_' + next_layer.name

                # advance tensor
                if isinstance(layer, Multiply):
                    x = next_layer([x, multiply_input.get()])
                elif isinstance(layer, Add):
                    x = next_layer([x, output_residual.get()])
                else:
                    x = next_layer(x)
                
                # save necessary outputs
                if 'activation_' in next_layer.name:
                    if multiply_input.full():
                        multiply_input.get()
                    multiply_input.put(x)
                if "/project/BatchNorm" in layer.name:
                    if output_residual.full():
                        output_residual.get()
                    output_residual.put(x)
                if "/Add" in layer.name:
                    if output_residual.full():
                        output_residual.get()
                    output_residual.put(x)


            model_reduced = Model(input_net, x)

            for j in range(2,len(model_reduced.layers)):
                layer = model_reduced.layers[j]
                name_in_original_model = ""

                i = 0
                while i < 10:
                    if len(layer.name.split('_')) > i:
                        if layer.name.split('_')[i] == 'block':
                            i += 2
                            continue
                        name_in_original_model += '_' + layer.name.split('_')[i]
                    i += 1
                if name_in_original_model[0] == '_': name_in_original_model = name_in_original_model[1:]
                weights = mobilenet.get_layer(name=name_in_original_model).get_weights()
                if len(weights) > 0:
                    model_reduced.layers[j].set_weights(weights)

            x = GlobalAveragePooling2D()(x)
            if self.is_small:
                x = Reshape((1,1,576))(x)
            else:
                x = Reshape((1,1,960))(x)
            x = Conv2D(1024, kernel_size=1, padding='same')(x)
            layer = Activation(hard_swish)
            x = Dropout(rate=0.2)(x)
            x = Conv2D(num_classes, kernel_size=1, padding='same')(x)
            x = Flatten()(x)
            x = Softmax()(x)

            return MobileNetV3_extended(inputs=input_net, outputs=x)

        else:
            if is_small:
                mobilenet = MobileNetV3Small(weights=None, include_top=True, classes=num_classes, input_shape=mobilenet_shape, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
            else:
                mobilenet = MobileNetV3Large(weights=None, include_top=True, classes=num_classes, input_shape=mobilenet_shape, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)

            scale_factor = mobilenet_shape[0] // input_shape[0]

            block_num = 1
            output_residual = queue.Queue(2)
            multiply_input = queue.Queue(2)

            input_net = Input(input_shape)
            
            if scale_factor > 1:
                x = UpSampling2D((scale_factor, scale_factor))(input_net)
            else:
                x = input_net
            
            # CIFAR10 changes
            number_layers_stride_changed = 0

            # add first 4 layers which are not part of blocks
            for layer in mobilenet.layers[1:5]:
                config = layer.get_config()
                if 'strides' in config and number_layers_stride_changed > 0:
                    if config['strides'] == (2,2):
                        config['strides'] = (1,1)
                        number_layers_stride_changed -= 1
                next_layer = layer_from_config({'class_name': layer.__class__.__name__, 'config': config})
                x = next_layer(x)
            
            for layer in mobilenet.layers[5:-6]:

                config = layer.get_config()
                if 'strides' in config and number_layers_stride_changed > 0:
                    if config['strides'] == (2,2):
                        config['strides'] = (1,1)
                        number_layers_stride_changed -= 1
                next_layer = layer_from_config({'class_name': layer.__class__.__name__, 'config': config})

                # look if next block startsF
                if next_layer.name.endswith('/expand'):
                    block_num += 1 

                # add block name in front of layer name
                next_layer._name = 'block_' + str(block_num) + '_' + next_layer.name

                # advance tensor
                if isinstance(layer, Multiply):
                    x = next_layer([x, multiply_input.get()])
                elif isinstance(layer, Add):
                    x = next_layer([x, output_residual.get()])
                else:
                    x = next_layer(x)
                
                # save necessary outputs
                if 'activation' in next_layer.name:
                    if multiply_input.full():
                        multiply_input.get()
                    multiply_input.put(x)
                if "/project/BatchNorm" in layer.name:
                    if output_residual.full():
                        output_residual.get()
                    output_residual.put(x)
                if "/Add" in layer.name:
                    if output_residual.full():
                        output_residual.get()
                    output_residual.put(x)

            for layer in mobilenet.layers[-6:]:
                config = layer.get_config()
                next_layer = layer_from_config({'class_name': layer.__class__.__name__, 'config': config})
                x = next_layer(x)

        return MobileNetV3_extended(inputs=input_net, outputs=x)

    def getKnowledgeQuotients(self, data, val_acc_model):

        (x_test, y_test) = data

        know_quot = {}
        residual_block_indexes = []
        if self.is_small:
            residual_block_indexes = [3, 5, 6, 8, 10, 11]
        else:
            residual_block_indexes = [1, 3, 5, 6, 8, 9, 10, 12, 14, 15]

        for block_index in residual_block_indexes:

            model_reduced = self.dropBlocks([block_index])

            model_reduced.compile(loss='categorical_crossentropy', optimizer=SGD(lr=2e-2, momentum=0.9, decay=0.0, nesterov=False), metrics=['accuracy'])
        
            val_loss, val_acc = model_reduced.evaluate(x_test, y_test, verbose=1)
            print('Test loss for block {}: {:.5f}'.format(block_index, val_loss))
            print('Test accuracy for block {}: {:.5f}'.format(block_index, val_acc))

            know_quot[block_index] = val_acc / val_acc_model

        return know_quot


    def dropBlocks(self, block_indexes):
        get_custom_objects().update({'hard_swish': Activation(hard_swish)})

        output_residual = queue.Queue(2)
        multiply_input = queue.Queue(2)

        input_net = Input(self.input_shape[1:])
        x = input_net
        
        for layer in self.layers[1:]:

            block_to_delete = False
            for block_index in block_indexes:
                block_name = 'block_' + str(block_index) + '_'
                if block_name in layer.name:
                    block_to_delete = True             

            if not block_to_delete:

                config = layer.get_config()

                if 'activation' in config:
                    if type(config['activation']) is not str:
                        next_layer = Activation(hard_swish, name=layer.name)
                    else:
                        next_layer = layer_from_config({'class_name': layer.__class__.__name__, 'config': config})  
                else:
                    next_layer = layer_from_config({'class_name': layer.__class__.__name__, 'config': config})

                if isinstance(layer, Multiply):
                    x = next_layer([x, multiply_input.get()])
                elif isinstance(layer, Add):
                    x = next_layer([x, output_residual.get()])
                else:
                    x = next_layer(x)
                
                if 'activation' in next_layer.name:
                    if multiply_input.full():
                        multiply_input.get()
                    multiply_input.put(x)
                if "block" in layer.name and "/project/BatchNorm" in layer.name:
                    if output_residual.full():
                        output_residual.get()
                    output_residual.put(x)
                if "block" in layer.name and "/Add" in layer.name:
                    if output_residual.full():
                        output_residual.get()
                    output_residual.put(x)

        model_reduced = Model(input_net, x)

        for j in range(1,len(model_reduced.layers)):
            layer = model_reduced.layers[j]
            weights = self.get_layer(name=layer.name).get_weights()
            if len(weights) > 0:
                model_reduced.layers[j].set_weights(weights)

        return model_reduced

    def getFeatureMaps(self, locs, x_train):
        get_custom_objects().update({'hard_swish': Activation(hard_swish)})

        if locs[0] == 1:
            layer_loc1 = self.get_layer(name='block_1_expanded_conv/depthwise')
            layer_loc2 = self.get_layer(name='block_1_expanded_conv/Add')
        else:
            layer_loc1 = self.get_layer(name='block_' + str(locs[0]) + '_expanded_conv_' + str(locs[0]-1) + '/expand')
            layer_loc2 = self.get_layer(name='block_' + str(locs[0]) + '_expanded_conv_' + str(locs[0]-1) + '/Add')
        

        model_loc1 = Model(inputs=self.input, outputs=layer_loc1.input)
        model_loc2 = Model(inputs=self.input, outputs=layer_loc2.output)

        for j in range(1,len(model_loc1.layers)):
            layer = model_loc1.layers[j]
            weights = self.get_layer(name=layer.name).get_weights()
            if len(weights) > 0:
                model_loc1.layers[j].set_weights(weights)

        for j in range(1,len(model_loc2.layers)):
            layer = model_loc2.layers[j]
            weights = self.get_layer(name=layer.name).get_weights()
            if len(weights) > 0:
                model_loc2.layers[j].set_weights(weights)

        datagen = ImageDataGenerator(
            featurewise_center=False, 
            featurewise_std_normalization=False, 
            rotation_range=0.0,
            width_shift_range=0.2, 
            height_shift_range=0.2, 
            vertical_flip=False,
            horizontal_flip=True)
        datagen.fit(x_train)

        first_batch = True
        fm1 = fm2 = None

        number_maps = len(x_train)

        for x_batch in datagen.flow(x_train, None, batch_size=128):
            if first_batch:
                first_batch = False

                fm1 = np.array(model_loc1.predict(x_batch))
                fm2 = np.array(model_loc2.predict(x_batch))

            else:
                fm1 = np.append(fm1, model_loc1.predict(x_batch), axis=0)
                fm2 = np.append(fm2, model_loc2.predict(x_batch), axis=0)

            print('Extract feature maps: {} maps done of {}'.format(fm1.shape[0], number_maps), end='\r')
            if(fm1.shape[0] > number_maps):
                break

        return (fm1, fm2)


    def insertShunt(self, shunt, block_to_replace_indexes):

        is_shunt_inserted = False
        output_residual = queue.Queue(2)
        multiply_input = queue.Queue(2)

        input_net = Input(self.input_shape[1:])
        x = input_net

        for layer in self.layers[1:]:
            
            block_to_delete = False
            for block_index in block_to_replace_indexes:
                block_name = 'block_' + str(block_index) + '_'
                if block_name in layer.name:
                    block_to_delete = True             

            if block_to_delete:
                if not is_shunt_inserted:
                    is_shunt_inserted = True
                    x = shunt(x)
                    if output_residual.full():
                        output_residual.get()
                    output_residual.put(x)

            else:
                config = layer.get_config()

                if 'activation' in config:
                    if type(config['activation']) is not str:
                        next_layer = Activation(hard_swish, name=layer.name)
                    else:
                        next_layer = layer_from_config({'class_name': layer.__class__.__name__, 'config': config})  
                else:
                    next_layer = layer_from_config({'class_name': layer.__class__.__name__, 'config': config})

                if isinstance(layer, Multiply):
                    x = next_layer([x, multiply_input.get()])
                elif isinstance(layer, Add):
                    x = next_layer([x, output_residual.get()])
                else:
                    x = next_layer(x)
                
                if 'activation' in next_layer.name:
                    if multiply_input.full():
                        multiply_input.get()
                    multiply_input.put(x)
                if "block" in layer.name and "/project/BatchNorm" in layer.name:
                    if output_residual.full():
                        output_residual.get()
                    output_residual.put(x)
                if "block" in layer.name and "/Add" in layer.name:
                    if output_residual.full():
                        output_residual.get()
                    output_residual.put(x)

        finished_model = MobileNetV3_extended(input_net, x)

        for j in range(1,len(finished_model.layers)):
            layer = finished_model.layers[j]
            if layer.name == 'shunt':
                weights = shunt.get_weights()
            else:
                weights = self.get_layer(name=layer.name).get_weights()
                finished_model.layers[j].trainable = False
            if len(weights) > 0:
                finished_model.layers[j].set_weights(weights)

        return finished_model