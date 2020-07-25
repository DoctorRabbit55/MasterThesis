import tensorflow as tf

from keras.applications import MobileNetV2
from keras.layers import Input, UpSampling2D, GlobalAveragePooling2D, Dense, Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, InputLayer, Add
from keras import Model
from keras.layers import deserialize as layer_from_config
from keras.optimizers import SGD

import queue

class MobileNetV2_extended(Model):

    def __init__(self, inputs, outputs):
        super().__init__(inputs=inputs, outputs=outputs, name='MobileNetV2')

    def call(self):
        super().call()

    @classmethod
    def create(self, input_shape=(32,32,3), num_classes=10, is_pretrained=False, mobilenet_shape=(224,224,3)):

        assert(input_shape[0] == input_shape[1])
        assert(224 % input_shape[0] == 0)

        mobilenet = None

        if is_pretrained:
        
            mobilenet = MobileNetV2(input_shape=mobilenet_shape, include_top=False, alpha=1.0, weights='imagenet')

            output_residual = queue.Queue(2)

            for layer in mobilenet.layers:
                layer.trainable = False

            input_net = Input(input_shape)
            scale_factor = 224 // input_shape[0]
            x = UpSampling2D((scale_factor, scale_factor))(input_net)

            for layer in mobilenet.layers[1:]:

                config = layer.get_config()
                next_layer = layer_from_config({'class_name': layer.__class__.__name__, 'config': config})

                if isinstance(layer, Add):
                    x = next_layer([x, output_residual.get()])
                else:
                    x = next_layer(x)
                
                if "block_" in layer.name and "_project_BN" in layer.name:
                    if output_residual.full():
                        output_residual.get()
                    output_residual.put(x)
                if "block_" in layer.name and "_add" in layer.name:
                    if output_residual.full():
                        output_residual.get()
                    output_residual.put(x)


            model_reduced = Model(input_net, x)

            for j in range(2,len(model_reduced.layers)):
                layer = model_reduced.layers[j]
                weights = mobilenet.get_layer(name=layer.name).get_weights()
                if len(weights) > 0:
                    model_reduced.layers[j].set_weights(weights)

            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dense(512, activation='relu')(x)
            preds = Dense(num_classes, activation='softmax')(x)

            return MobileNetV2_extended(inputs=input_net, outputs=preds)

        else:
            mobilenet = MobileNetV2(input_shape=mobilenet_shape, include_top=True, weights=None, classes=num_classes)

            scale_factor = mobilenet_shape[0] // input_shape[0]

            if scale_factor > 1:

                output_residual = queue.Queue(2)

                input_net = Input(input_shape)
                x = UpSampling2D((scale_factor, scale_factor))(input_net)

                for layer in mobilenet.layers[1:]:

                    config = layer.get_config()
                    next_layer = layer_from_config({'class_name': layer.__class__.__name__, 'config': config})

                    if isinstance(layer, Add):
                        x = next_layer([x, output_residual.get()])
                    else:
                        x = next_layer(x)
                    
                    if "block_" in layer.name and "_project_BN" in layer.name:
                        if output_residual.full():
                            output_residual.get()
                        output_residual.put(x)
                    if "block_" in layer.name and "_add" in layer.name:
                        if output_residual.full():
                            output_residual.get()
                        output_residual.put(x)

                return MobileNetV2_extended(inputs=input_net, outputs=x)
            else:
                return MobileNetV2_extended(mobilenet.input, mobilenet.output)

    def getKnowledgeQuotients(self, data):

        (x_test, y_test) = data

        know_quot = {}
        
        residual_block_indexes = [2, 4, 5, 7, 8, 9, 11, 12, 14, 15]

        for block_index in residual_block_indexes:

            model_reduced = self.dropBlocks([block_index])

            model_reduced.compile(loss='categorical_crossentropy', optimizer=SGD(lr=2e-2, momentum=0.9, decay=0.0, nesterov=False), metrics=['accuracy'])
        
            val_loss, val_acc = model_reduced.evaluate(x_test, y_test, verbose=1)
            print('Test loss for block {}: {:.5f}'.format(block_index, val_loss))
            print('Test accuracy for block {}: {:.5f}'.format(block_index, val_acc))

            know_quot[block_index] = val_acc

        return know_quot


    def dropBlocks(self, block_indexes):
            
        output_residual = queue.Queue(2)

        input_net = Input(self.input_shape[1:])
        x = input_net
        
        for layer in self.layers[1:]:

            block_to_delete = False
            for block_index in block_indexes:
                block_name = 'block_' + str(block_index)
                if block_name in layer.name:
                    block_to_delete = True             

            if not block_to_delete:
                config = layer.get_config()
                next_layer = layer_from_config({'class_name': layer.__class__.__name__, 'config': config})

                if isinstance(layer, Add):
                    x = next_layer([x, output_residual.get()])
                else:
                    x = next_layer(x)
                
                if "block_" in layer.name and "_project_BN" in layer.name:
                    if output_residual.full():
                        output_residual.get()
                    output_residual.put(x)
                if "block_" in layer.name and "_add" in layer.name:
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


    def insertShunt(self, shunt, block_to_replace_indexes):

        is_shunt_inserted = False
        output_residual = queue.Queue(2)

        input_net = Input(self.input_shape[1:])
        x = input_net

        for layer in self.layers[1:]:
            
            block_to_delete = False
            for block_index in block_to_replace_indexes:
                block_name = 'block_' + str(block_index)
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
                next_layer = layer_from_config({'class_name': layer.__class__.__name__, 'config': config})

                if isinstance(layer, Add):
                    x = next_layer([x, output_residual.get()])
                else:
                    x = next_layer(x)
                
                if "block_" in layer.name and "_project_BN" in layer.name:
                    if output_residual.full():
                        output_residual.get()
                    output_residual.put(x)
                if "block_" in layer.name and "_add" in layer.name:
                    if output_residual.full():
                        output_residual.get()
                    output_residual.put(x)

        finished_model = MobileNetV2_extended(input_net, x)

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