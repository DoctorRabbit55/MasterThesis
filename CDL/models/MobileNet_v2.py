import tensorflow as tf

from keras.applications import MobileNetV2
from keras.layers import Input, UpSampling2D, GlobalAveragePooling2D, Dense, Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, InputLayer, Add
from keras import Model
from keras.layers import deserialize as layer_from_config
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

import queue

class MobileNetV2_extended(Model):

    def __init__(self, inputs, outputs):
        super().__init__(inputs=inputs, outputs=outputs, name='MobileNetV2')

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

            output_residual = queue.Queue(2)

            input_net = Input(input_shape)
            if scale_factor > 1:
                x = UpSampling2D((scale_factor, scale_factor))(input_net)
            else:
                x = input_net

            layer = mobilenet.layers[2]
            config = layer.get_config()
            config['strides'] = (1,1)
            config['padding'] = 'same'
            next_layer = layer_from_config({'class_name': layer.__class__.__name__, 'config': config})
            x = next_layer(x)

            for layer in mobilenet.layers[3:]:

                config = layer.get_config()

                # CIFAR10 changes
                if layer.name == 'block_1_depthwise':
                    config['strides'] = (1,1)
                    config['padding'] = 'same'
                if layer.name == 'block_1_pad':
                    continue

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

    def getFeatureMaps(self, locs, x_train):

        layer_loc1 = self.get_layer(name='block_' + str(locs[0]) + '_expand')
        layer_loc2 = self.get_layer(name='block_' + str(locs[1]) + '_add')

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