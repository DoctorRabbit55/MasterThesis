from keras import Model
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

def getFeatureMaps(model, locs, x_train):

    layer_loc1 = model.get_layer(name='block_' + str(locs[0]) + '_expand')
    layer_loc2 = model.get_layer(name='block_' + str(locs[1]) + '_add')

    model_loc1 = Model(inputs=model.input, outputs=layer_loc1.input)
    model_loc2 = Model(inputs=model.input, outputs=layer_loc2.output)

    for j in range(1,len(model_loc1.layers)):
        layer = model_loc1.layers[j]
        weights = model.get_layer(name=layer.name).get_weights()
        if len(weights) > 0:
            model_loc1.layers[j].set_weights(weights)

    for j in range(1,len(model_loc2.layers)):
        layer = model_loc2.layers[j]
        weights = model.get_layer(name=layer.name).get_weights()
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

    number_maps = 2*len(x_train)

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