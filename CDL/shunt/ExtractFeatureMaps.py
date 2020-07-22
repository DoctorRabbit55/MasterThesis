from keras import Model

def getFeatureMaps(model, locs, data):

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

    fm1 = model_loc1.predict(data, verbose=1)
    fm2 = model_loc2.predict(data, verbose=1)

    return (fm1, fm2)