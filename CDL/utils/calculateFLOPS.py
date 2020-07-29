
from keras.layers import Conv2D, DepthwiseConv2D, Add, Input
from keras.models import Model
from keras.applications import VGG16

def calculateFLOPs_model(model):
    
    flops_dic = {'conv2d':0, 'depthwise_conv2d':0, 'add':0, 'total':0}

    for layer in model.layers:

        if layer.__class__ is Model:
            for layer_ in layer.layers:
                flops_dic = calculateFLOPs_layer(layer_, flops_dic)

        else:
            flops_dic = calculateFLOPs_layer(layer, flops_dic)

    return flops_dic

def calculateFLOPs_blocks(model, block_to_measure_indexes):

    flops_dic = {'conv2d':0, 'depthwise_conv2d':0, 'add':0, 'total':0}

    for layer in model.layers:

        if layer.__class__ is Model:
            for layer_ in layer.layers:
                
                block_to_measure = False
                for block_index in block_to_measure_indexes:
                    block_name = 'block_' + str(block_index) + '_'
                    if block_name in layer_.name:
                        block_to_measure = True  
                
                if block_to_measure:
                    flops_dic = calculateFLOPs_layer(layer_, flops_dic)
        else:

            block_to_measure = False
            for block_index in block_to_measure_indexes:
                block_name = 'block_' + str(block_index) + '_'
                if block_name in layer.name:
                    block_to_measure = True  
            
            if block_to_measure:
                flops_dic = calculateFLOPs_layer(layer, flops_dic)

    return flops_dic

def calculateFLOPs_layer(layer, flops_dic):
    
    config = layer.get_config()
    flops_layer = 0
    if layer.__class__ is Conv2D:

        filters_input = layer.input_shape[3]
        image_size = (layer.input_shape[1], layer.input_shape[2])
        filters_output = config['filters']
        kernel_size = config['kernel_size'][0]
        strides = config['strides'][0]

        flops_layer = filters_input * image_size[0] * image_size[1] * filters_output * kernel_size * kernel_size / strides / strides
        flops_dic['total'] += flops_layer
        flops_dic['conv2d'] += flops_layer

    if layer.__class__ is DepthwiseConv2D:

        filters_input = layer.input_shape[3]
        image_size = (layer.input_shape[1], layer.input_shape[2])
        filters_output = filters_input
        kernel_size = config['kernel_size'][0]
        strides = config['strides'][0]

        flops_layer = filters_input * image_size[0] * image_size[1] * kernel_size * kernel_size / strides / strides
        flops_dic['total'] += flops_layer
        flops_dic['depthwise_conv2d'] += flops_layer

    if layer.__class__ is Add:
        filters_input = layer.input_shape[0][3]
        image_size = layer.input_shape[0][1]
        
        flops_layer = image_size * image_size * filters_input
        flops_dic['total'] += flops_layer
        flops_dic['add'] += flops_layer

    return flops_dic



if __name__ == '__main__':

    input_model = Input(shape=(8,8,3))
    x = Conv2D(filters=256, kernel_size=5, strides=1, padding='same')(input_model)
    model = Model(input_model, x)
    
    flops = calculateFLOPs_model(model)
    assert(flops['total'] == 1228800)

    x = DepthwiseConv2D(kernel_size=5, strides=(1,1), padding='same')(input_model)
    x = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(x)
    model = Model(input_model, x)

    flops = calculateFLOPs_model(model)
    assert(flops['total'] == 53952)

    model = VGG16(weights='imagenet', include_top=False, input_shape=(126,224,3))
    flops = calculateFLOPs_model(model)
    print(flops)