from keras.callbacks import Callback
from keras.layers import BatchNormalization
from keras.optimizers import SGD
import numpy as np

class UnfreezeLayersCallback(Callback):

    def __init__(self, epochs, num_layers, learning_rate):
        self.unfreeze_count_per_epoch = int(np.ceil(num_layers / epochs))
        self.learning_rate = learning_rate

    def on_epoch_begin(self, epoch, logs=None):
        
        num_layers = len(self.model.layers)
        already_unfreezed_until = num_layers - 1 - epoch*self.unfreeze_count_per_epoch
        unfreezed = []

        for i in range(already_unfreezed_until-self.unfreeze_count_per_epoch, already_unfreezed_until):
            if not isinstance(self.model.layers[i], BatchNormalization):
                self.model.layers[i].trainable = True
                unfreezed.append(i)

        self.model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=self.learning_rate, momentum=0.9, decay=0.0, nesterov=False), metrics=['accuracy'])

        print("Layers unfreezed: ", unfreezed)