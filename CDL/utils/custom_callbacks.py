from keras.callbacks import Callback
import numpy as np

class UnfreezeLayersCallback(Callback):

    def __init__(self, epochs, num_layers):
        self.unfreeze_count_per_epoch = int(np.ceil(num_layers / epochs))

    def on_epoch_begin(self, epoch, logs=None):
        
        num_layers = len(self.model.layers)
        already_unfreezed_until = num_layers - 1 - epoch*self.unfreeze_count_per_epoch
        unfreezed = []

        for i in range(already_unfreezed_until-self.unfreeze_count_per_epoch, already_unfreezed_until):
            self.model.layers[i].trainable = True
            unfreezed.append(i)

        print("Layers unfreezed: ", unfreezed)