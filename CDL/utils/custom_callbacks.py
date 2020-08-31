from keras.callbacks import Callback
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras import backend
import numpy as np

class UnfreezeLayersCallback(Callback):

    def __init__(self, epochs, epochs_per_unfreeze, learning_rate, unfreeze_to_index=0, start_at=0, direction=-1):
        self.epochs_per_unfreeze = epochs_per_unfreeze
        self.learning_rate = learning_rate
        self.unfreeze_to_index = unfreeze_to_index
        self.start_at = start_at
        self.unfreezed_index = start_at
        self.increment = direction


    def on_epoch_begin(self, epoch, logs=None):

        unfreezed = []

        if epoch % self.epochs_per_unfreeze == 0 and self.unfreezed_index >= self.unfreeze_to_index:

            self.unfreezed_index += self.increment

            #if isinstance( self.model.layers[self.unfreezed_index], BatchNormalization):
            #    self.unfreezed_index += self.increment

            try:
                self.model.layers[self.unfreezed_index].trainable = True
                unfreezed.append(self.model.layers[self.unfreezed_index].name)
            except:
                pass

        self.model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=self.learning_rate, momentum=0.9, decay=0.0, nesterov=False), metrics=['accuracy'])

        print("Layers unfreezed: ", unfreezed)

class LearningRateSchedulerCallback(Callback):

    def __init__(self, epochs_first_cycle, learning_rate_second_cycle):
        super(LearningRateSchedulerCallback, self).__init__()
        self.epochs_first_cycle = epochs_first_cycle
        self.learning_rate_second_cycle = learning_rate_second_cycle

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.epochs_first_cycle:
            print("Activated second cycle with learning rate = {}".format(self.learning_rate_second_cycle))
            backend.set_value(self.model.optimizer.lr, self.learning_rate_second_cycle)