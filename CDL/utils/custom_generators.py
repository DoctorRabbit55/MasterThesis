from keras.utils import Sequence

import numpy as np
import cv2

class VOC2012_generator(Sequence):

    def __init__(self, x_dir, file_names, y_dir=None, batch_size=32, num_classes=None,shuffle=True):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.file_names = file_names
        self.indices = range(len(file_names))
        self.index = self.indices
        self.on_epoch_end()
        



    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]
        X, y = self.__get_data(batch)
        return X, y

    def __get_data(self, batch):

        X = np.zeros((len(batch),512,512,3))
        y = np.zeros((len(batch),512,512,21))

        for i, id in enumerate(batch):

            X[i,:,:,:] = np.array(cv2.imread(self.x_dir + "/" + self.file_names[id] + ".jpg"))
            X[i,:,:,:] = X[i,:,:,:] / 127.5 - 1
        
            #y[i,:,:] = np.array(cv2.imread(self.y_dir + "/" + self.file_names[id] + ".png"))
            label = np.array(cv2.imread(self.y_dir + "/" + self.file_names[id] + ".png"))

            for j, unique_value in enumerate(np.unique(label)):
                if unique_value == 255:
                    y[i,:,:,j] = np.where(label[:,:,0] == unique_value, -1, 0)
                else:
                    y[i,:,:,j] = np.where(label[:,:,0] == unique_value, 1, 0)

        return X, y