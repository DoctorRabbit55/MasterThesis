import tensorflow as tf
from keras.utils import Sequence, to_categorical
from keras.preprocessing import image
from keras.applications import imagenet_utils
import keras

from CDL.utils.keras_utils import modify_model

from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def preprocess_feature_maps(file_path, model):

    #img = tf.io.read_file(file_path)
    #img = tf.image.decode_jpeg(img)
    #img = tf.image.resize(img, (224,224))
    #img = keras.applications.mobilenet.preprocess_input(img)

    #maps = model.predict(img)[:-1]
    maps = model.predict(file_path)[:-1]

    return maps[0], maps[1]


def create_feature_map_ds(model, shunt_locations, epochs, batch_size, glob_pattern):
    tf.compat.v1.enable_eager_execution()
    model_reduced = modify_model(model, layer_indexes_to_output=shunt_locations)

    ds = tf.data.Dataset.from_tensors(glob_pattern)
    #if glob_pattern is np.ndarray:
    #    ds = tf.data.Dataset.from_tensors(glob_pattern)
    #else:
    #    ds = tf.data.Dataset.list_files(glob_pattern)

    num_threads = 5 
    ds = ds.map(lambda x: preprocess_feature_maps(x, model_reduced), num_parallel_calls=num_threads)

    ds = ds.shuffle(buffer_size=10000)
    ds = ds.repeat(epochs)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds

class feature_map_generator(Sequence):

    def __init__(self, model, batch_size, shunt_locations, datagen_train, x_data, len_x_data, flow_from_directory=False):
        self.reduced_model = modify_model(model, layer_indexes_to_output=shunt_locations)
        self.datagen = datagen_train
        self.batch_size = batch_size
        self.x_data = x_data
        self.len_x_data = len_x_data
        self.flow_from_directory = flow_from_directory

    def __len__(self):
        return self.len_x_data // self.batch_size

    def __getitem__(self, index):

        if self.flow_from_directory:
            maps = self.reduced_model.predict(self.datagen.flow_from_directory(self.x_data, batch_size=self.batch_size), steps=1)[:-1]
        else:
            maps = self.reduced_model.predict(self.datagen.flow(self.x_data, batch_size=self.batch_size), steps=1)[:-1]

        X = maps[0]
        y = maps[1]

        return X,y


class Imagenet_generator(Sequence):

    def __init__(self, x_dir, label_file_path, shuffle=True, batch_size=64):
        self.batch_size = batch_size
        self.x_dir = x_dir
        self.x_file_names = sorted(os.listdir(x_dir))
        self.shuffle = shuffle
        self.current_indices = np.arange(len(self.x_file_names))

        self.labels = np.zeros((len(self.x_file_names)), dtype=np.int32)

        with open(label_file_path, 'r') as label_file: 
            lines = label_file.readlines()
            for i, line in enumerate(lines):
                if i == len(self.x_file_names):
                    break
                self.labels[i] = int(line.split()[1])

        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.current_indices)
            
    def __len__(self):
        return len(self.current_indices) // self.batch_size

    def __getitem__(self, index):
        batch = self.current_indices[index*self.batch_size:(index+1)*self.batch_size]
        X,y = self.__get_data(batch)
        return X,y

    def __get_data(self, batch):

        X = np.zeros((len(batch),224,224,3))
        y = np.zeros((len(batch),1000))

        for i, id in enumerate(batch):
            #print(str(self.x_dir / self.x_file_names[id]))
            X[i,:,:,:] = image.load_img(str(self.x_dir / self.x_file_names[id]), target_size=(224,224))
            X[i,:,:,:] = image.img_to_array(X[i,:,:,:])
            X[i,:,:,:] = keras.applications.mobilenet.preprocess_input(X[i,:,:,:])
            y[i,:] = np.expand_dims(to_categorical(self.labels[id], num_classes=1000), 0)
            #print(self.labels[id])

        return X,y
       

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

            X[i,:,:,:] = np.array(cv2.imread(str(self.x_dir / (self.file_names[id] + ".jpg"))))
            X[i,:,:,:] = X[i,:,:,:] / 127.5 - 1
        
            #y[i,:,:] = np.array(cv2.imread(self.y_dir + "/" + self.file_names[id] + ".png"))
            label = np.array(cv2.imread(str(self.y_dir / (self.file_names[id] + ".png"))))

            for j in range(21):
                #if unique_value == 255:
                #    y[i,:,:,j] = np.where(label[:,:,0] == unique_value, 1, 0)
                #else:
                if j != 20:
                    y[i,:,:,j] = np.where(label[:,:,0] == j, 1, 0)
                else:
                    y[i,:,:,j] = np.where(label[:,:,0] == 255, 1, 0)

        return X, y