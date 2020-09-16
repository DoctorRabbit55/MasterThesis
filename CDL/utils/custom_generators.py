import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
import tensorflow.keras as keras

from CDL.utils.keras_utils import modify_model

from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image

class Imagenet_train_shunt_generator(Sequence):

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
        #y = np.zeros((len(batch),shunt_output_size))
        y = None

        for i, id in enumerate(batch):
            #print(str(self.x_dir / self.x_file_names[id]))
            img = image.load_img(str(self.x_dir / self.x_file_names[id]))
            #img = image.img_to_array(img)

            height, width, _ = img.shape
            new_height = height * 256 // min(img.shape[:2])
            new_width = width * 256 // min(img.shape[:2])
            img = image.img_to_array((img.resize((new_width, new_height), Image.BICUBIC))
            
            # Crop
            height, width, _ = img.shape
            startx = width//2 - (224//2)
            starty = height//2 - (224//2)
            img = img[starty:starty+224,startx:startx+224]

            X[i,:,:,:] = keras.applications.mobilenet.preprocess_input(img)

            #print(self.labels[id])

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
            img = image.load_img(str(self.x_dir / self.x_file_names[id]))
            #img = image.img_to_array(img)

            height, width, _ = img.size
            new_height = height * 256 // min(img.size[:2])
            new_width = width * 256 // min(img.size[:2])
            img = image.img_to_array(img.resize((new_width, new_height), Image.BICUBIC))
            
            # Crop
            height, width, _ = img.shape
            startx = width//2 - (224//2)
            starty = height//2 - (224//2)
            img = img[starty:starty+224,startx:startx+224]

            X[i,:,:,:] = keras.applications.mobilenet.preprocess_input(img)
            
            y[i,:] = np.expand_dims(to_categorical(self.labels[id], num_classes=1000), 0)
            #print(self.labels[id])

        return X,y
       

class VOC2012_generator(Sequence):

    def __init__(self, x_dir, file_names, y_dir=None, batch_size=32, num_classes=None, shuffle=True, include_labels=True):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.file_names = file_names
        self.include_labels = include_labels
        self.indices = np.arange(len(file_names))
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
        if self.include_labels: y = np.zeros((len(batch),512,512,21)) 
        else: y = None

        for i, id in enumerate(batch):

            X[i,:,:,:] = np.array(cv2.imread(str(self.x_dir / (self.file_names[id] + ".jpg"))))
            X[i,:,:,:] = X[i,:,:,:] / 127.5 - 1
        
            #y[i,:,:] = np.array(cv2.imread(self.y_dir + "/" + self.file_names[id] + ".png"))
            label = np.array(cv2.imread(str(self.y_dir / (self.file_names[id] + ".png"))))

            if self.include_labels:
                for j in range(21):
                    y[i,:,:,j] = np.where(label[:,:,0] == j, 1, 0)

        return X, y