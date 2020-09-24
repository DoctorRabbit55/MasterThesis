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
import matplotlib.pyplot as plt
       

def create_imagenet_dataset(file_path, batch_size=64):

    if not isinstance(file_path, Path):     # convert str to Path
        file_path = Path(file_path)

    record_file_list = list(map(str, file_path.glob("*")))
    ds = tf.data.TFRecordDataset(record_file_list)
    
    def parse_function(example):
        feature_descriptor = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
        }

        data = tf.io.parse_single_example(example, feature_descriptor)
        img = data['image_raw']
        label = data['label']

        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        #img = tf.image.central_crop(img, 0.875)
        #img = tf.expand_dims(img, 0)
        #img = tf.compat.v1.image.resize_bilinear(img, (224,224), align_corners=False)
        #img = tf.squeeze(img, [0])
        img = tf.subtract(img, 0.5)
        img = tf.multiply(img, 2.0)

        label = tf.one_hot(indices=label, depth=1000)
        return img, label
    ds = ds.map(parse_function)
    
    ds = ds.shuffle(2000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds

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