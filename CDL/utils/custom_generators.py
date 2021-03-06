import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
import tensorflow.keras as keras

from CDL.utils.keras_utils import modify_model
from CDL.utils.dataset_utils import cityscapes_preprocess_image_and_label

from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image
import matplotlib.pyplot as plt
       

def create_imagenet_dataset(file_path, should_repeat=True, batch_size=64):

    if not isinstance(file_path, Path):     # convert str to Path
        file_path = Path(file_path)

    record_file_list = list(map(str, file_path.glob("*")))
    ds = tf.data.TFRecordDataset(record_file_list, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    
    def parse_function(example):
        feature_descriptor = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
        }

        data = tf.io.parse_single_example(example, feature_descriptor)
        
        label = data['label']
        img = tf.image.decode_jpeg(data['image_raw'], channels=3)
        img = tf.cast(img, tf.float32)
        #img = tf.image.convert_image_dtype(img, tf.float32)
        #img = tf.image.central_crop(img, 0.875)
        #img = tf.expand_dims(img, 0)
        #img = tf.compat.v1.image.resize_bilinear(img, (224,224), align_corners=False)
        #img = tf.squeeze(img, [0])
        #img = tf.subtract(img, 0.5)
        #img = tf.multiply(img, 2.0)
        img = keras.applications.mobilenet_v2.preprocess_input(img)

        label = tf.one_hot(indices=label, depth=1000)
        return img, label

    ds = ds.shuffle(2024)
    ds = ds.cache()
    ds = ds.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size)
    if should_repeat:
        ds = ds.repeat()
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def create_cityscape_dataset(file_path, is_training=True, batch_size=64):
    if not isinstance(file_path, Path):     # convert str to Path
        file_path = Path(file_path)

    if is_training:
        preamble = 'train'
    else:
        preamble = 'val'
    record_file_list = list(map(str, file_path.glob(preamble + "*")))

    def parse_function(example_proto):
        def _decode_image(content, channels):
            return tf.cond(
                tf.image.is_jpeg(content),
                lambda: tf.image.decode_jpeg(content, channels),
                lambda: tf.image.decode_png(content, channels))

        features = {
            'image/encoded':
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            'image/filename':
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            'image/format':
                tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/height':
                tf.io.FixedLenFeature((), tf.int64, default_value=0),
            'image/width':
                tf.io.FixedLenFeature((), tf.int64, default_value=0),
            'image/segmentation/class/encoded':
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            'image/segmentation/class/format':
                tf.io.FixedLenFeature((), tf.string, default_value='png'),
        }

        parsed_features = tf.io.parse_single_example(example_proto, features)

        image = _decode_image(parsed_features['image/encoded'], channels=3)

        label = _decode_image(parsed_features['image/segmentation/class/encoded'], channels=1)

        image_name = parsed_features['image/filename']
        if image_name is None:
            image_name = tf.constant('')

        if label is not None:
            if label.get_shape().ndims == 2:
                label = tf.expand_dims(label, 2)
            elif label.get_shape().ndims == 3 and label.shape.dims[2] == 1:
                pass
            else:
                raise ValueError('Input label shape must be [height, width], or '
                            '[height, width, 1].')

        label.set_shape([None, None, 1])

        image, label = cityscapes_preprocess_image_and_label(image, label, is_training=is_training)

        image = tf.multiply(image, 1/127.5)
        image = tf.subtract(image, 1)

        return image, label

    ds = tf.data.TFRecordDataset(record_file_list, num_parallel_reads=tf.data.experimental.AUTOTUNE) \
         .map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds.shuffle(1000)
    ds.batch(batch_size)
    if is_training:
        ds = ds.repeat()
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
