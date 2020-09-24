from keras.datasets import cifar10
from keras.utils import to_categorical

import cv2
import tensorflow as tf
import multiprocessing, random
from itertools import repeat
from pathlib import Path

def load_and_preprocess_CIFAR10():
        
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_test = x_test.astype('float32')  #argmax_ch
    x_train = x_train.astype('float32')  #argmax_ch

    def ch_wise_normalization(X_type, ch):
        mean_ch = X_type[:, :, :, ch].mean()
        std_ch = X_type[:, :, :, ch].std()
        X_type[:, :, :, ch] = (X_type[:, :, :, ch] - mean_ch) / std_ch
        return X_type[:, :, :, ch]

    x_test[:, :, :, 0]  = ch_wise_normalization(x_test, 0)
    x_test[:, :, :, 1]  = ch_wise_normalization(x_test, 1)
    x_test[:, :, :, 2]  = ch_wise_normalization(x_test, 2)
    x_train[:, :, :, 0]  = ch_wise_normalization(x_train, 0)
    x_train[:, :, :, 1]  = ch_wise_normalization(x_train, 1)
    x_train[:, :, :, 2]  = ch_wise_normalization(x_train, 2)

    y_test  = to_categorical(y_test, 10)
    y_train = to_categorical(y_train, 10)

    return (x_train, y_train), (x_test, y_test)


def prepare_imagenet():

    image_folder = Path('E:/Masterarbeit/MasterThesis/saved/datasets/imagenet/val/')
    record_path = Path('E:/Masterarbeit/MasterThesis/saved/datasets/imagenet/val/records/')
    identifier = 'val'

    label_map = {}
    with open('E:/Masterarbeit/MasterThesis/saved/datasets/imagenet/imagenet_label_map.txt') as label_list:
        lines = label_list.readlines()
        for i, line in enumerate(lines):
            label_map[line[:-1]] = i
    create_tf_record(image_folder, record_path, identifier, label_map)

def create_tf_record(image_folder, record_path, identifier, label_map, size=224, split_number=1000, image_quality=90, tf_record_options=None):
    print("creating " + identifier + " records")
 
    files = list(image_folder.glob('n*/*.JPEG'))
 
    random.shuffle(files)
 
    split_file_list = [files[x:x + split_number] for x in range(0, len(files), split_number)]
 
    tf_record_paths = []
 
    for i in range(len(split_file_list)):
        tf_record_paths.append(record_path / (identifier + "-" + str(i) + ".tfrecord"))
 
    master_tf_write(split_file_list, tf_record_paths, size, image_quality, label_map, tf_record_options)

def master_tf_write(split_file_list, tf_record_paths, size, image_quality, label_map, tf_record_options):
    cpu_core = multiprocessing.cpu_count()
 
    p = multiprocessing.Pool(cpu_core)
    results = p.starmap(worker_tf_write,
                        zip(split_file_list, tf_record_paths, repeat(label_map), repeat(size), repeat(image_quality), repeat(tf_record_options),
                            list(range(len(tf_record_paths)))))
    p.close()
    p.join()


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
 
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
 
def worker_tf_write(files, tf_record_path, label_map, size, image_quality, tf_record_options, number):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), image_quality]
    tf_record_options = tf.io.TFRecordOptions(compression_type=tf_record_options)
 
    with tf.io.TFRecordWriter(str(tf_record_path), tf_record_options) as tf_writer:
        for i, file in enumerate(files):
            #print(file)
            image = process_image(cv2.imread(str(file)), size)
            is_success, im_buf_arr = cv2.imencode(".jpg", image, encode_param)
 
            if is_success:
                label_str = file.parts[-2]
 
                label_number = label_map[label_str]
 
                image_raw = im_buf_arr.tobytes()
                row = tf.train.Example(features=tf.train.Features(feature={
                    'label': _int64_feature(label_number),
                    'image_raw': _bytes_feature(image_raw)
                }))
 
                tf_writer.write(row.SerializeToString())
            else:
                print("Error processing " + file)

def process_image(image, size):
    image = scale_image(image, size)
    image = center_crop(image, size)
    return image
	

def scale_image(image, size):
    image_height, image_width = image.shape[:2]
 
    if image_height <= image_width:
        ratio = image_width / image_height
        h = size
        w = int(ratio * 256)
 
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
 
    else:
        ratio = image_height / image_width
        w = size
        h = int(ratio * 256)
 
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
 
    return image

def center_crop(image, size):
    image_height, image_width = image.shape[:2]
 
    if image_height <= image_width and abs(image_width - size) > 1:
 
        dx = int((image_width - size) / 2)
        image = image[:, dx:-dx, :]
    elif abs(image_height - size) > 1:
        dy = int((image_height - size) / 2)
        image = image[dy:-dy, :, :]
 
    image_height, image_width = image.shape[:2]

    if image_height is not size or image_width is not size:
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    #print(image.shape[:2])
    return image

if __name__ == '__main__':
    prepare_imagenet()