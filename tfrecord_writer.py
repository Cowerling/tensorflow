import os
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

BASE_PATH = 'E:/PycharmProjects'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


mnist = input_data.read_data_sets(
    os.path.join(BASE_PATH, 'tensorflow/mnist'), dtype=tf.uint8, one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
num_examples = mnist.train.num_examples

with tf.Session() as sess:
    print(labels)
    l = sess.run(tf.argmax(labels, 1))
    print(l)

filename = os.path.join(BASE_PATH, 'tensorflow/tfrecord/output.tfrecords')
writer = tf.python_io.TFRecordWriter(filename)

for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())

writer.close()
