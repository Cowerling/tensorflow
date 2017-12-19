import os

import tensorflow as tf

BASE_PATH = 'E:\\PycharmProjects\\'

reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer([
    os.path.join(BASE_PATH, 'tensorflow\\tfrecord\\output.tfrecords')])

_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)})

images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    for i in range(1):
        image, label, pixel = sess.run([images, labels, pixels])
        print(image)
