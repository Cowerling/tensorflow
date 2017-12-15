import os

import tensorflow as tf

BASE_PATH = 'E:\\PycharmProjects\\tensorflow'

files = tf.train.match_filenames_once(os.path.join(BASE_PATH, 'tfrecord\\data.tfrecords-*'))

filename_queue = tf.train.string_input_producer(files, shuffle=True, num_epochs=1)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64)})

with tf.Session() as sess:
    init_op = tf.local_variables_initializer()
    sess.run(init_op)

    sess.run(files)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for _ in range(4):
        print(sess.run([features['i'], features['j']]))

    coord.request_stop()
    coord.join(threads)
