import os

import tensorflow as tf

BASE_PATH = 'E:\\PycharmProjects\\tensorflow'

files = tf.train.match_filenames_once(os.path.join(BASE_PATH, 'tfrecord\\data.tfrecords-*'))

filename_queue = tf.train.string_input_producer(files, shuffle=True)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64)})
example, label = features['i'], features['j']

batch_size = 3
capacity = 1000 + 3 * batch_size

#example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size, capacity=capacity)
example_batch, label_batch = tf.train.shuffle_batch(
    [example, label],
    batch_size=batch_size,
    capacity=capacity, min_after_dequeue=30)

with tf.Session() as sess:
    init_op = tf.local_variables_initializer()
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for _ in range(2):
        cur_example_batch, cur_label_example = sess.run([example_batch, label_batch])
        print(cur_example_batch, cur_label_example)

    coord.request_stop()
    coord.join(threads)
