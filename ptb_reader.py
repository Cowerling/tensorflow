import tensorflow as tf
from tensorflow.models.tutorials.rnn.ptb import reader

DATA_PATH = 'E:\PycharmProjects\\tensorflow\\ptb'
train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)

result = reader.ptb_producer(train_data, 4, 5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(3):
        x, y = sess.run(result)
        print('X%d:' % i, x)
        print('Y%d:' % i, y)
