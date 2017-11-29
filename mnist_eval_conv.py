import time
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference_conv
import mnist_train_conv

EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        xv = mnist.validation.images

        x = tf.placeholder(tf.float32, [
            xv.shape[0],
            mnist_inference_conv.IMAGE_SIZE,
            mnist_inference_conv.IMAGE_SIZE,
            mnist_inference_conv.NUM_CHANNELS],
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference_conv.OUTPUT_NODE], name='y-input')

        reshaped_xv = np.reshape(xv, (xv.shape[0],
                                      mnist_inference_conv.IMAGE_SIZE,
                                      mnist_inference_conv.IMAGE_SIZE,
                                      mnist_inference_conv.NUM_CHANNELS))
        validate_feed = {x: reshaped_xv, y_: mnist.validation.labels}

        y = mnist_inference_conv.inference(x, train=False, regularizer=None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train_conv.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train_conv.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print('After %s training step(s), validation accuracy = %g' % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return

                time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets(mnist_train_conv.BASE_PATH + 'mnist/', one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()