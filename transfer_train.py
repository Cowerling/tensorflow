import os

import tensorflow as tf
from tensorflow.python.platform import gfile

import transfer_image
import transfer_bottleneck

BOTTLENECK_TENSOR_SIZE = 2048

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
IMAGE_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

MODEL_DIR = 'E:\\PycharmProjects\\tensorflow\\model'
MODEL_FILE = 'tensorflow_inception_graph.pb'

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100


def main(argv=None):
    image_lists = transfer_image.create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())

    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    bottleneck_tensor, image_data_tensor = tf.import_graph_def(
        graph_def,
        return_elements=[BOTTLENECK_TENSOR_NAME, IMAGE_DATA_TENSOR_NAME])
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='bottleneck-input')
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='ground_truth-input')

    with tf.variable_scope('final_training_ops'):
        weights = tf.get_variable('weights',
                                  shape=[BOTTLENECK_TENSOR_SIZE, n_classes],
                                  initializer=tf.truncated_normal_initializer(stddev=0.002))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 initializer=tf.constant_initializer(0))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)

    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(ground_truth_input, 1))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEPS):
            train_bottlenecks, train_ground_truth = \
                transfer_bottleneck.get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, BATCH, 'training',
                    image_data_tensor, bottleneck_tensor)
            sess.run(train_step,
                     feed_dict={bottleneck_input: train_bottlenecks,
                                ground_truth_input: train_ground_truth})

            if i % 100 == 0 or i + 1 == STEPS:
                validation_bottlenecks, validation_ground_truth = \
                    transfer_bottleneck.get_random_cached_bottlenecks(
                        sess, n_classes, image_lists, BATCH, 'validation',
                        image_data_tensor, bottleneck_tensor)
                validation_accuracy = sess.run(
                    evaluation_step,
                    feed_dict={bottleneck_input: validation_bottlenecks,
                               ground_truth_input: validation_ground_truth})
                print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%' %
                      (i, BATCH, validation_accuracy * 100))

        test_bottlenecks, test_ground_truth = \
            transfer_bottleneck.get_test_bottlenecks(
                sess, image_lists, n_classes,
                image_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(
            evaluation_step,
            feed_dict={bottleneck_input: test_bottlenecks,
                       ground_truth_input: test_ground_truth})
        print('Final test accuray = %.1f%%' % (test_accuracy * 100))


if __name__ == '__main__':
    tf.app.run()
