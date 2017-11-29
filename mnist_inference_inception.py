import tensorflow as tf
import tensorflow.contrib.slim as slim

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

FC_SIZE = 512


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable('weights', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def get_biases_variable(shape):
    biases = tf.get_variable('biases', shape=shape, initializer=tf.constant_initializer(0.1))
    return biases


def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = get_weight_variable([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], None)
        conv1_biases = get_biases_variable([CONV1_DEEP])
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer3-conv2'):
        conv2_weights = get_weight_variable([CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], None)
        conv2_biases = get_biases_variable([CONV2_DEEP])
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
        with tf.variable_scope('Mixed_7c'):
            with tf.variable_scope('Branch_0'):
                branch_0 = slim.conv2d(pool2, 320, [1, 1], scope='Conv2d_0a_1x1')

            with tf.variable_scope('Branch_1'):
                branch_1 = slim.conv2d(pool2, 284, [1, 1], scope='Conv2d_0a_1x1')
                branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                                      slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0c_3x1')], 3)

            with tf.variable_scope('Branch_2'):
                branch_2 = slim.conv2d(pool2, 448, [1, 1], scope='Conv2d_0a_1x1')
                branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                                      slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)

            with tf.variable_scope('Branch_3'):
                branch_3 = slim.avg_pool2d(pool2, [3, 3], scope='AvgPool_0a_3x3')
                branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')

            pool2 = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = get_weight_variable([nodes, FC_SIZE], regularizer)
        fc1_biases = get_biases_variable([FC_SIZE])
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = get_weight_variable([FC_SIZE, NUM_LABELS], regularizer)
        fc2_biases = get_biases_variable([NUM_LABELS])
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit
