import os

import tensorflow as tf

import image_preprocess
import mnist_inference
import mnist_train

REGULARIZATION_RATE = 0.0001

BASE_PATH = 'E:\\PycharmProjects'

files = tf.train.match_filenames_once(os.path.join(BASE_PATH, 'tensorflow\\tfrecord\\output.tfrecords*'))
filename_queue = tf.train.string_input_producer(files, shuffle=True)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)})
image, label = features['image_raw'], features['label']
#height = width = features['pixels']
height = width = 28
channels = 1

decoded_image = tf.decode_raw(image, tf.uint8)
decoded_image.set_shape([height * width, ])
decoded_image = tf.reshape(decoded_image, [height, width, channels])

image_size = 28
distorted_image = image_preprocess.preprocess_for_train(decoded_image, image_size, image_size)
distorted_image = tf.reshape(distorted_image, [image_size * image_size, ])

min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size
image_batch, label_batch = tf.train.shuffle_batch(
    [distorted_image, label], batch_size=batch_size,
    capacity=capacity, min_after_dequeue=min_after_dequeue)

global_step = tf.Variable(0, trainable=False)
num_examples = 55000

regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
logits = mnist_inference.inference(image_batch, regularizer)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_batch)
cross_entropy_mean = tf.reduce_mean(cross_entropy)
loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
learning_rate = tf.train.exponential_decay(
    mnist_train.LEARNING_RATE_BASE,
    global_step, num_examples / batch_size,
    mnist_train.LEARNING_RATE_DECAY)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

TRAINING_ROUNDS = 5000

with tf.Session() as sess:
    local_init_op = tf.local_variables_initializer()
    global_init_op = tf.global_variables_initializer()
    sess.run([global_init_op, local_init_op])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(TRAINING_ROUNDS):
        _, loss_value, step = sess.run([train_step, loss, global_step])

        if i % 1000 == 0:
            print('After %d training step(s), loss on training batch is %g.' % (step - 1, loss_value))

    coord.request_stop()
    coord.join(threads)
