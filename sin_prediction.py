import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

HIDDEN_SIZE = 30
NUM_LAYERS = 2
TIMESTEPS = 0
TRAINING_STEPS = 10000
BATCH_SIZE = 32

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01


def generate_data(seq):
    x = []
    y = []

    for i in range(len(seq) - TIMESTEPS):
        x.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


def lstm_model(x, y, is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        for _ in range(NUM_LAYERS)])

    output, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    output = output[:, -1, :]

    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)

    if not is_training:
        return predictions, None, None

    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)

    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.train.get_global_step(), optimizer='Adagrad', learning_rate=0.1)

    return predictions, loss, train_op


def run_eval(sess, test_x, test_y):
    dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    dataset = dataset.batch(1)
    x, y = dataset.make_one_shot_iterator().get_next()

    with tf.variable_scope('model', reuse=True):
        prediction, _, _ = lstm_model(x, None, False)

    predictions = []
    labels = []
    for i in range(TESTING_EXAMPLES):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)

    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt((predictions - labels) ** 2).mean(axis=0)
    print('Mean Square Error is: %f' % rmse)

    plt.figure()
    plt.plot(predictions, label='predictions')
    plt.plot(labels, label='real_sin')
    plt.legend()
    plt.show()


test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
train_x, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
test_x, test_y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)))

dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
dataset = dataset.repeat().shuffle(1000).batch(BATCH_SIZE)
x, y = dataset.make_one_shot_iterator().get_next()

with tf.variable_scope('model'):
    _, loss, train_op = lstm_model(x, y, True)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print('Evaluate model before training.')
    run_eval(sess, test_x, test_y)

    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op, loss])
        if i % 1000 == 0:
            print('train step: %d, loss: %f' % (i, l))

    print('Evaluate model after training.')
    run_eval(sess, test_x, test_y)
