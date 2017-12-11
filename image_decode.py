import os

import matplotlib.pyplot as plt
import tensorflow as tf

BASE_PATH = '/home/cowerling'
image_path = os.path.join(BASE_PATH, 'PycharmProjects/tensorflow/picture/picture.jpg')

image_raw_data = tf.gfile.FastGFile(image_path)

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    print(img_data.eval())

    plt.imshow(img_data.eval())
    plt.show()

    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

    encode_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile(os.path.join(BASE_PATH, 'PycharmProjects/tensorflow/picture/picture_n.jpg')) as f:
        f.write(encode_image.eval())
