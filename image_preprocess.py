import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


def distort_color(image):
    processes = [tf.image.random_brightness,
                 tf.image.random_saturation,
                 tf.image.random_hue,
                 tf.image.random_contrast]
    parameters = {0: 32.0 / 255, 1: (0.5, 1.5), 2: 0.2, 3: (0.5, 1.5)}
    ordering = [0, 1, 2, 3]

    np.random.shuffle(ordering)

    for index in ordering:
        process = processes[index]
        parameter = parameters[index]

        if index == 0 or index == 2:
            image = process(image, max_delta=parameter)
        else:
            image = process(image, lower=parameter[0], upper=parameter[1])


def preprocess_for_train(image, height, width, bbox=None):
    if bbox is None:
        bbox = tf.constant([0, 0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    bbox_begin, bbox_size, _, = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox, min_object_covered=0.1)

    distorted_image = tf.slice(image, bbox_begin, bbox_size)
    distorted_image = tf.image.resize_images(distorted_image, [height, width], method=tf.image.ResizeMethod.BICUBIC)

    height = distorted_image.eval().shape[0]
    width = distorted_image.eval().shape[1]
    channels = distorted_image.eval().shape[2]
    for i in range(height):
        for j in range(width):
            for x in range(channels):
                if distorted_image.eval()[i, j, x] < 0 or distorted_image.eval()[i, j, x] >= 1:
                    print(distorted_image.eval()[i, j, x])

    distort_color(distorted_image)

    return distorted_image


BASE_PATH = 'E:/PycharmProjects'
image_path = os.path.join(BASE_PATH, 'tensorflow/picture/picture.jpg')
image_raw_data = tf.gfile.FastGFile(image_path, 'rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])

    for i in range(6):
        result = preprocess_for_train(img_data, 10, 10, boxes)
        #result = tf.abs(result)
        print(result.eval())
        plt.imshow(result.eval())
        plt.show()
