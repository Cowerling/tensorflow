import os

import matplotlib.pyplot as plt
import tensorflow as tf

BASE_PATH = 'E:/PycharmProjects'
image_path = os.path.join(BASE_PATH, 'tensorflow/picture/picture.jpg')

image_raw_data = tf.gfile.FastGFile(image_path, 'rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    print(img_data.eval())

    plt.imshow(img_data.eval())
    plt.show()

    '''
    img_data2 = tf.image.convert_image_dtype(img_data, dtype=tf.uint8)

    resized = tf.image.resize_images(img_data, [300, 300], method=0)
    resized = tf.cast(resized, tf.uint8)
    plt.imshow(resized.eval())
    plt.show()

    croped = tf.image.resize_image_with_crop_or_pad(img_data, 300, 300)
    plt.imshow(croped.eval())
    plt.show()

    central_cropped = tf.image.central_crop(img_data, 0.5);
    plt.imshow(central_cropped.eval())
    plt.show()

    flipped = tf.image.random_flip_up_down(img_data)
    flipped = tf.image.random_flip_left_right(flipped)
    transposed = tf.image.transpose_image(flipped)
    plt.imshow(transposed.eval())
    plt.show()

    adjusted = tf.image.random_brightness(img_data, 0.5)
    plt.imshow(adjusted.eval())
    plt.show()

    adjusted = tf.image.adjust_hue(img_data, 0.1)
    plt.imshow(adjusted.eval())
    plt.show()

    adjusted = tf.image.per_image_standardization(img_data)
    print(adjusted.eval())
    adjusted = tf.abs(adjusted)
    plt.imshow(adjusted.eval())
    plt.show()
    '''
    img_data = tf.image.resize_images(img_data, [400, 300], method=1)
    batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0)
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    result = tf.image.draw_bounding_boxes(batched, boxes)
    tf.image.sample_distorted_bounding_box(tf.shape(img_data), bounding_boxes=boxes, min_object_covered=0.1)
    plt.imshow(result[0].eval())
    plt.show()

    encode_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile(os.path.join(BASE_PATH, 'tensorflow/picture/picture_n.jpg'), 'wb') as f:
        f.write(encode_image.eval())
