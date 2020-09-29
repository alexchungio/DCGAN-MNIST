#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : train.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/27 下午5:39
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import PIL
from tensorflow.keras import layers
import time

from IPython import display


Z_DIM = 100
BATCH_SIZE = 128
BUFFER_SIZE = 6000






if __name__ == "__main__":
    # load mnist
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()


    print(train_images.shape, train_labels.shape)


    def noise_generator(dim):
        while True:
            yield np.random.normal(0.0, 1.0, (dim)).astype('float32')


    def noise_dataset(batch_size):

        dataset = tf.data.Dataset.from_generator(generator=noise_generator,
                                                 output_types=(tf.float32),
                                                 args=(batch_size,))

        dataset = dataset.batch(batch_size)

        return dataset

    noise_batch = noise_dataset(BATCH_SIZE)


    def image_process(image, label):
        # reshape (768,) => (28, ,28, 1)
        # image = tf.cast(image, dtype=tf.float32)
        image = image.reshape((28, 28, 1)).astype('float32')
        # scale (0, 255) = (-1, 1)
        image = (image - 127.5) / 127.5

        label = label.astype('int32')

        return image, label


    def mnist_generator(images, labels, batch_size, buffer_size=60000):

        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(lambda item1, item2: tf.numpy_function(image_process,
                                                                 inp=[item1, item2],
                                                                 Tout=[tf.float32, tf.int32]),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size)

        return dataset

    mnist_batch = mnist_generator(train_images, train_labels, batch_size=BATCH_SIZE)

    sample_mnist = next(iter(mnist_batch))

    print(np.min(sample_mnist[0].numpy()), np.max(sample_mnist[0].numpy()), sample_mnist[0].shape)
    print(np.min(sample_mnist[1].numpy()), np.max(sample_mnist[1].numpy()), sample_mnist[1].shape)




