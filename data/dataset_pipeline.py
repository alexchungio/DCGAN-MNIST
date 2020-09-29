#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : dataset_pipeline.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/29 下午6:27
# @ Software   : PyCharm
#-------------------------------------------------------
import os
import numpy as np
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import PIL

from libs.configs import cfgs
from utils.tools import makedir


def noise_generator(dim):
    while True:
        yield np.random.normal(0.0, 1.0, (dim)).astype('float32')


def noise_dataset(batch_size, dim):
    dataset = tf.data.Dataset.from_generator(generator=noise_generator,
                                             output_types=(tf.float32),
                                             args=(dim,))

    dataset = dataset.batch(batch_size)

    return dataset


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


# print(np.min(sample_mnist[0].numpy()), np.max(sample_mnist[0].numpy()), sample_mnist[0].shape)
# print(np.min(sample_mnist[1].numpy()), np.max(sample_mnist[1].numpy()), sample_mnist[1].shape)

def show_save_image_grid(images, save_dir=None, batch_size=128, id=None):
    fig = plt.figure(figsize=(8, batch_size / 32))  # width height
    fig.suptitle("Epoch {}".format(id))
    gs = plt.GridSpec(nrows=int(batch_size / 16), ncols=16)
    gs.update(wspace=0.05, hspace=0.05)

    for i, image in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(image * 127.5 + 127.5, cmap='Greys_r')

    makedir(save_dir)
    plt.savefig(os.path.join(save_dir, 'epoch_{:04d}.png'.format(id)))
    plt.show()


def generate_gif(image_path, anim_file):
    """

    :param image_path:
    :param anim_file:
    :return:
    """
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(os.path.join(image_path, '*.png'))
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)


if __name__ == "__main__":
    pass
