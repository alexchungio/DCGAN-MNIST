#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : model.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/9/29 下午6:23
# @ Software   : PyCharm
#-------------------------------------------------------
import tensorflow as tf


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = tf.keras.layers.Dense(units=7*7*256, use_bias=False, input_dim=(100,))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.leak_relu1 = tf.keras.layers.LeakyReLU()

        # (-1, 7*7*256) => (-1, 7, 7, 256)
        self.reshape = tf.keras.layers.Reshape((7, 7, 256))

        # transpose convd (-1, 7, 7, 256) => (-1, 7, 7, 128)
        self.conv_trans1 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(5, 5), strides=(1, 1),
                                                           padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.leak_relu2 = tf.keras.layers.LeakyReLU()

        # transpose convd (-1, 7, 7, 128) => (-1, 14, 14, 64)
        self.conv_trans2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2),
                                                           padding='same', use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.leak_relu3 = tf.keras.layers.LeakyReLU()

        # transpose convd (-1, 14, 14, 64) => (-1, 28, 28, 1)
        self.conv_trans3 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(5, 5), strides=(2, 2),
                                                           padding='same', use_bias=False, activation='tanh')

    def call(self, noise, training=False):
        """
        :param random_noise: (batch_size, z_dim)
        :return:
        """

        # assert noise.shape == (cfgs.BATCH_SIZE, cfgs.Z_DIM)

        model = self.fc1(noise)
        model = self.bn1(model)
        model = self.leak_relu1(model)

        # assert model.shape == (None, 7*7*256)
        # reshape
        model = self.reshape(model)
        # assert model.shape == (None, 7, 7, 256)

        # transpose convolution block
        model = self.conv_trans1(model)
        model = self.bn2(model)
        model = self.leak_relu2(model)
        # assert model.shape == (None, 7, 7, 128)

        # transpose convolution block
        model = self.conv_trans2(model)
        model = self.bn3(model)
        model = self.leak_relu3(model)
        # assert model.shape == (None, 14, 14, 64)

        # transpose convolution block
        model = self.conv_trans3(model)
        # assert model.shape == (None, 28, 28, 1)

        return model


class Discriminator(tf.keras.Model):
    def __init__(self, dropout_rate=0.0):
        super(Discriminator, self).__init__()

        # (-1, 28, 28, 1) => (-1, 14, 14, 64)
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same')
        self.leak_relu1 = tf.keras.layers.LeakyReLU()
        self.dropout1 = tf.keras.layers.Dropout(rate=dropout_rate)

        # (-1, 14, 14, 64) => (-1, 7, 7, 128)
        self.conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.leak_relu2 = tf.keras.layers.LeakyReLU()
        self.dropout2 = tf.keras.layers.Dropout(rate=dropout_rate)


        # (-1, 7, 7, 128) => (-1, 7*7*128)
        self.flatten = tf.keras.layers.Flatten()

        # (-1, 1024) => (-1, 1)
        self.fc1 = tf.keras.layers.Dense(units=1)

    def call(self, image, training=False):
        """

        :param image: (-1, 28, 28, 1)
        :return:
        """
        # conv block 1 (-1, 28, 28, 1) => (-1, 14, 14, 64)
        model = self.conv1(image)
        model = self.leak_relu1(model)
        model = self.dropout1(model, training=training)

        # conv block 2 (-1, 14, 14, 1) => (-1, 7, 7, 128)
        model = self.conv2(model)
        model = self.bn1(model)
        model = self.leak_relu2(model)
        model = self.dropout2(model, training=training)

        # flatten
        model = self.flatten(model)

        # full connect block 2
        model = self.fc1(model)

        return model