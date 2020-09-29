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
import tensorflow as tf
import time


from libs.configs import cfgs
from utils.tools import makedir
from libs.nets.model import Generator, Discriminator
from data.dataset_pipeline import noise_dataset, mnist_generator, show_save_image_grid, generate_gif


# sample_noise = next(iter(noise_batch))
# sample_mnist = next(iter(mnist_batch))
#
generator = Generator()
discriminator = Discriminator()
#
# generated_image = generator(sample_noise)
# # show_image_grid(generated_image, batch_size=cfgs.BATCH_SIZE)
# # plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# show_image_grid(generated_image)
#
# decision = discriminator(generated_image)
# print(decision)


# ---------------------- loss function-----------------------------

# Define loss functions and optimizers for both models.
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# discriminator loss
def discriminator_loss(real_output, fake_output):
    """

    :param real_output: (batch_size, 1)
    :param fake_output: (batch_size, 1)
    :return:
    """
    real_loss = cross_entropy(y_true=tf.ones_like(real_output, dtype=tf.float32),
                              y_pred=real_output)

    fake_loss = cross_entropy(y_true=tf.zeros_like(fake_output, dtype=tf.float32),
                              y_pred=fake_output)
    loss = real_loss + fake_loss

    return loss


# generator loss
def generator_loss(fake_output):
    """
    The generator's loss quantifies how well it was able to trick the discriminator
    :param fake_output: (batch_size, 1)
    :return:
    """
    non_fake_loss = cross_entropy(y_true=tf.ones_like(fake_output, dtype=tf.float32),
                                  y_pred=fake_output)

    return non_fake_loss


# ----------------------------------optimizer--------------------------------
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=cfgs.GENERATOR_LEARNING_RATE)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=cfgs.DISCRIMINATOR_LEARNING_RATE)

# ----------------------------------trian log---------------------------------------
# checkpoint
ckpt = tf.train.Checkpoint(generator=generator,
                           discriminator=discriminator,
                           generator_optimizer=generator_optimizer,
                           discriminator_optimizer=discriminator_optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, directory=cfgs.TRAINED_CKPT, max_to_keep=5)

# --------------------------train start with latest checkpoint----------------------------
start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    ckpt.restore(ckpt_manager.latest_checkpoint)

summary_writer = tf.summary.create_file_writer(cfgs.SUMMARY_PATH)
# -------------------------------train step---------------------------------------
@tf.function
def train_step(real_image, noise):
    """

    :param real_image:
    :param fake_image:
    :return:
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_image = generator(noise, training=True)

        real_output = discriminator(real_image, training=True)
        fake_output = discriminator(generated_image, training=True)

        gen_loss = generator_loss(fake_output=fake_output)
        disc_loss = discriminator_loss(real_output=real_output, fake_output=fake_output)

        gen_gradient = gen_tape.gradient(gen_loss, generator.trainable_variables)
        disc_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gen_gradient, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(disc_gradient, discriminator.trainable_variables))

    return gen_loss, disc_loss


# --------------------------------- train-------------------------------
tf.random.set_seed(0)
noise_seed = tf.random.normal(mean=0.0, stddev=1.0, shape=[cfgs.BATCH_SIZE, cfgs.Z_DIM], dtype=tf.float32)


def train(dataset, noise, epochs):
    """

    :param dataset:
    :param epochs:
    :return:
    """
    global_step = 0
    for epoch in range(epochs):
        start_time = time.time()
        epoch_steps = 0
        gen_losses = 0
        disc_losses = 0
        for (batch, (image_batch, _)) in enumerate(dataset):
            noise_batch = next(iter(noise))
            gen_loss, disc_loss = train_step(image_batch, noise_batch)

            gen_losses += gen_loss
            disc_losses += disc_loss
            epoch_steps += 1
            global_step += 1


            if batch % cfgs.SHOW_TRAIN_INFO_INTE == 0:
                print('Epoch {} Batch {} Generator Loss {:.4f} Discriminator Loss {:.4f}'.format(
                    epoch + 1, batch, gen_loss / cfgs.BATCH_SIZE, disc_loss / cfgs.BATCH_SIZE))

            if global_step % cfgs.SMRY_ITER == 0:
                with summary_writer.as_default():
                    tf.summary.scalar('generator_loss', (gen_losses / epoch_steps), step=global_step)
                    tf.summary.scalar('discriminator_loss', (disc_losses / epoch_steps), step=global_step)

        if epoch % 5 == 0:
            ckpt_manager.save()



        print('Epoch {} Generator Loss {:.4f} Discriminator Loss {:.4f}'.format(epoch + 1,
                                                                                gen_losses / epoch_steps,
                                                                                disc_losses / epoch_steps))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start_time))

        generated_image = generator(noise_seed, training=False)

        show_save_image_grid(generated_image, save_dir=cfgs.IMAGE_SAVE_PATH, batch_size=cfgs.BATCH_SIZE, id=epoch)


def main():
    # load mnist
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    noise_batch = noise_dataset(cfgs.BATCH_SIZE, cfgs.Z_DIM)
    mnist_batch = mnist_generator(train_images, train_labels, batch_size=cfgs.BATCH_SIZE)

    train(mnist_batch, noise_batch, cfgs.NUM_EPOCH)

    generate_gif(image_path=cfgs.IMAGE_SAVE_PATH,
                 anim_file=os.path.join(cfgs.IMAGE_SAVE_PATH, 'dcgan_mnist.gif'))


if __name__ == "__main__":
    main()







