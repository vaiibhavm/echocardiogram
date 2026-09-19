"""
PatchGAN Discriminator for the Pix2Pix GAN.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config


def build_discriminator(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)):
    input_image = layers.Input(shape=input_shape, name="input_image")
    target_image = layers.Input(shape=input_shape, name="target_image")

    x = layers.Concatenate()([input_image, target_image])

    x = layers.Conv2D(
        64, kernel_size=4, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02)
    )(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(
        128, kernel_size=4, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(
        256, kernel_size=4, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.ZeroPadding2D()(x)
    x = layers.Conv2D(
        512, kernel_size=4, strides=1, padding='valid',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.ZeroPadding2D()(x)
    output = layers.Conv2D(
        1, kernel_size=4, strides=1, padding='valid',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        activation='sigmoid'
    )(x)

    model = Model(inputs=[input_image, target_image], outputs=output,
                  name="PatchGAN_Discriminator")
    return model
