"""
UNET Generator for the Pix2Pix GAN.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config


def downsample_block(filters, kernel_size=4, apply_batchnorm=True):
    block = tf.keras.Sequential()
    block.add(layers.Conv2D(
        filters, kernel_size, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        use_bias=False
    ))
    if apply_batchnorm:
        block.add(layers.BatchNormalization())
    block.add(layers.LeakyReLU(0.2))
    return block


def upsample_block(filters, kernel_size=4, apply_dropout=False):
    block = tf.keras.Sequential()
    block.add(layers.Conv2DTranspose(
        filters, kernel_size, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        use_bias=False
    ))
    block.add(layers.BatchNormalization())
    if apply_dropout:
        block.add(layers.Dropout(0.5))
    block.add(layers.ReLU())
    return block


def build_generator(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
                    output_channels=config.OUTPUT_CHANNELS):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    down1 = downsample_block(64, apply_batchnorm=False)(inputs)
    down2 = downsample_block(128)(down1)
    down3 = downsample_block(256)(down2)
    down4 = downsample_block(512)(down3)
    down5 = downsample_block(512)(down4)
    down6 = downsample_block(512)(down5)
    down7 = downsample_block(512)(down6)
    down8 = downsample_block(512, apply_batchnorm=False)(down7)

    # Decoder
    up1 = upsample_block(512, apply_dropout=True)(down8)
    up1 = layers.Concatenate()([up1, down7])

    up2 = upsample_block(512, apply_dropout=True)(up1)
    up2 = layers.Concatenate()([up2, down6])

    up3 = upsample_block(512, apply_dropout=True)(up2)
    up3 = layers.Concatenate()([up3, down5])

    up4 = upsample_block(512)(up3)
    up4 = layers.Concatenate()([up4, down4])

    up5 = upsample_block(256)(up4)
    up5 = layers.Concatenate()([up5, down3])

    up6 = upsample_block(128)(up5)
    up6 = layers.Concatenate()([up6, down2])

    up7 = upsample_block(64)(up6)
    up7 = layers.Concatenate()([up7, down1])

    # Output layer
    output = layers.Conv2DTranspose(
        output_channels, kernel_size=4, strides=2, padding='same',
        kernel_initializer=tf.random_normal_initializer(0., 0.02),
        activation='sigmoid'
    )(up7)

    model = Model(inputs=inputs, outputs=output, name="UNET_Generator")
    return model
