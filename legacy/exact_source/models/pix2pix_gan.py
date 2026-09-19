import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import config
from models.generator import build_generator
from models.discriminator import build_discriminator

class Pix2PixGAN:
    def __init__(self):
        self.generator = build_generator()
        self.discriminator = build_discriminator()

        self.gen_optimizer = Adam(
            learning_rate=config.LEARNING_RATE_G,
            beta_1=config.BETA_1, beta_2=config.BETA_2
        )
        self.disc_optimizer = Adam(
            learning_rate=config.LEARNING_RATE_D,
            beta_1=config.BETA_1, beta_2=config.BETA_2
        )

        self.bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.lambda_pixel = config.LAMBDA_PIXEL
        self.lambda_dice = config.LAMBDA_DICE
        self.label_smoothing = config.LABEL_SMOOTHING

    @staticmethod
    def dice_loss(target, prediction):
        smooth = 1e-6
        target_flat = tf.reshape(target, [-1])
        pred_flat = tf.reshape(prediction, [-1])
        intersection = tf.reduce_sum(target_flat * pred_flat)
        dice = (2.0 * intersection + smooth) / (
            tf.reduce_sum(target_flat) + tf.reduce_sum(pred_flat) + smooth
        )
        return 1.0 - dice

    def discriminator_loss(self, disc_real_output, disc_fake_output):
        real_labels = tf.ones_like(disc_real_output) * self.label_smoothing
        fake_labels = tf.zeros_like(disc_fake_output)

        if config.NOISE_LABELS_PROB > 0:
            noise_mask = tf.random.uniform(tf.shape(real_labels)) < config.NOISE_LABELS_PROB
            real_labels = tf.where(noise_mask, 1.0 - real_labels, real_labels)
            noise_mask_f = tf.random.uniform(tf.shape(fake_labels)) < config.NOISE_LABELS_PROB
            fake_labels = tf.where(noise_mask_f, 1.0 - fake_labels, fake_labels)

        real_loss = self.bce_loss(real_labels, disc_real_output)
        fake_loss = self.bce_loss(fake_labels, disc_fake_output)
        total_loss = real_loss + fake_loss
        return total_loss, real_loss, fake_loss

    def generator_loss(self, disc_fake_output, gen_output, target):
        adversarial_loss = self.bce_loss(tf.ones_like(disc_fake_output), disc_fake_output)
        pixel_loss = tf.reduce_mean(tf.abs(target - gen_output))
        d_loss = self.dice_loss(target, gen_output)
        total_loss = adversarial_loss + self.lambda_pixel * pixel_loss + self.lambda_dice * d_loss
        return total_loss, adversarial_loss, pixel_loss, d_loss

    @tf.function
    def train_step_full(self, input_image, target_mask):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)
            disc_real_output = self.discriminator([input_image, target_mask], training=True)
            disc_fake_output = self.discriminator([input_image, gen_output], training=True)

            disc_loss, d_real_loss, d_fake_loss = self.discriminator_loss(disc_real_output, disc_fake_output)
            gen_loss, g_adv_loss, g_pixel_loss, g_dice_loss = self.generator_loss(disc_fake_output, gen_output, target_mask)

        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        disc_gradients, _ = tf.clip_by_global_norm(disc_gradients, 5.0)
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gen_gradients, _ = tf.clip_by_global_norm(gen_gradients, 5.0)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        return {
            "disc_loss": disc_loss, "disc_real_loss": d_real_loss, "disc_fake_loss": d_fake_loss,
            "gen_loss": gen_loss, "gen_adv_loss": g_adv_loss, "gen_pixel_loss": g_pixel_loss, "gen_dice_loss": g_dice_loss,
        }

    @tf.function
    def train_step_gen_only(self, input_image, target_mask):
        with tf.GradientTape() as gen_tape:
            gen_output = self.generator(input_image, training=True)
            disc_real_output = self.discriminator([input_image, target_mask], training=False)
            disc_fake_output = self.discriminator([input_image, gen_output], training=False)
            disc_loss, d_real_loss, d_fake_loss = self.discriminator_loss(disc_real_output, disc_fake_output)
            gen_loss, g_adv_loss, g_pixel_loss, g_dice_loss = self.generator_loss(disc_fake_output, gen_output, target_mask)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gen_gradients, _ = tf.clip_by_global_norm(gen_gradients, 5.0)
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))

        return {
            "disc_loss": disc_loss, "disc_real_loss": d_real_loss, "disc_fake_loss": d_fake_loss,
            "gen_loss": gen_loss, "gen_adv_loss": g_adv_loss, "gen_pixel_loss": g_pixel_loss, "gen_dice_loss": g_dice_loss,
        }

    def train_step(self, input_image, target_mask, update_discriminator=True):
        if update_discriminator: return self.train_step_full(input_image, target_mask)
        else: return self.train_step_gen_only(input_image, target_mask)

    def save_models(self, checkpoint_dir, epoch):
        gen_path = os.path.join(checkpoint_dir, f"generator_epoch_{epoch}.h5")
        disc_path = os.path.join(checkpoint_dir, f"discriminator_epoch_{epoch}.h5")
        self.generator.save_weights(gen_path)
        self.discriminator.save_weights(disc_path)

    def load_models(self, gen_path, disc_path=None):
        self.generator.load_weights(gen_path)
        if disc_path: self.discriminator.load_weights(disc_path)
