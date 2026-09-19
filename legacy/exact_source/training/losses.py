import tensorflow as tf

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def discriminator_real_loss(disc_real_output):
    return bce(tf.ones_like(disc_real_output), disc_real_output)

def discriminator_fake_loss(disc_fake_output):
    return bce(tf.zeros_like(disc_fake_output), disc_fake_output)

def discriminator_loss(disc_real_output, disc_fake_output):
    real_loss = discriminator_real_loss(disc_real_output)
    fake_loss = discriminator_fake_loss(disc_fake_output)
    return real_loss + fake_loss, real_loss, fake_loss

def adversarial_loss(disc_fake_output):
    return bce(tf.ones_like(disc_fake_output), disc_fake_output)

def pixel_loss(target, generated):
    return tf.reduce_mean(tf.abs(target - generated))

def dice_loss(target, prediction):
    smooth = 1e-6
    target_flat = tf.reshape(target, [-1])
    pred_flat = tf.reshape(prediction, [-1])
    intersection = tf.reduce_sum(target_flat * pred_flat)
    dice = (2.0 * intersection + smooth) / (
        tf.reduce_sum(target_flat) + tf.reduce_sum(pred_flat) + smooth
    )
    return 1.0 - dice

def generator_loss(disc_fake_output, gen_output, target, lambda_pixel=100, lambda_dice=50):
    adv_loss = adversarial_loss(disc_fake_output)
    pix_loss = pixel_loss(target, gen_output)
    d_loss = dice_loss(target, gen_output)
    total_loss = adv_loss + lambda_pixel * pix_loss + lambda_dice * d_loss
    return total_loss, adv_loss, pix_loss, d_loss
