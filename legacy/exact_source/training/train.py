"""
Training script for the Pix2Pix GAN — IMPROVED VERSION v2.

Key improvements:
    - Discriminator throttling (per-batch, based on running D_loss)
    - Validation Dice computed every N epochs
    - Best model saved by Dice score (not just G_loss)
    - Learning rate scheduling on plateau
    - Early stopping when Dice stops improving (FIXED: separate counters)
    - Automatic saturation detection (stops when model stops learning)
    - Data augmentation applied per epoch

Usage:
    python training/train.py --view 2CH --structure LVendo --epochs 300
"""

import os
import sys
import argparse
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tensorflow as tf

# GPU memory growth — prevents TF from allocating all VRAM at once
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
else:
    print("WARNING: No GPU found. Training will be slow on CPU.")

import config
from data.dataset import load_train_test
from data.preprocessing import preprocess_dataset, augment_dataset
from models.pix2pix_gan import Pix2PixGAN
from utils.visualization import plot_loss_curves, plot_sample_predictions


def compute_dice_on_batch(model, images, masks):
    """Compute average Dice score on a batch of data.

    Parameters
    ----------
    model : tf.keras.Model
        The generator model.
    images : np.ndarray
        Input images, shape (N, H, W, 1).
    masks : np.ndarray
        Ground truth masks, shape (N, H, W, 1).

    Returns
    -------
    float
        Average Dice coefficient across all samples.
    """
    dice_scores = []
    for i in range(len(images)):
        pred = model(images[i:i+1], training=False).numpy()
        pred_binary = (pred[0, :, :, 0] > 0.5).astype(np.float32)
        gt = masks[i, :, :, 0]

        intersection = np.sum(pred_binary * gt)
        total = np.sum(pred_binary) + np.sum(gt)

        if total == 0:
            dice = 1.0  # Both empty = perfect match
        else:
            dice = (2.0 * intersection) / total

        dice_scores.append(dice)

    return np.mean(dice_scores)


def train(view="2CH", structure="LVendo", epochs=None, resume_from=None):
    """Train the Pix2Pix GAN model with all improvements.

    Automatic saturation detection stops training when validation Dice
    plateaus, preventing wasted compute time.

    Parameters
    ----------
    view : str
        Camera view: '2CH' or '4CH'.
    structure : str
        Cardiac structure: 'LVendo', 'LVmyo', or 'LA'.
    epochs : int
        Maximum training epochs (default from config). Training may stop
        earlier via early stopping if Dice score saturates.
    resume_from : str, optional
        Path to generator weights to resume training from.
    """
    if epochs is None:
        epochs = config.EPOCHS

    print("=" * 60)
    print(f"  Pix2Pix GAN Training (IMPROVED v2)")
    print(f"  View: {view} | Structure: {structure}")
    print(f"  Max Epochs: {epochs} (auto-stops at saturation)")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  LR (G): {config.LEARNING_RATE_G} | LR (D): {config.LEARNING_RATE_D}")
    print(f"  Lambda pixel: {config.LAMBDA_PIXEL} | Lambda Dice: {config.LAMBDA_DICE}")
    print(f"  Label smoothing: {config.LABEL_SMOOTHING}")
    print(f"  D throttle threshold: {config.D_THROTTLE_THRESHOLD}")
    print(f"  Augmentation: {config.AUGMENT_TRAIN}")
    print(f"  Early stop patience: {config.EARLY_STOP_PATIENCE} epochs")
    print(f"  LR reduce patience: {config.LR_REDUCE_PATIENCE} epochs")
    print("=" * 60)

    # =========================================================================
    # 1. Load and preprocess data
    # =========================================================================
    print("\n[1/4] Loading dataset...")
    train_images, train_masks, test_images, test_masks = load_train_test(
        view=view, structure=structure
    )

    print("\n[2/4] Preprocessing...")
    X_train, Y_train = preprocess_dataset(train_images, train_masks)
    X_test, Y_test = preprocess_dataset(test_images, test_masks)

    print(f"  Training set:  {X_train.shape}")
    print(f"  Test set:      {X_test.shape}")

    n_samples = X_train.shape[0]
    n_batches = n_samples // config.BATCH_SIZE

    if n_batches == 0:
        print("ERROR: Not enough samples for even 1 batch. Check your dataset.")
        return

    # =========================================================================
    # 2. Build model
    # =========================================================================
    print("\n[3/4] Building Pix2Pix GAN...")
    gan = Pix2PixGAN()

    if resume_from:
        gan.generator.load_weights(resume_from)
        print(f"  Resumed from: {resume_from}")

    print(f"  Generator parameters:     {gan.generator.count_params():,}")
    print(f"  Discriminator parameters: {gan.discriminator.count_params():,}")
    print(f"  Total parameters:         "
          f"{gan.generator.count_params() + gan.discriminator.count_params():,}")

    # =========================================================================
    # 3. Training loop
    # =========================================================================
    print("\n[4/4] Training...")

    run_name = f"{view}_{structure}"
    run_checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, run_name)
    run_plot_dir = os.path.join(config.PLOT_DIR, run_name)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    os.makedirs(run_plot_dir, exist_ok=True)

    # Loss & Dice history
    history = {
        "gen_loss": [],
        "gen_adv_loss": [],
        "gen_pixel_loss": [],
        "gen_dice_loss": [],
        "disc_loss": [],
        "disc_real_loss": [],
        "disc_fake_loss": [],
        "val_dice": [],
    }

    best_dice = 0.0
    best_gen_loss = float("inf")
    d_skipped_total = 0
    start_time = time.time()

    # --- Early stopping & LR scheduling state ---
    # These are INDEPENDENT counters (fixing the critical bug from v1)
    epochs_since_improvement = 0    # For early stopping (never reset except on improvement)
    lr_reductions_done = 0          # Track how many times we've reduced LR
    current_lr_g = config.LEARNING_RATE_G
    current_lr_d = config.LEARNING_RATE_D

    # Running D_loss for per-batch throttling (exponential moving average)
    running_d_loss = 1.0  # Start high = don't throttle initially

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Apply augmentation per epoch (fresh random transforms)
        if config.AUGMENT_TRAIN:
            X_epoch, Y_epoch = augment_dataset(X_train, Y_train)
        else:
            X_epoch, Y_epoch = X_train, Y_train

        # Shuffle training data
        indices = np.random.permutation(n_samples)
        X_shuffled = X_epoch[indices]
        Y_shuffled = Y_epoch[indices]

        epoch_losses = {k: [] for k in history.keys() if k != "val_dice"}
        d_skipped_epoch = 0

        # Mini-batch learning loop
        for batch_idx in range(n_batches):
            start = batch_idx * config.BATCH_SIZE
            end = start + config.BATCH_SIZE

            batch_X = tf.constant(X_shuffled[start:end], dtype=tf.float32)
            batch_Y = tf.constant(Y_shuffled[start:end], dtype=tf.float32)

            # Per-batch D throttling using running average
            update_d = running_d_loss >= config.D_THROTTLE_THRESHOLD

            # Dispatch to correct train step
            losses = gan.train_step(batch_X, batch_Y,
                                    update_discriminator=update_d)

            if not update_d:
                d_skipped_epoch += 1

            # Update running D_loss (exponential moving average, α=0.1)
            batch_d_loss = float(losses["disc_loss"])
            running_d_loss = 0.9 * running_d_loss + 0.1 * batch_d_loss

            for k in epoch_losses:
                epoch_losses[k].append(float(losses[k]))

        d_skipped_total += d_skipped_epoch

        # Average losses for this epoch
        for k in epoch_losses:
            avg = np.mean(epoch_losses[k])
            history[k].append(avg)

        epoch_time = time.time() - epoch_start

        # Log progress
        if epoch % config.DISPLAY_INTERVAL == 0 or epoch == 1:
            d_skip_pct = (d_skipped_epoch / n_batches * 100) if n_batches > 0 else 0
            elapsed_min = (time.time() - start_time) / 60
            print(f"  Epoch {epoch:4d}/{epochs} [{epoch_time:.1f}s, {elapsed_min:.0f}m total] | "
                  f"G: {history['gen_loss'][-1]:.4f} "
                  f"(adv:{history['gen_adv_loss'][-1]:.3f} "
                  f"pix:{history['gen_pixel_loss'][-1]:.3f} "
                  f"dice:{history['gen_dice_loss'][-1]:.3f}) | "
                  f"D: {history['disc_loss'][-1]:.4f} "
                  f"(skip:{d_skip_pct:.0f}%) | "
                  f"run_D:{running_d_loss:.3f}")

        # =====================================================================
        # Validation Dice (the real metric — checked every N epochs)
        # =====================================================================
        if epoch % config.VALIDATE_INTERVAL == 0 or epoch == 1:
            val_dice = compute_dice_on_batch(gan.generator, X_test, Y_test)
            history["val_dice"].append(val_dice)
            print(f"  ► Val Dice: {val_dice:.4f} (best: {best_dice:.4f}) | "
                  f"no-improve: {epochs_since_improvement} epochs")

            if val_dice > best_dice:
                # === IMPROVEMENT FOUND ===
                best_dice = val_dice
                gan.save_models(run_checkpoint_dir, epoch="best")
                print(f"  ★ New best Dice: {best_dice:.4f}")
                epochs_since_improvement = 0
                lr_reductions_done = 0
            else:
                # === NO IMPROVEMENT ===
                epochs_since_improvement += config.VALIDATE_INTERVAL

            # --- LR reduction on plateau (can trigger multiple times) ---
            # Trigger at: LR_REDUCE_PATIENCE, 2*LR_REDUCE_PATIENCE, etc.
            lr_reduce_threshold = config.LR_REDUCE_PATIENCE * (lr_reductions_done + 1)
            if epochs_since_improvement >= lr_reduce_threshold:
                if current_lr_g > config.MIN_LR:
                    lr_reductions_done += 1
                    current_lr_g = max(
                        current_lr_g * config.LR_REDUCE_FACTOR, config.MIN_LR)
                    current_lr_d = max(
                        current_lr_d * config.LR_REDUCE_FACTOR, config.MIN_LR)
                    gan.gen_optimizer.learning_rate.assign(current_lr_g)
                    gan.disc_optimizer.learning_rate.assign(current_lr_d)
                    print(f"  ↓ LR reduced (#{lr_reductions_done}) → "
                          f"G:{current_lr_g:.2e} D:{current_lr_d:.2e}")

            # --- Early stopping (INDEPENDENT of LR reduction) ---
            if epochs_since_improvement >= config.EARLY_STOP_PATIENCE:
                print(f"\n  {'='*50}")
                print(f"  ⊘ EARLY STOPPING at epoch {epoch}")
                print(f"    Dice has not improved for "
                      f"{epochs_since_improvement} epochs")
                print(f"    Best Dice: {best_dice:.4f}")
                print(f"    LR reductions done: {lr_reductions_done}")
                print(f"  {'='*50}")
                break

        # Also track best G_loss (secondary metric)
        current_gen_loss = history["gen_loss"][-1]
        if current_gen_loss < best_gen_loss:
            best_gen_loss = current_gen_loss

        # Periodic checkpoint
        if epoch % config.SAVE_INTERVAL == 0:
            gan.save_models(run_checkpoint_dir, epoch=epoch)

    # =========================================================================
    # 4. Save final results
    # =========================================================================
    total_time = time.time() - start_time
    gan.save_models(run_checkpoint_dir, epoch="final")

    # Save loss history
    history_path = os.path.join(run_checkpoint_dir, "loss_history.npy")
    np.save(history_path, history)
    print(f"\nLoss history saved to: {history_path}")

    # Plot loss curves
    try:
        plot_loss_curves(history,
                         save_path=os.path.join(run_plot_dir, "loss_curves.png"))
        print(f"Loss curves saved to: {run_plot_dir}/loss_curves.png")
    except Exception as e:
        print(f"Warning: Could not plot loss curves: {e}")

    # Plot sample predictions using BEST model
    try:
        best_gen_path = os.path.join(run_checkpoint_dir, "generator_epoch_best.h5")
        if os.path.exists(best_gen_path):
            gan.generator.load_weights(best_gen_path)

        n_samples_to_show = min(5, len(X_test))
        predictions = gan.generator(X_test[:n_samples_to_show], training=False)
        plot_sample_predictions(
            X_test[:n_samples_to_show],
            Y_test[:n_samples_to_show],
            predictions.numpy(),
            save_path=os.path.join(run_plot_dir, "sample_predictions.png")
        )
        print(f"Sample predictions saved to: {run_plot_dir}/sample_predictions.png")
    except Exception as e:
        print(f"Warning: Could not plot sample predictions: {e}")

    print("\n" + "=" * 60)
    print(f"  Training complete!")
    print(f"  Total time: {total_time/3600:.1f} hours ({total_time/60:.0f} minutes)")
    print(f"  Final epoch: {epoch}")
    print(f"  Best Dice score: {best_dice:.4f}")
    print(f"  Best generator loss: {best_gen_loss:.4f}")
    print(f"  LR reductions: {lr_reductions_done}")
    print(f"  D updates skipped: {d_skipped_total} total")
    print(f"  Checkpoints: {run_checkpoint_dir}")
    print(f"  Plots: {run_plot_dir}")
    print("=" * 60)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Pix2Pix GAN for echocardiography segmentation"
    )
    parser.add_argument(
        "--view", type=str, default="2CH",
        choices=["2CH", "4CH"],
        help="Camera view: 2CH or 4CH (default: 2CH)"
    )
    parser.add_argument(
        "--structure", type=str, default="LVendo",
        choices=["LVendo", "LVmyo", "LA"],
        help="Cardiac structure to segment (default: LVendo)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=f"Max training epochs (default: {config.EPOCHS}). "
             "Training auto-stops at saturation."
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to generator weights to resume from"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        view=args.view,
        structure=args.structure,
        epochs=args.epochs,
        resume_from=args.resume,
    )

