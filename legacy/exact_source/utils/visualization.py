"""
Visualization utilities for the Pix2Pix GAN — IMPROVED VERSION.

Provides plotting functions for:
    - Loss curves with Dice overlay
    - Sample segmentation predictions vs ground truth
    - EF correlation scatter plots
    - Dice score boxplots
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt


def plot_loss_curves(history, save_path=None, title="Training Loss Curves"):
    """Plot training loss curves with optional Dice score overlay.

    Shows G/D losses on top subplot and validation Dice on bottom subplot.

    Parameters
    ----------
    history : dict
        Dictionary with keys: 'gen_loss', 'disc_real_loss', 'disc_fake_loss',
        and optionally 'val_dice', 'gen_dice_loss'.
    save_path : str, optional
        Path to save the figure.
    title : str
        Plot title.
    """
    has_dice = 'val_dice' in history and len(history['val_dice']) > 0
    n_plots = 2 if has_dice else 1

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 5 * n_plots))
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(history['gen_loss']) + 1)

    # --- Top plot: Losses ---
    ax = axes[0]

    # Discriminator losses
    ax.plot(epochs, history['disc_real_loss'], 'r-', alpha=0.6,
            label='D Real Loss', linewidth=1.5)
    ax.plot(epochs, history['disc_fake_loss'], 'b-', alpha=0.6,
            label='D Fake Loss', linewidth=1.5)
    ax.plot(epochs, history['disc_loss'], 'purple', alpha=0.5,
            label='D Total Loss', linewidth=1.0, linestyle='--')

    # Generator loss
    ax.plot(epochs, history['gen_loss'], color='orange', alpha=0.8,
            label='G Loss', linewidth=2)

    # Generator components
    if 'gen_dice_loss' in history and len(history['gen_dice_loss']) > 0:
        ax.plot(epochs, history['gen_dice_loss'], color='green', alpha=0.5,
                label='G Dice Loss', linewidth=1.0, linestyle=':')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    # D throttle reference line
    ax.axhline(y=0.3, color='gray', linestyle=':', alpha=0.5,
               label='D throttle threshold')

    # --- Bottom plot: Validation Dice ---
    if has_dice:
        ax2 = axes[1]
        val_dice = history['val_dice']
        # Dice is sampled every VALIDATE_INTERVAL epochs
        n_dice = len(val_dice)
        n_epochs = len(history['gen_loss'])
        dice_epochs = np.linspace(1, n_epochs, n_dice, dtype=int)

        ax2.plot(dice_epochs, val_dice, 'g-o', linewidth=2,
                 markersize=4, label='Validation Dice')
        ax2.axhline(y=max(val_dice), color='green', linestyle='--',
                     alpha=0.5, label=f'Best: {max(val_dice):.4f}')

        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Dice Coefficient', fontsize=12)
        ax2.set_title('Validation Dice Score', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_sample_predictions(images, ground_truths, predictions,
                            save_path=None, title="Segmentation Results",
                            n_samples=None):
    """Plot input images alongside ground truth and predicted segmentation.

    Parameters
    ----------
    images : np.ndarray
        Input images, shape (N, H, W, 1).
    ground_truths : np.ndarray
        Ground truth masks, shape (N, H, W, 1).
    predictions : np.ndarray
        Predicted masks, shape (N, H, W, 1).
    save_path : str, optional
        Path to save the figure.
    title : str
        Plot title.
    n_samples : int, optional
        Number of samples to show (default: all).
    """
    if n_samples is None:
        n_samples = min(len(images), 5)

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

    if n_samples == 1:
        axes = axes[np.newaxis, :]

    column_titles = ['Input Image', 'Ground Truth', 'Prediction', 'Overlay']

    for col, col_title in enumerate(column_titles):
        axes[0, col].set_title(col_title, fontsize=13, fontweight='bold')

    for i in range(n_samples):
        img = images[i].squeeze()
        gt = ground_truths[i].squeeze()
        pred = (predictions[i].squeeze() > 0.5).astype(float)

        # Compute per-sample Dice for display
        intersection = np.sum(pred * gt)
        total = np.sum(pred) + np.sum(gt)
        dice = (2.0 * intersection / total) if total > 0 else 1.0

        # Input image
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].axis('off')

        # Ground truth mask
        axes[i, 1].imshow(gt, cmap='gray')
        axes[i, 1].axis('off')

        # Predicted mask
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].axis('off')

        # Overlay: input with GT contour (red) and Pred contour (green)
        axes[i, 3].imshow(img, cmap='gray')
        axes[i, 3].contour(gt, levels=[0.5], colors='red',
                           linewidths=1.5, linestyles='dashed')
        axes[i, 3].contour(pred, levels=[0.5], colors='lime',
                           linewidths=1.5)
        axes[i, 3].set_title(f'Dice: {dice:.3f}', fontsize=11, color='green')
        axes[i, 3].axis('off')

    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_ef_correlation(ef_gt, ef_pred, save_path=None,
                        title="EF Correlation"):
    """Plot EF correlation scatter plot.

    Parameters
    ----------
    ef_gt : np.ndarray
        Ground truth EF values.
    ef_pred : np.ndarray
        Predicted EF values.
    save_path : str, optional
        Path to save the figure.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    ax.scatter(ef_gt * 100, ef_pred * 100, alpha=0.6, s=40,
               edgecolors='navy', facecolors='cornflowerblue')

    # Identity line
    lims = [0, 100]
    ax.plot(lims, lims, 'r--', alpha=0.7, linewidth=1.5, label='Identity')

    # Correlation
    corr = np.corrcoef(ef_gt, ef_pred)[0, 1]
    mae = np.mean(np.abs(ef_gt - ef_pred)) * 100

    ax.set_xlabel('Ground Truth EF (%)', fontsize=12)
    ax.set_ylabel('Predicted EF (%)', fontsize=12)
    ax.set_title(f'{title}\nCorrelation: {corr:.3f} | MAE: {mae:.1f}%',
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_dice_boxplot(dice_scores_dict, save_path=None,
                      title="Dice Scores by Training Size"):
    """Plot boxplot of Dice scores across training sizes.

    Parameters
    ----------
    dice_scores_dict : dict
        Keys are training sizes (int), values are lists of Dice scores.
    save_path : str, optional
        Path to save the figure.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    labels = sorted(dice_scores_dict.keys())
    data = [dice_scores_dict[k] for k in labels]

    bp = ax.boxplot(data, labels=[str(l) for l in labels],
                    patch_artist=True)

    # Style
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Training Size', fontsize=12)
    ax.set_ylabel('Dice Coefficient', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

