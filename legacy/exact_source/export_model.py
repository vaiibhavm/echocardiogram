"""
Export trained model for use on another system.

Saves the generator as:
    1. Full Keras model (.keras or SavedModel) — easy to load anywhere
    2. Weights only (.h5) — smaller file, needs code to rebuild architecture

Usage:
    python export_model.py --weights outputs/checkpoints/2CH_LVendo/generator_epoch_best.h5 --output exported_model/
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import tensorflow as tf

# GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import config
from models.pix2pix_gan import Pix2PixGAN


def export_model(weights_path, output_dir="exported_model"):
    """Export the trained generator model for deployment on another system.

    Creates three export formats:
        1. SavedModel (TensorFlow standard format — most portable)
        2. .h5 weights file (lightweight, needs architecture code)
        3. Standalone predict script that works without the full project

    Parameters
    ----------
    weights_path : str
        Path to the trained generator weights.
    output_dir : str
        Directory to save exported files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build and load the trained generator
    print("Building generator and loading weights...")
    gan = Pix2PixGAN()
    gan.generator.load_weights(weights_path)
    generator = gan.generator

    print(f"  Parameters: {generator.count_params():,}")
    print(f"  Input shape:  {generator.input_shape}")
    print(f"  Output shape: {generator.output_shape}")

    # =========================================================================
    # Export 1: SavedModel format (most portable)
    # =========================================================================
    saved_model_path = os.path.join(output_dir, "saved_model")
    generator.save(saved_model_path)
    print(f"\n[1] SavedModel exported to: {saved_model_path}")
    print(f"    Load with: tf.keras.models.load_model('{saved_model_path}')")

    # =========================================================================
    # Export 2: .h5 weights only (lightweight)
    # =========================================================================
    h5_path = os.path.join(output_dir, "generator_weights.h5")
    generator.save_weights(h5_path)
    print(f"\n[2] Weights exported to: {h5_path}")

    # =========================================================================
    # Export 3: Standalone prediction script
    # =========================================================================
    standalone_script = '''"""
Standalone prediction script — works without the full project.
Just needs: tensorflow, numpy, scikit-image, matplotlib

Usage:
    python standalone_predict.py --input your_image.png --model saved_model/
"""
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from skimage.io import imread
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IMG_SIZE = 256

def load_and_preprocess(image_path):
    """Load and preprocess an echocardiography image."""
    img = imread(image_path).astype(np.float32)
    if img.ndim == 3:
        img = np.mean(img[:, :, :3], axis=2)
    resized = resize(img, (IMG_SIZE, IMG_SIZE), mode='reflect',
                     anti_aliasing=True, preserve_range=True).astype(np.float32)
    img_min, img_max = resized.min(), resized.max()
    if img_max - img_min > 0:
        resized = (resized - img_min) / (img_max - img_min)
    return resized[np.newaxis, :, :, np.newaxis], img

def predict_and_save(image_path, model, output_path):
    """Run segmentation and save result."""
    input_tensor, original = load_and_preprocess(image_path)
    prediction = model(input_tensor, training=False).numpy()
    mask = (prediction[0, :, :, 0] > 0.5).astype(float)
    display = resize(original, (IMG_SIZE, IMG_SIZE), mode='reflect',
                     anti_aliasing=True, preserve_range=True)
    dmin, dmax = display.min(), display.max()
    if dmax - dmin > 0:
        display = (display - dmin) / (dmax - dmin)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(display, cmap='gray')
    axes[0].set_title('Input')
    axes[0].axis('off')
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Segmentation')
    axes[1].axis('off')
    axes[2].imshow(display, cmap='gray')
    axes[2].contour(mask, levels=[0.5], colors='lime', linewidths=2)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--model", required=True, help="Path to saved_model/ directory")
    parser.add_argument("--output", default="result.png", help="Output image path")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    predict_and_save(args.input, model, args.output)
'''

    standalone_path = os.path.join(output_dir, "standalone_predict.py")
    with open(standalone_path, 'w') as f:
        f.write(standalone_script)
    print(f"\n[3] Standalone script exported to: {standalone_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("  Export complete!")
    print("=" * 60)
    print(f"\nTo use on another system:")
    print(f"  1. Copy the '{output_dir}/' folder to the new machine")
    print(f"  2. Install: pip install tensorflow numpy scikit-image matplotlib")
    print(f"  3. Run: python standalone_predict.py --input your_image.png --model saved_model/")
    print("=" * 60)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export trained model for use on another system"
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to trained generator weights (.h5)"
    )
    parser.add_argument(
        "--output", type=str, default="exported_model",
        help="Output directory (default: exported_model/)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_model(
        weights_path=args.weights,
        output_dir=args.output,
    )

