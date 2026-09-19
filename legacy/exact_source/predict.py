"""
Custom inference script — run the trained model on your own echocardiography images.

Supports:
    - Single image files (PNG, JPG, BMP, TIFF, DICOM)
    - Folders of images
    - Video files (AVI, MP4) — processes each frame

Usage:
    python predict.py --input path/to/image.png --weights outputs/checkpoints/2CH_LVendo/generator_epoch_best.h5
    python predict.py --input path/to/folder/ --weights outputs/checkpoints/2CH_LVendo/generator_epoch_best.h5
    python predict.py --input path/to/video.avi --weights outputs/checkpoints/2CH_LVendo/generator_epoch_best.h5
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import tensorflow as tf

# GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import config
from models.pix2pix_gan import Pix2PixGAN

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =========================================================================
# Image loading utilities
# =========================================================================

def load_image_file(filepath):
    """Load a single image file (PNG, JPG, BMP, TIFF, or DICOM).

    Parameters
    ----------
    filepath : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        Grayscale image as 2-D numpy array.
    """
    ext = os.path.splitext(filepath)[1].lower()

    # DICOM files
    if ext in ['.dcm', '.dicom']:
        try:
            import pydicom
            ds = pydicom.dcmread(filepath)
            img = ds.pixel_array.astype(np.float32)
        except ImportError:
            raise ImportError("Install pydicom for DICOM support: pip install pydicom")
    # MHD files (CAMUS format)
    elif ext in ['.mhd']:
        try:
            import SimpleITK as sitk
            image = sitk.ReadImage(filepath)
            img = sitk.GetArrayFromImage(image).astype(np.float32)
            if img.ndim == 3:
                img = img[0]
        except ImportError:
            raise ImportError("Install SimpleITK for .mhd support: pip install SimpleITK")
    # Standard image files
    else:
        from skimage.io import imread
        img = imread(filepath).astype(np.float32)

    # Convert RGB/RGBA to grayscale if needed
    if img.ndim == 3:
        img = np.mean(img[:, :, :3], axis=2)

    return img


def load_video_frames(video_path, max_frames=None):
    """Load frames from a video file.

    Parameters
    ----------
    video_path : str
        Path to video file (AVI, MP4, etc.).
    max_frames : int, optional
        Maximum number of frames to extract.

    Returns
    -------
    list of np.ndarray
        List of grayscale frames.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("Install OpenCV for video support: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame.astype(np.float32))
        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break

    cap.release()
    print(f"Loaded {len(frames)} frames from video")
    return frames


# =========================================================================
# Preprocessing (same as training pipeline)
# =========================================================================

def preprocess_single_image(image):
    """Preprocess a single image for inference.

    Parameters
    ----------
    image : np.ndarray
        Raw grayscale image of any size.

    Returns
    -------
    np.ndarray
        Preprocessed image of shape (1, 256, 256, 1).
    """
    from skimage.transform import resize

    # Resize to 256x256
    resized = resize(image, (config.IMG_HEIGHT, config.IMG_WIDTH),
                     mode='reflect', anti_aliasing=True,
                     preserve_range=True).astype(np.float32)

    # Normalize to [0, 1]
    img_min = resized.min()
    img_max = resized.max()
    if img_max - img_min > 0:
        resized = (resized - img_min) / (img_max - img_min)
    else:
        resized = np.zeros_like(resized)

    # Add batch and channel dims: (H, W) -> (1, H, W, 1)
    return resized[np.newaxis, :, :, np.newaxis]


# =========================================================================
# Prediction and visualization
# =========================================================================

def predict_and_save(image, model, save_path, original_path=""):
    """Run prediction on a single image and save the result.

    Parameters
    ----------
    image : np.ndarray
        Raw grayscale image.
    model : tf.keras.Model
        Trained generator model.
    save_path : str
        Path to save the output image.
    original_path : str
        Name of the original file (for display).
    """
    # Preprocess
    input_tensor = preprocess_single_image(image)

    # Predict
    prediction = model(input_tensor, training=False).numpy()
    pred_mask = (prediction[0, :, :, 0] > 0.5).astype(np.float32)

    # Resize input for display
    from skimage.transform import resize
    display_img = resize(image, (config.IMG_HEIGHT, config.IMG_WIDTH),
                         mode='reflect', anti_aliasing=True,
                         preserve_range=True)
    # Normalize for display
    dmin, dmax = display_img.min(), display_img.max()
    if dmax - dmin > 0:
        display_img = (display_img - dmin) / (dmax - dmin)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(display_img, cmap='gray')
    axes[0].set_title('Input Image', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    # Predicted mask
    axes[1].imshow(pred_mask, cmap='gray')
    axes[1].set_title('Predicted Segmentation', fontsize=13, fontweight='bold')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(display_img, cmap='gray')
    axes[2].contour(pred_mask, levels=[0.5], colors='lime', linewidths=2)
    axes[2].set_title('Overlay', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    basename = os.path.basename(original_path) if original_path else "Custom Image"
    plt.suptitle(f'Segmentation Result: {basename}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def run_prediction(input_path, weights_path, output_dir=None, max_video_frames=50):
    """Run prediction on images, folders, or videos.

    Parameters
    ----------
    input_path : str
        Path to image file, folder of images, or video file.
    weights_path : str
        Path to trained generator weights (.h5 file).
    output_dir : str, optional
        Directory to save results (default: outputs/predictions/).
    max_video_frames : int
        Max frames to process from video.
    """
    if output_dir is None:
        output_dir = os.path.join(config.OUTPUT_DIR, "predictions")
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    gan = Pix2PixGAN()
    gan.generator.load_weights(weights_path)
    model = gan.generator
    print(f"  Loaded weights: {weights_path}")

    # Determine input type
    if os.path.isdir(input_path):
        # Folder of images
        extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.mhd', '.dcm'}
        files = [f for f in os.listdir(input_path)
                 if os.path.splitext(f)[1].lower() in extensions]
        files.sort()
        print(f"\nProcessing {len(files)} images from folder...")

        for fname in files:
            fpath = os.path.join(input_path, fname)
            try:
                img = load_image_file(fpath)
                save_name = os.path.splitext(fname)[0] + "_segmented.png"
                save_path = os.path.join(output_dir, save_name)
                predict_and_save(img, model, save_path, fpath)
            except Exception as e:
                print(f"  ERROR on {fname}: {e}")

    elif input_path.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
        # Video file
        print(f"\nProcessing video: {input_path}")
        frames = load_video_frames(input_path, max_frames=max_video_frames)

        for i, frame in enumerate(frames):
            save_name = f"frame_{i:04d}_segmented.png"
            save_path = os.path.join(output_dir, save_name)
            predict_and_save(frame, model, save_path, f"frame_{i:04d}")

    else:
        # Single image file
        print(f"\nProcessing image: {input_path}")
        img = load_image_file(input_path)
        save_name = os.path.splitext(os.path.basename(input_path))[0] + "_segmented.png"
        save_path = os.path.join(output_dir, save_name)
        predict_and_save(img, model, save_path, input_path)

    print(f"\nAll results saved to: {output_dir}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run segmentation on custom echocardiography images"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to image file, folder of images, or video file"
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to trained generator weights (.h5 file)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: outputs/predictions/)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=50,
        help="Max video frames to process (default: 50)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prediction(
        input_path=args.input,
        weights_path=args.weights,
        output_dir=args.output,
        max_video_frames=args.max_frames,
    )

