"""
Configuration file for the Pix2Pix GAN Echocardiography Segmentation project.

All hyperparameters are set according to the paper:
Fatima et al., "Automatic Segmentation of 2-D Echocardiography Ultrasound
Images by Means of Generative Adversarial Network", IEEE TUFFC, 2024.

Dataset: CAMUS_public (NIfTI format, .nii.gz files)

IMPROVED: Tuned for stable GAN training and better Dice scores.
"""

import os

# ==============================================================================
# Paths
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- CAMUS dataset location (NIfTI format) ---
DATASET_DIR = r"C:\Users\tejas\Desktop\code\projects\DATASET\PRINC\CAMUS_public"
DATA_DIR = os.path.join(DATASET_DIR, "database_nifti")
SPLIT_DIR = os.path.join(DATASET_DIR, "database_split")

# Split text files (provided by CAMUS)
TRAIN_SPLIT_FILE = os.path.join(SPLIT_DIR, "subgroup_training.txt")
TEST_SPLIT_FILE = os.path.join(SPLIT_DIR, "subgroup_testing.txt")
VAL_SPLIT_FILE = os.path.join(SPLIT_DIR, "subgroup_validation.txt")

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# Create output directories if they don't exist
for directory in [CHECKPOINT_DIR, PLOT_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==============================================================================
# Dataset Configuration
# ==============================================================================
VIEWS = ["2CH", "4CH"]
PHASES = ["ED", "ES"]

STRUCTURE_LABELS = {
    "LVendo": 1,
    "LVmyo": 2,
    "LA": 3,
}
STRUCTURES = list(STRUCTURE_LABELS.keys())

# ==============================================================================
# Image Configuration
# ==============================================================================
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 1
OUTPUT_CHANNELS = 1

# ==============================================================================
# Model Hyperparameters (IMPROVED from paper defaults)
# ==============================================================================
PATCH_SIZE = 70
BATCH_SIZE = 1
EPOCHS = 300               # More epochs, but early stopping prevents waste

# Learning rates — D gets HALF the LR of G to prevent dominance
LEARNING_RATE_G = 0.0002   # Generator learning rate
LEARNING_RATE_D = 0.0001   # Discriminator learning rate (halved!)
BETA_1 = 0.5
BETA_2 = 0.999

# Loss weights
LAMBDA_PIXEL = 100         # L1 pixel loss weight (Eq. 3)
LAMBDA_DICE = 50           # Dice loss weight (NEW — directly optimizes Dice)

# Training stability
LABEL_SMOOTHING = 0.9      # Real labels → 0.9 instead of 1.0
D_THROTTLE_THRESHOLD = 0.3 # Skip D update when D_loss < this value
NOISE_LABELS_PROB = 0.05   # Probability of flipping real/fake labels (noise)

# Data augmentation
AUGMENT_TRAIN = True       # Enable augmentation during training
AUG_FLIP_H = True          # Random horizontal flip
AUG_FLIP_V = False         # Random vertical flip (disabled — anatomy matters)
AUG_ROTATE_MAX = 10        # Max random rotation in degrees
AUG_BRIGHTNESS = 0.1       # Random brightness jitter
AUG_CONTRAST = 0.1         # Random contrast jitter

# Generator (UNET) architecture
GENERATOR_FILTERS = [64, 128, 256, 512, 512, 512, 512, 512]

# Discriminator (PatchGAN) architecture
DISCRIMINATOR_FILTERS = [64, 128, 256, 512]

# ==============================================================================
# Training Configuration
# ==============================================================================
SAVE_INTERVAL = 25
DISPLAY_INTERVAL = 5
SAVE_BEST_ONLY = True
VALIDATE_INTERVAL = 10     # Run validation Dice every N epochs
EARLY_STOP_PATIENCE = 60   # Stop if no Dice improvement for N epochs
LR_REDUCE_PATIENCE = 30    # Halve LR if no improvement for N epochs
LR_REDUCE_FACTOR = 0.5     # Multiply LR by this factor
MIN_LR = 1e-6              # Minimum learning rate floor
