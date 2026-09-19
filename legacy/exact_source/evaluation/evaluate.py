"""
Evaluation / inference script for the trained Pix2Pix GAN.

Loads a trained model, runs inference on the test set, computes
all geometric and clinical metrics, and saves results.

Usage:
    python evaluation/evaluate.py --view 2CH --structure LVendo
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from data.dataset import load_dataset
from data.preprocessing import preprocess_dataset
from models.pix2pix_gan import Pix2PixGAN
from evaluation.metrics import (
    evaluate_segmentation,
    calculate_volume_simpson,
    ejection_fraction,
    ef_correlation,
    ef_mae,
)
from utils.visualization import plot_sample_predictions, plot_ef_correlation


def evaluate(view="2CH", structure="LVendo", weights_path=None):
    """Run evaluation on the test set.

    Parameters
    ----------
    view : str
        Camera view: '2CH' or '4CH'.
    structure : str
        Cardiac structure: 'LVendo', 'LVmyo', or 'LA'.
    weights_path : str, optional
        Path to generator weights. If None, uses best checkpoint.
    """
    print("=" * 60)
    print(f"  Evaluation: {view} | {structure}")
    print("=" * 60)

    # =========================================================================
    # 1. Determine weights path
    # =========================================================================
    run_name = f"{view}_{structure}"
    run_checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, run_name)
    run_results_dir = os.path.join(config.RESULTS_DIR, run_name)
    run_plot_dir = os.path.join(config.PLOT_DIR, run_name)
    os.makedirs(run_results_dir, exist_ok=True)
    os.makedirs(run_plot_dir, exist_ok=True)

    if weights_path is None:
        weights_path = os.path.join(run_checkpoint_dir, "generator_epoch_best.h5")

    if not os.path.exists(weights_path):
        print(f"ERROR: Weights not found at: {weights_path}")
        print("Please train the model first using: python training/train.py")
        return

    # =========================================================================
    # 2. Load model
    # =========================================================================
    print("\nLoading model...")
    gan = Pix2PixGAN()
    gan.generator.load_weights(weights_path)
    print(f"  Loaded weights from: {weights_path}")

    # =========================================================================
    # 3. Load and preprocess test data for each phase
    # =========================================================================
    all_results = {}

    for phase in config.PHASES:
        print(f"\n{'─' * 40}")
        print(f"  Phase: {phase}")
        print(f"{'─' * 40}")

        # Load test data
        test_images, test_masks = load_dataset(
            view=view, phase=phase, structure=structure, split="test"
        )

        if len(test_images) == 0:
            print(f"  No test data found for {view}/{phase}/{structure}")
            continue

        X_test, Y_test = preprocess_dataset(test_images, test_masks)
        print(f"  Test samples: {X_test.shape[0]}")

        # Run inference
        print("  Running inference...")
        predictions = gan.generator(X_test, training=False).numpy()

        # Evaluate geometric metrics
        print(f"\n  Geometric metrics for {phase}:")
        results = evaluate_segmentation(predictions, Y_test, verbose=True)
        all_results[phase] = results

        # Plot sample predictions
        n_show = min(5, len(X_test))
        try:
            plot_sample_predictions(
                X_test[:n_show], Y_test[:n_show], predictions[:n_show],
                save_path=os.path.join(run_plot_dir, f"predictions_{phase}.png"),
                title=f"{view} - {structure} - {phase}"
            )
        except Exception as e:
            print(f"  Warning: Could not plot: {e}")

    # =========================================================================
    # 4. Clinical parameters (EF analysis for LVendo)
    # =========================================================================
    if structure == "LVendo" and "ED" in all_results and "ES" in all_results:
        print(f"\n{'─' * 40}")
        print("  Clinical Parameters (EF Analysis)")
        print(f"{'─' * 40}")

        # Load ED and ES test data
        ed_images, ed_masks = load_dataset(view, "ED", structure, "test")
        es_images, es_masks = load_dataset(view, "ES", structure, "test")
        X_ed, Y_ed = preprocess_dataset(ed_images, ed_masks)
        X_es, Y_es = preprocess_dataset(es_images, es_masks)

        pred_ed = (gan.generator(X_ed, training=False).numpy().squeeze(-1) > 0.5).astype(float)
        pred_es = (gan.generator(X_es, training=False).numpy().squeeze(-1) > 0.5).astype(float)
        gt_ed = Y_ed.squeeze(-1)
        gt_es = Y_es.squeeze(-1)

        n_patients = min(len(pred_ed), len(pred_es))

        ef_pred_list = []
        ef_gt_list = []

        for i in range(n_patients):
            # Predicted volumes and EF
            vol_ed_pred = calculate_volume_simpson(pred_ed[i])
            vol_es_pred = calculate_volume_simpson(pred_es[i])
            ef_pred = ejection_fraction(vol_ed_pred, vol_es_pred)

            # Ground truth volumes and EF
            vol_ed_gt = calculate_volume_simpson(gt_ed[i])
            vol_es_gt = calculate_volume_simpson(gt_es[i])
            ef_gt = ejection_fraction(vol_ed_gt, vol_es_gt)

            ef_pred_list.append(ef_pred)
            ef_gt_list.append(ef_gt)

        ef_pred_arr = np.array(ef_pred_list)
        ef_gt_arr = np.array(ef_gt_list)

        corr = ef_correlation(ef_pred_arr, ef_gt_arr)
        mae = ef_mae(ef_pred_arr, ef_gt_arr)

        print(f"\n  EF Correlation: {corr:.4f}")
        print(f"  EF MAE:         {mae:.4f}")

        # Plot EF correlation
        try:
            plot_ef_correlation(
                ef_gt_arr, ef_pred_arr,
                save_path=os.path.join(run_plot_dir, "ef_correlation.png"),
                title=f"{view} - EF Correlation"
            )
        except Exception as e:
            print(f"  Warning: Could not plot EF correlation: {e}")

    # =========================================================================
    # 5. Save summary to CSV
    # =========================================================================
    summary_rows = []
    for phase, results in all_results.items():
        summary_rows.append({
            "View": view,
            "Structure": structure,
            "Phase": phase,
            "Dice_Mean": results["dice_mean"],
            "Dice_Std": results["dice_std"],
            "MAD_Mean": results["mad_mean"],
            "MAD_Std": results["mad_std"],
            "HD_Mean": results["hd_mean"],
            "HD_Std": results["hd_std"],
        })

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(run_results_dir, "evaluation_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        print("\n" + df.to_string(index=False))

    print("\n" + "=" * 60)
    print("  Evaluation complete!")
    print("=" * 60)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained Pix2Pix GAN for echocardiography segmentation"
    )
    parser.add_argument(
        "--view", type=str, default="2CH",
        choices=["2CH", "4CH"],
        help="Camera view (default: 2CH)"
    )
    parser.add_argument(
        "--structure", type=str, default="LVendo",
        choices=["LVendo", "LVmyo", "LA"],
        help="Cardiac structure (default: LVendo)"
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to generator weights (default: best checkpoint)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        view=args.view,
        structure=args.structure,
        weights_path=args.weights,
    )

