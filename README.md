# Echo Research Paper Kit

A research-oriented continuation of the college major project that reproduced the 2024 IEEE TUFFC Pix2Pix-GAN echocardiography segmentation paper.

The ZIP contains **two things on purpose**:

1. `legacy/` preserves the supplied TensorFlow project and its context exactly enough to return to it later.
2. `echo_research/` is a clean PyTorch research framework designed for reproducible experiments on an NVIDIA A4000 or Kaggle GPU.

The primary paper track is intentionally focused on **LV endocardium segmentation** first. It extends Pix2Pix with:

- strict patient-level train/validation/test isolation;
- region supervision (BCE + Dice);
- boundary-aware supervision;
- paired ED/ES functional-consistency supervision;
- robustness experiments;
- patient-level clinical evaluation hooks;
- ablations and five-seed experiments;
- reproducibility manifests and paper templates.

> Important: this kit is designed to make a publishable study *possible*. It cannot guarantee acceptance or guarantee that a novelty claim is unique. The novelty gate in `paper/CLAIMS_AND_EVIDENCE.md` must be completed immediately before submission because the literature changes.

## First run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
python scripts/smoke_test.py
pytest
```

Check CAMUS before training:

```bash
python scripts/check_dataset.py \
  --config configs/proposed_binary.yaml \
  --data-root /path/to/CAMUS_public
```

Train the proposed model:

```bash
python scripts/train.py \
  --config configs/proposed_binary.yaml \
  --data-root /path/to/CAMUS_public
```

Final test evaluation only after model choices are frozen:

```bash
python scripts/evaluate.py \
  --config configs/proposed_binary.yaml \
  --data-root /path/to/CAMUS_public \
  --checkpoint runs/proposed_boundary_functional_pix2pix_lv/seed_2026/best.pt \
  --output runs/proposed_boundary_functional_pix2pix_lv/seed_2026/test_metrics.csv
```

For the complete experimental order, read **`CONTEXT.md` first**, then `paper/EXPERIMENT_PROTOCOL.md`.
