# Validation Report

**Validation date:** 11 August 2026

## Checks performed in the build environment

- Python bytecode compilation for the new package, scripts, and tests: **PASS**.
- Pytest suite: **10 tests PASS**.
- Binary U-Net/PatchGAN forward pass: **PASS**.
- Multiclass U-Net/PatchGAN forward pass: **PASS**.
- Binary segmentation + boundary + functional losses: forward/backward **PASS**.
- Multiclass segmentation + boundary + functional losses: forward/backward **PASS**.
- One complete synthetic paired ED/ES binary trainer epoch including generator update, discriminator update, validation, checkpointing, and early-stopping state: **PASS**.
- One complete synthetic paired ED/ES multiclass trainer epoch: **PASS**.
- Automatic split tests verify patient-disjoint train/validation/test behavior: **PASS**.
- Explicit test split exclusion from auto-derived validation: **PASS**.
- Editable package installation with local build isolation disabled: **PASS**.

## Important things that cannot be truthfully validated without the real dataset/GPU

The conversation supplied project source/context and the reference paper, not the CAMUS image files themselves. Therefore the following still require the user's actual CAMUS installation:

1. exact compatibility with the local CAMUS directory variant;
2. real NIfTI metadata/spacing values;
3. full A4000/Kaggle VRAM usage and training speed;
4. final Dice/HD95/ASD numbers;
5. clinical EDV/ESV/EF validity;
6. cross-domain performance on EchoNet-Dynamic.

The repository includes `scripts/check_dataset.py` specifically to validate the first three data assumptions before a long training run.

## Clinical-volume warning

`echo_research/metrics/clinical.py` contains an experimental biplane Simpson-style estimator so the full research workflow is scaffolded. It is intentionally marked **not publication-validated**. Before any paper reports mL or EF from it, compare ground-truth-mask-derived values against trusted CAMUS clinical references or replace it with a validated/official implementation.

## Legacy code

`legacy/exact_source/` is preserved from the uploaded project for traceability. It is not presented as the corrected research pipeline. The corrected pipeline is `echo_research/` + `scripts/` + `configs/`.
