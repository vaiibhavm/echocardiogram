# CONTEXT: Read This First When Continuing the Project

**Snapshot date:** 11 August 2026  
**Purpose:** turn the existing Pix2Pix echocardiography college major project into a rigorous research study while preserving the original implementation.

---

## 1. What existed before this kit

The supplied project was a TensorFlow implementation of Fatima et al., *Automatic Segmentation of 2-D Echocardiography Ultrasound Images by Means of Generative Adversarial Network*, IEEE TUFFC 2024, DOI `10.1109/TUFFC.2024.3393026`.

It used:

- CAMUS echocardiography data;
- 2CH and 4CH views;
- ED and ES phases;
- LV endocardium, LV myocardium, and left atrium targets;
- Pix2Pix with an 8-level U-Net generator and PatchGAN discriminator;
- adversarial + L1 losses, later extended in the supplied code with Dice loss and GAN-stability heuristics.

The exact supplied context is kept under `legacy/current_project_context.md`, and the code blocks were extracted into `legacy/exact_source/`. The source paper supplied in the conversation is in `references/Fatima_2024_Pix2Pix_Echocardiography.pdf`.

### Critical issue found in the supplied training pipeline

The old training code loaded `X_test/Y_test` and used them as the validation data for checkpoint selection, learning-rate scheduling, and early stopping. That creates **test-set leakage**. The new framework never does that. Validation is patient-disjoint from test, and the final test split is not touched by the trainer.

---

## 2. What this kit changes

The new research framework is in PyTorch. This does **not** erase the TensorFlow implementation. The legacy version remains available as the historical baseline.

PyTorch was chosen for the research track because it is easy to debug, works well on the college A4000 and Kaggle, and makes custom paired-phase losses/ablation experiments straightforward.

### Primary research hypothesis

> Adding explicit cardiac-boundary supervision and paired ED/ES functional consistency to a Pix2Pix-style segmentation model can improve boundary quality and the stability of clinically relevant LV measurements without requiring a large video foundation model.

This is a **hypothesis**, not a publication claim yet.

### Proposed training objective

For the primary binary LV endocardium track:

`L_total = L_region + λ_pixel L1 + λ_adv L_adv + λ_boundary L_boundary + λ_functional L_functional`

Where:

- `L_region` = BCE + soft Dice;
- `L_boundary` = differentiable tolerant boundary F1 loss;
- `L_functional` = paired ED/ES **LV area-change consistency** plus a small penalty for predicted ES area exceeding ED area;
- `L_adv` = PatchGAN adversarial loss;
- `L1` keeps a direct pixel-space constraint.

### Important wording rule

The functional training loss is **not called an EF loss**. It uses a 2-D differentiable LV area-change surrogate. EF is a 3-D volume-derived clinical quantity and must be evaluated with a validated biplane protocol. This distinction is intentional and should remain in the paper.

---

## 3. Why the primary paper track is binary LV endocardium first

A multi-structure version is already supported in `configs/proposed_multiclass.yaml`, but it should be treated as a later experiment rather than changing three research variables at once.

The cleanest first paper story is:

1. reproduce/implement a Pix2Pix LV baseline;
2. fix the experimental protocol;
3. add region + boundary supervision;
4. add ED/ES functional consistency;
5. show which component matters through ablation;
6. test robustness and clinical consequences.

If the multiclass model is added to the same main paper, it must have its own ablation because otherwise reviewers cannot tell whether gains came from multiclass context or from the new losses.

---

## 4. Non-negotiable scientific rules

1. **Never use the test split for early stopping, hyperparameter choice, threshold choice, or model selection.**
2. Split by **patient**, never by individual image/frame.
3. Keep the exact split manifest for every run.
4. Use at least five random seeds for final comparison unless the target venue explicitly allows less and compute is prohibitive.
5. Report mean ± standard deviation and patient-level confidence intervals for primary metrics.
6. Evaluate both overlap and boundaries: Dice/IoU plus HD95/ASD.
7. Do not report a clinical `mL` result until the biplane estimator has been validated against trusted reference values or replaced with official/validated evaluation code.
8. Do not call a method "state of the art" from one run or one dataset.
9. Do not claim a method component is novel until the novelty search is refreshed near submission.
10. Save negative results. They belong in the experiment log even if they do not appear in the paper.

---

## 5. Dataset expectations

### CAMUS

Expected structure is compatible with:

```text
CAMUS_public/
├── database_nifti/
│   ├── patient0001/
│   │   ├── patient0001_2CH_ED.nii.gz
│   │   ├── patient0001_2CH_ED_gt.nii.gz
│   │   ├── patient0001_2CH_ES.nii.gz
│   │   ├── ...
└── database_split/
    ├── subgroup_training.txt
    ├── subgroup_validation.txt
    └── subgroup_testing.txt
```

If an explicit validation file is missing, the new loader derives validation **only from the training pool**. It never derives validation from test.

### EchoNet-Dynamic

Use it later for external/domain-shift testing, especially the LV segmentation/EF track. **Do not put the EchoNet dataset inside this ZIP and do not publicly redistribute it.** Its research use agreement requires individual access and prohibits redistribution.

---

## 6. Exact experiment sequence

Do not jump directly to the fanciest model. Run in this order:

### Gate 0: infrastructure

- `python scripts/smoke_test.py`
- `pytest`
- `scripts/check_dataset.py`
- verify split manifest by hand once

### Gate 1: baseline

Run `configs/baseline_binary.yaml` for seed 2026. Confirm training behaves sensibly. Then obtain a final frozen test result.

### Gate 2: ablations

Run:

1. `a0_controlled_pix2pix.yaml`
2. `a1_plus_dice.yaml`
3. `a2_plus_boundary.yaml`
4. `a3_plus_functional.yaml`
5. `proposed_binary.yaml`

Start with one seed for debugging. After the code and hyperparameters are frozen, repeat the final ablation set for seeds 2026–2030.

### Gate 3: robustness

Use `scripts/robustness_eval.py` on the same frozen test patients with controlled:

- Gaussian noise;
- speckle noise;
- blur;
- contrast reduction;
- shadow-like attenuation.

Report degradation relative to clean images, not just absolute values.

### Gate 4: clinical evaluation

Use `scripts/clinical_biplane_eval.py` only as an **experimental estimator** initially. Validate its ground-truth-mask derived EDV/ESV/EF against trusted CAMUS clinical values before putting `mL` values in the paper.

### Gate 5: external validation

Add EchoNet-Dynamic as a separate adapter/module. Do not contaminate CAMUS training with EchoNet if the experiment is advertised as zero-shot cross-domain generalization.

### Gate 6: optional multiclass extension

Run `configs/proposed_multiclass.yaml` only after the binary paper track is stable. This predicts background/LV cavity/myocardium/LA jointly.

---

## 7. What counts as evidence for the paper

A strong paper result is not merely "Dice went up." The evidence package should include:

- per-patient test predictions;
- Dice, IoU, HD95, ASD;
- ED and ES results separately;
- 2CH and 4CH separately;
- five-seed mean ± SD;
- 95% bootstrap confidence intervals;
- paired statistical comparison against the strongest baseline;
- qualitative good/medium/poor cases;
- failure cases;
- robustness curves;
- parameter count and inference time/FPS;
- clinical metric analysis after validation of the volume pipeline;
- cross-domain result if feasible.

---

## 8. Current code map

- `echo_research/data/camus.py`: patient-safe split handling, NIfTI metadata/spacing, frame and ED/ES-pair datasets.
- `echo_research/models/unet.py`: Pix2Pix-style U-Net.
- `echo_research/models/patchgan.py`: conditional PatchGAN.
- `echo_research/losses/segmentation.py`: BCE/CE + Dice.
- `echo_research/losses/boundary.py`: boundary loss.
- `echo_research/losses/functional.py`: ED/ES LV functional consistency.
- `echo_research/metrics/segmentation.py`: Dice, IoU, precision, recall, HD95, ASD.
- `echo_research/metrics/clinical.py`: experimental biplane volume estimator with explicit warning.
- `echo_research/engine/trainer.py`: AMP, separate G/D optimizers, validation checkpointing, early stopping.
- `scripts/train.py`: main training CLI.
- `scripts/evaluate.py`: final test evaluation CLI.
- `scripts/run_ablation.py`: 5-seed experiment launcher.
- `scripts/robustness_eval.py`: corruption study.
- `scripts/clinical_biplane_eval.py`: clinical experiment scaffold.
- `paper/`: manuscript and experiment planning.
- `docs/`: environment and reproducibility instructions.

---

## 9. Hardware guidance

An NVIDIA A4000 is appropriate for the proposed 256×256 model. Start with batch size 2 and AMP. If memory permits, test batch size 4, but do **not** change batch size between compared methods after the experiment protocol is frozen unless all methods are rerun consistently.

Kaggle is also suitable for development and repeat runs. Keep data licensing in mind, especially EchoNet-Dynamic.

---

## 10. Immediate next action after unpacking

```bash
pip install -e .
python scripts/smoke_test.py
pytest
python scripts/check_dataset.py --config configs/proposed_binary.yaml --data-root /path/to/CAMUS_public
```

Then run the baseline once.

---

## 11. Instructions to any future AI or collaborator

Before modifying the model:

1. Read this `CONTEXT.md`.
2. Read `paper/RESEARCH_QUESTION.md` and `paper/EXPERIMENT_PROTOCOL.md`.
3. Inspect `runs/.../resolved_config.json` and `split_manifest.json` for the experiment being discussed.
4. Do not silently change the split, metrics, primary endpoint, or test threshold.
5. If proposing a new component, explain which ablation proves its contribution.
6. If changing the clinical volume estimator, document exactly what changed and rerun all clinical comparisons.
7. Refresh `paper/LITERATURE_LANDSCAPE_2026.md` before making novelty claims.

The project is designed so the scientific story remains understandable even months later, when everyone has forgotten why a particular lambda was set to 2.0, as humans inevitably do.
