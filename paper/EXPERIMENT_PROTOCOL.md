# Experiment Protocol

## 1. Freeze before final test

Freeze the following using training + validation only:

- input resolution;
- normalization;
- augmentation policy;
- decision threshold for binary segmentation;
- architecture;
- loss weights;
- optimizer and learning rates;
- early stopping;
- number of seeds;
- primary metric.

The test split is then used for final evaluation, not tuning.

## 2. Required model matrix

| ID | Model | Region | L1 | GAN | Boundary | Functional |
|---|---|---:|---:|---:|---:|---:|
| B0 | Original-style Pix2Pix | No | Yes | Yes | No | No |
| A1 | Pix2Pix + Dice | Yes | Yes | Yes | No | No |
| A2 | + Boundary | Yes | Yes | Yes | Yes | No |
| A3 | + Functional | Yes | Yes | Yes | No | Yes |
| P | Proposed full model | Yes | Yes | Yes | Yes | Yes |

Recommended final seeds: 2026, 2027, 2028, 2029, 2030.

## 3. Required segmentation results

For every model report by view and phase:

- Dice;
- IoU;
- HD95 in physical units where valid spacing is available;
- ASD;
- precision;
- recall.

Report mean ± SD across patients and confidence intervals for primary comparisons.

## 4. Statistical comparison

Use patient-matched metrics for paired comparisons. A Wilcoxon signed-rank test is provided in `echo_research/metrics/statistics.py`. If many hypotheses are tested, apply a multiple-comparison correction and state it.

Do not treat individual pixels as independent observations for significance testing.

## 5. Robustness

Run clean plus controlled severity levels for:

- Gaussian noise;
- speckle noise;
- blur;
- low contrast;
- shadow-like attenuation.

Primary robustness statistic: relative Dice drop and HD95 increase from clean to each severity.

## 6. Efficiency

Record:

- trainable parameters;
- GPU model;
- peak VRAM if available;
- inference milliseconds/image;
- FPS;
- training wall time.

Warm up GPU before timing and report batch size.

## 7. Clinical evaluation

Do not use resized-mask pixel counts as mL.

Before reporting EDV/ESV/EF:

1. preserve original image geometry and spacing;
2. pair 2CH and 4CH by patient and phase;
3. validate the implemented biplane volume procedure against trusted reference values;
4. calculate EDV, ESV, EF from predicted and ground-truth contours using the same validated procedure;
5. report MAE, correlation, and Bland-Altman analysis.

The bundled `clinical_biplane_eval.py` is a scaffold, not automatically a validated clinical tool.

## 8. External validation

If time permits, add EchoNet-Dynamic only after the CAMUS method is frozen. Clearly distinguish:

- within-domain training/testing;
- zero-shot cross-domain transfer;
- fine-tuned external-domain performance.

Do not mix these in one table without labels.
