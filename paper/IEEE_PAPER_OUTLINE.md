# IEEE-Style Paper Outline

## Candidate working title

**Boundary-Aware and Functionally Consistent Adversarial Segmentation of the Left Ventricle in 2-D Echocardiography**

Do not lock the title until results exist.

## Abstract

1. Clinical problem and why LV contour reliability matters.
2. Limitation of purely pixel/region-driven segmentation.
3. Proposed boundary + ED/ES functional-consistency method.
4. Datasets and evaluation protocol.
5. Main numerical results only after final experiments.
6. Clinical/robustness result.

## I. Introduction

- 2-D echo and LV assessment.
- segmentation difficulty: speckle, weak boundaries, anatomical variability.
- existing CNN/GAN/video approaches.
- gap addressed by this study.
- research question.
- 2–3 contribution bullets, each backed by experiments.

## II. Related Work

- echocardiography segmentation;
- GAN-based segmentation;
- boundary-aware medical segmentation;
- temporal/phase consistency;
- clinical index estimation and domain generalization.

## III. Materials and Methods

### A. Datasets
CAMUS, split protocol, patient counts, views/phases, external dataset if used.

### B. Preprocessing
Normalization, resolution, augmentation, physical-spacing preservation.

### C. Baseline
Pix2Pix/U-Net/PatchGAN.

### D. Proposed Boundary Supervision
Define loss precisely.

### E. ED/ES Functional Consistency
Define soft LV area-change ratio and physiology penalty. Explicitly distinguish from EF.

### F. Total Objective
List weights and how selected using validation only.

### G. Evaluation Metrics
Dice, IoU, HD95, ASD, precision/recall, and validated clinical metrics.

### H. Statistical Analysis
Seeds, confidence intervals, paired tests.

## IV. Experiments and Results

- baseline comparison;
- ablation;
- view/phase analysis;
- robustness;
- clinical analysis;
- efficiency;
- cross-domain result if available.

## V. Discussion

- what improved and why;
- whether boundary gains translate clinically;
- cases where method fails;
- dependence on static ED/ES rather than full video;
- limitations of public datasets and scanner/domain diversity.

## VI. Conclusion

Only claims directly supported by the final tables.
