# Claims and Evidence Gate

This file exists to stop the paper from saying more than the experiments prove. A surprisingly popular research hobby.

## Claim 1: "Our method improves segmentation accuracy"

Required evidence:

- same patient split and preprocessing for baseline/proposed;
- multiple seeds;
- per-patient Dice/HD95;
- paired statistical comparison;
- no test-set tuning.

Status: **NOT YET ESTABLISHED**.

## Claim 2: "Boundary supervision improves contour accuracy"

Required evidence:

- A1 vs A2 ablation;
- HD95/ASD improvement;
- boundary-focused qualitative figure;
- failure cases.

Status: **NOT YET ESTABLISHED**.

## Claim 3: "Functional consistency improves clinical relevance"

Required evidence:

- A1 vs A3 and full-model ablations;
- validated clinical estimator;
- EDV/ESV/EF comparison;
- evidence that gains are not only segmentation-threshold artifacts.

Status: **NOT YET ESTABLISHED**.

## Claim 4: "The method is robust"

Required evidence:

- predefined corruption types and severities;
- degradation curves;
- baseline vs proposed under identical corruptions.

Status: **NOT YET ESTABLISHED**.

## Claim 5: "The method is novel"

This can only be written after a literature search immediately before submission. Search at minimum:

- IEEE Xplore;
- PubMed/Medline for clinical imaging context;
- Google Scholar / Semantic Scholar;
- arXiv for very recent preprints;
- recent MICCAI, CVPR, ICCV, ECCV, AAAI, IEEE IUS, TMI, TUFFC, JBHI literature.

Search combinations of:

- echocardiography segmentation boundary loss;
- cardiac ultrasound ED ES consistency;
- functional consistency segmentation ejection fraction;
- Pix2Pix echocardiography clinical loss;
- paired phase cardiac segmentation;
- boundary-aware echocardiography segmentation;
- multi-domain echocardiography segmentation.

Status as of kit creation: **NOVELTY NOT GUARANTEED**.

## Required wording discipline

Prefer:

> "We investigate whether..."

until experiments and novelty review are complete.

Do not write:

> "This is the first..."

without a documented search supporting it.
