# Literature Landscape Snapshot (August 2026)

This is a planning snapshot, not a substitute for the final literature review.

## Foundation of this project

### Fatima et al., IEEE TUFFC 2024

The supplied source paper applies Pix2Pix GAN to 2-D echocardiography using U-Net as generator and PatchGAN as discriminator, evaluating CAMUS segmentation and clinical/geometric indices. The paper also evaluates generalization to EchoNet and identifies video-level real-time segmentation as future work.

DOI: `10.1109/TUFFC.2024.3393026`

## Dataset benchmark

### Leclerc et al., IEEE TMI 2019 / CAMUS

CAMUS contains 500 patients with 2CH and 4CH acquisitions and ED/ES manual annotations. The official CAMUS project explicitly targets segmentation and clinical volume/EF estimation. It also contains difficult and poor-quality clinical cases.

Official: `https://www.creatis.insa-lyon.fr/Challenge/camus/`

DOI: `10.1109/TMI.2019.2900516`

## External validation dataset

### EchoNet-Dynamic

EchoNet-Dynamic provides 10,030 A4C videos, EF/EDV/ESV labels, and expert LV tracings. This is valuable for external validation and video follow-up work.

Official: `https://echonet.github.io/dynamic/`

## Recent video segmentation direction

### GDKVM, ICCV 2025

GDKVM addresses echocardiography video segmentation with spatiotemporal memory, explicitly discussing blur/noise robustness and reporting results on CAMUS and EchoNet-Dynamic. This means a 2026 paper should not present "we added video frames" as sufficient novelty by itself.

CVF open access: `https://openaccess.thecvf.com/content/ICCV2025/html/Wang_GDKVM_Echocardiography_Video_Segmentation_via_Spatiotemporal_Key-Value_Memory_with_Gated_ICCV_2025_paper.html`

## Emerging 2026 themes to re-check before submission

Recent preprints are actively studying:

- cross-domain/domain-shift behavior in echocardiography segmentation;
- partially labelled multi-domain training;
- ultrasound-specific denoising and boundary/semantic stabilization.

Because these are moving quickly, the final novelty section must be refreshed. The proposed project therefore emphasizes **rigorous, low-compute boundary + functional consistency and clinically grounded evaluation**, not a claim that static Pix2Pix itself is new.

## Practical implication

A credible paper should contribute more than architecture cosmetics. The strongest version of this project will combine:

1. a clearly motivated loss/constraint;
2. rigorous patient-safe evaluation;
3. ablation;
4. robustness;
5. clinical consequences;
6. external validation if feasible;
7. reproducible code and experiment manifests.
