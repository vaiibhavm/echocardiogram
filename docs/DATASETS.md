# Datasets

## CAMUS

Use CAMUS as the primary dataset. Its official project describes 500 patients with 2CH and 4CH echocardiographic sequences, ED/ES annotations, and clinically realistic heterogeneity including poor-quality cases. The official project also states that the views were selected to support LV EF estimation using Simpson's biplane method.

Official project: `https://www.creatis.insa-lyon.fr/Challenge/camus/`

Required citation:

S. Leclerc et al., "Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography," IEEE Transactions on Medical Imaging, 2019. DOI: `10.1109/TMI.2019.2900516`.

### Important evaluation note

The CAMUS project evaluates both segmentation accuracy and derived clinical indices. Keep physical pixel spacing and original geometry for any volume work. Do not compute mL directly from a resized 256×256 mask using `pixel_spacing=1`.

## EchoNet-Dynamic

Use EchoNet-Dynamic later for external validation. The official dataset has 10,030 A4C videos with EF, EDV/ESV, and LV tracings.

Official project: `https://echonet.github.io/dynamic/`

**License warning:** the EchoNet research agreement prohibits redistributing the dataset or download link. Every user must register individually. If using Kaggle, keep any dataset copy private and ensure that this complies with your institution's interpretation of the agreement.

## Data is intentionally absent from this ZIP

This repository contains code, metadata expectations, and citations. It does not redistribute CAMUS or EchoNet data.
