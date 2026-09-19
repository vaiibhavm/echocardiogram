# Echo LV Render API

Memory-conscious FastAPI deployment for left-ventricle endocardium segmentation in 2-D echocardiography.

## Endpoints

- `GET /health` returns runtime and model readiness.
- `POST /predict` accepts one `file` field containing a PNG or JPG and returns a PNG overlay.
- `GET /docs` opens the generated OpenAPI interface.

The API uses a frozen FP16 ONNX export of the trained Pix2Pix-style U-Net with FP32 input/output. Preprocessing and post-processing mirror the research policy: 256 x 256 bilinear resize, 1st/99th percentile normalization, threshold `0.625`, largest 8-connected component, and hole filling.

## Free deployment

The Render blueprint runs a single CPU worker on the free plan. During the build, `download_model.py` downloads the public model bundle from the project's Google Drive folder and verifies SHA-256 checksum `dd4494a66f8cb8e01d930a57d8c66627eede46b1d947d1ebdb5811be79e8f5b1` before extraction. The 100 MB model archive is intentionally kept out of Git so the repository stays small.

Create a Render Blueprint from this repository and set its root directory to `render-api`. The public API then exposes `/health`, `/predict`, and `/docs`.

Research use only. This prototype is not a medical device and does not provide a diagnosis.
