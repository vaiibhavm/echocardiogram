# NVIDIA A4000 Setup

Recommended: Ubuntu/Linux workstation, recent NVIDIA driver, Python 3.10/3.11, CUDA-compatible PyTorch.

```bash
nvidia-smi
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install the CUDA build of PyTorch from the official PyTorch selector for your driver/CUDA.
pip install -e .
python - <<'PY'
import torch
print(torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available(): print(torch.cuda.get_device_name(0))
PY
```

Start with:

- image size: 256×256;
- AMP: enabled;
- batch size: 2;
- base channels: 64;
- workers: 4.

If VRAM is comfortable, test batch size 4 **before freezing the final experiment protocol**. Do not compare one model trained at batch 1 against another at batch 4 and then pretend architecture alone caused the difference.

For reproducibility, save `nvidia-smi`, PyTorch version, CUDA version, Python version, and the resolved config with every final run.
