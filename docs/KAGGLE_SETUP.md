# Kaggle Setup

1. Create a Kaggle notebook with GPU enabled.
2. Upload this ZIP or repository as a private dataset/repository input.
3. Add CAMUS according to its data terms. Do not publish a licensed dataset just to make a notebook convenient.
4. In a notebook cell:

```bash
!pip install -q nibabel PyYAML
!pip install -q -e /kaggle/working/echo_research_paper_kit
```

Then:

```bash
!python /kaggle/working/echo_research_paper_kit/scripts/check_dataset.py \
  --config /kaggle/working/echo_research_paper_kit/configs/proposed_binary.yaml \
  --data-root /kaggle/input/YOUR-CAMUS-DATASET
```

Write runs to `/kaggle/working/runs`, then download only checkpoints, CSVs, figures, configs, and logs. The raw medical dataset should not be copied into the experiment artifact.

A ready notebook is in `notebooks/kaggle_quickstart.ipynb`.
