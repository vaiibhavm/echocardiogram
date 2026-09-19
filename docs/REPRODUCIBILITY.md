# Reproducibility Protocol

Every final experiment should archive:

- git commit hash;
- immutable YAML configuration;
- `resolved_config.json`;
- `split_manifest.json`;
- random seed;
- package versions (`pip freeze`);
- GPU/driver information (`nvidia-smi`);
- training history;
- best checkpoint;
- per-patient test CSV;
- summary CSV;
- robustness CSV if applicable.

IEEE's Author Center explicitly encourages detailed methodology and sharing code/data where permitted so independent researchers can reproduce the work. Medical data licensing can limit raw-data sharing, so share code, split IDs when allowed, derived metrics, and exact acquisition/preprocessing descriptions instead.

## Five-seed final protocol

Recommended seeds: `2026 2027 2028 2029 2030`.

```bash
python scripts/run_ablation.py --data-root /path/to/CAMUS_public --execute
```

Do not add or remove seeds after looking at which runs make the result prettier.
