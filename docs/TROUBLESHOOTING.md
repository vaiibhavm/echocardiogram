# Troubleshooting

## `No CAMUS patient folders found`
Point `--data-root` either to `CAMUS_public` or directly to `database_nifti`. The loader accepts both.

## Missing validation split file
The loader will split only the training pool into train/validation using the configured seed and `val_fraction`. It will never use the test patient list for validation.

## CUDA out of memory
Reduce `training.batch_size` from 2 to 1. Keep AMP enabled. If comparing models, rerun all compared methods under the same batch-size protocol.

## GAN discriminator dominates
First inspect region/boundary performance. Avoid immediately adding label flipping, arbitrary discriminator throttling, or more losses. Change one variable and ablate it. The configuration supports label smoothing but defaults to conservative settings.

## Clinical volume numbers look absurd
Do not report them. Verify NIfTI spacing, original-size restoration, view pairing, patient IDs, and the biplane estimator against trusted reference values.
