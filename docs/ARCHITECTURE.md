# Architecture

## Primary binary research track

```text
             ED image ─────┐
                           ├─> shared U-Net generator ─> ED LV logits ─┐
             ES image ─────┘                                           │
                                                                       ├─ region loss
                                                                       ├─ boundary loss
                                                                       ├─ functional ED/ES loss
image + real/predicted mask ─────────────> conditional PatchGAN ────────┴─ adversarial loss
```

The same generator processes ED and ES. The functional loss compares the predicted relative LV area change with the ground-truth relative area change and discourages the pathological ordering `predicted ES area > predicted ED area`.

## Why not train directly on EF?

EF depends on ventricular volumes, not just 2-D pixel area. A differentiable 2-D area surrogate is safe to optimize as long as it is named correctly. Clinical EDV/ESV/EF are evaluated separately from physical geometry using both views.

## Multiclass extension

The optional model outputs four channels: background, LV cavity, myocardium, left atrium. The discriminator receives a one-hot/probability map. This extension should be independently ablated.
