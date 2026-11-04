# Baseline results, 2026-11-04 (user's RTX 4070 run, hrc_train.py, 100 epochs, small CNN, last epoch)

Datasets: safety 40 episodes (32 train / 8 test), actions 29 episodes (23 / 6); cameras spot_front, wall0, wall1.

## Safety (8304 test samples; majority class "safe" = 61.7 %)
| modality | acc | macro-F1 | clearance MAE [m] | ADE [m] | FDE 1.5 s [m] | acc spot_front / wall0 / wall1 |
|---|---|---|---|---|---|---|
| events | 0.552 | 0.361 | 1.43 | 3.28 | 3.30 | 0.581 / 0.576 / 0.499 |
| RGB (single frame) | 0.502 | 0.354 | 1.38 | 3.34 | 3.31 | 0.664 / 0.455 / 0.386 |
Danger recall: events 62/969, RGB 3/969. Train loss -> 0.003 (events) / 0.21 (RGB): strong overfitting.

## Actions (14898 test samples, 12 classes)
| modality | acc | macro-F1 | frame acc | edit | F1@10/25/50 | early rec. 10/20/30/50/75/100 % |
|---|---|---|---|---|---|---|
| events (1 s window) | 0.631 | 0.630 | 62.1 | 25.5 | 35.2 / 29.8 / 23.4 | .50 .66 .73 .69 .68 .70 |
| RGB (single frame) | 0.314 | 0.200 | 30.7 | 20.9 | 19.0 / 14.7 / 7.6 | .25 .28 .26 .22 .26 .24 |
Main confusions (events): walk <-> walk_past, point <-> wave <-> handover, sit <-> fall, pick -> place.

## Caveats for the paper
- The RGB baseline sees one frame, the event model 1 s of history: not a fair temporal comparison.
- Safety risk is below the majority-class baseline; robot-frame trajectory targets are ill-posed for wall cameras.
- No validation split / early stopping; the numbers are the last of 100 epochs.
