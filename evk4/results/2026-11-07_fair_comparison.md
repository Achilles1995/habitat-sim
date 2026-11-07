# Fair events / RGB / fusion comparison, 2026-11-07 (user's RTX 4070 run)

hrc_train.py --modality all --epochs 40 (patience 6, val 15 % of train episodes, best-val checkpoint tested). Safety: 40 episodes, actions: 29 episodes; cameras spot_front, wall0, wall1.

Raw log (tables at the end of each app):

```
# safety: events vs RGB vs fusion (same samples, model, window, schedule, seed)

| metric | events | rgb | both |
|---|---|---|---|
| input channels | 30 | 45 | 75 |
| window [s] | 0.50 | 0.50 | 0.50 |
| best epoch | 3 | 6 | 3 |
| accuracy [%] | 49.2 | 43.4 | 37.9 |
| macro-F1 [%] | 35.5 | 30.3 | 27.1 |
| balanced accuracy [%] | 36.3 | 30.4 | 29.4 |
| recall safe [%] | 60.1 | 56.9 | 41.2 |
| recall caution [%] | 43.2 | 28.9 | 46.7 |
| recall danger [%] | 5.8 | 5.6 | 0.1 |
| clearance MAE [m] | 1.279 | 1.769 | 1.635 |
| ADE robot cams [m] | 3.359 | 4.189 | 3.844 |
| FDE 1.5 s robot cams [m] | 3.364 | 4.188 | 3.884 |

Majority-class baseline (safe): accuracy 61.7 %, macro-F1 25.4 %.
Mean-clearance baseline: MAE 1.539 m.
Trajectory references (robot cams): constant velocity from the TRUE positions ADE 0.207 / FDE 0.361 m; mean position ADE 3.254 / FDE 3.233 m.

Test episodes: ep0111, ep0127, ep0104, ep0124, ep0123, ep0102, ep0103, ep0134 (8304 samples per modality); 14013 train / 1740 val samples.

Per camera (accuracy / macro-F1 %):

| camera | events | rgb | both |
|---|---|---|---|
| spot_front | 52.1 / 36.2 | 57.2 / 39.5 | 44.2 / 30.8 |
| wall0 | 52.1 / 34.4 | 44.2 / 32.6 | 32.0 / 23.1 |
| wall1 | 43.6 / 35.0 | 29.0 / 17.8 | 37.4 / 27.3 |
# actions: events vs RGB vs fusion (same samples, model, window, schedule, seed)

| metric | events | rgb | both |
|---|---|---|---|
| input channels | 60 | 90 | 150 |
| window [s] | 1.00 | 1.00 | 1.00 |
| best epoch | 6 | 38 | 11 |
| accuracy [%] | 54.6 | 39.2 | 52.4 |
| macro-F1 [%] | 51.8 | 35.2 | 51.7 |
| balanced accuracy [%] | 54.1 | 36.6 | 52.6 |
| recall idle [%] | 44.2 | 48.5 | 50.0 |
| recall walk [%] | 55.9 | 40.9 | 58.3 |
| recall approach [%] | 99.7 | 60.5 | 90.9 |
| recall walk_past [%] | 50.5 | 38.3 | 40.5 |
| recall handover [%] | 20.2 | 8.7 | 38.8 |
| recall point [%] | 37.4 | 54.5 | 35.7 |
| recall wave [%] | 60.5 | 14.2 | 51.8 |
| recall pick [%] | 62.5 | 35.0 | 75.3 |
| recall place [%] | 88.3 | 62.4 | 70.4 |
| recall sit [%] | 21.0 | 12.5 | 45.7 |
| recall stand_up [%] | 56.7 | 17.5 | 56.7 |
| recall fall [%] | 52.5 | 46.3 | 16.6 |
| frame acc (smoothed) [%] | 54.1 | 40.8 | 51.1 |
| edit | 29.0 | 24.3 | 28.8 |
| F1@10 / 25 / 50 | 38.7 / 33.5 / 25.4 | 30.0 / 24.8 / 16.4 | 38.4 / 32.3 / 24.1 |
| early 10% | 41.4 | 34.1 | 41.0 |
| early 20% | 49.4 | 41.4 | 50.2 |
| early 30% | 55.0 | 43.8 | 53.8 |
| early 50% | 59.0 | 42.6 | 59.8 |
| early 75% | 55.8 | 46.6 | 58.2 |
| early 100% | 51.8 | 42.2 | 51.8 |

Majority-class baseline (walk): accuracy 26.4 %, macro-F1 3.5 %.

Test episodes: ep0020, ep0004, ep0011, ep0016, ep0010, ep0002 (14898 samples per modality); 19116 train / 2817 val samples.

Per camera (accuracy / macro-F1 %):

| camera | events | rgb | both |
|---|---|---|---|
| spot_front | 47.8 / 45.7 | 49.4 / 47.6 | 52.0 / 51.0 |
| wall0 | 58.6 / 56.7 | 36.7 / 32.0 | 56.8 / 57.2 |
| wall1 | 57.3 / 53.1 | 31.6 / 25.5 | 48.4 / 46.4 |```

## Notes
- Actions: events beat RGB by about 17 macro-F1 points with everything else identical (51.8 vs 35.2). Fusion is the same as events (51.7).
- The gap is large on the static wall cameras (wall0: 56.7 vs 32.0 F1) and gone on the moving Spot camera (45.7 vs 47.6), where ego-motion makes the whole background fire.
- RGB was still improving at epoch 40 (best epoch 38); events and fusion stopped early at epochs 6 and 11 on a noisy 3-episode validation set. Re-run with more patience before quoting numbers.
- Safety is not learned by any modality: danger recall is ≤ 6 %, accuracy is below the majority class, and ADE (3.36 m) is no better than predicting the mean position (3.25 m). The likely cause is that labels use every person, including people the camera cannot see.
