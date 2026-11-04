# safety: events vs RGB vs fusion (same samples, model, window, schedule, seed)

| metric | events | rgb | both |
|---|---|---|---|
| input channels | 30 | 45 | 75 |
| window [s] | 0.50 | 0.50 | 0.50 |
| best epoch | - | - | - |
| accuracy [%] | 0.0 | 44.8 | 0.0 |
| macro-F1 [%] | 0.0 | 32.1 | 0.0 |
| balanced accuracy [%] | 0.0 | 25.0 | 0.0 |
| recall safe [%] | - | - | - |
| recall caution [%] | 0.0 | 50.0 | 0.0 |
| recall danger [%] | 0.0 | 0.0 | 0.0 |
| clearance MAE [m] | 0.354 | 0.355 | 0.353 |
| ADE robot cams [m] | 2.539 | 2.539 | 2.524 |
| FDE 1.5 s robot cams [m] | 2.601 | 2.596 | 2.520 |

Majority-class baseline (safe): accuracy 0.0 %, macro-F1 0.0 %.
Mean-clearance baseline: MAE 0.353 m.
Trajectory references (robot cams): constant velocity from the TRUE positions ADE 0.249 / FDE 0.482 m; mean position ADE 2.534 / FDE 2.572 m.

Test episodes: ep0000 (212 samples per modality); 106 train / 0 val samples.

Per camera (accuracy / macro-F1 %):

| camera | events | rgb | both |
|---|---|---|---|
| spot_front | 0.0 / 0.0 | 89.6 / 47.3 | 0.0 / 0.0 |
| wall0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
