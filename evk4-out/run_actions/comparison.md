# actions: events vs RGB vs fusion (same samples, model, window, schedule, seed)

| metric | events | rgb | both |
|---|---|---|---|
| input channels | 60 | 90 | 150 |
| window [s] | 1.00 | 1.00 | 1.00 |
| best epoch | - | - | - |
| accuracy [%] | 13.1 | 15.5 | 16.2 |
| macro-F1 [%] | 3.5 | 3.0 | 5.8 |
| balanced accuracy [%] | 9.4 | 11.1 | 11.0 |
| recall idle [%] | 3.0 | 0.0 | 48.7 |
| recall walk [%] | 0.0 | 0.0 | 0.0 |
| recall approach [%] | - | - | - |
| recall walk_past [%] | - | - | - |
| recall handover [%] | 0.0 | 0.0 | 0.0 |
| recall point [%] | - | - | - |
| recall wave [%] | 0.0 | 0.0 | 0.0 |
| recall pick [%] | 0.0 | 0.0 | 0.0 |
| recall place [%] | 0.0 | 0.0 | 0.0 |
| recall sit [%] | 0.0 | 0.0 | 0.0 |
| recall stand_up [%] | 0.0 | 0.0 | 0.0 |
| recall fall [%] | 81.4 | 100.0 | 50.0 |
| frame acc (smoothed) [%] | 13.2 | 15.5 | 16.3 |
| edit | 17.3 | 6.7 | 20.0 |
| F1@10 / 25 / 50 | 7.4 / 7.4 / 7.4 | 12.5 / 0.0 / 0.0 | 15.8 / 7.5 / 0.0 |
| early 10% | 6.7 | 6.7 | 23.3 |
| early 20% | 0.0 | 6.7 | 23.3 |
| early 30% | 0.0 | 6.7 | 23.3 |
| early 50% | 10.0 | 6.7 | 23.3 |
| early 75% | 10.0 | 6.7 | 23.3 |
| early 100% | 10.0 | 6.7 | 20.0 |

Majority-class baseline (walk_past): accuracy 0.0 %, macro-F1 0.0 %.

Test episodes: ep0000 (1358 samples per modality); 682 train / 0 val samples.

Per camera (accuracy / macro-F1 %):

| camera | events | rgb | both |
|---|---|---|---|
| spot_head | 14.6 / 4.3 | 15.5 / 3.0 | 15.5 / 3.3 |
| wall0 | 11.6 / 2.6 | 15.5 / 3.0 | 16.9 / 3.3 |
