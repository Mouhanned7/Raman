Preprocessing: baseline=False, norm=area

| Protocol | Model | RMSE | MAE | R² |
|---|---|---|---|---|
| A: official split | PLS (12 LV) | 1.827 | 1.350 | 0.571 |
| A: official split | Ridge | 1.810 | 1.340 | 0.579 |
| A: official split | 1D CNN | 1.382 | 0.928 | 0.754 |
| B: leave labs out | PLS | 2.360 | 1.682 | 0.261 |
| B: leave labs out | Ridge | 2.173 | 1.590 | 0.374 |
| B: leave labs out | 1D CNN | 1.951 | 1.397 | 0.495 |
