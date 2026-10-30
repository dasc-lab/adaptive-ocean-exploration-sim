# Corrected Monte Carlo aggregation

Source: Slurm job `62968580`; 30 seeds for each of 36 `(Ls, Lt, W)` settings; seven strategies; 7,560 trials.

## Ergo adaptation relative to uniform target
- In-target sampling (percentage points): mean +15.50, range [+3.72, +33.62], positive in 36/36 and negative in 0/36 settings.
- Global RMSE change (%): mean +3.44, range [-0.21, +8.37], positive in 34/36 and negative in 2/36 settings.
- Ground-truth clarity-deficit change (%): mean +10.04, range [-0.82, +32.67], positive in 35/36 and negative in 1/36 settings.

## Bb Ipp adaptation relative to uniform target
- In-target sampling (percentage points): mean +14.85, range [+1.18, +23.75], positive in 36/36 and negative in 0/36 settings.
- Global RMSE change (%): mean +3.77, range [-5.31, +20.09], positive in 21/36 and negative in 15/36 settings.
- Ground-truth clarity-deficit change (%): mean +2.79, range [-18.48, +42.47], positive in 17/36 and negative in 19/36 settings.

## Temporal-scale pattern

- Lt=30 min, ergo: in-target +12.82 pp; RMSE +1.71%; GT deficit +4.02%.
- Lt=30 min, bb_ipp: in-target +12.46 pp; RMSE -0.74%; GT deficit -8.87%.
- Lt=75 min, ergo: in-target +16.27 pp; RMSE +3.77%; GT deficit +11.30%.
- Lt=75 min, bb_ipp: in-target +15.87 pp; RMSE +3.98%; GT deficit +1.94%.
- Lt=120 min, ergo: in-target +17.40 pp; RMSE +4.84%; GT deficit +14.78%.
- Lt=120 min, bb_ipp: in-target +16.23 pp; RMSE +8.07%; GT deficit +15.30%.

## Oracle-target diagnostic

- Ergo ground-truth target vs estimated adaptive: RMSE +5.76%, GT deficit -8.61%, in-target +0.29 pp.
- Bb Ipp ground-truth target vs estimated adaptive: RMSE +5.22%, GT deficit -7.81%, in-target +9.63 pp.
