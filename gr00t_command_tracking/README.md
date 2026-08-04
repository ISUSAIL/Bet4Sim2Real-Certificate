# The certificate for GR00T Command Tracking

Measurement: `err_norm` — the norm of the velocity-command tracking error
(`err_vx`, `err_vy`, `err_yaw`), one sample per ~1 s step of a rollout.

## Certificate comparison

Run the same certificate methods and hyperparameters used by the ASTM and NIST
demos:

```bash
cd gr00t_command_tracking
python demo.py
python plot_demo.py
```

`demo.py` picks up the newest `data/vel_error_*_err.csv`, so dropping in a new
measurement file and re-running is enough.

`err_norm` is normalized only while computing certificates, using fixed bounds
of `[0, 1.2]`. Reported and plotted certificate widths are converted back to
error-norm units. Generated CSV results and plots are saved under `data/`.

## Dataset

`data/vel_error_<timestamp>_err.csv` — per-step command-tracking error:

| column | meaning |
| --- | --- |
| `t` | time (s) |
| `err_vx`, `err_vy` | linear velocity tracking error (m/s) |
| `err_yaw` | yaw-rate tracking error (rad/s) |
| `err_norm` | error norm — **the only column used by the certificate** |

The current drop (`vel_error_20260804_181805_err.csv`) has 38 samples,
`err_norm` in `[0.0096, 1.0407]`, mean `0.2032`.

## Outputs

| file | contents |
| --- | --- |
| `data/normalization_metadata.csv` | normalization bounds and raw/normalized statistics |
| `data/certificate_widths.csv` | per-`n` bounds and widths for every method (17 series × 38 samples) |
| `data/summary.csv` | final/mean widths plus snapshots at n = 5, 10, 20, 30, 38 |
| `data/width_curves.png` | certificate width vs. samples (log y) |
| `data/normalized_sequences.png` | the normalized measurement sequence |

## Methods compared

- **Proposed (sim2real)** over 5 simulator banks imported from
  `synthetic/demo.py` — `Sim_35`, `Sim_252`, `Sim_756`, `Sim_10080`,
  `Sim_7_biased` — each with its tuned `eta`
- **e-value** — WSR, and constant `lambda_t` of 0.25 and 0.5
- **Concentration** — Hoeffding, empirical Bernstein
- **p-value** — t-test, z-test, sequential t-test
- **Vincent et al.** at sim-to-real gaps 0.05 / 0.10 / 0.20 / 0.30, against a
  `BetaSkewed(2, 10)` simulator (mean 0.167, matching the normalized data)

All at 95% confidence.

## Notes

- **Normalization bound** — `1.2` sits just above the observed max (`1.0407`),
  so no sample is clipped. Widening it would shrink every normalized width
  proportionally and make the comparison look artificially tight.
- **The n = 28 outlier** — sample 28 is `err_norm = 1.0407`, ~5x the mean. It
  is visible as the spike in `normalized_sequences.png` and as the width jump
  at n = 28 in every anytime-valid method's curve.
- **Small-n artifact** — the t-test and z-test curves dip to near-zero width at
  n = 2 before recovering. Those bounds are not meaningful at that sample size;
  the same artifact appears in the ASTM and NIST demos.
