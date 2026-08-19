# The certificate for GR00T Command Tracking

Measurements: `err_lin` — the norm of the linear velocity tracking error
(`err_vx`, `err_vy`) — and `err_yaw`, the yaw-rate tracking error. One sample
per control step of a rollout.

## Certificate comparison

Run the same certificate methods and hyperparameters used by the ASTM and NIST
demos:

```bash
cd gr00t_command_tracking
python demo.py
python plot_demo.py
```

Both measures are normalized only while computing certificates. Reported and
plotted certificate widths are converted back to their original units (m/s for
`err_lin`, rad/s for `err_yaw`). Generated CSV results and plots are saved
under `data/`.

## Dataset

`data/2_err.csv` — per-step command-tracking error:

| column | meaning |
| --- | --- |
| `t` | time (s) |
| `err_vx`, `err_vy` | linear velocity tracking error components (m/s) |
| `err_lin` | linear velocity error norm (m/s) — **certified** |
| `err_yaw` | yaw-rate tracking error (rad/s) — **certified** |

The current drop has 911 samples:

| measure | raw range | raw mean | bounds | normalized mean |
| --- | --- | --- | --- | --- |
| `err_lin` | `[0.0008, 0.4306]` | `0.1539` | `[0, 0.45]` | `0.342` |
| `err_yaw` | `[-1.5664, 2.0794]` | `-0.0511` | `[-2.1, 2.1]` | `0.488` |

## Outputs

| file | contents |
| --- | --- |
| `data/normalization_metadata.csv` | normalization bounds and raw/normalized statistics |
| `data/certificate_widths.csv` | per-`n` bounds and widths for every method (17 series × 911 samples × 2 measures) |
| `data/summary.csv` | final/mean widths plus snapshots at n = 5, 10, 20, 30, 100, 300, 900 |
| `data/width_curves.png` | certificate width vs. samples (log y) |
| `data/normalized_sequences.png` | the normalized measurement sequences |

## Methods compared

- **Proposed (sim2real)** over 5 simulator banks imported from
  `synthetic/demo.py` — `Sim_35`, `Sim_252`, `Sim_756`, `Sim_10080`,
  `Sim_7_biased` — each with its tuned `eta`
- **e-value** — WSR, and constant `lambda_t` of 0.25 and 0.5
- **Concentration** — Hoeffding, empirical Bernstein
- **p-value** — t-test, z-test, sequential t-test
- **Vincent et al.** at sim-to-real gaps 0.05 / 0.10 / 0.20 / 0.30, against
  `BetaSkewed(2, 4)` for `err_lin` (mean 0.333) and `BetaSkewed(2, 2)` for
  `err_yaw` (mean 0.5), each matching its normalized data mean

All at 95% confidence.

## Notes

- **Normalization** — `err_lin` is a magnitude, so its window starts at 0.
  `err_yaw` is signed (531 of 911 samples are negative), so it gets a symmetric
  window; the certified quantity is therefore the signed mean yaw-rate bias,
  not an error magnitude, and widths in rad/s carry the full `4.2` scale.
  Neither window clips a single sample. Widening a window shrinks every
  normalized width proportionally and makes the comparison look artificially
  tight.

- **Full-width spikes at large n** — this rollout is ~900 samples, an order of
  magnitude longer than the ASTM/NIST drops. Past n ≈ 500 six of the 34 series
  (`sim2real` on most banks, and `e_value_wsr` on `err_yaw`) spike to full
  width and stay there through the last sample. The grid-based methods reject
  every candidate mean once the confidence sequence is narrower than the `0.02`
  spacing of `method.e_value.DEFAULT_GRID`, and both `method/e_value.py` and
  `method/sim2real.py` report that all-rejected case as `[0, 1]` — the widest
  possible interval, rather than the empty set it actually is. These spikes are
  visible in `width_curves.png` and dominate the `final_width_*` columns of
  `summary.csv`, so read those series at an earlier horizon instead.

- **Small-n artifact** — the t-test and z-test curves dip to near-zero width at
  n = 2 before recovering. Those bounds are not meaningful at that sample size;
  the same artifact appears in the ASTM and NIST demos.
