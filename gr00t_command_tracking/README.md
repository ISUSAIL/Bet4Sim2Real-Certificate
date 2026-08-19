# GR00T Command Tracking — Unitree G1

This one is unfinished, we need details on suresim implementation

This folder certifies the command-following accuracy of a Unitree G1 humanoid
running the GR00T locomotion controller under joystick commands (category **C2**
in the paper). Unlike the ASTM and NIST replays, a matched simulator exists here:
the same commands are reproduced in MuJoCo. See the paper for the setup —
**Fig. 3b**.

## Reproducibility

### Measure

`err2` — the weighted sum of per-axis velocity tracking error magnitudes, one
sample per control step. Non-negative, normalized to `[0, 1.1]`. Reported widths 
are converted back to the original units.

### Run

```bash
python demo.py
python plot_demo.py
```

### Outputs

Everything is written to `data/`:

| file | contents | paper figure |
| --- | --- | --- |
| `normalization_metadata.csv` | bounds plus raw and normalized statistics, one row per sequence | — |
| `certificate_widths.csv` | per-`n` bounds and widths for every method and bank | — |
| `summary.csv` | final and mean widths, plus snapshots at n = 5, 10, 20, 30, 100, 300, 600 | — |
| `width_curves.png` | certificate width vs. samples | **Fig. 5** |
| `normalized_sequences.png` | the normalized measurement sequences | — |

### Data

| file | rows | role |
| --- | --- | --- |
| `data/vel_error_real.csv` | 690 | the real rollout, column `err2` — the only sequence certified |
| `data/Simulators/{0,1,2}/vel_error_paired_sim.csv` | 690 each | three MuJoCo variants over the same commands, column `err_weighted` |
| `data/Simulators/2/vel_error_all_err.csv` | 7228 | augmented sim rollout, used as SureSim's unlabeled pool |

All sequences share the `err2` normalization window so they stay directly
comparable.

## Method

See the paper for the methods themselves; the modules are the same ones the
synthetic folder documents. `demo.py` adds `../synthetic` to `sys.path` and
imports them, along with the simulator banks, from there.

Configuration specific to this study:

- `CONFIDENCE = 0.95`, `HORIZONS = (5, 10, 20, 30, 100, 300, 600)`.
- `GRID = arange(0.0001, 1.0, 0.002)` with `refine=True, tol=1e-5`. The shared
  `0.02` default is coarser than the certificates this dataset reaches (~0.005
  wide), which would leave the accepted region between two nodes and collapse
  the width to full range.
- **`Sim_3_mujoco`** — the three MuJoCo variants, each reduced to the `(mean,
  variance)` of its normalized sequence and passed to
  `sim2real.bounds_from_samples_mujoco()`, with `eta = 5.0`. The variants differ
  in physical parameters such as joint damping and are configured to be
  separated, which is the trust gap Lemma 2 needs.
- The synthetic banks are also run, except `Sim_7_biased`.
- **SureSim** (`method/suresim.py`) — prediction-powered intervals via
  `ppi_uniform`, with the real rollout as `Y_gold`, the paired sim rollout from
  `Simulators/2` as `Y_gold_sim`, and the augmented rollout as the unlabeled
  pool. `alpha = 0.05`, `c = 0.05`.
- Vincent et al. at gaps 0.05 / 0.10 / 0.20 / 0.30, against `BetaSkewed(2, 11)`
  (mean 0.154, matching the normalized data mean).

## Notes

- **`method/suresim.py` is loaded by path**, not imported. `synthetic/method` is
  a real package and this folder holds a same-named `method/` directory, so a
  plain `from method.suresim import ...` would bind the local directory first and
  break every other `from method import ...`.
- **SureSim is re-randomized per call.** `ppi_uniform` shuffles its stacked
  sample through the global numpy RNG, so `demo.py` seeds each step to keep runs
  reproducible and each step independent of the sweep length.
- **SureSim intervals are clipped to `[0, 1]`.** PPI's rectified sample lives on
  `[-(1 + N/n), 2 + N/n]`, about 1400 wide at `n = 1`, so without clipping the
  early steps dominate the axis.
