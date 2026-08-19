# Synthetic Certificates

This folder holds the synthetic experiments for comparing certificate methods
(category **C1** in the paper), along with other "toy" demos used in Section I.

The synthetic distributions are adapted and extended from previous publications [1] and [2]
and their corresponding code bases.

[1] Weng, Capito, Castillo, Khor. *Rethink repeatable measures of robot
performance with statistical query.* IEEE Transactions on Robotics, 42:561–578,
2025.

```bibtex
@article{weng2025rethink,
  title={Rethink repeatable measures of robot performance with statistical query},
  author={Weng, Bowen and Capito, Linda and Castillo, Guillermo A and Khor, Dylan},
  journal={IEEE Transactions on Robotics},
  volume={42},
  pages={561--578},
  year={2025},
  publisher={IEEE}
}
```

[2] Chen, Mahboob, Weng. *Betting for Sim-to-Real Performance Evaluation.*
Robotics: Science and Systems (RSS), 2026.
https://roboticsconference.org/program/papers/90/

```bibtex
@inproceedings{chen2026betting,
  author    = {Chen, Yujia and Mahboob, Zaid and Weng, Bowen},
  title     = {Betting for Sim-to-Real Performance Evaluation},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2026},
}
```

Code bases:

- [1] https://github.com/ISUSAIL/RepeatableSQ
- [2] https://github.com/ISUSAIL/Bet4Sim2Real

The six distribution families in `method/distributions.py` are inherited from
that line of work, and the empirical-Bernstein baseline in
`method/concentration.py` uses the bound form from [1].

## Reproducibility

### Entry points

- `demo.py`: runs every certificate method over every real set and bank; writes the CSVs.
- `plot_demo.py`: reads those CSVs, writes the PNGs — **Fig. 3a** and **Fig. 4a-c**.
- `baseline_demo.ipynb`: the Section I.A toy example — **Fig. 1**.
- `theory_demo.ipynb`: empirical wealth regret against the Theorem 1 and
  Theorem 2 bounds — **Fig. 2**.

### Run

```bash
python demo.py
python plot_demo.py
```

```bash
jupyter nbconvert --execute --inplace baseline_demo.ipynb   # Fig. 1
jupyter nbconvert --execute --inplace theory_demo.ipynb     # Fig. 2
```

### Outputs

Everything is written to `data/`:

| file | contents | paper figure |
| --- | --- | --- |
| `summary.csv` | width and coverage averaged by real set, method, and bank | — |
| `per_distribution.csv` | the same fields, one row per distribution | — |
| `width_curves.csv` | mean width at each horizon | — |
| `coverage_curves.csv` | empirical coverage at each horizon | — |
| `eta_ablation.csv` | `eta` sensitivity by bank and horizon | — |
| `width_curves.png` | width over samples | **Fig. 4a** |
| `coverage_curves.png` | coverage heatmap | **Fig. 4b** |
| `eta_ablation.png` | eta-ablation heatmap | **Fig. 4c** |
| `distribution_geometry_by_bank.png` | real vs. simulator moment pairs per bank | **Fig. 3a** |
| `example_certificate_paths.png` | written by `baseline_demo.ipynb` | **Fig. 1** |
| `theory_demo.png` | written by `theory_demo.ipynb` | **Fig. 2** |

### Determinism

Every certificate function calls `np.random.seed(seed)` before sampling, so for
a fixed `(distribution, seed)` all methods see the same real sequence — the
comparison is paired. Vincent et al. is the exception: it consumes simulator
samples, drawn from `vincent_shifted_simulator()`. The `Sim_10080` bank comes
from a fixed generator, so bank membership is fixed too.

### Dependencies

`numpy`, `scipy`, `matplotlib`, plus `numba` for `theory_demo.ipynb`.

## Method

See the paper for the method itself. Each module implements one of the
certificates compared there:

| module | paper |
| --- | --- |
| `method/sim2real.py` | Algorithm 1, the proposed sim-to-real betting certificate |
| `method/e_value.py` | the e-process certificates without simulators (WSR, constant `lambda`) |
| `method/concentration.py` | Hoeffding and empirical Bernstein |
| `method/p_value.py` | `t`-test, `z`-test, sequential `t`-test |
| `method/vincent.py` | Vincent et al., with the lower bound built by reflection |
| `method/distributions.py` | the six `[0,1]` families used for both real sets and banks |

Implementation details:

- Candidates are inverted on `DEFAULT_GRID = arange(0.02, 1.0, 0.02)`, then the
  bracketing pair around each boundary is refined by bisection (`tol = 1e-3`,
  `max_iter = 10`) — the procedure in Remark 1.
- Stake clipping uses `DELTA = 0.05`.

### Simulator banks

Defined in `sim_banks()`, with `eta` from `SIM2REAL_ETA_BY_BANK`:

| bank | size | construction | `eta` |
| --- | --- | --- | --- |
| `Sim_35` | 35 | sparse spread across all six families | 5.0 |
| `Sim_252` | 252 | one perturbation of each real distribution, padded | 2.0 |
| `Sim_756` | 756 | three perturbations of each real distribution, padded | 1.0 |
| `Sim_10080` | 10,080 | Betas over the whole `(mean, variance)` region | 25.0 |
| `Sim_7_biased` | 7 | misspecified on purpose | 5.0 |

`Ideal` uses the real distribution itself for the Ideal Kelly oracle.
