# Synthetic Certificates

This folder will hold synthetic experiments for comparing certificate methods.
The default experiment uses 100 seeds, matching the RSS synthetic examples.

Planned method files:

- `method/distributions.py`: synthetic data distributions and rollout cost models.
- `method/sim2real.py`: the proposed sim-to-real certificate method.
- `method/concentration.py`: standard concentration-inequality certificates, including Hoeffding and empirical Bernstein bounds.
- `method/p_value.py`: classical p-value based tests, including t-tests and z-tests.
- `method/e_value.py`: e-process based certificates, including WSR-style methods.
- `method/vincent.py`: finite-sample rollout certificates based on Vincent, Feldman, and Schwager, "Guarantees on Robot System Performance Using Stochastic Simulation Rollouts."

Entry points:

- `demo.py`: run synthetic certificate comparisons.
- `plot_demo.py`: plot synthetic certificate comparison results.

Run:

```bash
python demo.py
python plot_demo.py
```

Outputs are written to `data/`:

- `summary.csv`: final width, mean width, final coverage, and anytime coverage by distribution and method.
- `width_curves.csv`: mean certificate width at selected sample counts.
- `coverage_curves.csv`: empirical coverage at selected sample counts.
- `eta_ablation.csv`: proposed-method eta sensitivity at selected sample counts.
- `width_curves.png`: width-over-samples plot.
- `coverage_curves.png`: coverage heatmap centered at the target confidence.
- `eta_ablation.png`: eta-ablation heatmap.
