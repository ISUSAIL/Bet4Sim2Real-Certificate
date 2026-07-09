# Synthetic Certificates

This folder will hold synthetic experiments for comparing certificate methods.

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
