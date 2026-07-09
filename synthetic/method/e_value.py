import numpy as np


DEFAULT_GRID = np.round(np.arange(0.02, 1.0, 0.02), 2)


def truncate_stake(stake, candidate_mean, delta=0.05):
    """Clip stakes so 1 + lambda * (x - m) stays positive for x in [0, 1]."""
    candidate_mean = np.asarray(candidate_mean, dtype=float)
    lower = -(1.0 - delta) / np.maximum(1.0 - candidate_mean, 1e-12)
    upper = (1.0 - delta) / np.maximum(candidate_mean, 1e-12)
    return np.clip(stake, lower, upper)


def running_moments(samples, initial_mean=0.5, initial_variance=0.25, variance_floor=1e-4):
    """Predictable empirical mean and variance before each sample arrives."""
    samples = np.asarray(samples, dtype=float).ravel()
    means = np.empty(samples.size, dtype=float)
    variances = np.empty(samples.size, dtype=float)
    sum_x = 0.0
    sum_x2 = 0.0

    for t, x in enumerate(samples):
        if t == 0:
            means[t] = initial_mean
            variances[t] = initial_variance
        else:
            mean = sum_x / t
            means[t] = mean
            variances[t] = max(sum_x2 / t - mean**2, variance_floor)
        sum_x += x
        sum_x2 += x**2

    return means, variances


def log_wealth_grid(samples, grid=DEFAULT_GRID, means=None, variances=None, kappa=0.5):
    """Compute log e-process wealth for each candidate mean in the grid."""
    samples = np.asarray(samples, dtype=float).ravel()
    grid = np.asarray(grid, dtype=float).ravel()

    if means is None or variances is None:
        means, variances = running_moments(samples)
    means = np.asarray(means, dtype=float).ravel()
    variances = np.asarray(variances, dtype=float).ravel()

    if means.size != samples.size or variances.size != samples.size:
        raise ValueError("means and variances must have the same length as samples")

    candidates = grid[:, None]
    edge = means[None, :] - candidates
    raw_stakes = kappa * edge / (variances[None, :] + edge**2)
    stakes = truncate_stake(raw_stakes, candidates)
    factors = np.maximum(1.0 + stakes * (samples[None, :] - candidates), 1e-12)
    return np.cumsum(np.log(factors), axis=1)


def confidence_sequence_from_log_wealth(grid, log_wealth, alpha=0.05):
    """Return grid-based anytime confidence intervals from e-process wealth."""
    grid = np.asarray(grid, dtype=float).ravel()
    log_wealth = np.asarray(log_wealth, dtype=float)
    threshold = np.log(1.0 / alpha)

    if log_wealth.shape[0] != grid.size:
        raise ValueError("log_wealth must have one row per grid point")

    lower = np.full(log_wealth.shape[1], np.nan, dtype=float)
    upper = np.full(log_wealth.shape[1], np.nan, dtype=float)

    for t in range(log_wealth.shape[1]):
        accepted = np.flatnonzero(log_wealth[:, t] < threshold)
        if accepted.size == 0:
            continue
        lower[t] = grid[accepted[0]]
        upper[t] = grid[accepted[-1]]

    return lower, upper


def bounds_from_samples(samples, grid=DEFAULT_GRID, confidence=0.95, kappa=0.5):
    """Raw WSR-style confidence sequence using only data-driven moments."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    alpha = 1.0 - confidence
    samples = np.asarray(samples, dtype=float).ravel()
    means, variances = running_moments(samples)
    log_wealth = log_wealth_grid(samples, grid, means, variances, kappa)
    lower, upper = confidence_sequence_from_log_wealth(grid, log_wealth, alpha)
    return np.column_stack((lower, upper))


def certificate(real_distribution, seed, n_samples, confidence=0.95, grid=DEFAULT_GRID, kappa=0.5):
    """Return per-sample WSR e-process certificate bounds for a real distribution."""
    np.random.seed(seed)
    samples = real_distribution.sample(n_samples)
    return bounds_from_samples(samples, grid=grid, confidence=confidence, kappa=kappa)


if __name__ == "__main__":
    try:
        from .distributions import BetaSkewed
    except ImportError:
        from distributions import BetaSkewed

    distribution = BetaSkewed(alpha=0.5, beta=2.0)
    certificates = certificate(distribution, seed=0, n_samples=100)

    print("WSR e-process example")
    print(f"true mean: {distribution.true_mean():.4f}")
    print(f"certificates shape: {certificates.shape}")
    print(f"final 95% CS: [{certificates[-1, 0]:.4f}, {certificates[-1, 1]:.4f}]")
