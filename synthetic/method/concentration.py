import numpy as np


def _alpha(confidence):
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    return 1.0 - confidence


def _bounds(mean, radius):
    lower = np.maximum(mean - radius, 0.0)
    upper = np.minimum(mean + radius, 1.0)
    return np.column_stack((lower, upper))


def hoeffding_bounds_from_samples(samples, confidence=0.95):
    """Hoeffding confidence bounds for bounded samples in [0, 1]."""
    samples = np.asarray(samples, dtype=float).ravel()
    n = np.arange(1, samples.size + 1, dtype=float)
    means = np.cumsum(samples) / n
    radius = np.sqrt(np.log(2.0 / _alpha(confidence)) / (2.0 * n))
    return _bounds(means, radius)


def empirical_bernstein_bounds_from_samples(samples, confidence=0.95):
    """Empirical Bernstein bounds using the RepeatableSQ bound form."""
    samples = np.asarray(samples, dtype=float).ravel()
    bounds = np.empty((samples.size, 2), dtype=float)
    log_term = np.log(2.0 / _alpha(confidence))

    for n in range(1, samples.size + 1):
        prefix = samples[:n]
        mean = np.mean(prefix)

        if n <= 1:
            radius = np.inf
        else:
            empirical_var = np.mean((prefix - mean) ** 2)
            variance_term = np.sqrt(2.0 * empirical_var * log_term / n)
            range_term = 7.0 * log_term / (3.0 * (n - 1))
            radius = variance_term + range_term

        bounds[n - 1] = [max(mean - radius, 0.0), min(mean + radius, 1.0)]

    return bounds


def certificate(real_distribution, seed, n_samples, confidence=0.95, method="empirical_bernstein"):
    """Return per-sample concentration certificate bounds for a real distribution."""
    np.random.seed(seed)
    samples = real_distribution.sample(n_samples)

    if method in ("hoeffding", "hoeffding_ineq"):
        return hoeffding_bounds_from_samples(samples, confidence=confidence)
    if method in ("empirical_bernstein", "bernstein"):
        return empirical_bernstein_bounds_from_samples(samples, confidence=confidence)
    raise ValueError(f"unknown concentration method: {method}")


if __name__ == "__main__":
    try:
        from .distributions import BetaSkewed
    except ImportError:
        from distributions import BetaSkewed

    distribution = BetaSkewed(alpha=0.5, beta=2.0)
    hoeffding = certificate(distribution, seed=0, n_samples=100, method="hoeffding")
    bernstein = certificate(distribution, seed=0, n_samples=100, method="empirical_bernstein")

    print("Concentration certificate example")
    print(f"true mean: {distribution.true_mean():.4f}")
    print(f"Hoeffding final 95% CI: [{hoeffding[-1, 0]:.4f}, {hoeffding[-1, 1]:.4f}]")
    print(f"Empirical Bernstein final 95% CI: [{bernstein[-1, 0]:.4f}, {bernstein[-1, 1]:.4f}]")
