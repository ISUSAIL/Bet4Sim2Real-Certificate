import numpy as np

try:
    from scipy.stats import t
except ImportError:
    t = None


def _t_quantile(confidence, df):
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if t is None:
        raise ImportError("scipy is required for the Student t p-value baseline")
    return t.ppf(1.0 - (1.0 - confidence) / 2.0, df)


def bounds_from_samples(samples, confidence=0.95):
    """Two-sided one-sample Student t confidence intervals for the mean."""
    samples = np.asarray(samples, dtype=float).ravel()
    certificates = np.empty((samples.size, 2), dtype=float)

    for n in range(1, samples.size + 1):
        prefix = samples[:n]
        mean = np.mean(prefix)

        if n < 2:
            certificates[n - 1] = [0.0, 1.0]
            continue

        sample_std = np.std(prefix, ddof=1)
        radius = _t_quantile(confidence, n - 1) * sample_std / np.sqrt(n)
        certificates[n - 1] = [max(mean - radius, 0.0), min(mean + radius, 1.0)]

    return certificates


def certificate(real_distribution, seed, n_samples, confidence=0.95):
    """Return per-sample p-value/t-test certificate bounds for a real distribution."""
    np.random.seed(seed)
    samples = real_distribution.sample(n_samples)
    return bounds_from_samples(samples, confidence=confidence)


if __name__ == "__main__":
    try:
        from .distributions import BetaSkewed
    except ImportError:
        from distributions import BetaSkewed

    distribution = BetaSkewed(alpha=0.5, beta=2.0)
    certificates = certificate(distribution, seed=0, n_samples=100)

    print("P-value t-test certificate example")
    print(f"true mean: {distribution.true_mean():.4f}")
    print(f"certificates shape: {certificates.shape}")
    print(f"final 95% CI: [{certificates[-1, 0]:.4f}, {certificates[-1, 1]:.4f}]")
