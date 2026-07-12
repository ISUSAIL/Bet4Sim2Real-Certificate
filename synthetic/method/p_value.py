import numpy as np

try:
    from scipy.stats import norm
    from scipy.stats import t
except ImportError:
    norm = None
    t = None


def _check_confidence(confidence):
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")


def _t_quantile(confidence, df):
    _check_confidence(confidence)
    if t is None:
        raise ImportError("scipy is required for the Student t p-value baseline")
    return t.ppf(1.0 - (1.0 - confidence) / 2.0, df)


def _z_quantile(confidence):
    _check_confidence(confidence)
    if norm is None:
        raise ImportError("scipy is required for the normal p-value baseline")
    return norm.ppf(1.0 - (1.0 - confidence) / 2.0)


def _radius(prefix, confidence, method):
    n = prefix.size
    if n < 2:
        return None

    sample_std = np.std(prefix, ddof=1)
    if method == "student_t":
        critical = _t_quantile(confidence, n - 1)
    elif method == "normal":
        critical = _z_quantile(confidence)
    elif method == "sequential_t":
        alpha = 1.0 - confidence
        confidence_n = 1.0 - alpha / (n * (n + 1.0))
        critical = _t_quantile(confidence_n, n - 1)
    else:
        raise ValueError(f"unknown p-value method: {method}")

    return critical * sample_std / np.sqrt(n)


def bounds_from_samples(samples, confidence=0.95, method="student_t"):
    """Two-sided real-sample p-value intervals for the mean.

    student_t is the usual one-sample t interval with sample standard deviation.
    normal is a plug-in Wald/z interval with sample standard deviation, not an
    oracle known-variance interval. sequential_t uses a simple alpha-spending
    schedule alpha_t = alpha / (t * (t + 1)).
    """
    samples = np.asarray(samples, dtype=float).ravel()
    certificates = np.empty((samples.size, 2), dtype=float)

    for n in range(1, samples.size + 1):
        prefix = samples[:n]
        mean = np.mean(prefix)

        if n < 2:
            certificates[n - 1] = [0.0, 1.0]
            continue

        radius = _radius(prefix, confidence, method)
        certificates[n - 1] = [max(mean - radius, 0.0), min(mean + radius, 1.0)]

    return certificates


def certificate(real_distribution, seed, n_samples, confidence=0.95, method="student_t"):
    """Return per-sample p-value certificate bounds for a real distribution."""
    np.random.seed(seed)
    samples = real_distribution.sample(n_samples)
    return bounds_from_samples(samples, confidence=confidence, method=method)


if __name__ == "__main__":
    try:
        from .distributions import BetaSkewed
    except ImportError:
        from distributions import BetaSkewed

    distribution = BetaSkewed(alpha=0.5, beta=2.0)
    certificates = certificate(distribution, seed=0, n_samples=100, method="student_t")
    sequential = certificate(distribution, seed=0, n_samples=100, method="sequential_t")

    print("P-value t-test certificate example")
    print(f"true mean: {distribution.true_mean():.4f}")
    print(f"certificates shape: {certificates.shape}")
    print(f"final 95% CI: [{certificates[-1, 0]:.4f}, {certificates[-1, 1]:.4f}]")
    print(f"sequential t final 95% CI: [{sequential[-1, 0]:.4f}, {sequential[-1, 1]:.4f}]")
