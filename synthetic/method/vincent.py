import numpy as np

def _failure_probability(confidence):
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    return 1.0 - confidence


def _vincent_mean_upper(samples, confidence, sim2real_gap):
    """Robust expected-value upper bound from Vincent et al., Theorem 2."""
    n = len(samples)
    delta = _failure_probability(confidence)
    if n < -0.5 * np.log(delta):
        return 1.0

    epsilon = np.sqrt(-np.log(delta) / (2.0 * n)) + sim2real_gap
    if epsilon >= 1.0:
        return 1.0

    ordered = np.sort(samples)
    k = min(max(int(np.ceil(n * epsilon)), 1), n)
    return epsilon + ((k / n) - epsilon) * ordered[k - 1] + np.sum(ordered[k:]) / n


def certificate(
    real_distribution,
    sim_distribution,
    seed,
    n_samples,
    confidence=0.95,
    sim2real_gap_upper=0.0,
    sim2real_gap_lower=None,
):
    """Return Vincent mean certificates using simulator samples.
    """
    if sim2real_gap_upper < 0.0:
        raise ValueError("sim2real_gap_upper must be nonnegative")
    if sim2real_gap_lower is not None and sim2real_gap_lower < 0.0:
        raise ValueError("sim2real_gap_lower must be nonnegative")

    _ = real_distribution
    np.random.seed(seed)
    samples = np.asarray(sim_distribution.sample(n_samples), dtype=float).ravel()
    samples = np.clip(samples, 0.0, 1.0)

    one_sided_confidence = confidence if sim2real_gap_lower is None else 1.0 - (1.0 - confidence) / 2.0
    certificates = np.zeros((n_samples, 2), dtype=float)
    for n in range(1, n_samples + 1):
        upper = _vincent_mean_upper(samples[:n], one_sided_confidence, sim2real_gap_upper)
        certificates[n - 1, 1] = min(upper, 1.0)

        if sim2real_gap_lower is not None:
            lower_upper = _vincent_mean_upper(1.0 - samples[:n], one_sided_confidence, sim2real_gap_lower)
            certificates[n - 1, 0] = max(1.0 - lower_upper, 0.0)

    return certificates


if __name__ == "__main__":
    try:
        from .distributions import BetaSkewed
    except ImportError:
        from distributions import BetaSkewed

    real = BetaSkewed(alpha=0.5, beta=2.0)
    sim = BetaSkewed(alpha=0.7, beta=2.4)
    certs = certificate(
        real,
        sim,
        seed=0,
        n_samples=100,
        sim2real_gap_upper=0.05,
        sim2real_gap_lower=0.05,
    )

