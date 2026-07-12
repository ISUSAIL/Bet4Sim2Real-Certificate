import numpy as np

try:
    from .e_value import DEFAULT_GRID
    from .e_value import confidence_sequence_from_log_wealth
    from .e_value import log_wealth_grid
    from .e_value import refined_confidence_sequence_from_log_wealth
except ImportError:
    from e_value import DEFAULT_GRID
    from e_value import confidence_sequence_from_log_wealth
    from e_value import log_wealth_grid
    from e_value import refined_confidence_sequence_from_log_wealth


_SIMULATOR_MOMENT_CACHE = {}


def gaussian_log_likelihood(samples, means, variances, variance_floor=1e-8):
    """Gaussian score used to update simulator-bank trust weights online."""
    samples = np.asarray(samples, dtype=float)
    means = np.asarray(means, dtype=float)
    variances = np.maximum(np.asarray(variances, dtype=float), variance_floor)
    return -0.5 * (np.log(2.0 * np.pi * variances) + (samples - means) ** 2 / variances)


def simulator_moments(simulators):
    """Extract mean and variance from simulator distributions."""
    cache_key = tuple(id(sim) for sim in simulators)
    if cache_key in _SIMULATOR_MOMENT_CACHE:
        return _SIMULATOR_MOMENT_CACHE[cache_key]

    means = np.array([sim.true_mean() for sim in simulators], dtype=float)
    variances = np.array([sim.true_variance() for sim in simulators], dtype=float)
    result = (means, np.maximum(variances, 1e-8))
    _SIMULATOR_MOMENT_CACHE[cache_key] = result
    return result


def simulator_bank_mixture_moments(samples, simulators, eta=5.0):
    """Predictable moments from an online mixture over simulator distributions.

    At each real sample, simulator weights are computed before observing that
    sample. Afterward, each simulator is scored by a Gaussian likelihood using
    its known synthetic mean and variance. This is the proposed sim-to-real
    replacement for the raw WSR data-driven moments.
    """
    samples = np.asarray(samples, dtype=float).ravel()
    sim_means, sim_variances = simulator_moments(simulators)
    log_weights = np.zeros(len(simulators), dtype=float)
    mixture_means = np.empty(samples.size, dtype=float)
    mixture_variances = np.empty(samples.size, dtype=float)
    weights_history = np.empty((samples.size, len(simulators)), dtype=float)

    for t, sample in enumerate(samples):
        stable = log_weights - np.max(log_weights)
        weights = np.exp(stable)
        weights /= np.sum(weights)

        mixture_means[t] = float(weights @ sim_means)
        mixture_variances[t] = float(np.maximum(weights @ sim_variances, 1e-8))
        weights_history[t] = weights

        scores = gaussian_log_likelihood(sample, sim_means, sim_variances)
        log_weights += eta * scores

    return mixture_means, mixture_variances, weights_history


def bounds_from_samples(
    samples,
    simulators,
    grid=DEFAULT_GRID,
    confidence=0.95,
    eta=5.0,
    kappa=1.0,
    refine=True,
    tol=1e-3,
):
    """Sim-to-real betting confidence sequence for a bounded mean in [0, 1]."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    alpha = 1.0 - confidence
    samples = np.asarray(samples, dtype=float).ravel()
    means, variances, weights = simulator_bank_mixture_moments(
        samples=samples,
        simulators=simulators,
        eta=eta,
    )
    log_wealth = log_wealth_grid(
        samples=samples,
        grid=grid,
        means=means,
        variances=variances,
        kappa=kappa,
    )
    if refine:
        log_wealth_fn = lambda candidates: log_wealth_grid(
            samples=samples,
            grid=candidates,
            means=means,
            variances=variances,
            kappa=kappa,
        )
        lower, upper = refined_confidence_sequence_from_log_wealth(
            grid,
            log_wealth,
            log_wealth_fn,
            alpha=alpha,
            tol=tol,
        )
    else:
        lower, upper = confidence_sequence_from_log_wealth(grid, log_wealth, alpha)
    return np.column_stack((lower, upper))


def certificate(
    real_distribution,
    seed,
    n_samples,
    simulators,
    grid=DEFAULT_GRID,
    confidence=0.95,
    eta=5.0,
    kappa=1.0,
    refine=True,
    tol=1e-3,
):
    """Return per-sample sim-to-real certificate bounds for a real distribution."""
    np.random.seed(seed)
    samples = real_distribution.sample(n_samples)
    return bounds_from_samples(
        samples=samples,
        simulators=simulators,
        grid=grid,
        confidence=confidence,
        eta=eta,
        kappa=kappa,
        refine=refine,
        tol=tol,
    )


if __name__ == "__main__":
    try:
        from .distributions import BetaSkewed
    except ImportError:
        from distributions import BetaSkewed

    real_distribution = BetaSkewed(alpha=0.5, beta=2.0)
    simulator_bank = [
        BetaSkewed(alpha=0.7, beta=2.4),
        BetaSkewed(alpha=0.4, beta=1.6),
        BetaSkewed(alpha=2.0, beta=0.5),
        BetaSkewed(alpha=2.0, beta=2.0),
        BetaSkewed(alpha=1.0, beta=5.0),
    ]
    certificates = certificate(
        real_distribution=real_distribution,
        seed=0,
        n_samples=100,
        simulators=simulator_bank,
    )

    print("Sim-to-real certificate example")
    print(f"true mean: {real_distribution.true_mean():.4f}")
    print(f"certificates shape: {certificates.shape}")
    print(f"final 95% CS: [{certificates[-1, 0]:.4f}, {certificates[-1, 1]:.4f}]")
