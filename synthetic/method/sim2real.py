import numpy as np

try:
    from .e_value import DEFAULT_GRID
    from .e_value import DEFAULT_KAPPA
except ImportError:
    from e_value import DEFAULT_GRID
    from e_value import DEFAULT_KAPPA


_SIMULATOR_MOMENT_CACHE = {}
DELTA = 0.05


def truncate(stake, candidate_mean):
    """Algorithm line 5: keep every factor positive for outcomes in [0, 1]."""
    return np.clip(
        stake,
        -(1.0 - DELTA) / np.maximum(1.0 - candidate_mean, 1e-6),
        (1.0 - DELTA) / np.maximum(candidate_mean, 1e-6),
    )


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


def simulator_moments_from_pairs(simulators):
    """Extract mean and variance when simulators are already (mean, variance) pairs.
    """
    means = np.array([mean for mean, _ in simulators], dtype=float)
    variances = np.array([variance for _, variance in simulators], dtype=float)
    return means, np.maximum(variances, 1e-8)


def simulator_bank_mixture_moments(samples, simulators, eta=5.0, moments_fn=simulator_moments):
    """Predictable moments from an online mixture over simulator distributions.
    """
    samples = np.asarray(samples, dtype=float).ravel()
    sim_means, sim_variances = moments_fn(simulators)
    log_weights = np.zeros(len(simulators), dtype=float)
    mixture_means = np.empty(samples.size, dtype=float)
    mixture_variances = np.empty(samples.size, dtype=float)

    for t, sample in enumerate(samples):
        stable = log_weights - np.max(log_weights)
        weights = np.exp(stable)
        weights /= np.sum(weights)

        mixture_means[t] = float(weights @ sim_means)
        mixture_variances[t] = float(np.maximum(weights @ sim_variances, 1e-8))

        scores = gaussian_log_likelihood(sample, sim_means, sim_variances)
        log_weights += eta * scores

    return mixture_means, mixture_variances


def grid_wealth(samples, grid, means, variances, kappa):
    """Algorithm lines 7: log wealth for every grid candidate."""
    samples = np.asarray(samples, dtype=float).ravel()
    grid = np.asarray(grid, dtype=float).ravel()
    means = np.asarray(means, dtype=float).ravel()
    variances = np.asarray(variances, dtype=float).ravel()

    candidates = grid[:, None]
    edge = means[None, :] - candidates
    stakes = truncate(kappa * edge / (variances[None, :] + edge**2), candidates)
    factors = np.maximum(1.0 + stakes * (samples[None, :] - candidates), 1e-12)
    return np.cumsum(np.log(factors), axis=1)


def _wealth_diag(candidates, samples, means, variances, kappa):
    """Maximum log wealth of candidate c[n] through time n, vectorized."""
    samples = np.asarray(samples, dtype=float).ravel()
    candidates = np.asarray(candidates, dtype=float).ravel()
    times = samples.size
    grid = candidates[:, None]
    edge = means[None, :] - grid
    stakes = truncate(kappa * edge / (variances[None, :] + edge**2), grid)
    log_factors = np.log(np.maximum(1.0 + stakes * (samples[None, :] - grid), 1e-12))
    cumulative_log_wealth = np.cumsum(log_factors, axis=1)
    observed = np.tril(np.ones((times, times), dtype=bool))
    return np.max(np.where(observed, cumulative_log_wealth, -np.inf), axis=1)


def confidence_sequence(grid, log_wealth, samples, means, variances, kappa, alpha=0.05, tol=1e-3, max_iter=10):
    """Algorithm line 9: bisection-refined confidence interval endpoints."""
    threshold = np.log(1.0 / alpha)
    grid = np.asarray(grid, dtype=float).ravel()
    log_wealth = np.asarray(log_wealth, dtype=float)
    grid_size, times = log_wealth.shape

    lower = np.zeros(times, dtype=float)
    upper = np.ones(times, dtype=float)

    lower_a = np.zeros(times, dtype=float)
    lower_b = np.zeros(times, dtype=float)
    refine_lower = np.zeros(times, dtype=bool)
    upper_a = np.zeros(times, dtype=float)
    upper_b = np.zeros(times, dtype=float)
    refine_upper = np.zeros(times, dtype=bool)

    running_max_log_wealth = np.maximum.accumulate(log_wealth, axis=1)
    for time_index in range(times):
        column = running_max_log_wealth[:, time_index]
        accepted = np.flatnonzero(column < threshold)
        if accepted.size == 0:
            continue

        first = accepted[0]
        last = accepted[-1]
        if first == 0:
            lower[time_index] = 0.0
        else:
            lower_a[time_index] = grid[first - 1]
            lower_b[time_index] = grid[first]
            refine_lower[time_index] = True

        if last == grid_size - 1:
            upper[time_index] = 1.0
        else:
            upper_a[time_index] = grid[last]
            upper_b[time_index] = grid[last + 1]
            refine_upper[time_index] = True

    refine = refine_lower | refine_upper
    if refine.any():
        if max_iter <= 0:
            lower = np.where(refine_lower, lower_a, lower)
            upper = np.where(refine_upper, upper_b, upper)
            return lower, upper

        for _ in range(max_iter):
            if refine_lower.any():
                candidates = np.where(refine_lower, 0.5 * (lower_a + lower_b), grid[0])
                rejected = refine_lower & (_wealth_diag(candidates, samples, means, variances, kappa) >= threshold)
                lower_a = np.where(rejected, candidates, lower_a)
                lower_b = np.where(refine_lower & ~rejected, candidates, lower_b)

            if refine_upper.any():
                candidates = np.where(refine_upper, 0.5 * (upper_a + upper_b), grid[0])
                rejected = refine_upper & (_wealth_diag(candidates, samples, means, variances, kappa) >= threshold)
                upper_b = np.where(rejected, candidates, upper_b)
                upper_a = np.where(refine_upper & ~rejected, candidates, upper_a)

            lower_width = np.where(refine_lower, lower_b - lower_a, 0.0)
            upper_width = np.where(refine_upper, upper_b - upper_a, 0.0)
            if max(np.max(lower_width), np.max(upper_width)) <= tol:
                break

        lower = np.where(refine_lower, lower_a, lower)
        upper = np.where(refine_upper, upper_b, upper)

    return lower, upper


def bounds_from_samples(
    samples,
    simulators,
    grid=DEFAULT_GRID,
    confidence=0.95,
    eta=5.0,
    kappa=DEFAULT_KAPPA,
    refine=True,
    tol=1e-3,
):
    """Sim-to-real betting confidence sequence for a bounded mean in [0, 1]."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    alpha = 1.0 - confidence
    samples = np.asarray(samples, dtype=float).ravel()
    means, variances = simulator_bank_mixture_moments(
        samples=samples,
        simulators=simulators,
        eta=eta,
    )
    log_wealth = grid_wealth(
        samples=samples,
        grid=grid,
        means=means,
        variances=variances,
        kappa=kappa,
    )
    if refine:
        lower, upper = confidence_sequence(
            grid,
            log_wealth,
            samples=samples,
            means=means,
            variances=variances,
            kappa=kappa,
            alpha=alpha,
            tol=tol,
        )
    else:
        lower, upper = confidence_sequence(
            grid,
            log_wealth,
            samples=samples,
            means=means,
            variances=variances,
            kappa=kappa,
            alpha=alpha,
            tol=0.0,
            max_iter=0,
        )
    return np.column_stack((lower, upper))

def bounds_from_samples_mujoco(
    samples,
    simulators,
    grid=DEFAULT_GRID,
    confidence=0.95,
    eta=5.0,
    kappa=DEFAULT_KAPPA,
    refine=True,
    tol=1e-3,
):
    """Sim-to-real certificate for a bank given as (mean, variance) computed from mujoco rollout.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    alpha = 1.0 - confidence
    samples = np.asarray(samples, dtype=float).ravel()
    means, variances = simulator_bank_mixture_moments(
        samples=samples,
        simulators=simulators,
        eta=eta,
        moments_fn=simulator_moments_from_pairs,
    )
    log_wealth = grid_wealth(
        samples=samples,
        grid=grid,
        means=means,
        variances=variances,
        kappa=kappa,
    )
    if refine:
        lower, upper = confidence_sequence(
            grid,
            log_wealth,
            samples=samples,
            means=means,
            variances=variances,
            kappa=kappa,
            alpha=alpha,
            tol=tol,
        )
    else:
        lower, upper = confidence_sequence(
            grid,
            log_wealth,
            samples=samples,
            means=means,
            variances=variances,
            kappa=kappa,
            alpha=alpha,
            tol=0.0,
            max_iter=0,
        )
    return np.column_stack((lower, upper))

def certificate(
    real_distribution,
    seed,
    n_samples,
    simulators,
    grid=DEFAULT_GRID,
    confidence=0.95,
    eta=5.0,
    kappa=DEFAULT_KAPPA,
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
