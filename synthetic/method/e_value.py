import numpy as np


DEFAULT_GRID = np.round(np.arange(0.02, 1.0, 0.02), 2)
DEFAULT_KAPPA = 1.0
DEFAULT_CONSTANT_LAMBDAS = (0.25, 0.5, 0.75, 1.0)


def truncate_stake(stake, candidate_mean, delta=0.05):
    """Clip stakes so that the resulting wealth stays positive for x in [0, 1]."""
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


def log_wealth_grid(samples, grid=DEFAULT_GRID, means=None, variances=None, kappa=DEFAULT_KAPPA):
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


def constant_log_wealth_grid(samples, grid=DEFAULT_GRID, lambdas=DEFAULT_CONSTANT_LAMBDAS):
    """Log wealth from a fixed mixture over positive and negative constant stakes."""
    samples = np.asarray(samples, dtype=float).ravel()
    grid = np.asarray(grid, dtype=float).ravel()
    lambdas = np.asarray(lambdas, dtype=float).ravel()

    if np.any(lambdas <= 0):
        raise ValueError("constant lambdas must be positive")

    candidates = grid[:, None, None]
    signed_lambdas = np.concatenate((-lambdas, lambdas))
    stakes = truncate_stake(signed_lambdas[None, :, None], candidates)
    factors = np.maximum(1.0 + stakes * (samples[None, None, :] - candidates), 1e-12)
    log_wealths = np.cumsum(np.log(factors), axis=2)

    max_log = np.max(log_wealths, axis=1)
    mixture_log = max_log + np.log(np.mean(np.exp(log_wealths - max_log[:, None, :]), axis=1))
    return mixture_log


def confidence_sequence_from_log_wealth(grid, log_wealth, alpha=0.05):
    """Return nested grid-based confidence intervals from e-process wealth."""
    grid = np.asarray(grid, dtype=float).ravel()
    log_wealth = np.asarray(log_wealth, dtype=float)
    threshold = np.log(1.0 / alpha)

    if log_wealth.shape[0] != grid.size:
        raise ValueError("log_wealth must have one row per grid point")

    lower = np.zeros(log_wealth.shape[1], dtype=float)
    upper = np.ones(log_wealth.shape[1], dtype=float)

    running_max_log_wealth = np.maximum.accumulate(log_wealth, axis=1)
    for t in range(log_wealth.shape[1]):
        accepted = np.flatnonzero(running_max_log_wealth[:, t] < threshold)
        if accepted.size == 0:
            continue
        if accepted[0] > 0:
            lower[t] = grid[accepted[0] - 1]
        if accepted[-1] < grid.size - 1:
            upper[t] = grid[accepted[-1] + 1]

    return lower, upper


def refined_confidence_sequence_from_log_wealth(
    grid,
    log_wealth,
    log_wealth_fn,
    alpha=0.05,
    tol=1e-3,
    max_iter=10,
):
    """Refine grid-bracketed confidence bounds through bisection."""
    grid = np.asarray(grid, dtype=float).ravel()
    log_wealth = np.asarray(log_wealth, dtype=float)
    threshold = np.log(1.0 / alpha)
    lower, upper = confidence_sequence_from_log_wealth(grid, log_wealth, alpha)

    running_max_log_wealth = np.maximum.accumulate(log_wealth, axis=1)
    accepted_by_time = [
        np.flatnonzero(running_max_log_wealth[:, time_index] < threshold)
        for time_index in range(log_wealth.shape[1])
    ]

    lower_times = []
    lower_a = []
    lower_b = []
    upper_times = []
    upper_a = []
    upper_b = []

    for time_index, accepted in enumerate(accepted_by_time):
        if accepted.size == 0:
            continue

        first = accepted[0]
        if first > 0:
            lower_times.append(time_index)
            lower_a.append(grid[first - 1])
            lower_b.append(grid[first])

        last = accepted[-1]
        if last < grid.size - 1:
            upper_times.append(time_index)
            upper_a.append(grid[last])
            upper_b.append(grid[last + 1])

    def refine_side(times, a_values, b_values, side):
        if not times:
            return None

        times = np.asarray(times, dtype=int)
        a_values = np.asarray(a_values, dtype=float)
        b_values = np.asarray(b_values, dtype=float)

        for _ in range(max_iter):
            mids = 0.5 * (a_values + b_values)
            candidate_log_wealth = log_wealth_fn(mids)
            observed = np.arange(candidate_log_wealth.shape[1])[None, :] <= times[:, None]
            mid_log_wealth = np.max(np.where(observed, candidate_log_wealth, -np.inf), axis=1)
            rejected = mid_log_wealth >= threshold

            if side == "lower":
                a_values = np.where(rejected, mids, a_values)
                b_values = np.where(rejected, b_values, mids)
            else:
                a_values = np.where(rejected, a_values, mids)
                b_values = np.where(rejected, mids, b_values)

            if np.all(b_values - a_values <= tol):
                break

        values = a_values if side == "lower" else b_values
        return times, values

    refined_lower = refine_side(lower_times, lower_a, lower_b, "lower")
    if refined_lower is not None:
        times, values = refined_lower
        lower[times] = values

    refined_upper = refine_side(upper_times, upper_a, upper_b, "upper")
    if refined_upper is not None:
        times, values = refined_upper
        upper[times] = values

    return lower, upper


def bounds_from_samples(
    samples,
    grid=DEFAULT_GRID,
    confidence=0.95,
    method="wsr",
    kappa=DEFAULT_KAPPA,
    constant_lambdas=DEFAULT_CONSTANT_LAMBDAS,
    refine=True,
    tol=1e-3,
):
    """WSR-style confidence sequence using data-driven moments."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    alpha = 1.0 - confidence
    samples = np.asarray(samples, dtype=float).ravel()

    if method == "wsr":
        means, variances = running_moments(samples)
        log_wealth = log_wealth_grid(samples, grid, means, variances, kappa)
        log_wealth_fn = lambda candidates: log_wealth_grid(samples, candidates, means, variances, kappa)
    elif method == "constant":
        log_wealth = constant_log_wealth_grid(samples, grid, constant_lambdas)
        log_wealth_fn = lambda candidates: constant_log_wealth_grid(samples, candidates, constant_lambdas)
    else:
        raise ValueError(f"unknown e-value method: {method}")

    if refine:
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
    confidence=0.95,
    grid=DEFAULT_GRID,
    method="wsr",
    kappa=DEFAULT_KAPPA,
    constant_lambdas=DEFAULT_CONSTANT_LAMBDAS,
    refine=True,
    tol=1e-3,
):
    """Return per-sample e-process certificate bounds for a real distribution."""
    np.random.seed(seed)
    samples = real_distribution.sample(n_samples)
    return bounds_from_samples(
        samples,
        grid=grid,
        confidence=confidence,
        method=method,
        kappa=kappa,
        constant_lambdas=constant_lambdas,
        refine=refine,
        tol=tol,
    )


if __name__ == "__main__":
    try:
        from .distributions import BetaSkewed
    except ImportError:
        from distributions import BetaSkewed

    distribution = BetaSkewed(alpha=0.5, beta=2.0)
    wsr = certificate(distribution, seed=0, n_samples=100, method="wsr")
    constant = certificate(distribution, seed=0, n_samples=100, method="constant")

    print("E-process certificate example")
    print(f"true mean: {distribution.true_mean():.4f}")
    print(f"WSR final 95% CS: [{wsr[-1, 0]:.4f}, {wsr[-1, 1]:.4f}]")
    print(f"Constant final 95% CS: [{constant[-1, 0]:.4f}, {constant[-1, 1]:.4f}]")
