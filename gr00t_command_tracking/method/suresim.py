"""
Suresim methods for computing confidence intervals.
src/suresim/problem_data.py
src/suresim/intervals/ppi.py
"""
import numpy as np


def MARTINGALE(m, c, lam_t, Z_norm):
    coeff_plus = np.minimum(lam_t, c / m)
    coeff_minus = np.minimum(lam_t, c / (1 - m))
    M1 = np.cumsum(np.log(1 + coeff_plus * (Z_norm - m)))
    M2 = np.cumsum(np.log(1 - coeff_minus * (Z_norm - m)))
    # return 0.5 * np.maximum(M1, M2)
    return np.maximum(M1, M2) - np.log(2)

def wsr_iid(
    x_n, alpha, intersection: bool = True, theta: float = 0.5, c: float = 0.75, L=0, U=1
):
    x_n = (x_n - L) / (U - L)

    grid_spacing = 1e-4  # Grid spacing for CI computation based on WSR algorithm
    grid = np.arange(grid_spacing, 1, step=grid_spacing)
    n = x_n.shape[0]
    t_n = np.arange(1, n + 1)
    muhat_n = (0.5 + np.cumsum(x_n)) / (1 + t_n)
    sigma2hat_n = (0.25 + np.cumsum(np.power(x_n - muhat_n, 2))) / (
        1 + t_n
    )  # Line with error due to cumsum.
    sigma2hat_tminus1_n = np.append(0.25, sigma2hat_n[:-1])
    assert np.all(sigma2hat_tminus1_n > 0)
    lambda_n = np.sqrt(2 * np.log(2 / alpha) / (n * sigma2hat_tminus1_n))

    def M(m):
        lambdaplus_n = np.minimum(lambda_n, c / m)
        lambdaminus_n = np.minimum(lambda_n, c / (1 - m))
        return np.maximum(
            theta * np.exp(np.cumsum(np.log(1 + lambdaplus_n * (x_n - m)))),
            (1 - theta) * np.exp(np.cumsum(np.log(1 - lambdaminus_n * (x_n - m)))),
        )

    indicators_gxn = np.zeros([grid.size, n])
    found_lb = False
    for m_idx, m in enumerate(grid):
        m_n = M(m)
        indicators_gxn[m_idx] = m_n < 1 / alpha

        if not found_lb and np.prod(indicators_gxn[m_idx]):
            found_lb = True
        if found_lb and not np.prod(indicators_gxn[m_idx]):
            break  # since interval, once find a value that fails, stop searching
    if intersection:
        ci_full = grid[np.where(np.prod(indicators_gxn, axis=1))[0]]
    else:
        ci_full = grid[np.where(indicators_gxn[:, -1])[0]]

    # if len(x_n) >= 800:
    #     st()
    if ci_full.size == 0:  # grid maybe too coarse
        idx = np.argmax(np.sum(indicators_gxn, axis=1))
        if idx == 0:
            CI = np.array([grid[0], grid[1]])
        else:
            CI = np.array([grid[idx - 1], grid[idx]])
    else:
        CI = np.array([ci_full.min(), ci_full.max()])
    CI = CI * (U - L) + L
    return CI  # only output the interval

def mean_ci_eff_corrected_membership_accelerated(
    Z, alpha, c=0.5, L=0, U=1, method_string=""
):
    n = len(Z)
    Z_norm = np.array([(zi - L) / (U - L) for zi in Z])

    initial_mean = np.mean(Z_norm)

    grid_spacing = 1e-4
    total_m = int(1 / grid_spacing)
    Mgrid = np.arange(grid_spacing, 1, step=grid_spacing)
    A = Mgrid
    # M = np.zeros((total_m, n))

    t_n = np.arange(1, n + 1)

    mu_hat = (0.5 + np.cumsum(Z_norm)) / (1.0 + t_n)
    sigma2hat_n = (0.25 + np.cumsum(np.power(Z_norm - mu_hat, 2))) / (1.0 + t_n)
    sigma2hat_tminus1_n = np.append(0.25, sigma2hat_n[:-1])

    lam_t = np.sqrt(2.0 * np.log(2.0 / alpha) / (n * sigma2hat_tminus1_n))
    
    # def M(m):
    #     coeff_plus = np.minimum(lam_t, c / m)
    #     coeff_minus = np.minimum(lam_t, c / (1 - m))
    #     M1 = np.exp(np.cumsum(np.log(1 + coeff_plus * (Z_norm - m))))
    #     M2 = np.exp(np.cumsum(np.log(1 - coeff_minus * (Z_norm - m))))
    #     return 0.5 * np.maximum(M1, M2)

    C_alpha = []

    # Run lower bound
    current_lb = initial_mean / 2.0
    tightest_valid_lb = 0.0
    gap_lb = np.abs(initial_mean - current_lb)
    while gap_lb > 1e-6:
        gap_lb /= 2.0
        current_martingale = MARTINGALE(current_lb, c, lam_t, Z_norm)
        
        if np.max(current_martingale) >= np.log(1.0 / alpha):
            # Tighten
            # INCREASE valid_lb
            tightest_valid_lb = float(current_lb)
            # print("Current ceiling: ", current_lb + 2*gap_lb)
            # Set new current_lb HIGHER
            current_lb += gap_lb
        else:
            # Can't tighten this far
            # Set new current_lb LOWER
            # print("Current ceiling: ", current_lb)
            current_lb -= gap_lb
        
    # print(np.exp(np.max(MARTINGALE(tightest_valid_lb, c, lam_t, Z_norm))))
    C_alpha.append(tightest_valid_lb * (U - L) + L)

    # Run upper bound
    current_ub = (1.0 + initial_mean) / 2.0
    tightest_valid_ub = 1.0
    gap_ub = np.abs(current_ub - initial_mean)
    while gap_ub > 1e-6:
        gap_ub /= 2.0
        current_martingale = MARTINGALE(current_ub, c, lam_t, Z_norm)
        if np.max(current_martingale) >= np.log(1.0 / alpha):
            # Tighten
            # DECREASE valid_ub
            tightest_valid_ub = float(current_ub)
            # Set new current_ub LOWER
            current_ub -= gap_ub
        else:
            # Can't tighten this far
            # Set new current_ub HIGHER
            current_ub += gap_ub

    C_alpha.append(tightest_valid_ub * (U - L) + L)
    return C_alpha

###################################################################
# Prediction powered confidence interval
def ppi_uniform(Y_gold, Y_gold_sim, Y_sim, alpha=0.05, c=0.95):
    """
    Y_gold_sim is sampled from Y_sim.
    """
    # Simple assertions
    # assert len(Y_gold) <= len(Y_sim)
    assert len(Y_gold) == len(Y_gold_sim)
    n = len(Y_gold)
    N = len(Y_sim)
    L = -(1 + N / n)
    U = 2 + N / n

    Y_rect = Y_gold_sim + (Y_gold - Y_gold_sim) * ((N + n) / n)
    Y = np.concatenate((Y_sim, Y_rect))
    np.random.shuffle(Y)
    C_pp_alpha = mean_ci_eff_corrected_membership_accelerated(
        Y, alpha, c=c, L=L, U=U, method_string="PPI_unif"
    )
    if C_pp_alpha == set():
        C_pp_alpha = wsr_iid(Y, alpha, intersection=True, c=c, L=L, U=U)
    C_pp_alpha = [min(C_pp_alpha), max(C_pp_alpha)]

    # C_pp_wsr_iid = wsr_iid(Y, alpha, intersection=True, c=c, L=L, U=U)
    return C_pp_alpha
