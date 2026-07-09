
"""kelly_approx.py

Approximate the ideal Kelly rule in `kelly.py` while using a simulator expert bank.

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from distributions import Distribution01
from kelly import KellySample, KellyState, update_wealth


@dataclass
class SimulatorExpert:
    expert_id: int
    dist: Distribution01
    mu: float
    var: float
    log_wealth: float = 0.0

    @property
    def wealth(self) -> float:
        return float(math.exp(self.log_wealth))


def gaussian_nll(y: float, mu: float, var: float, var_floor: float = 1e-8) -> float:
    v = max(var, var_floor)
    return 0.5 * (math.log(2.0 * math.pi * v) + ((y - mu) ** 2) / v)


def logsumexp(a: Sequence[float]) -> float:
    m = max(a)
    return m + math.log(sum(math.exp(x - m) for x in a))


def softmax_log_weights(log_w: Sequence[float]) -> np.ndarray:
    lse = logsumexp(log_w)
    return np.array([math.exp(x - lse) for x in log_w], dtype=float)


def build_experts(sim_dists: Sequence[Distribution01]) -> List[SimulatorExpert]:
    experts: List[SimulatorExpert] = []
    for k, d in enumerate(sim_dists):
        experts.append(
            SimulatorExpert(
                expert_id=k,
                dist=d,
                mu=float(d.true_mean()),
                var=float(d.true_variance()),
                log_wealth=0.0,
            )
        )
    return experts


def mixture_mean_and_aleatoric_var(experts: Sequence[SimulatorExpert], pi: np.ndarray) -> Tuple[float, float]:
    mus = np.array([e.mu for e in experts], dtype=float)
    vars_ = np.array([e.var for e in experts], dtype=float)
    m = float(np.sum(pi * mus))
    v = float(np.sum(pi * vars_))
    return m, max(v, 0.0)


def compute_kelly_bet_from_moments(
    m: float,
    v: float,
    tau: float,
    lambda_kelly: float = 1.0,
    b_max: float = 1.0,
    var_floor: float = 1e-8,
) -> Tuple[float, float, float, int]:
    advantage = m - tau
    denom = max(v, var_floor)
    bet = float(np.clip(lambda_kelly * abs(advantage) / denom, 0.0, b_max))
    if advantage > 0:
        return bet, 0.0, bet, +1
    if advantage < 0:
        return 0.0, bet, bet, -1
    return 0.0, 0.0, 0.0, 0


def run_kelly_approx(
    real_dist: Distribution01,
    sim_dists: Sequence[Distribution01],
    n_rounds: int = 200,
    lambda_kelly: float = 1.0,
    eta_score: float = 5.0,
    seed: Optional[int] = None,
    initial_tau: float = 0.5,
    var_floor: float = 1e-8,
    b_max: float = 1.0,
) -> Dict[str, Any]:
    if seed is not None:
        np.random.seed(seed)

    experts = build_experts(sim_dists)
    state = KellyState(wealth=1.0, tau=float(initial_tau), samples=[])

    history: List[Dict[str, Any]] = []

    for t in range(1, n_rounds + 1):
        # posterior weights BEFORE observing y_t
        pi = softmax_log_weights([e.log_wealth for e in experts])

        # plug-in moments (mean + aleatoric variance)
        m_t, v_t = mixture_mean_and_aleatoric_var(experts, pi)

        # kelly-like bet mirroring oracle
        bet_up, bet_down, bet, direction = compute_kelly_bet_from_moments(
            m=m_t, v=v_t, tau=state.tau,
            lambda_kelly=lambda_kelly, b_max=b_max, var_floor=var_floor
        )

        # real sample
        y = float(real_dist.sample(1)[0])

        # wealth update (same payoff rule)
        new_wealth, payoff, bet_correct = update_wealth(
            wealth=state.wealth, bet=bet, x_observed=y, tau=state.tau,
            bet_up=bet_up, bet_down=bet_down
        )

        state.samples.append(KellySample(x=y, bet=bet, payoff=payoff))
        state.wealth = new_wealth
        state.update_tau()

        # update expert log-wealth with proper score
        for e in experts:
            e.log_wealth += -eta_score * gaussian_nll(y, e.mu, e.var, var_floor=var_floor)

        history.append({
            "round": t, "x": y, "tau": state.tau, "bet": bet,
            "mixture_mean": m_t, "aleatoric_var": v_t,
            "pi_max": float(np.max(pi)), "top": int(np.argmax(pi))
        })

    bet_weighted_estimate = state.get_bet_weighted_estimate()
    mc_estimate = float(np.mean([s.x for s in state.samples]))
    return {
        "bet_weighted_estimate": float(bet_weighted_estimate),
        "mc_estimate": mc_estimate,
        "true_mean": float(real_dist.true_mean()),
        "true_variance": float(real_dist.true_variance()),
        "history": history,
        "experts": experts,
        "state": state,
    }


def make_default_sim_bank() -> List[Distribution01]:
    """A reasonably diverse simulator bank.
    """
    from distributions import (
        BernoulliRare,
        BetaSkewed,
        TruncatedNormal,
        BimodalMixture,
        UniformSpike,
        GaussianMixture,
    )

    bank: List[Distribution01] = []

    # Bernoulli
    for p in [0.05, 0.2, 0.5, 0.8]:
        bank.append(BernoulliRare(p=p))

    # Beta family
    for a, b in [(0.5, 2.0), (2.0, 0.5), (5.0, 1.0), (1.0, 5.0), (2.0, 2.0), (0.5, 0.5)]:
        bank.append(BetaSkewed(alpha=a, beta=b))

    # Truncated normals
    for mu, sig in [(0.2, 0.1), (0.5, 0.1), (0.8, 0.1), (0.5, 0.05), (0.5, 0.2)]:
        bank.append(TruncatedNormal(mu=mu, sigma=sig))

    # Mixtures
    for w in [0.3, 0.5, 0.7, 0.9]:
        bank.append(BimodalMixture(weight=w))

    # Uniform spike
    for loc, prob in [(0.1, 0.3), (0.1, 0.6), (0.5, 0.4), (0.9, 0.3), (0.9, 0.6)]:
        bank.append(UniformSpike(spike_location=loc, spike_prob=prob))

    bank.append(GaussianMixture(means=[0.3, 0.7], sigmas=[0.12, 0.12], weights=[0.5, 0.5]))
    bank.append(GaussianMixture(means=[0.2, 0.5], sigmas=[0.10, 0.10], weights=[0.6, 0.4]))
    bank.append(GaussianMixture(means=[0.5, 0.8], sigmas=[0.10, 0.10], weights=[0.4, 0.6]))
    bank.append(GaussianMixture(means=[0.2, 0.5, 0.8], sigmas=[0.08, 0.08, 0.08], weights=[0.3, 0.4, 0.3]))

    return bank


