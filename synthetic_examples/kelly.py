"""
Ideal Kelly betting with ground truth information.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
from distributions import Distribution01


@dataclass
class KellySample:
    """Stores a single sample with its associated betting weight"""
    x: float  # The actual sample from P (psi(x) value)
    bet: float  # The bet size used for this sample
    payoff: float = 0.0  # The payoff Y_t for this sample


@dataclass
class KellyState:
    """Tracks the current state of the ideal Kelly betting process"""
    wealth: float = 1.0  # Current wealth W
    samples: List[KellySample] = field(default_factory=list)
    tau: float = 0.5  # Current running mean/reference value

    def add_sample(self, x: float, bet: float, payoff: float):
        """Add a new sample to the betting history"""
        self.samples.append(KellySample(x=x, bet=bet, payoff=payoff))

    def update_tau(self):
        """Update τ_t as bet-weighted estimate from observed samples"""
        if len(self.samples) > 0:
            total_bet = sum(s.bet for s in self.samples)
            if total_bet > 0:
                # Use bet-weighted estimate (samples with higher bets get more weight)
                self.tau = sum(s.bet * s.x for s in self.samples) / total_bet
            else:
                # Fallback to simple mean if no bets were placed
                self.tau = np.mean([s.x for s in self.samples])

    def get_bet_weighted_estimate(self) -> float:
        """Compute bet-weighted estimate: Σ(b_i * x_i) / Σ(b_i)"""
        if len(self.samples) == 0:
            return 0.0

        total_bet = sum(s.bet for s in self.samples)
        if total_bet == 0:
            return 0.0

        weighted_sum = sum(s.bet * s.x for s in self.samples)
        return weighted_sum / total_bet


def compute_ideal_kelly_bet(
    mu_true: float,
    tau: float,
    sigma_sq_true: float,
    lambda_kelly: float = 1.0
) -> Tuple[float, float, float]:
    """
    Compute ideal Kelly bet using ground truth mean and variance.

    Args:
        mu_true: Ground truth mean E[ψ(X)] from distribution
        tau: τ_{t-1}, current running mean estimate
        sigma_sq_true: True variance Var[ψ(X)] of the distribution
        lambda_kelly: Kelly fraction (0 < λ ≤ 1)

    Returns:
        (bet_up, bet_down, bet): Tuple of (b_t^↑, b_t^↓, b_t)
    """
    # Compute advantage: μ_true - τ_{t-1}
    advantage = mu_true - tau

    # Ideal Kelly bet: b_t = λ * |advantage| / σ²
    if sigma_sq_true > 1e-10:  # Avoid division by zero
        bet = lambda_kelly * abs(advantage) / sigma_sq_true
    else:
        bet = 0.0

    # Ensure bet is non-negative and clip to [0, 1]
    bet = np.clip(bet, 0.0, 1.0)

    # Determine betting direction based on advantage
    if advantage > 0:
        # μ_true > τ → bet UP (predicting ψ(x_t) > τ)
        bet_up = bet
        bet_down = 0.0
    elif advantage < 0:
        # μ_true < τ → bet DOWN (predicting ψ(x_t) < τ)
        bet_up = 0.0
        bet_down = bet
    else:
        # μ_true == τ → no advantage, no bet
        bet_up = 0.0
        bet_down = 0.0
        bet = 0.0

    return bet_up, bet_down, bet


def update_wealth(
    wealth: float,
    bet: float,
    x_observed: float,
    tau: float,
    bet_up: float,
    bet_down: float
) -> Tuple[float, float, bool]:
    """
    Update wealth based on betting outcome (Algorithm 1, lines 12-13).

    Args:
        wealth: Current wealth W
        bet: Bet size b_t
        x_observed: Observed value ψ(x_t) from real distribution
        tau: Reference value τ_{t-1}
        bet_up: b_t^↑, bet size on ψ(x_t) > τ
        bet_down: b_t^↓, bet size on ψ(x_t) < τ

    Returns:
        (new_wealth, payoff, bet_correct): Updated wealth, payoff Y_t, and whether bet was correct
    """
    # Compute payoff magnitude
    deviation = abs(x_observed - tau)

    # Determine which direction we bet and whether it was correct
    if bet_up > bet_down:
        # bet_up larger → bet UP (predicting ψ(x_t) > τ)
        bet_correct = (x_observed > tau)
    elif bet_down > bet_up:
        # bet_down larger → bet DOWN (predicting ψ(x_t) < τ)
        bet_correct = (x_observed < tau)
    else:
        bet_correct = True  # Consider any outcome as "correct" in symmetric case

    # Payoff is the deviation, with sign based on correctness
    if bet_correct:
        payoff = deviation
    else:
        payoff = -deviation

    # Update wealth: W ← W(1 + b_t * Y_t)
    new_wealth = wealth * (1.0 + bet * payoff)

    # Ensure wealth doesn't go negative (bankruptcy protection)
    new_wealth = max(new_wealth, 1e-10)

    return new_wealth, payoff, bet_correct


def run_ideal_kelly(
    distribution: Distribution01,
    n_rounds: int = 100,
    lambda_kelly: float = 0.5
) -> Dict[str, Any]:
    """
    Run ideal Kelly betting with ground truth information.

    Args:
        distribution: Distribution to sample from (must have true_mean() and true_variance())
        n_rounds: Number of betting rounds T
        lambda_kelly: Kelly fraction (0 < λ ≤ 1)

    Returns:
        Dictionary containing:
            - bet_weighted_estimate: Final bet-weighted estimate
            - mc_estimate: Standard Monte Carlo estimate (equal-weight)
            - true_mean: True mean from distribution
            - state: Final KellyState object
            - history: List of results per round
    """
    # Get ground truth information
    mu_true = distribution.true_mean()
    sigma_sq_true = distribution.true_variance()

    # Initialize state
    state = KellyState()

    # History tracking
    history = []

    # Sequential betting loop (Algorithm 1)
    for t in range(1, n_rounds + 1):
        # Compute ideal Kelly bet using ground truth
        bet_up, bet_down, bet = compute_ideal_kelly_bet(
            mu_true=mu_true,
            tau=state.tau,
            sigma_sq_true=sigma_sq_true,
            lambda_kelly=lambda_kelly
        )

        # Sample from real distribution (line 10)
        x_observed = distribution.sample(1)[0]

        # Update wealth (lines 12-13)
        new_wealth, payoff, bet_correct = update_wealth(
            wealth=state.wealth,
            bet=bet,
            x_observed=x_observed,
            tau=state.tau,
            bet_up=bet_up,
            bet_down=bet_down
        )

        # Update state (lines 10-11 in Algorithm 1)
        state.add_sample(x=x_observed, bet=bet, payoff=payoff)
        state.wealth = new_wealth
        state.update_tau()  # line 14

        # Store round result
        history.append({
            "round": t,
            "bet_up": bet_up,
            "bet_down": bet_down,
            "bet": bet,
            "payoff": payoff,
            "bet_correct": bet_correct,
            "x_observed": x_observed,
            "tau": state.tau,
            "wealth": state.wealth,
            "advantage": mu_true - state.tau  # Ground truth advantage
        })

    # Compute final estimates
    bet_weighted_estimate = state.get_bet_weighted_estimate()
    mc_estimate = np.mean([s.x for s in state.samples])

    return {
        "bet_weighted_estimate": bet_weighted_estimate,
        "mc_estimate": mc_estimate,
        "true_mean": mu_true,
        "state": state,
        "history": history
    }
