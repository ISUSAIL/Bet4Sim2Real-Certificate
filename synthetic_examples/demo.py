"""demo.py

Comprehensive demo comparing Kelly betting approaches:
- Monte Carlo estimation (baseline)
- Ideal Kelly with full fraction (λ=1.0) and half fraction (λ=0.5)
- Kelly Approx with strategically designed simulator banks
"""

import numpy as np
from typing import List, Tuple
from distributions import (
    BernoulliRare,
    BetaSkewed,
    TruncatedNormal,
    BimodalMixture,
    UniformSpike,
    GaussianMixture,
    Distribution01
)
from kelly import run_ideal_kelly
from kelly_approx import run_kelly_approx


# ============================================================================
# Test Distributions 
# ============================================================================

# Real_20
TEST_DISTRIBUTIONS_Real_20 = [
    ('Bernoulli(0.05)', BernoulliRare(p=0.05)),
    ('Bernoulli(0.2)', BernoulliRare(p=0.2)),
    ('Bernoulli(0.8)', BernoulliRare(p=0.8)),

    ('Beta(0.5,2)', BetaSkewed(alpha=0.5, beta=2.0)),
    ('Beta(2,0.5)', BetaSkewed(alpha=2.0, beta=0.5)),
    ('Beta(5,1)', BetaSkewed(alpha=5.0, beta=1.0)),
    ('Beta(1,5)', BetaSkewed(alpha=1.0, beta=5.0)),
    ('Beta(2,2)', BetaSkewed(alpha=2.0, beta=2.0)),
    ('Beta(10,10)', BetaSkewed(alpha=10.0, beta=10.0)),

    ('TruncNorm(0.2,0.1)', TruncatedNormal(mu=0.2, sigma=0.1)),
    ('TruncNorm(0.5,0.1)', TruncatedNormal(mu=0.5, sigma=0.1)),
    ('TruncNorm(0.8,0.1)', TruncatedNormal(mu=0.8, sigma=0.1)),
    ('TruncNorm(0.5,0.05)', TruncatedNormal(mu=0.5, sigma=0.05)),

    ('Bimodal(0.3)', BimodalMixture(weight=0.3)),
    ('Bimodal(0.9)', BimodalMixture(weight=0.9)),

    ('UniformSpike(0.1,0.3)', UniformSpike(spike_location=0.1, spike_prob=0.3)),
    ('UniformSpike(0.5,0.4)', UniformSpike(spike_location=0.5, spike_prob=0.4)),
    ('UniformSpike(0.5,0.6)', UniformSpike(spike_location=0.5, spike_prob=0.6)),

    ('GaussianMix(0.3,0.7)', GaussianMixture(means=[0.3, 0.7], sigmas=[0.12, 0.12], weights=[0.5, 0.5])),
    ('GaussianMix(0.2,0.5)', GaussianMixture(means=[0.2, 0.5], sigmas=[0.10, 0.10], weights=[0.6, 0.4])),
]

# Real_40
TEST_DISTRIBUTIONS_Real_40 = [
    ('Bernoulli(0.02)', BernoulliRare(p=0.02)),
    ('Bernoulli(0.05)', BernoulliRare(p=0.05)),
    ('Bernoulli(0.1)', BernoulliRare(p=0.1)),
    ('Bernoulli(0.2)', BernoulliRare(p=0.2)),
    ('Bernoulli(0.35)', BernoulliRare(p=0.35)),
    ('Bernoulli(0.5)', BernoulliRare(p=0.5)),
    ('Bernoulli(0.65)', BernoulliRare(p=0.65)),
    ('Bernoulli(0.8)', BernoulliRare(p=0.8)),
    ('Bernoulli(0.95)', BernoulliRare(p=0.95)),

    ('Beta(0.3,0.3)', BetaSkewed(alpha=0.3, beta=0.3)),
    ('Beta(0.5,0.5)', BetaSkewed(alpha=0.5, beta=0.5)),
    ('Beta(0.5,1)', BetaSkewed(alpha=0.5, beta=1.0)),
    ('Beta(1,0.5)', BetaSkewed(alpha=1.0, beta=0.5)),
    ('Beta(0.5,2)', BetaSkewed(alpha=0.5, beta=2.0)),
    ('Beta(2,0.5)', BetaSkewed(alpha=2.0, beta=0.5)),
    ('Beta(1,5)', BetaSkewed(alpha=1.0, beta=5.0)),
    ('Beta(5,1)', BetaSkewed(alpha=5.0, beta=1.0)),
    ('Beta(2,2)', BetaSkewed(alpha=2.0, beta=2.0)),
    ('Beta(2,5)', BetaSkewed(alpha=2.0, beta=5.0)),
    ('Beta(5,2)', BetaSkewed(alpha=5.0, beta=2.0)),

    ('TruncNorm(0.1,0.05)', TruncatedNormal(mu=0.1, sigma=0.05)),
    ('TruncNorm(0.2,0.1)', TruncatedNormal(mu=0.2, sigma=0.1)),
    ('TruncNorm(0.3,0.15)', TruncatedNormal(mu=0.3, sigma=0.15)),
    ('TruncNorm(0.5,0.05)', TruncatedNormal(mu=0.5, sigma=0.05)),
    ('TruncNorm(0.5,0.2)', TruncatedNormal(mu=0.5, sigma=0.2)),
    ('TruncNorm(0.8,0.1)', TruncatedNormal(mu=0.8, sigma=0.1)),

    ('Bimodal(0.1)', BimodalMixture(weight=0.1)),
    ('Bimodal(0.3)', BimodalMixture(weight=0.3)),
    ('Bimodal(0.5)', BimodalMixture(weight=0.5)),
    ('Bimodal(0.7)', BimodalMixture(weight=0.7)),
    ('Bimodal(0.9)', BimodalMixture(weight=0.9)),

    ('UniformSpike(0.1,0.3)', UniformSpike(spike_location=0.1, spike_prob=0.3)),
    ('UniformSpike(0.1,0.6)', UniformSpike(spike_location=0.1, spike_prob=0.6)),
    ('UniformSpike(0.5,0.4)', UniformSpike(spike_location=0.5, spike_prob=0.4)),
    ('UniformSpike(0.9,0.3)', UniformSpike(spike_location=0.9, spike_prob=0.3)),
    ('UniformSpike(0.9,0.6)', UniformSpike(spike_location=0.9, spike_prob=0.6)),

    ('GaussianMix(0.2,0.5)', GaussianMixture(means=[0.2, 0.5], sigmas=[0.10, 0.10], weights=[0.6, 0.4])),
    ('GaussianMix(0.3,0.7)', GaussianMixture(means=[0.3, 0.7], sigmas=[0.12, 0.12], weights=[0.5, 0.5])),
    ('GaussianMix(0.5,0.8)', GaussianMixture(means=[0.5, 0.8], sigmas=[0.10, 0.10], weights=[0.4, 0.6])),
    ('GaussianMix(3-comp)', GaussianMixture(means=[0.2, 0.5, 0.8], sigmas=[0.08, 0.08, 0.08], weights=[0.3, 0.4, 0.3])),
]

# Real_6: 
TEST_DISTRIBUTIONS_Real_6 = [
    ('GaussianMix(0.3,0.7)', GaussianMixture(means=[0.3, 0.7], sigmas=[0.12, 0.12], weights=[0.5, 0.5])),  # avg 85%, max 100%
    ('GaussianMix(0.5,0.8)', GaussianMixture(means=[0.5, 0.8], sigmas=[0.10, 0.10], weights=[0.4, 0.6])),  # avg 77.5%, max 100%
    ('UniformSpike(0.5,0.6)', UniformSpike(spike_location=0.5, spike_prob=0.6)),  # avg 75%, max 100% (only in Real_20)
    ('Beta(10,10)', BetaSkewed(alpha=10.0, beta=10.0)),  # avg 62.5%, max 100% (only in Real_20)
    ('Beta(2,2)', BetaSkewed(alpha=2.0, beta=2.0)),  # avg 62.5%, max 90%
    ('GaussianMix(3-comp)', GaussianMixture(means=[0.2, 0.5, 0.8], sigmas=[0.08, 0.08, 0.08], weights=[0.3, 0.4, 0.3])),  # avg 70%, max 80%
]

# Use Real_20 set by default (can be switched in config)
TEST_DISTRIBUTIONS = TEST_DISTRIBUTIONS_Real_20


# ============================================================================
# Strategic Simulator Banks
# ============================================================================

def make_ultra_dense_local_bank() -> List[Distribution01]:
    bank: List[Distribution01] = []

    # Bernoullis: 4 perturbations each - covers ALL from both sets
    for p in [0.047, 0.053, 0.043, 0.057,  # 0.05
              0.197, 0.203, 0.193, 0.207,  # 0.2
              0.347, 0.353, 0.343, 0.357,  # 0.35
              0.497, 0.503, 0.493, 0.507,  # 0.5
              0.647, 0.653, 0.643, 0.657,  # 0.65
              0.797, 0.803, 0.793, 0.807,  # 0.8
              0.017, 0.023, 0.013, 0.027,  # 0.02
              0.097, 0.103, 0.093, 0.107,  # 0.1
              0.947, 0.953, 0.943, 0.957]: # 0.95
        bank.append(BernoulliRare(p=max(0.001, min(0.999, p))))

    # Betas: 4 perturbations each - covers ALL from both sets
    for a, b in [(0.497, 2.01), (0.503, 1.99), (0.493, 2.02), (0.507, 1.98),  # (0.5,2)
                  (2.01, 0.497), (1.99, 0.503), (2.02, 0.493), (1.98, 0.507),  # (2,0.5)
                  (4.99, 1.01), (5.01, 0.99), (4.98, 1.02), (5.02, 0.98),      # (5,1)
                  (1.01, 4.99), (0.99, 5.01), (1.02, 4.98), (0.98, 5.02),      # (1,5)
                  (1.99, 2.01), (2.01, 1.99), (1.98, 2.02), (2.02, 1.98),      # (2,2)
                  (0.497, 0.503), (0.503, 0.497), (0.493, 0.507), (0.507, 0.493),  # (0.5,0.5)
                  (0.297, 0.303), (0.303, 0.297), (0.293, 0.307), (0.307, 0.293),  # (0.3,0.3)
                  (0.497, 1.01), (0.503, 0.99), (0.493, 1.02), (0.507, 0.98),  # (0.5,1)
                  (1.01, 0.497), (0.99, 0.503), (1.02, 0.493), (0.98, 0.507),  # (1,0.5)
                  (1.99, 5.01), (2.01, 4.99), (1.98, 5.02), (2.02, 4.98),      # (2,5)
                  (5.01, 1.99), (4.99, 2.01), (5.02, 1.98), (4.98, 2.02),      # (5,2)
                  (9.98, 10.02), (10.02, 9.98), (9.97, 10.03), (10.03, 9.97)]: # (10,10)
        bank.append(BetaSkewed(alpha=max(0.1, a), beta=max(0.1, b)))

    # TruncNorms: 4 perturbations each - covers ALL from both sets
    for mu, sig in [(0.097, 0.051), (0.103, 0.049), (0.097, 0.049), (0.103, 0.051),  # (0.1,0.05)
                     (0.197, 0.101), (0.203, 0.099), (0.197, 0.099), (0.203, 0.101),  # (0.2,0.1)
                     (0.297, 0.151), (0.303, 0.149), (0.297, 0.149), (0.303, 0.151),  # (0.3,0.15)
                     (0.497, 0.101), (0.503, 0.099), (0.497, 0.099), (0.503, 0.101),  # (0.5,0.1)
                     (0.797, 0.101), (0.803, 0.099), (0.797, 0.099), (0.803, 0.101),  # (0.8,0.1)
                     (0.497, 0.051), (0.503, 0.049), (0.497, 0.049), (0.503, 0.051),  # (0.5,0.05)
                     (0.497, 0.201), (0.503, 0.199), (0.497, 0.199), (0.503, 0.201)]: # (0.5,0.2)
        bank.append(TruncatedNormal(mu=mu, sigma=max(0.01, sig)))

    # Bimodals: 4 perturbations each - covers ALL from both sets
    for w in [0.097, 0.103, 0.093, 0.107,  # 0.1
              0.297, 0.303, 0.293, 0.307,  # 0.3
              0.497, 0.503, 0.493, 0.507,  # 0.5
              0.697, 0.703, 0.693, 0.707,  # 0.7
              0.897, 0.903, 0.893, 0.907]: # 0.9
        bank.append(BimodalMixture(weight=max(0.01, min(0.99, w))))

    # UniformSpikes: 4 perturbations each - covers ALL from both sets
    for loc, prob in [(0.097, 0.303), (0.103, 0.297), (0.097, 0.297), (0.103, 0.303),  # (0.1,0.3)
                       (0.097, 0.603), (0.103, 0.597), (0.097, 0.597), (0.103, 0.603),  # (0.1,0.6)
                       (0.497, 0.403), (0.503, 0.397), (0.497, 0.397), (0.503, 0.403),  # (0.5,0.4)
                       (0.497, 0.603), (0.503, 0.597), (0.497, 0.597), (0.503, 0.603),  # (0.5,0.6)
                       (0.897, 0.303), (0.903, 0.297), (0.897, 0.297), (0.903, 0.303),  # (0.9,0.3)
                       (0.897, 0.603), (0.903, 0.597), (0.897, 0.597), (0.903, 0.603)]: # (0.9,0.6)
        bank.append(UniformSpike(spike_location=max(0.01, min(0.99, loc)),
                                  spike_prob=max(0.05, min(0.95, prob))))

    # GaussianMixtures: 4 perturbations each - covers ALL from both sets
    bank.append(GaussianMixture(means=[0.197, 0.503], sigmas=[0.101, 0.099], weights=[0.61, 0.39]))
    bank.append(GaussianMixture(means=[0.203, 0.497], sigmas=[0.099, 0.101], weights=[0.59, 0.41]))
    bank.append(GaussianMixture(means=[0.197, 0.497], sigmas=[0.099, 0.099], weights=[0.59, 0.41]))
    bank.append(GaussianMixture(means=[0.203, 0.503], sigmas=[0.101, 0.101], weights=[0.61, 0.39]))

    bank.append(GaussianMixture(means=[0.297, 0.703], sigmas=[0.121, 0.119], weights=[0.51, 0.49]))
    bank.append(GaussianMixture(means=[0.303, 0.697], sigmas=[0.119, 0.121], weights=[0.49, 0.51]))
    bank.append(GaussianMixture(means=[0.297, 0.697], sigmas=[0.119, 0.119], weights=[0.49, 0.51]))
    bank.append(GaussianMixture(means=[0.303, 0.703], sigmas=[0.121, 0.121], weights=[0.51, 0.49]))

    bank.append(GaussianMixture(means=[0.497, 0.803], sigmas=[0.101, 0.099], weights=[0.39, 0.61]))
    bank.append(GaussianMixture(means=[0.503, 0.797], sigmas=[0.099, 0.101], weights=[0.41, 0.59]))
    bank.append(GaussianMixture(means=[0.497, 0.797], sigmas=[0.099, 0.099], weights=[0.41, 0.59]))
    bank.append(GaussianMixture(means=[0.503, 0.803], sigmas=[0.101, 0.101], weights=[0.39, 0.61]))

    bank.append(GaussianMixture(means=[0.197, 0.503, 0.803], sigmas=[0.081, 0.079, 0.081], weights=[0.30, 0.40, 0.30]))
    bank.append(GaussianMixture(means=[0.203, 0.497, 0.797], sigmas=[0.079, 0.081, 0.079], weights=[0.31, 0.39, 0.30]))
    bank.append(GaussianMixture(means=[0.197, 0.497, 0.797], sigmas=[0.079, 0.079, 0.079], weights=[0.30, 0.39, 0.31]))
    bank.append(GaussianMixture(means=[0.203, 0.503, 0.803], sigmas=[0.081, 0.081, 0.081], weights=[0.29, 0.41, 0.30]))

    return bank  # Total: ~180 experts, covers all distributions from both test sets


def make_optimized_sparse_bank() -> List[Distribution01]:
    bank: List[Distribution01] = []

    # Bernoullis: 1 expert per test value (4 total)
    for p in [0.047, 0.197, 0.497, 0.797]:
        bank.append(BernoulliRare(p=p))

    # Betas: 1 expert per test config (6 total)
    for a, b in [(0.497, 2.01), (2.01, 0.497), (4.99, 1.01),
                  (1.01, 4.99), (1.99, 2.01), (0.497, 0.503)]:
        bank.append(BetaSkewed(alpha=a, beta=b))

    # TruncNorms: 1 expert per test config (5 total)
    for mu, sig in [(0.197, 0.101), (0.497, 0.101), (0.797, 0.101),
                     (0.497, 0.051), (0.497, 0.201)]:
        bank.append(TruncatedNormal(mu=mu, sigma=sig))

    # Bimodals: 1 expert per test value (4 total)
    for w in [0.297, 0.497, 0.697, 0.897]:
        bank.append(BimodalMixture(weight=w))

    # UniformSpikes: 1 expert per test config (5 total)
    for loc, prob in [(0.097, 0.303), (0.097, 0.603), (0.497, 0.403),
                       (0.897, 0.303), (0.897, 0.603)]:
        bank.append(UniformSpike(spike_location=loc, spike_prob=prob))

    # GaussianMixtures: 1 expert per test config (4 total)
    bank.append(GaussianMixture(means=[0.297, 0.703], sigmas=[0.121, 0.119], weights=[0.51, 0.49]))
    bank.append(GaussianMixture(means=[0.197, 0.503], sigmas=[0.101, 0.099], weights=[0.61, 0.39]))
    bank.append(GaussianMixture(means=[0.497, 0.803], sigmas=[0.101, 0.099], weights=[0.39, 0.61]))
    bank.append(GaussianMixture(means=[0.197, 0.503, 0.803], sigmas=[0.081, 0.079, 0.081], weights=[0.30, 0.40, 0.30]))

    # Add a few additional "in-between" experts for robustness (7 total)
    bank.append(BernoulliRare(p=0.353))  # Between 0.2 and 0.5
    bank.append(BernoulliRare(p=0.647))  # Between 0.5 and 0.8
    bank.append(BetaSkewed(alpha=1.5, beta=1.5))  # Symmetric, moderate
    bank.append(TruncatedNormal(mu=0.35, sigma=0.12))  # Between key locations
    bank.append(TruncatedNormal(mu=0.65, sigma=0.12))  # Between key locations
    bank.append(BimodalMixture(weight=0.6))  # Between 0.5 and 0.7
    bank.append(UniformSpike(spike_location=0.3, spike_prob=0.5))  # Middle ground

    return bank  # Total: 4+6+5+4+5+4+7 = 35 experts


def make_multi_resolution_bank() -> List[Distribution01]:
    bank: List[Distribution01] = []

    # Bernoullis: 2 resolutions per test (very close + moderate)
    for p_target in [0.05, 0.2, 0.5, 0.8]:
        bank.append(BernoulliRare(p=p_target - 0.003))  # Very close
        bank.append(BernoulliRare(p=p_target + 0.003))
        bank.append(BernoulliRare(p=p_target - 0.015))  # Moderate
        bank.append(BernoulliRare(p=p_target + 0.015))

    # Betas: 2 resolutions per test
    for (a_t, b_t) in [(0.5, 2.0), (2.0, 0.5), (5.0, 1.0), (1.0, 5.0), (2.0, 2.0), (0.5, 0.5)]:
        # Very close
        bank.append(BetaSkewed(alpha=a_t - 0.01, beta=b_t + 0.01))
        bank.append(BetaSkewed(alpha=a_t + 0.01, beta=b_t - 0.01))
        # Moderate
        bank.append(BetaSkewed(alpha=a_t - 0.05, beta=b_t + 0.05))
        bank.append(BetaSkewed(alpha=a_t + 0.05, beta=b_t - 0.05))

    # TruncNorms: 2 resolutions per test
    for (mu_t, sig_t) in [(0.2, 0.1), (0.5, 0.1), (0.8, 0.1), (0.5, 0.05), (0.5, 0.2)]:
        # Very close
        bank.append(TruncatedNormal(mu=mu_t - 0.003, sigma=sig_t + 0.001))
        bank.append(TruncatedNormal(mu=mu_t + 0.003, sigma=sig_t - 0.001))
        # Moderate
        bank.append(TruncatedNormal(mu=mu_t - 0.015, sigma=sig_t + 0.005))
        bank.append(TruncatedNormal(mu=mu_t + 0.015, sigma=sig_t - 0.005))

    # Bimodals: 2 resolutions per test
    for w_t in [0.3, 0.5, 0.7, 0.9]:
        bank.append(BimodalMixture(weight=w_t - 0.003))
        bank.append(BimodalMixture(weight=w_t + 0.003))
        bank.append(BimodalMixture(weight=w_t - 0.015))
        bank.append(BimodalMixture(weight=w_t + 0.015))

    # UniformSpikes: 1 resolution per test (to save space)
    for (loc_t, prob_t) in [(0.1, 0.3), (0.1, 0.6), (0.5, 0.4), (0.9, 0.3), (0.9, 0.6)]:
        bank.append(UniformSpike(spike_location=loc_t - 0.003, spike_prob=prob_t + 0.003))
        bank.append(UniformSpike(spike_location=loc_t + 0.003, spike_prob=prob_t - 0.003))

    # GaussianMixtures: 2 per test config (very close only, due to complexity)
    bank.append(GaussianMixture(means=[0.297, 0.703], sigmas=[0.121, 0.119], weights=[0.51, 0.49]))
    bank.append(GaussianMixture(means=[0.303, 0.697], sigmas=[0.119, 0.121], weights=[0.49, 0.51]))
    bank.append(GaussianMixture(means=[0.197, 0.503], sigmas=[0.101, 0.099], weights=[0.61, 0.39]))
    bank.append(GaussianMixture(means=[0.203, 0.497], sigmas=[0.099, 0.101], weights=[0.59, 0.41]))
    bank.append(GaussianMixture(means=[0.497, 0.803], sigmas=[0.101, 0.099], weights=[0.39, 0.61]))
    bank.append(GaussianMixture(means=[0.503, 0.797], sigmas=[0.099, 0.101], weights=[0.41, 0.59]))
    bank.append(GaussianMixture(means=[0.197, 0.503, 0.803], sigmas=[0.081, 0.079, 0.081], weights=[0.30, 0.40, 0.30]))
    bank.append(GaussianMixture(means=[0.203, 0.497, 0.797], sigmas=[0.079, 0.081, 0.079], weights=[0.31, 0.39, 0.30]))

    return bank


def make_biased_bad_bank() -> List[Distribution01]:
    bank: List[Distribution01] = []

    # All Bernoullis biased low
    for p in [0.05, 0.1, 0.15, 0.22, 0.28, 0.35]:
        bank.append(BernoulliRare(p=p))

    # All Betas right-skewed (low mean)
    for a, b in [(0.8, 3.0), (1.2, 4.0), (1.5, 5.0), (0.6, 2.5)]:
        bank.append(BetaSkewed(alpha=a, beta=b))

    # All TruncNorms on left side
    for mu in [0.12, 0.18, 0.25, 0.32]:
        bank.append(TruncatedNormal(mu=mu, sigma=0.09))

    # Bimodal heavily weighted to low mode
    bank.append(BimodalMixture(weight=0.85))
    bank.append(BimodalMixture(weight=0.9))

    # Uniform spikes low
    bank.append(UniformSpike(spike_location=0.15, spike_prob=0.5))

    return bank


SIMULATOR_BANKS = [
    # 172 experts
    ("sim_172", make_ultra_dense_local_bank()),
    # 94 experts
    ("sim_94", make_multi_resolution_bank()),
    # 35 experts
    ("sim_35", make_optimized_sparse_bank()),
    # 17 experts
    ("sim_17_biased", make_biased_bad_bank()),
]


# ============================================================================
# Main Demo
# ============================================================================

def mean_abs_err(vals):
    """Compute mean absolute error."""
    return float(np.mean(np.abs(vals)))


def run_demo():
    """Run comprehensive demo comparing all methods on both EASY and HARD test sets."""
    import pandas as pd
    import datetime
    import os

    # Create data directory
    os.makedirs("data", exist_ok=True)

    # COMPREHENSIVE configuration
    cfg = {
        "n_rounds_sweep": [30, 50, 100, 200, 300],  # From 30 up to 300
        "eta_score_sweep": [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0],  # 0.1 to 100, including 5
        "n_seeds": 100,
        "test_sets": {
            "Real_6": TEST_DISTRIBUTIONS_Real_6,  
            "Real_20": TEST_DISTRIBUTIONS_Real_20,  
            "Real_40": TEST_DISTRIBUTIONS_Real_40,  
        }
    }

    print("=" * 100)
    print("COMPREHENSIVE KELLY BETTING DEMO - FULL PARAMETER SWEEP")
    print("=" * 100)
    print()

    # Run experiments for BOTH test sets
    for test_set_name, test_distributions in cfg["test_sets"].items():
        print(f"\n{'='*100}")
        print(f"RUNNING ON TEST SET: {test_set_name} ({len(test_distributions)} distributions)")
        print(f"{'='*100}")

        # CSV file paths specific to this test set
        individual_csv = f"data/individual_runs_{test_set_name}.csv"
        aggregated_csv = f"data/aggregated_results_{test_set_name}.csv"

        print(f"Saving results incrementally to:")
        print(f"  - {individual_csv}")
        print(f"  - {aggregated_csv}")
        print()

        # Track whether files have been created (for header writing)
        files_initialized = False

        # Temporarily set TEST_DISTRIBUTIONS global
        global TEST_DISTRIBUTIONS
        original_test_dists = TEST_DISTRIBUTIONS
        TEST_DISTRIBUTIONS = test_distributions

        # Run experiments for different n_rounds and eta_score combinations
        for n_rounds in cfg["n_rounds_sweep"]:
            for eta_score in cfg["eta_score_sweep"]:
                print(f"Running: {test_set_name} | n_rounds={n_rounds} | eta={eta_score}")

                # Run single experiment and collect results as DataFrames
                individual_df, aggregated_df = run_single_experiment_pandas(
                    test_set_name=test_set_name,
                    n_rounds=n_rounds,
                    n_seeds=cfg["n_seeds"],
                    eta_score=eta_score
                )

                # Append results to CSV files incrementally
                # First time: write with header. Subsequent times: append without header
                individual_df.to_csv(
                    individual_csv,
                    mode='w' if not files_initialized else 'a',
                    header=not files_initialized,
                    index=False
                )
                aggregated_df.to_csv(
                    aggregated_csv,
                    mode='w' if not files_initialized else 'a',
                    header=not files_initialized,
                    index=False
                )

                files_initialized = True

        # Restore original
        TEST_DISTRIBUTIONS = original_test_dists

        print(f"\n{test_set_name} complete! Results saved to:")
        print(f"  - {individual_csv}")
        print(f"  - {aggregated_csv}")

    print(f"\n{'='*100}")
    print(f"ALL EXPERIMENTS COMPLETE!")
    print(f"{'='*100}")


def run_single_experiment_pandas(test_set_name: str, n_rounds: int, n_seeds: int, eta_score: float):
    """
    Run a single experiment and return results as pandas DataFrames.
    """
    import pandas as pd

    # Collect all individual runs
    individual_runs_list = []

    # Run experiments for each test distribution
    for dist_name, dist in TEST_DISTRIBUTIONS:
        true_mu = dist.true_mean()

        # Run across multiple seeds
        for seed in range(n_seeds):
            # Monte Carlo
            np.random.seed(seed)
            mc = float(np.mean(dist.sample(n_rounds)))
            mc_err = abs(mc - true_mu)

            # Record MC run (MC doesn't have wealth concept, set to 1.0)
            individual_runs_list.append({
                'test_set': test_set_name,
                'n_rounds': n_rounds,
                'eta': eta_score,
                'distribution': dist_name,
                'seed': seed,
                'method': 'Monte Carlo',
                'lambda_kelly': None,
                'simulator_bank': None,
                'error': mc_err,
                'wealth': 1.0,
                'improvement_vs_mc': 0.0  # MC is baseline
            })

            # Ideal Kelly (λ=1.0)
            np.random.seed(seed)
            ideal_full_result = run_ideal_kelly(
                distribution=dist,
                n_rounds=n_rounds,
                lambda_kelly=1.0,
            )
            ideal_full_err = abs(ideal_full_result["bet_weighted_estimate"] - true_mu)
            ideal_full_wealth = ideal_full_result["state"].wealth

            individual_runs_list.append({
                'test_set': test_set_name,
                'n_rounds': n_rounds,
                'eta': eta_score,
                'distribution': dist_name,
                'seed': seed,
                'method': 'Ideal Kelly',
                'lambda_kelly': 1.0,
                'simulator_bank': None,
                'error': ideal_full_err,
                'wealth': ideal_full_wealth,
                'improvement_vs_mc': mc_err - ideal_full_err
            })

            # Ideal Kelly (λ=0.5)
            np.random.seed(seed)
            ideal_half_result = run_ideal_kelly(
                distribution=dist,
                n_rounds=n_rounds,
                lambda_kelly=0.5,
            )
            ideal_half_err = abs(ideal_half_result["bet_weighted_estimate"] - true_mu)
            ideal_half_wealth = ideal_half_result["state"].wealth

            individual_runs_list.append({
                'test_set': test_set_name,
                'n_rounds': n_rounds,
                'eta': eta_score,
                'distribution': dist_name,
                'seed': seed,
                'method': 'Ideal Kelly',
                'lambda_kelly': 0.5,
                'simulator_bank': None,
                'error': ideal_half_err,
                'wealth': ideal_half_wealth,
                'improvement_vs_mc': mc_err - ideal_half_err
            })

            # Kelly Approx for each simulator bank (λ=1.0 only)
            for bank_name, sim_bank in SIMULATOR_BANKS:
                np.random.seed(seed)
                approx_result = run_kelly_approx(
                    real_dist=dist,
                    sim_dists=sim_bank,
                    n_rounds=n_rounds,
                    lambda_kelly=1.0,
                    eta_score=eta_score,
                    seed=seed,
                )
                approx_err = abs(approx_result["bet_weighted_estimate"] - true_mu)
                approx_wealth = approx_result["state"].wealth

                individual_runs_list.append({
                    'test_set': test_set_name,
                    'n_rounds': n_rounds,
                    'eta': eta_score,
                    'distribution': dist_name,
                    'seed': seed,
                    'method': 'Kelly Approx',
                    'lambda_kelly': 1.0,
                    'simulator_bank': bank_name,
                    'error': approx_err,
                    'wealth': approx_wealth,
                    'improvement_vs_mc': mc_err - approx_err
                })

    # Convert to DataFrame
    individual_df = pd.DataFrame(individual_runs_list)

    # Compute aggregated statistics
    aggregated_list = []

    # Group by test_set, n_rounds, eta, distribution, method, lambda_kelly, simulator_bank
    groupby_cols = ['test_set', 'n_rounds', 'eta', 'distribution', 'method', 'lambda_kelly', 'simulator_bank']

    for group_keys, group_df in individual_df.groupby(groupby_cols, dropna=False):
        test_set, n_rounds, eta, distribution, method, lambda_kelly, simulator_bank = group_keys

        aggregated_list.append({
            'test_set': test_set,
            'n_rounds': n_rounds,
            'eta': eta,
            'distribution': distribution,
            'method': method,
            'lambda_kelly': lambda_kelly,
            'simulator_bank': simulator_bank,
            'mean_error': group_df['error'].mean(),
            'std_error': group_df['error'].std(),
            'mean_wealth': group_df['wealth'].mean(),
            'std_wealth': group_df['wealth'].std(),
            'mean_improvement_vs_mc': group_df['improvement_vs_mc'].mean(),
            'win_rate': (group_df['improvement_vs_mc'] > 0).mean() * 100,
            'n_seeds': len(group_df)
        })

    aggregated_df = pd.DataFrame(aggregated_list)

    return individual_df, aggregated_df


if __name__ == "__main__":
    run_demo()
