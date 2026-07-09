"""
Toy Distributions on [0,1]
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List
import scipy.stats as stats

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class Distribution01(ABC):
    """Abstract base class for distributions on [0,1]"""
    
    @abstractmethod
    def sample(self, n: int = 1) -> np.ndarray:
        """Generate n samples from the distribution"""
        pass
    
    @abstractmethod
    def true_mean(self) -> float:
        """Return the true analytical mean"""
        pass

    @abstractmethod
    def true_variance(self) -> float:
        """Return the true analytical variance"""
        pass

    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function (or PMF for discrete)"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Distribution name"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Distribution description"""
        pass
    
    def suggest_importance_distribution(self) -> 'Distribution01':
        """
        Suggest an importance sampling distribution for this target distribution.
        Default implementation returns self (no importance sampling).
        """
        return self


class BernoulliRare(Distribution01):
    """Binary distribution on {0,1} with rare success events"""
    
    def __init__(self, p: float = 0.05):
        """
        Bernoulli distribution with success probability p.
        
        Args:
            p: Probability of success (getting 1). Default 0.05 for rare events.
        """
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0,1]")
        self.p = p
    
    def sample(self, n: int = 1) -> np.ndarray:
        """Sample from Bernoulli(p)"""
        return np.random.binomial(1, self.p, size=n).astype(float)
    
    def true_mean(self) -> float:
        """E[X] = p for Bernoulli(p)"""
        return self.p

    def true_variance(self) -> float:
        """Var[X] = p(1-p) for Bernoulli(p)"""
        return self.p * (1.0 - self.p)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """PMF for Bernoulli distribution"""
        pmf = np.zeros_like(x)
        pmf[x == 0] = 1 - self.p
        pmf[x == 1] = self.p
        return pmf
    
    @property
    def name(self) -> str:
        return f"Bernoulli({self.p:.3f})"
    
    @property
    def description(self) -> str:
        return f"Binary distribution: P(X=1)={self.p:.3f}, P(X=0)={1-self.p:.3f}. " + \
               "Demonstrates rare event sampling challenges."
    
    def suggest_importance_distribution(self) -> 'Distribution01':
        """For rare events, suggest optimal importance probability for variance reduction"""
        if self.p <= 0.1:  # For rare events
            # Optimal importance probability balances variance reduction with weight variance
            # For Bernoulli, optimal p_importance ≈ √(p * true_mean) for mean estimation
            optimal_p = min(0.3, max(0.1, 3 * self.p))  # More conservative scaling
            return BernoulliRare(optimal_p)
        else:
            return self  # No improvement expected for common events


class BetaSkewed(Distribution01):
    """Beta distribution with various shapes on [0,1]"""
    
    def __init__(self, alpha: float = 0.5, beta: float = 2.0):
        """
        Beta distribution on [0,1].
        
        Args:
            alpha, beta: Shape parameters
            - alpha < 1, beta > 1: Right-skewed (most mass near 0)
            - alpha > 1, beta < 1: Left-skewed (most mass near 1) 
            - alpha = beta = 1: Uniform
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("Alpha and beta must be positive")
        self.alpha = alpha
        self.beta = beta
        self._dist = stats.beta(alpha, beta)
    
    def sample(self, n: int = 1) -> np.ndarray:
        """Sample from Beta(alpha, beta)"""
        return self._dist.rvs(size=n)
    
    def true_mean(self) -> float:
        """E[X] = alpha/(alpha + beta) for Beta(alpha, beta)"""
        return self.alpha / (self.alpha + self.beta)

    def true_variance(self) -> float:
        """Var[X] = αβ / ((α+β)²(α+β+1)) for Beta(alpha, beta)"""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """PDF for Beta distribution"""
        return self._dist.pdf(x)
    
    @property
    def name(self) -> str:
        return f"Beta({self.alpha:.1f}, {self.beta:.1f})"
    
    @property
    def description(self) -> str:
        skew = "right-skewed" if self.alpha < self.beta else "left-skewed" if self.alpha > self.beta else "symmetric"
        return f"Beta distribution on [0,1], {skew}. True mean = {self.true_mean():.3f}."
    
    def suggest_importance_distribution(self) -> 'Distribution01':
        """Suggest importance distribution for Beta that reduces variance"""
        # For very skewed distributions, use a less skewed version
        if self.alpha < 1 and self.beta > 2:  # Right-skewed
            # Use a less skewed Beta that still emphasizes the important region
            return BetaSkewed(alpha=max(0.8, self.alpha * 2), beta=max(1.5, self.beta * 0.7))
        elif self.alpha > 2 and self.beta < 1:  # Left-skewed  
            return BetaSkewed(alpha=max(1.5, self.alpha * 0.7), beta=max(0.8, self.beta * 2))
        else:
            return self  # Not skewed enough to benefit


class TruncatedNormal(Distribution01):
    """Truncated normal distribution on [0,1]"""
    
    def __init__(self, mu: float = 0.3, sigma: float = 0.2):
        """
        Normal distribution truncated to [0,1].
        
        Args:
            mu: Mean of underlying normal (before truncation)
            sigma: Standard deviation of underlying normal
        """
        self.mu = mu
        self.sigma = sigma
        # Create truncated normal using scipy
        a, b = (0 - mu) / sigma, (1 - mu) / sigma  # Standardized bounds
        self._dist = stats.truncnorm(a, b, loc=mu, scale=sigma)
    
    def sample(self, n: int = 1) -> np.ndarray:
        """Sample from truncated normal"""
        return self._dist.rvs(size=n)
    
    def true_mean(self) -> float:
        """Analytical mean of truncated normal"""
        return self._dist.mean()

    def true_variance(self) -> float:
        """Analytical variance of truncated normal"""
        return self._dist.var()

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """PDF for truncated normal"""
        return self._dist.pdf(x)
    
    @property
    def name(self) -> str:
        return f"TruncNorm(μ={self.mu:.2f}, σ={self.sigma:.2f})"
    
    @property
    def description(self) -> str:
        return f"Normal(μ={self.mu:.2f}, σ={self.sigma:.2f}) truncated to [0,1]. " + \
               f"True mean = {self.true_mean():.3f}."
    
    def suggest_importance_distribution(self) -> 'Distribution01':
        """Suggest uniform for truncated normal importance sampling"""
        # Uniform is often effective for truncated distributions
        return BetaSkewed(alpha=1.0, beta=1.0)  # Uniform


class BimodalMixture(Distribution01):
    """Mixture of two Beta distributions creating bimodal shape"""
    
    def __init__(self, weight: float = 0.3):
        """
        Mixture: weight*Beta(0.5,2) + (1-weight)*Beta(2,0.5)
        Creates bimodal distribution with modes near 0 and 1.
        
        Args:
            weight: Weight for first component (mode near 0)
        """
        if not 0 <= weight <= 1:
            raise ValueError("Weight must be in [0,1]")
        self.weight = weight
        self.beta1 = stats.beta(0.5, 2.0)  # Mode near 0
        self.beta2 = stats.beta(2.0, 0.5)  # Mode near 1
    
    def sample(self, n: int = 1) -> np.ndarray:
        """Sample from mixture distribution"""
        # Randomly choose which component for each sample
        component_choice = np.random.binomial(1, 1-self.weight, size=n)
        
        samples = np.zeros(n)
        n1 = np.sum(component_choice == 0)  # Number from first component
        n2 = n - n1  # Number from second component
        
        if n1 > 0:
            samples[component_choice == 0] = self.beta1.rvs(size=n1)
        if n2 > 0:
            samples[component_choice == 1] = self.beta2.rvs(size=n2)
            
        return samples
    
    def true_mean(self) -> float:
        """E[X] = weight*E[X1] + (1-weight)*E[X2]"""
        mean1 = self.beta1.mean()  # 0.5/(0.5+2) = 0.2
        mean2 = self.beta2.mean()  # 2/(2+0.5) = 0.8
        return self.weight * mean1 + (1 - self.weight) * mean2

    def true_variance(self) -> float:
        """Var[X] = E[Var[X|component]] + Var[E[X|component]]"""
        mean1 = self.beta1.mean()
        mean2 = self.beta2.mean()
        var1 = self.beta1.var()
        var2 = self.beta2.var()

        # E[Var[X|component]] = weight*Var[X1] + (1-weight)*Var[X2]
        expected_variance = self.weight * var1 + (1 - self.weight) * var2

        # Var[E[X|component]] = E[E[X|component]^2] - (E[E[X|component]])^2
        # where E[E[X|component]] = overall mean
        overall_mean = self.weight * mean1 + (1 - self.weight) * mean2
        expected_mean_squared = self.weight * mean1**2 + (1 - self.weight) * mean2**2
        variance_of_means = expected_mean_squared - overall_mean**2

        return expected_variance + variance_of_means

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """PDF for mixture distribution"""
        return self.weight * self.beta1.pdf(x) + (1 - self.weight) * self.beta2.pdf(x)
    
    @property
    def name(self) -> str:
        return f"BiModal(w={self.weight:.2f})"
    
    @property
    def description(self) -> str:
        return f"Bimodal mixture: {self.weight:.2f}*Beta(0.5,2) + {1-self.weight:.2f}*Beta(2,0.5). " + \
               f"True mean = {self.true_mean():.3f}."
    
    def suggest_importance_distribution(self) -> 'Distribution01':
        """For bimodal, suggest uniform to sample both modes effectively"""
        return BetaSkewed(alpha=1.0, beta=1.0)  # Uniform


class UniformSpike(Distribution01):
    """Uniform with probability spike at specific value"""
    
    def __init__(self, spike_location: float = 0.95, spike_prob: float = 0.1):
        """
        Distribution with uniform base and probability spike.
        
        Args:
            spike_location: Location of probability spike in [0,1]
            spike_prob: Probability mass at spike location
        """
        if not 0 <= spike_location <= 1:
            raise ValueError("Spike location must be in [0,1]")
        if not 0 <= spike_prob <= 1:
            raise ValueError("Spike probability must be in [0,1]")
        
        self.spike_location = spike_location
        self.spike_prob = spike_prob
    
    def sample(self, n: int = 1) -> np.ndarray:
        """Sample from uniform + spike distribution"""
        # Decide which samples come from spike vs uniform
        from_spike = np.random.binomial(1, self.spike_prob, size=n).astype(bool)
        
        samples = np.zeros(n)
        samples[from_spike] = self.spike_location
        samples[~from_spike] = np.random.uniform(0, 1, size=np.sum(~from_spike))
        
        return samples
    
    def true_mean(self) -> float:
        """E[X] = spike_prob * spike_location + (1-spike_prob) * 0.5"""
        uniform_mean = 0.5
        return self.spike_prob * self.spike_location + (1 - self.spike_prob) * uniform_mean

    def true_variance(self) -> float:
        """Var[X] = E[Var[X|component]] + Var[E[X|component]]"""
        # Component means
        spike_mean = self.spike_location
        uniform_mean = 0.5

        # Component variances
        # Point mass has zero variance, Uniform[0,1] has variance 1/12
        uniform_var = 1.0 / 12.0

        # E[Var[X|component]] = spike_prob * 0 + (1-spike_prob) * (1/12)
        expected_variance = (1 - self.spike_prob) * uniform_var

        # Var[E[X|component]] = E[E[X|component]^2] - (E[E[X|component]])^2
        overall_mean = self.spike_prob * spike_mean + (1 - self.spike_prob) * uniform_mean
        expected_mean_squared = self.spike_prob * spike_mean**2 + (1 - self.spike_prob) * uniform_mean**2
        variance_of_means = expected_mean_squared - overall_mean**2

        return expected_variance + variance_of_means

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Mixed discrete-continuous PDF"""
        pdf_vals = np.ones_like(x) * (1 - self.spike_prob)  # Uniform part
        # Add spike (represented as very narrow peak for visualization)
        spike_mask = np.abs(x - self.spike_location) < 0.01
        pdf_vals[spike_mask] += self.spike_prob / 0.02  # Concentrate spike mass
        return pdf_vals
    
    @property
    def name(self) -> str:
        return f"UniformSpike({self.spike_location:.2f}, {self.spike_prob:.2f})"
    
    @property
    def description(self) -> str:
        return f"Uniform[0,1] with {self.spike_prob:.1%} probability spike at {self.spike_location:.2f}. " + \
               f"True mean = {self.true_mean():.3f}."
    
    def suggest_importance_distribution(self) -> 'Distribution01':
        """For spike distribution, create importance distribution that samples spike more efficiently"""
        if self.spike_prob >= 0.05:  # Only worthwhile for significant spikes
            # Create a distribution that puts more mass near the spike
            if self.spike_location > 0.7:
                # High spike: use left-skewed Beta to oversample upper region
                return BetaSkewed(alpha=2.0, beta=1.5)
            elif self.spike_location < 0.3:
                # Low spike: use right-skewed Beta to oversample lower region
                return BetaSkewed(alpha=1.5, beta=2.0)
            else:
                # Middle spike: uniform is reasonable
                return BetaSkewed(alpha=1.0, beta=1.0)
        else:
            return self  # Spike too small to benefit


class GaussianMixture(Distribution01):
    """
    Mixture of Gaussians (truncated to [0,1])

    Example usage:
        # 3-component mixture
        mixture = GaussianMixture(
            means=[0.2, 0.5, 0.8],
            sigmas=[0.1, 0.15, 0.1],
            weights=[0.3, 0.4, 0.3]
        )

        # Sample 1000 points
        samples = mixture.sample(1000)

        # Get true mean
        print(mixture.true_mean())
    """

    def __init__(self, means: List[float], sigmas: List[float], weights: List[float]):
        """
        Mixture of Gaussian distributions, each truncated to [0,1].

        Args:
            means: List of means for each Gaussian component
            sigmas: List of standard deviations for each Gaussian component
            weights: List of mixture weights (must sum to 1)

        Example:
            # Create a 3-component mixture
            mixture = GaussianMixture(
                means=[0.2, 0.5, 0.8],
                sigmas=[0.1, 0.15, 0.1],
                weights=[0.3, 0.4, 0.3]
            )
        """
        if len(means) != len(sigmas) or len(means) != len(weights):
            raise ValueError("means, sigmas, and weights must have the same length")

        if len(means) == 0:
            raise ValueError("Must provide at least one Gaussian component")

        if not np.isclose(sum(weights), 1.0):
            raise ValueError(f"Weights must sum to 1, got {sum(weights)}")

        if any(w < 0 for w in weights):
            raise ValueError("All weights must be non-negative")

        if any(s <= 0 for s in sigmas):
            raise ValueError("All sigmas must be positive")

        self.means = np.array(means)
        self.sigmas = np.array(sigmas)
        self.weights = np.array(weights)
        self.n_components = len(means)

        # Create truncated normal distributions for each component
        self._components = []
        for mu, sigma in zip(means, sigmas):
            a, b = (0 - mu) / sigma, (1 - mu) / sigma  # Standardized bounds
            self._components.append(stats.truncnorm(a, b, loc=mu, scale=sigma))

    def sample(self, n: int = 1) -> np.ndarray:
        """Sample from Gaussian mixture using Monte Carlo"""
        # Randomly choose which component for each sample based on weights
        component_indices = np.random.choice(
            self.n_components,
            size=n,
            p=self.weights
        )

        # Generate samples from chosen components
        samples = np.zeros(n)
        for i in range(self.n_components):
            mask = component_indices == i
            n_from_component = np.sum(mask)
            if n_from_component > 0:
                samples[mask] = self._components[i].rvs(size=n_from_component)

        return samples

    def true_mean(self) -> float:
        """E[X] = sum(weight_i * E[X_i]) for mixture"""
        component_means = np.array([comp.mean() for comp in self._components])
        return np.dot(self.weights, component_means)

    def true_variance(self) -> float:
        """Var[X] = E[Var[X|component]] + Var[E[X|component]]"""
        component_means = np.array([comp.mean() for comp in self._components])
        component_vars = np.array([comp.var() for comp in self._components])

        # E[Var[X|component]] = sum(weight_i * Var[X_i])
        expected_variance = np.dot(self.weights, component_vars)

        # Var[E[X|component]] = E[E[X|component]^2] - (E[E[X|component]])^2
        overall_mean = np.dot(self.weights, component_means)
        expected_mean_squared = np.dot(self.weights, component_means**2)
        variance_of_means = expected_mean_squared - overall_mean**2

        return expected_variance + variance_of_means

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """PDF for Gaussian mixture"""
        pdf_vals = np.zeros_like(x, dtype=float)
        for i in range(self.n_components):
            pdf_vals += self.weights[i] * self._components[i].pdf(x)
        return pdf_vals

    @property
    def name(self) -> str:
        return f"GaussianMixture({self.n_components} components)"

    @property
    def description(self) -> str:
        components_str = ", ".join([
            f"N({m:.2f},{s:.2f})×{w:.2f}"
            for m, s, w in zip(self.means, self.sigmas, self.weights)
        ])
        return f"Mixture of {self.n_components} truncated Gaussians: {components_str}. " + \
               f"True mean = {self.true_mean():.3f}."

    def with_adjusted_variance(self, factor: float) -> 'GaussianMixture':
        """
        Create a new GaussianMixture with variance scaled by factor.

        Args:
            factor: Multiplicative factor for VARIANCE (not std dev)
                   factor > 1: Increase variance (wider distribution)
                   factor < 1: Decrease variance (narrower distribution)

        Returns:
            New GaussianMixture instance with adjusted sigmas

        Example:
            wider_sim = sim_dist.with_adjusted_variance(1.2)  # 20% more variance
            narrower_sim = sim_dist.with_adjusted_variance(0.8)  # 20% less variance
        """
        # Apply sqrt since variance = sigma², so sigma_new = sigma * sqrt(factor)
        new_sigmas = self.sigmas * np.sqrt(factor)
        return GaussianMixture(
            means=self.means.tolist(),
            sigmas=new_sigmas.tolist(),
            weights=self.weights.tolist()
        )

    def suggest_importance_distribution(self) -> 'Distribution01':
        """
        Suggest importance distribution for Gaussian mixture.
        For multimodal distributions, uniform often works well.
        """
        if self.n_components > 1:
            # For multimodal, use uniform to sample all modes
            return BetaSkewed(alpha=1.0, beta=1.0)
        else:
            # Single component, no importance sampling needed
            return self
