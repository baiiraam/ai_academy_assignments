"""
Math4AI: Probability & Statistics — Assignment 4
Bayesian Inference & Networks
"""

from __future__ import annotations

import math
import itertools
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt

# Output directory
OUT_DIR = Path.cwd()

# ================================================================
# Task 4.1: Beta-Binomial Model
# ================================================================


def _log_beta(a: float, b: float) -> float:
    """log(Beta(a,b)) using log-gamma for numerical stability."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_pdf(theta: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Compute Beta(a,b) PDF on an array of theta values in (0,1).
    Uses a numerically stable log-space computation.
    """
    theta = np.asarray(theta, dtype=float)
    eps = 1e-12
    t = np.clip(theta, eps, 1 - eps)
    logp = (a - 1) * np.log(t) + (b - 1) * np.log(1 - t) - _log_beta(a, b)
    return np.exp(logp)


@dataclass
class BetaBinomialModel:
    """
    Maintains a Beta(alpha, beta) belief about theta = P(Heads).
    """

    alpha: float
    beta: float

    def update(self, heads: int, tails: int) -> None:
        """
        Perform the conjugate Bayesian update.
        """
        self.alpha += heads
        self.beta += tails

    def mean(self) -> float:
        """Posterior mean E[theta] = alpha / (alpha + beta)."""
        return float(self.alpha / (self.alpha + self.beta))

    def map(self) -> float:
        """
        MAP estimate for Beta(alpha,beta):
            (alpha-1)/(alpha+beta-2)  when alpha>1 and beta>1.
        If not defined (alpha<=1 or beta<=1), return the mean.
        """
        if self.alpha > 1 and self.beta > 1:
            return float((self.alpha - 1) / (self.alpha + self.beta - 2))
        return self.mean()

    def posterior_predictive(self, n_future: int, k_future: int) -> float:
        """
        Bonus Question 1: Compute P(exactly k_future heads in n_future flips)
        using the posterior predictive distribution.

        For Beta-Binomial model, this is the Beta-Binomial distribution:
        P(k|n, α, β) = C(n,k) * B(α+k, β+n-k) / B(α, β)
        where B is the Beta function.
        """
        # Using the formula: P(k|n) = C(n,k) * B(α+k, β+n-k) / B(α, β)
        log_combination = (
            math.lgamma(n_future + 1)
            - math.lgamma(k_future + 1)
            - math.lgamma(n_future - k_future + 1)
        )
        log_beta_num = _log_beta(self.alpha + k_future, self.beta + n_future - k_future)
        log_beta_denom = _log_beta(self.alpha, self.beta)

        log_prob = log_combination + log_beta_num - log_beta_denom
        return math.exp(log_prob)


def simulate_coin_flips(
    theta_true: float, n_flips: int, rng: np.random.Generator
) -> Tuple[int, int]:
    """Return (heads, tails) from n_flips Bernoulli trials."""
    flips = rng.random(n_flips) < theta_true
    heads = int(np.sum(flips))
    tails = int(n_flips - heads)
    return heads, tails


def plot_belief_evolution(
    thetas: np.ndarray, snapshots: Dict[str, Tuple[float, float]], filename: str
) -> None:
    """
    Plot PDFs for (alpha,beta) snapshots. 'snapshots' maps label -> (alpha,beta).
    """
    plt.figure(figsize=(12, 8))

    # Use a color map for better visualization
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(snapshots)))

    for i, (label, (a, b)) in enumerate(snapshots.items()):
        plt.plot(
            thetas, beta_pdf(thetas, a, b), label=label, color=colors[i], linewidth=2
        )

    # Add vertical line at true theta value
    plt.axvline(x=0.8, color="red", linestyle="--", alpha=0.7, label=f"True θ = 0.8")

    plt.xlabel(r"$\theta$ (Probability of Heads)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Bayesian Update: Beta Prior -> Beta Posterior", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    print(f"Plot saved as: {filename}")

    # Display the plot
    plt.show()
    plt.close()


# ================================================================
# Task 4.2: MLE vs MAP
# ================================================================


def compute_estimates(
    heads: int, tails: int, alpha_prior: float, beta_prior: float
) -> Dict[str, float]:
    """
    Return MLE and MAP estimates.
    """
    total_flips = heads + tails

    # MLE estimate
    mle = heads / total_flips if total_flips > 0 else 0.0

    # MAP estimate with fallback to posterior mean
    numerator = alpha_prior + heads - 1
    denominator = alpha_prior + beta_prior + heads + tails - 2

    if denominator > 0 and numerator > 0 and numerator < denominator:
        map_estimate = numerator / denominator
    else:
        # Fall back to posterior mean
        map_estimate = (alpha_prior + heads) / (
            alpha_prior + beta_prior + heads + tails
        )

    return {"mle": mle, "map": map_estimate}


# ================================================================
# Task 4.3: Bayesian Network (Wet Grass)
# ================================================================

Bool = bool
Assignment = Tuple[bool, bool, bool]  # (R, S, W)


class SimpleBayesNet:
    """
    A tiny Bayes Net for Wet Grass with variables:
      R (Rain), S (Sprinkler), W (WetGrass)
    """

    def __init__(self):
        # Priors / CPTs (given by the assignment)
        self.p_rain_true = 0.2

        # P(S=True | R)
        self.p_s_true_given_r = {True: 0.01, False: 0.4}

        # P(W=True | S, R)
        self.p_w_true_given_s_r = {
            (True, True): 0.99,
            (True, False): 0.90,
            (False, True): 0.80,
            (False, False): 0.0,
        }

        self.joint: Dict[Assignment, float] = {}

    def p_r(self, r: bool) -> float:
        return self.p_rain_true if r else 1.0 - self.p_rain_true

    def p_s_given_r(self, s: bool, r: bool) -> float:
        p_true = self.p_s_true_given_r[r]
        return p_true if s else 1.0 - p_true

    def p_w_given_s_r(self, w: bool, s: bool, r: bool) -> float:
        p_true = self.p_w_true_given_s_r[(s, r)]
        return p_true if w else 1.0 - p_true

    def compute_joint_distribution(self) -> Dict[Assignment, float]:
        """
        Fill self.joint with entries for all (R,S,W) assignments:
        P(R,S,W) = P(R) * P(S|R) * P(W|S,R)
        """
        for r in [False, True]:
            for s in [False, True]:
                for w in [False, True]:
                    prob = (
                        self.p_r(r)
                        * self.p_s_given_r(s, r)
                        * self.p_w_given_s_r(w, s, r)
                    )
                    self.joint[(r, s, w)] = prob

        # Verify probabilities sum to 1
        total = sum(self.joint.values())
        assert abs(total - 1.0) < 1e-10, f"Joint probabilities sum to {total}, not 1.0"

        return self.joint

    def query(self, query_var: str, evidence: Dict[str, bool]) -> float:
        """
        Compute P(query_var=True | evidence).
        """
        # Map variable names to indices in the assignment tuple
        var_to_idx = {"R": 0, "S": 1, "W": 2}

        # Get the index of the query variable
        query_idx = var_to_idx[query_var]

        # Create a list of evidence conditions
        evidence_conditions = []
        for var_name, var_value in evidence.items():
            evidence_conditions.append((var_to_idx[var_name], var_value))

        # Sum over all assignments
        prob_query_true_and_evidence = 0.0
        prob_evidence = 0.0

        for assignment, prob in self.joint.items():
            # Check if assignment satisfies evidence
            satisfies_evidence = True
            for idx, val in evidence_conditions:
                if assignment[idx] != val:
                    satisfies_evidence = False
                    break

            if satisfies_evidence:
                prob_evidence += prob
                # Check if query variable is True
                if assignment[query_idx]:
                    prob_query_true_and_evidence += prob

        # Handle case where evidence has zero probability
        if prob_evidence == 0:
            return 0.0

        return prob_query_true_and_evidence / prob_evidence


# ================================================================
# Main (runs all tasks and produces required outputs)
# ================================================================


def main() -> None:
    print("=" * 60)
    print("Assignment 4: Bayesian Inference & Networks")
    print("=" * 60)

    # ----------------------------
    # Task 4.1: belief evolution
    # ----------------------------
    print("\n--- Task 4.1: Bayesian Update Cycle ---")
    rng = np.random.default_rng(0)  # Fixed seed for reproducibility
    theta_true = 0.8
    model = BetaBinomialModel(alpha=1.0, beta=1.0)

    thetas = np.linspace(0.001, 0.999, 600)
    snapshots = {"Prior (0 flips)": (model.alpha, model.beta)}

    total_heads = 0
    total_tails = 0

    for step in range(1, 11):  # 10 batches x 50 = 500 flips
        heads, tails = simulate_coin_flips(theta_true, 50, rng)
        total_heads += heads
        total_tails += tails

        model.update(heads, tails)

        snapshots[f"Posterior ({50*step} flips)"] = (model.alpha, model.beta)

    plot_belief_evolution(
        thetas, snapshots, filename=str(OUT_DIR / "bayesian_update.png")
    )
    print(
        f"Final posterior alpha={model.alpha:.1f}, beta={model.beta:.1f}, mean={model.mean():.4f}, MAP={model.map():.4f}"
    )
    print(f"Total heads={total_heads}, total tails={total_tails}")

    # ----------------------------
    # Task 4.2: MLE vs MAP
    # ----------------------------
    print("\n--- Task 4.2: MLE vs MAP ---")
    est = compute_estimates(heads=5, tails=0, alpha_prior=10.0, beta_prior=10.0)
    print("Estimates for 5 Heads, 0 Tails with prior Beta(10,10):")
    print(f"  MLE: {est['mle']:.4f}")
    print(f"  MAP: {est['map']:.4f}")

    # ----------------------------
    # Task 4.3: Bayesian Network Inference
    # ----------------------------
    print("\n--- Task 4.3: Bayesian Network Inference ---")
    bn = SimpleBayesNet()
    bn.compute_joint_distribution()
    p_r_given_w = bn.query("R", {"W": True})
    print(f"P(R=True | W=True) = {p_r_given_w:.6f}")

    # Top-3 joint states
    top3 = sorted(bn.joint.items(), key=lambda kv: kv[1], reverse=True)[:3]
    print("\nTop-3 joint assignments (R,S,W):")
    for (r, s, w), p in top3:
        r_str = "T" if r else "F"
        s_str = "T" if s else "F"
        w_str = "T" if w else "F"
        print(f"  ({r_str}, {s_str}, {w_str}) -> {p:.6f}")

    # ----------------------------
    # Bonus Question 1: Posterior Predictive
    # ----------------------------
    print("\n" + "=" * 60)
    print("BONUS: Posterior Predictive Distribution")
    print("=" * 60)

    n_future = 10
    k_future = 7
    pred_prob = model.posterior_predictive(n_future, k_future)

    print(f"\nUsing final posterior Beta({model.alpha:.1f}, {model.beta:.1f})")
    print(f"P(exactly {k_future} heads in next {n_future} flips) = {pred_prob:.6f}")

    # Full predictive distribution
    print("\nFull predictive distribution for next 10 flips:")
    print("k\tP(K=k)")
    print("-" * 20)
    predictive_probs = []
    for k in range(11):
        prob = model.posterior_predictive(10, k)
        predictive_probs.append(prob)
        print(f"{k}\t{prob:.6f}")

    # Plot predictive distribution
    plt.figure(figsize=(10, 6))
    k_values = list(range(11))
    plt.bar(k_values, predictive_probs, alpha=0.7, color="skyblue", edgecolor="navy")
    plt.xlabel("Number of Heads (k)", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.title("Posterior Predictive Distribution (after 500 flips)", fontsize=14)
    plt.xticks(k_values)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    pred_plot_path = OUT_DIR / "posterior_predictive.png"
    plt.savefig(pred_plot_path, dpi=200)
    plt.show()
    plt.close()
    print(f"\nPredictive plot saved: {pred_plot_path}")

    # Compare with MLE plug-in
    mle_theta = total_heads / (total_heads + total_tails)
    mle_pred_prob = math.comb(10, 7) * (mle_theta**7) * ((1 - mle_theta) ** 3)
    print(f"\nComparison with MLE plug-in (θ={mle_theta:.4f}):")
    print(f"  MLE plug-in: P(K=7) = {mle_pred_prob:.6f}")
    print(f"  Bayesian:    P(K=7) = {pred_prob:.6f}")
    print("\nThe Bayesian predictive distribution accounts for parameter uncertainty,")
    print("while MLE plug-in ignores this uncertainty.")


if __name__ == "__main__":
    main()
