"""
Math4AI: Probability & Statistics
Assignment 2: Continuous Variables & Distributions
Completed Code
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.stats import binom  # For verification only
import math  # Import math for factorial

np.random.seed(42)

# ==========================================
# 1. Normal Distribution & Anomalies
# ==========================================


def gaussian_pdf(x, mu, sigma):
    """
    Implementation of univariate Gaussian PDF.
    """
    # Implement 1/(sigma * sqrt(2pi)) * exp(...)
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian_cdf(x, mu, sigma):
    """
    Implementation of CDF using Error Function.
    """
    # Implement 0.5 * (1 + erf(...))
    z = (x - mu) / (sigma * np.sqrt(2))
    return 0.5 * (1 + erf(z))


def run_section_1():
    print("--- Section 1: Anomaly Detection ---")
    # Generate synthetic server temp data
    data = np.concatenate(
        [
            np.random.normal(50, 5, 1000),  # Normal operations
            [25, 80, 78, 20],  # Anomalies
        ]
    )

    # Calculate MLE (Mean and Std) from data
    mu_hat = np.mean(data)
    sigma_hat = np.std(data)

    print(f"Estimated Mean: {mu_hat:.4f}")  # Record in Report
    print(f"Estimated Std Dev: {sigma_hat:.4f}")  # Record in Report

    # Detect Anomalies using 99% interval
    # (i.e., data points where CDF < 0.005 OR CDF > 0.995)
    anomalies = []
    for value in data:
        cdf_value = gaussian_cdf(value, mu_hat, sigma_hat)
        if cdf_value < 0.005 or cdf_value > 0.995:
            anomalies.append(value)

    print(f"Anomalies detected: {len(anomalies)}")
    print(f"Anomaly values: {anomalies}")

    # Plotting
    x_grid = np.linspace(min(data), max(data), 1000)
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, density=True, alpha=0.5, label="Data Hist")
    plt.plot(
        x_grid,
        gaussian_pdf(x_grid, mu_hat, sigma_hat),
        "k",
        linewidth=2,
        label="Fitted PDF",
    )

    # Plot red dots for anomalies
    if len(anomalies) > 0:
        plt.scatter(
            anomalies,
            np.zeros_like(anomalies),
            color="red",
            s=100,
            zorder=5,
            label=f"Anomalies (n={len(anomalies)})",
        )

    plt.title("Server Temperatures: Gaussian Fit & Anomaly Detection")
    plt.xlabel("Temperature")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("anomaly_detection.png", dpi=300, bbox_inches="tight")
    plt.show()

    return mu_hat, sigma_hat, anomalies


# ==========================================
# 2. Poisson vs Binomial Limit
# ==========================================


def poisson_pmf(k, lam):
    """
    Implementation of Poisson PMF.
    P(X=k) = (lam^k * e^-lam) / k!
    """
    # Implement formula.
    # Use math.factorial(k) from Python's math module
    return (lam**k * np.exp(-lam)) / math.factorial(k)


def run_section_2():
    print("\n--- Section 2: Poisson Limit Theorem ---")
    n = 10000
    p = 0.001
    lam = n * p
    target_k = 15

    print(f"Parameters: n={n}, p={p}, λ={lam}")

    # 1. Empirical (Simulation)
    # Simulate n coin flips, repeat 5000 times to get distribution of successes
    trials = 5000
    successes = np.random.binomial(n, p, trials)  # Using numpy's generator for speed
    empirical_prob = np.mean(successes == target_k)

    # 2. Exact Binomial (SciPy)
    exact_prob = binom.pmf(target_k, n, p)

    # 3. Poisson Approximation (Your Implementation)
    approx_prob = poisson_pmf(target_k, lam)

    print(f"\nFor k={target_k}:")
    print(f"  Empirical (Simulation): {empirical_prob:.6f}")
    print(f"  Exact Binomial:         {exact_prob:.6f}")
    print(f"  Poisson Approximation:  {approx_prob:.6f}")

    # Relative errors
    rel_error_empirical = abs(empirical_prob - exact_prob) / exact_prob * 100
    rel_error_poisson = abs(approx_prob - exact_prob) / exact_prob * 100
    print("\nRelative Errors:")
    print(f"  Empirical vs Exact: {rel_error_empirical:.2f}%")
    print(f"  Poisson vs Exact:   {rel_error_poisson:.2f}%")

    # Comparison Plot
    k_range = np.arange(0, 30)
    binom_probs = binom.pmf(k_range, n, p)
    poisson_probs = [poisson_pmf(k, lam) for k in k_range]

    plt.figure(figsize=(10, 6))
    plt.bar(
        k_range - 0.2,
        binom_probs,
        width=0.4,
        alpha=0.7,
        label="Binomial Exact",
        color="blue",
    )
    plt.bar(
        k_range + 0.2,
        poisson_probs,
        width=0.4,
        alpha=0.7,
        label="Poisson Approx",
        color="red",
    )
    plt.axvline(
        target_k, color="green", linestyle="--", alpha=0.5, label=f"k={target_k}"
    )

    plt.title(f"Law of Rare Events: Binomial vs Poisson (n={n}, p={p:.4f}, λ={lam})")
    plt.xlabel("Number of Clicks (k)")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.savefig("poisson_limit.png", dpi=300, bbox_inches="tight")
    plt.show()

    return empirical_prob, exact_prob, approx_prob


# ==========================================
# 3. Bivariate Distributions
# ==========================================


def bivariate_normal_pdf(x, y, mux, muy, sigx, sigy, rho):
    """
    Returns PDF value at (x,y)
    """
    # Implement Bivariate Gaussian Formula
    # z = ((x-mux)/sigx)^2 - 2*rho*... + ...
    # norm_const = 1 / (2 * pi * sigx * sigy * sqrt(1-rho^2))
    zx = (x - mux) / sigx
    zy = (y - muy) / sigy

    Q = (zx**2 - 2 * rho * zx * zy + zy**2) / (1 - rho**2)
    norm_const = 1 / (2 * np.pi * sigx * sigy * np.sqrt(1 - rho**2))

    return norm_const * np.exp(-Q / 2)


def run_section_3():
    print("\n--- Section 3: Bivariate Gaussian & Marginals ---")

    # Setup parameters
    mux, muy = 0, 0
    sigx, sigy = 1, 1
    rho = 0.7  # Correlation

    print(f"Parameters: μ_x={mux}, μ_y={muy}, σ_x={sigx}, σ_y={sigy}, ρ={rho}")

    # Create grid
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Calculate Joint PDF
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = bivariate_normal_pdf(x[i], y[j], mux, muy, sigx, sigy, rho)

    # Compute Marginal P(X) numerically
    # Sum over the Y axis (rows) and normalize
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    marginal_x = np.sum(Z, axis=0) * dy
    marginal_y = np.sum(Z, axis=1) * dx

    # Normalize to ensure area = 1
    marginal_x /= np.sum(marginal_x) * dx
    marginal_y /= np.sum(marginal_y) * dy

    # Theoretical univariate for comparison
    theory_x = gaussian_pdf(x, mux, sigx)
    theory_y = gaussian_pdf(y, muy, sigy)

    # Calculate RMSE between numeric and theoretical marginals
    rmse_x = np.sqrt(np.mean((marginal_x - theory_x) ** 2))
    rmse_y = np.sqrt(np.mean((marginal_y - theory_y) ** 2))

    print("\nMarginal Distribution Accuracy:")
    print(f"  RMSE for P(X): {rmse_x:.6f}")
    print(f"  RMSE for P(Y): {rmse_y:.6f}")

    # Plotting (Contour + Side Marginals)
    # Note: This is a complex plot, we have set it up for you.
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.05,
        hspace=0.05,
    )

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    # Contour
    ax.contourf(X, Y, Z, cmap="viridis", levels=20)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Bivariate Gaussian PDF (ρ={rho})")

    # Marginal X
    ax_histx.plot(x, theory_x, "k--", label="Theory")
    ax_histx.plot(x, marginal_x, "r-", label="Computed")
    ax_histx.set_title(f"Joint PDF (rho={rho})")
    ax_histx.legend(fontsize="small")
    ax_histx.grid(True, alpha=0.3)

    # Marginal Y
    ax_histy.plot(marginal_y, y, "r-", label="Computed")
    ax_histy.plot(theory_y, y, "k--", label="Theory")
    ax_histy.grid(True, alpha=0.3)
    ax_histy.legend(fontsize="small")

    plt.tight_layout()
    plt.savefig("bivariate_marginals.png", dpi=300, bbox_inches="tight")
    plt.show()

    return rmse_x, rmse_y


if __name__ == "__main__":
    print("Math4AI: Probability & Statistics - Assignment 2")
    print("=" * 60)

    mu_hat, sigma_hat, anomalies = run_section_1()
    empirical_prob, exact_prob, approx_prob = run_section_2()
    rmse_x, rmse_y = run_section_3()
