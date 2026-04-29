"""
Math4AI: Probability & Statistics — Assignment 5
Hypothesis Testing & A/B Testing

Complete implementation with Bonus 1 (Multiple Comparisons)
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Save all generated figures next to this file (robust to different working directories)
try:
    # Try to get the directory of the current script
    OUTPUT_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback for interactive environments (Jupyter, IPython)
    OUTPUT_DIR = Path.cwd()


# You may use scipy.stats for CDF/PPF/SF utilities (recommended).
try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


# ============================================================
# Provided Data (DO NOT CHANGE)
# ============================================================

# Task 5.1 example dataset for z-test (e.g., latency measurements in ms)
Z_TEST_DATA = np.array(
    [
        101.2,
        98.7,
        103.5,
        99.1,
        100.4,
        102.0,
        97.9,
        101.8,
        99.9,
        100.7,
        98.4,
        101.1,
        102.9,
        99.3,
        100.2,
        101.5,
        98.8,
        100.0,
        99.6,
        102.2,
    ],
    dtype=float,
)
MU0_Z = 100.0
SIGMA_Z = 15.0  # assumed known population sigma

# Task 5.2: model accuracies across random seeds (A/B test)
MODEL_A_ACCURACIES = np.array(
    [
        0.812,
        0.805,
        0.809,
        0.814,
        0.803,
        0.811,
        0.807,
        0.810,
        0.808,
        0.806,
        0.813,
        0.804,
        0.809,
        0.807,
        0.812,
        0.806,
        0.810,
        0.808,
        0.805,
        0.811,
    ],
    dtype=float,
)

MODEL_B_ACCURACIES = np.array(
    [
        0.821,
        0.816,
        0.819,
        0.823,
        0.815,
        0.822,
        0.818,
        0.820,
        0.817,
        0.816,
        0.824,
        0.814,
        0.820,
        0.818,
        0.823,
        0.816,
        0.821,
        0.819,
        0.815,
        0.822,
    ],
    dtype=float,
)

# Task 5.3: contingency table (rows = user segment, cols = churn status)
# Columns: [Retained, Churned]
OBSERVED_CONTINGENCY = np.array(
    [
        [50, 30],  # Segment 1
        [45, 55],  # Segment 2
        [20, 60],  # Segment 3
    ],
    dtype=float,
)


# ============================================================
# Small distribution helpers (optional)
# ============================================================


def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    if stats is not None:
        return float(stats.norm.cdf(x))
    # Fallback using erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard normal inverse CDF (quantile)."""
    if stats is None:
        raise RuntimeError("scipy is required for norm.ppf in this assignment.")
    return float(stats.norm.ppf(p))


# ============================================================
# Task 5.1 — Z-test and Power
# ============================================================


def z_test(data: np.ndarray, mu_0: float, sigma: float) -> tuple[float, float]:
    """
    Two-sided z-test for mean with known population sigma.
    Returns: (z_statistic, p_value_two_sided)
    """
    n = len(data)
    sample_mean = np.mean(data)
    standard_error = sigma / np.sqrt(n)

    # Calculate z-statistic
    z_stat = (sample_mean - mu_0) / standard_error

    # Calculate two-sided p-value
    # p = 2 * P(Z > |z_stat|) under H0
    p_value = 2 * (1 - _norm_cdf(abs(z_stat)))

    return z_stat, p_value


def compute_power(effect_size: float, alpha: float, n: int, sigma: float) -> float:
    """
    Compute power of a two-sided z-test with:
      H0: mu = mu0, H1: mu = mu0 + effect_size

    Under H1, the z-stat has mean:
      mean_z = effect_size / (sigma / sqrt(n))
    and variance 1.

    Returns power = P(reject H0 | H1 true).
    """
    # Critical values for two-sided test
    z_crit = _norm_ppf(1 - alpha / 2)

    # Mean of Z under H1
    mean_z = effect_size / (sigma / np.sqrt(n))

    # Power = P(Z <= -z_crit | H1) + P(Z >= z_crit | H1)
    # Under H1, Z ~ N(mean_z, 1)
    power_left = _norm_cdf(-z_crit - mean_z)
    power_right = 1 - _norm_cdf(z_crit - mean_z)

    return power_left + power_right


def plot_power_curve(
    effect_size: float,
    alpha: float,
    sigma: float,
    n_values: np.ndarray,
    filename: str = "power_curve.png",
) -> None:
    """
    Plot power vs. n and save to disk.
    """
    powers = []
    for n in n_values:
        powers.append(compute_power(effect_size, alpha, int(n), sigma))
    powers = np.array(powers, dtype=float)

    plt.figure(figsize=(9, 5))
    plt.plot(n_values, powers, linewidth=2)
    plt.axhline(0.8, linestyle="--", linewidth=1, color="red", label="80% power")

    # Find smallest n achieving 80% power
    idx_80 = np.where(powers >= 0.8)[0]
    if len(idx_80) > 0:
        n_80 = n_values[idx_80[0]]
        plt.axvline(n_80, linestyle=":", linewidth=1, color="green", alpha=0.7)
        plt.text(n_80 + 5, 0.5, f"n ≈ {n_80}", fontsize=10)

    plt.xlabel("Sample size n")
    plt.ylabel("Power (1 - beta)")
    plt.title(f"Power Curve (Two-sided z-test, δ={effect_size}, σ={sigma}, α={alpha})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


# ============================================================
# Task 5.2 — Welch's t-test and Confidence Interval
# ============================================================


def welch_t_test(data_a: np.ndarray, data_b: np.ndarray) -> tuple[float, float, float]:
    """
    Welch's two-sample t-test (unequal variances), two-sided.
    Returns: (t_statistic, dof, p_value_two_sided)
    """
    n_a = len(data_a)
    n_b = len(data_b)

    mean_a = np.mean(data_a)
    mean_b = np.mean(data_b)

    var_a = np.var(data_a, ddof=1)  # unbiased variance (sample variance)
    var_b = np.var(data_b, ddof=1)

    # Standard error for difference in means
    se = np.sqrt(var_a / n_a + var_b / n_b)

    # t-statistic
    t_stat = (mean_a - mean_b) / se

    # Degrees of freedom (Welch-Satterthwaite equation)
    numerator = (var_a / n_a + var_b / n_b) ** 2
    denominator = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    dof = numerator / denominator

    # Two-sided p-value
    if stats is not None:
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof))
    else:
        # Fallback approximation for large dof
        p_value = 2 * (1 - _norm_cdf(abs(t_stat)))

    return t_stat, dof, p_value


def compute_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> tuple[float, float, float]:
    """
    Two-sided confidence interval for the mean using t distribution.
    Returns: (mean, ci_low, ci_high)
    """
    n = len(data)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)

    # t critical value
    if stats is not None:
        t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
    else:
        # Normal approximation for large n
        t_crit = _norm_ppf((1 + confidence) / 2)

    ci_low = mean - t_crit * std_err
    ci_high = mean + t_crit * std_err

    return mean, ci_low, ci_high


# def plot_confidence_intervals(
#     mean_a: float, ci_a: tuple[float, float],
#     mean_b: float, ci_b: tuple[float, float],
#     filename: str = "confidence_intervals.png"
# ) -> None:
#     """
#     Error bar plot for two mean confidence intervals.
#     """
#     means = np.array([mean_a, mean_b], dtype=float)
#     lows = np.array([ci_a[0], ci_b[0]], dtype=float)
#     highs = np.array([ci_a[1], ci_b[1]], dtype=float)
#     yerr = np.vstack([means - lows, highs - means])

#     plt.figure(figsize=(7, 4))
#     plt.errorbar([0, 1], means, yerr=yerr, fmt="o", capsize=8, markersize=8)
#     plt.xticks([0, 1], ["Model A (Baseline)", "Model B (New)"])
#     plt.ylabel("Accuracy")
#     plt.title("95% Confidence Intervals for Mean Accuracy")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(OUTPUT_DIR / filename, dpi=200)
#     plt.close()


def plot_confidence_intervals(
    mean_a: float,
    ci_a: tuple[float, float],
    mean_b: float,
    ci_b: tuple[float, float],
    filename: str = "confidence_intervals.png",
) -> None:
    """
    Error bar plot for two mean confidence intervals with improved aesthetics.
    """
    # Data preparation
    models = ["Model A\n(Baseline)", "Model B\n(New)"]
    means = np.array([mean_a, mean_b], dtype=float)
    ci_lows = np.array([ci_a[0], ci_b[0]], dtype=float)
    ci_highs = np.array([ci_a[1], ci_b[1]], dtype=float)

    # Calculate error bar lengths
    yerr_lower = means - ci_lows
    yerr_upper = ci_highs - means
    np.array([yerr_lower, yerr_upper])

    # Create figure with better proportions
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define colors (using a professional color palette)
    colors = ["#2E86AB", "#A23B72"]  # Blue and purple

    # Plot means with error bars
    for i, (mean, low, high, color) in enumerate(zip(means, ci_lows, ci_highs, colors)):
        # Mean point
        ax.scatter(
            i,
            mean,
            color="white",
            edgecolor=color,
            s=200,
            linewidth=2.5,
            zorder=3,
            label="Mean" if i == 0 else "",
        )

        # Error bar
        ax.errorbar(
            i,
            mean,
            yerr=[[mean - low], [high - mean]],
            color=color,
            capsize=10,
            capthick=2.5,
            linewidth=2.5,
            zorder=2,
        )

    # Customize plot
    ax.set_xticks([0, 1])
    ax.set_xticklabels(models, fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")

    # Set y-axis limits with some padding
    y_min = min(ci_lows) - 0.01
    y_max = max(ci_highs) + 0.01
    ax.set_ylim(y_min, y_max)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, axis="y")
    ax.set_axisbelow(True)  # Grid behind data

    # Add title with statistical significance info
    t_stat, dof, p_val = welch_t_test(MODEL_A_ACCURACIES, MODEL_B_ACCURACIES)
    sig_text = (
        "Statistically Significant" if p_val < 0.05 else "Not Statistically Significant"
    )
    ax.set_title(
        f"95% Confidence Intervals for Mean Accuracy\n{p_val:.2e} ({sig_text})",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Add value labels near the means
    for i, (mean, low, high) in enumerate(zip(means, ci_lows, ci_highs)):
        ax.text(
            i,
            mean + 0.002,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            i,
            low - 0.002,
            f"{low:.4f}",
            ha="center",
            va="top",
            fontsize=8,
            color="gray",
        )
        ax.text(
            i,
            high + 0.002,
            f"{high:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="gray",
        )

    # Add difference annotation
    diff = mean_b - mean_a
    diff_text = f"Difference: {diff:.4f}"
    if p_val < 0.05:
        diff_text += " *"
    ax.annotate(
        diff_text,
        xy=(0.5, 0.95),
        xycoords="axes fraction",
        ha="center",
        fontsize=11,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.2),
    )

    # Add legend (custom handles)
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[0],
            markersize=10,
            label="Model A (Baseline)",
            markeredgecolor=colors[0],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[1],
            markersize=10,
            label="Model B (New)",
            markeredgecolor=colors[1],
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8,
            label="Mean",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9)

    # Add annotation for confidence interval interpretation
    ax.text(
        0.02,
        0.02,
        "95% CI: If we repeated this experiment 100 times,\n"
        "the true mean would fall in this interval ~95 times.",
        transform=ax.transAxes,
        fontsize=8,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3),
    )

    # Improve overall layout
    plt.tight_layout()

    # Save with high quality
    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()


# ============================================================
# Task 5.3 — Chi-square test (vectorized expected matrix)
# ============================================================


def chi_squared_test(observed_matrix: np.ndarray) -> tuple[float, int, float]:
    """
    Pearson chi-square test of independence.
    Returns: (chi2_statistic, dof, p_value)

    REQUIREMENT: compute E using np.outer(row_sums, col_sums) / N.
    """
    # Row sums and column sums
    row_sums = np.sum(observed_matrix, axis=1)
    col_sums = np.sum(observed_matrix, axis=0)
    N = np.sum(observed_matrix)

    # Expected matrix under independence (vectorized with outer product)
    expected_matrix = np.outer(row_sums, col_sums) / N

    # Chi-square statistic
    chi2_stat = np.sum((observed_matrix - expected_matrix) ** 2 / expected_matrix)

    # Degrees of freedom
    dof = (len(row_sums) - 1) * (len(col_sums) - 1)

    # p-value from chi-square distribution
    if stats is not None:
        p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
    else:
        # Rough approximation (not recommended)
        p_value = np.exp(-chi2_stat / 2) if chi2_stat > 0 else 1.0

    return chi2_stat, dof, p_value


# ============================================================
# Bonus 1: Multiple Comparisons with Holm-Bonferroni Correction
# ============================================================


def holm_bonferroni_correction(
    p_values: list[float], alpha: float = 0.05
) -> tuple[list[bool], list[float]]:
    """
    Apply Holm-Bonferroni correction for multiple hypothesis testing.

    Parameters:
    - p_values: list of raw p-values from multiple tests
    - alpha: family-wise error rate

    Returns:
    - rejected: boolean list indicating which null hypotheses are rejected
    - adjusted_p_values: Holm-adjusted p-values
    """
    m = len(p_values)  # number of hypotheses

    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Holm-Bonferroni procedure
    rejected = [False] * m
    adjusted_p = [0.0] * m

    # Compute Holm-adjusted p-values
    for i in range(m):
        # Adjusted p-value for the i-th smallest p-value
        adjusted_p[sorted_indices[i]] = min(1.0, sorted_p[i] * (m - i))

        # For Holm procedure, we compare sequentially
        if i == 0:
            # First (smallest) p-value
            if sorted_p[i] <= alpha / m:
                rejected[sorted_indices[i]] = True
            else:
                break  # If smallest isn't significant, none are
        else:
            # Subsequent p-values - need ALL previous to be significant
            if all(rejected[sorted_indices[j]] for j in range(i)):
                if sorted_p[i] <= alpha / (m - i):
                    rejected[sorted_indices[i]] = True
                else:
                    break  # Stop when we fail to reject

    # Alternative: compute adjusted p-values and compare to alpha directly
    # This is simpler: reject if adjusted_p[i] <= alpha
    rejected_simple = [adj <= alpha for adj in adjusted_p]

    return rejected_simple, adjusted_p


def bonus_multiple_comparisons_example():
    """
    Demonstrate the multiple comparisons problem and Holm-Bonferroni correction.
    """
    print("\n\n\n\n\nBONUS 1: Multiple Comparisons with Holm-Bonferroni Correction")

    # Simulate comparing 3 model variants against baseline
    # Using the provided data but creating synthetic variants for demonstration
    np.random.seed(42)  # For reproducibility

    baseline = MODEL_A_ACCURACIES

    # Create three variants with different effects
    # Variant 1: slightly better (should be significant)
    variant1 = baseline + np.random.normal(0.005, 0.001, size=len(baseline))

    # Variant 2: no improvement (should not be significant)
    variant2 = baseline + np.random.normal(0.0, 0.002, size=len(baseline))

    # Variant 3: slightly worse (might be significant depending on variance)
    variant3 = baseline + np.random.normal(-0.003, 0.0015, size=len(baseline))

    variants = [variant1, variant2, variant3]
    variant_names = ["Variant 1 (Better)", "Variant 2 (No effect)", "Variant 3 (Worse)"]

    # Perform t-tests for each variant vs baseline
    raw_p_values = []
    test_stats = []

    print("\nRaw comparisons:")
    print("-" * 50)

    for i, (variant, name) in enumerate(zip(variants, variant_names)):
        t_stat, dof, p_val = welch_t_test(baseline, variant)
        raw_p_values.append(p_val)
        test_stats.append(t_stat)

        significance = "SIGNIFICANT" if p_val < 0.05 else "not significant"
        print(f"{name}: \nt={t_stat:.4f}, p={p_val:.6f} ({significance})\n")

    # Apply Holm-Bonferroni correction
    alpha_family = 0.05
    rejected, adjusted_p = holm_bonferroni_correction(raw_p_values, alpha_family)

    print("\nAfter Holm-Bonferroni correction (family-wise α = 0.05):")
    print("-" * 50)

    for i, (name, raw_p, adj_p, rej) in enumerate(
        zip(variant_names, raw_p_values, adjusted_p, rejected)
    ):
        print(
            f"{name}: \nraw p={raw_p:.6f}, adjusted p={adj_p:.6f}, "
            + f"Reject H0: {rej}\n"
        )

    # Explanation
    print("\n" + "=" * 50)
    print("EXPLANATION: Why multiple comparisons inflate Type I error")
    print("=" * 50)
    print("""
When conducting multiple hypothesis tests simultaneously, the probability of
making at least one Type I error (false positive) increases.

Family-wise Error Rate (FWER) = 1 - (1 - α)^m ≈ m·α for small α

Where:
- α = significance level per test (e.g., 0.05)
- m = number of tests

For m = 3 tests:
FWER = 1 - (1 - 0.05)^3 = 1 - 0.857 = 0.143 (14.3% chance of at least one false positive)

The Holm-Bonferroni correction controls FWER by adjusting the significance
threshold for each test:
- Sort p-values: p₁ ≤ p₂ ≤ ... ≤ pₘ
- Reject H₁ if p₁ ≤ α/m
- Reject H₂ if p₂ ≤ α/(m-1) (and H₁ rejected)
- Reject H₃ if p₃ ≤ α/(m-2) (and H₁, H₂ rejected)
etc.

This is more powerful than the classic Bonferroni correction (which uses α/m
for all tests) while still maintaining FWER control.
    """)

    return raw_p_values, adjusted_p, rejected


# ============================================================
# Main (runs tasks + saves figures)
# ============================================================


def main() -> None:
    # ---------------- Task 5.1 ----------------
    z, pz = z_test(Z_TEST_DATA, MU0_Z, SIGMA_Z)
    print("Task 5.1 — Z-test")
    print("  z-stat:", z)
    print("  p-val :", pz)

    # Check significance at alpha=0.05
    if pz < 0.05:
        print("  → Reject H0: Mean latency differs from 100ms")
    else:
        print("  → Fail to reject H0: No evidence mean latency differs from 100ms")

    n_values = np.arange(5, 301, 5)
    plot_power_curve(
        effect_size=5.0,
        alpha=0.05,
        sigma=15.0,
        n_values=n_values,
        filename="power_curve.png",
    )
    print("  saved: power_curve.png")

    # ---------------- Task 5.2 ----------------
    t, dof, pt = welch_t_test(MODEL_A_ACCURACIES, MODEL_B_ACCURACIES)
    mean_a, lo_a, hi_a = compute_confidence_interval(
        MODEL_A_ACCURACIES, confidence=0.95
    )
    mean_b, lo_b, hi_b = compute_confidence_interval(
        MODEL_B_ACCURACIES, confidence=0.95
    )

    print("\nTask 5.2 — Welch's t-test")
    print("  mean(A):", mean_a, "CI:", (lo_a, hi_a))
    print("  mean(B):", mean_b, "CI:", (lo_b, hi_b))
    print("  t-stat :", t)
    print("  dof    :", dof)
    print("  p-val  :", pt)

    # Interpretation
    print("\n  Interpretation:")
    if pt < 0.05:
        print("  → Reject H0: There is a statistically significant difference")
        print("    between Model A and Model B accuracies.")
        print("    'Statistically significant' means the observed difference")
        print("    is unlikely to have occurred by random chance alone.")
    else:
        print("  → Fail to reject H0: No statistically significant difference")
        print("    between Model A and Model B accuracies.")

    print(
        "\n\nNote: Statistical significance does NOT imply practical/clinical significance (the effect may be small), causation (randomized experiment helps but doesn't guarantee) , and the fact that the result will replicate (p-values are sample-dependent)"
    )

    plot_confidence_intervals(
        mean_a, (lo_a, hi_a), mean_b, (lo_b, hi_b), filename="confidence_intervals.png"
    )
    print("  saved: confidence_intervals.png")

    # ---------------- Task 5.3 ----------------
    chi2, dof_c, pchi = chi_squared_test(OBSERVED_CONTINGENCY)
    print("\nTask 5.3 — Chi-square test of independence")
    print("  chi2  :", chi2)
    print("  dof   :", dof_c)
    print("  p-val :", pchi)

    if pchi < 0.05:
        print("  → Reject H0: Churn is dependent on user segment")
        print("    (variables are NOT independent)")
    else:
        print("  → Fail to reject H0: No evidence of dependence")
        print("    between user segment and churn")

    # ---------------- Bonus 1 ----------------
    bonus_multiple_comparisons_example()


if __name__ == "__main__":
    main()
