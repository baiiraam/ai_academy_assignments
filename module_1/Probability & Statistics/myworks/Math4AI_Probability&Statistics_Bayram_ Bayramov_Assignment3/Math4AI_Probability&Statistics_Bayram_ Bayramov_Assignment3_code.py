"""
Math4AI: Probability & Statistics
Assignment 3: Moments, Stability, and Correlation
Completed Code
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # For Heatmap

np.random.seed(42)

# ==========================================
# 1. Law of Large Numbers (Geometric Dist)
# ==========================================

class MomentEstimator:
    @staticmethod
    def geometric_theoretical_stats(p):
        """
        Returns theoretical Mean and Variance for Geometric distribution.
        """
        # Implement formulas for Mean (1/p) and Variance ((1-p)/p^2)
        mu = 1.0 / p
        var = (1 - p) / (p ** 2)

        return mu, var

    @staticmethod
    def analyze_convergence(data):
        """
        Computes running sample means to visualize convergence.
        Returns a list/array of means where index i is mean of data[0...i].
        """
        # Using np.cumsum() for running totals
        cumsum = np.cumsum(data)
        indices = np.arange(1, len(data) + 1)
        running_means = cumsum / indices

        return running_means

def run_section_1():
    print("--- Section 1: Law of Large Numbers ---")
    p = 0.2
    N = 10000

    # 1. Theoretical
    mu_theory, var_theory = MomentEstimator.geometric_theoretical_stats(p)

    # 2. Simulation (Generate N samples)
    # NumPy's geometric is usually number of trials to success
    data = np.random.geometric(p, N)

    # 3. Convergence
    running_means = MomentEstimator.analyze_convergence(data)
    final_mean = running_means[-1]

    print(f"Theoretical Mean: {mu_theory:.4f}")
    print(f"Theoretical Var:  {var_theory:.4f}")
    print(f"Final Sample Mean (N={N}): {final_mean:.4f}")

    # 4. Error Analysis
    ns = np.arange(1, N + 1)
    errors = np.abs(running_means - mu_theory)
    expected_error_scale = np.sqrt(var_theory) / np.sqrt(ns) # SE = sigma / sqrt(n)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot A: Convergence
    ax1.plot(ns, running_means, label='Running Sample Mean', color='blue', alpha=0.8)
    ax1.axhline(mu_theory, color='red', linestyle='--', label='Theoretical Mean')
    ax1.set_title("LLN Convergence: Geometric(p=0.2)")
    ax1.set_xlabel("Sample Size (N)")
    ax1.set_ylabel("Mean Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot B: Error Rates (Log-Log)
    ax2.loglog(ns, errors, label='Absolute Error', alpha=0.5)
    ax2.loglog(ns, expected_error_scale, 'r--', label=r'Theory $1/\sqrt{N}$')
    ax2.set_title("Standard Error Decay")
    ax2.set_xlabel("Sample Size (N)")
    ax2.set_ylabel("Absolute Error")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig("lln_convergence.png", dpi=300, bbox_inches='tight')
    plt.show()

    return mu_theory, var_theory, final_mean

# ==========================================
# 2. Higher-Order Moments (Robustness)
# ==========================================

def compute_central_moment(data, k):
    """
    Computes the k-th central moment: E[(X - mu)^k] / sigma^k
    """
    # Calculate sample mean and std (use ddof=1 for std)
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    # Standardize the data: z = (x - mean) / std
    z = (data - mean) / std

    # Compute mean of z^k
    moment = np.mean(z ** k)

    return moment

def run_section_2():
    print("\n--- Section 2: Skewness & Kurtosis ---")
    N = 5000

    # Dataset A: Normal
    data_a = np.random.normal(0, 1, N)

    # Dataset B: Student-t (Heavy tails)
    # df=3 implies finite variance but infinite skew/kurtosis theoretically,
    # but we will measure sample stats.
    data_b = np.random.standard_t(df=3, size=N)

    # Compute Moments
    skew_a = compute_central_moment(data_a, 3)
    kurt_a = compute_central_moment(data_a, 4)

    skew_b = compute_central_moment(data_b, 3)
    kurt_b = compute_central_moment(data_b, 4)

    print(f"{'Statistic':<15} | {'Normal':<10} | {'Student-t':<10}")
    print("-" * 45)
    print(f"{'Skewness':<15} | {skew_a:10.4f} | {skew_b:10.4f}")
    print(f"{'Kurtosis':<15} | {kurt_a:10.4f} | {kurt_b:10.4f}")

    return skew_a, kurt_a, skew_b, kurt_b

# ==========================================
# 3. Covariance & Matrix Algebra
# ==========================================

class MultivariateEstimator:
    @staticmethod
    def covariance_matrix(X):
        """
        Computes Covariance Matrix of X (shape N x D) using Matrix Algebra.
        Must return shape (D, D).
        """
        N, D = X.shape

        # Compute column means (shape D,)
        means = np.mean(X, axis=0)

        # Center the data (X_centered = X - means)
        X_centered = X - means

        # Compute (X_c.T @ X_c) / (N - 1)
        Sigma = (X_centered.T @ X_centered) / (N - 1)

        return Sigma

    @staticmethod
    def correlation_matrix(Sigma):
        """
        Converts Covariance Matrix Sigma to Correlation Matrix.
        Corr_ij = Sig_ij / (std_i * std_j)
        """
        # Extract variances (diagonal) and compute std devs
        variances = np.diag(Sigma)
        std = np.sqrt(variances)

        # Compute Outer Product of standard deviations
        outer_std = np.outer(std, std)

        # Element-wise division
        Corr = Sigma / outer_std

        return Corr

def run_section_3():
    print("\n--- Section 3: Multivariate Analysis ---")

    # Generate Synthetic Correlated Data
    # X1 = random
    # X2 = 2*X1 + noise (Strongly correlated)
    # X3 = random (Independent)
    N = 1000
    x1 = np.random.randn(N)
    x2 = 2 * x1 + np.random.randn(N) * 0.5
    x3 = np.random.randn(N)

    # Stack into Matrix X (N x 3)
    X = np.column_stack([x1, x2, x3])

    # 1. Compute Covariance
    Sigma = MultivariateEstimator.covariance_matrix(X)
    print("Covariance Matrix:")
    print(np.round(Sigma, 4))

    # 2. Compute Correlation
    Corr = MultivariateEstimator.correlation_matrix(Sigma)
    print("\nCorrelation Matrix:")
    print(np.round(Corr, 4))

    # 3. Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(Corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                xticklabels=['X1', 'X2', 'X3'], yticklabels=['X1', 'X2', 'X3'])
    plt.title("Correlation Matrix Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

    return Sigma, Corr

if __name__ == "__main__":
    print("Math4AI: Probability & Statistics - Assignment 3")
    print("=" * 60)

    mu_theory, var_theory, final_mean = run_section_1()
    skew_a, kurt_a, skew_b, kurt_b = run_section_2()
    Sigma, Corr = run_section_3()