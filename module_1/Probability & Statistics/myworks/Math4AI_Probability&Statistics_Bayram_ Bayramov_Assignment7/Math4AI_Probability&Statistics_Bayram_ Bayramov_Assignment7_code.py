"""
starter_as7_probstats_diamond.py
Assignment 7: Markov Chains & Sampling Methods
Math4AI: Probability & Statistics
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import multivariate_normal

# =============================================================================
# Task 7.1: PageRank (Power Iteration)
# =============================================================================


def pagerank_power_iteration(adj_matrix, d=0.85, tol=1e-6, max_iter=1000):
    """
    Compute PageRank using power iteration.

    Parameters:
    -----------
    adj_matrix : np.ndarray
        Adjacency matrix of the graph (N x N)
    d : float
        Damping factor (probability of following a link)
    tol : float
        Tolerance for convergence
    max_iter : int
        Maximum number of iterations

    Returns:
    --------
    pi : np.ndarray
        Stationary distribution (PageRank scores)
    n_iter : int
        Number of iterations performed
    """
    N = adj_matrix.shape[0]

    # Construct transition matrix P from adjacency matrix
    # P[i,j] = probability of moving from i to j
    # For nodes with outgoing links, probability is evenly distributed
    # For dangling nodes (no outgoing links), probability is evenly distributed to all nodes

    # Calculate row sums (outgoing links)
    row_sums = adj_matrix.sum(axis=1)

    # Initialize transition matrix
    P = np.zeros_like(adj_matrix, dtype=float)

    # Handle nodes with outgoing links
    for i in range(N):
        if row_sums[i] > 0:
            P[i] = adj_matrix[i] / row_sums[i]
        else:
            # Dangling node: equal probability to all nodes
            P[i] = np.ones(N) / N

    # Initialize uniform distribution
    pi = np.ones(N) / N

    # Power iteration
    for n_iter in range(max_iter):
        pi_old = pi.copy()

        # Apply PageRank formula: pi_new = d * pi_old @ P + (1-d) * (1/N)
        pi = d * pi_old @ P + (1 - d) * (np.ones(N) / N)

        # Check convergence
        if np.linalg.norm(pi - pi_old, 1) < tol:
            break

    return pi, n_iter + 1


def create_mini_internet_graph():
    """
    Create the "Mini-Internet" graph described in the assignment.

    Returns:
    --------
    adj_matrix : np.ndarray
        Adjacency matrix
    G : networkx.Graph
        NetworkX graph object for visualization
    """
    G = nx.DiGraph()

    # Add nodes (websites)
    nodes = ["A", "B", "C", "D", "E"]
    G.add_nodes_from(nodes)

    # Add edges (links between websites)
    edges = [
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
        ("C", "A"),
        ("D", "C"),
        ("D", "E"),
        ("E", "D"),
    ]
    G.add_edges_from(edges)

    # Create adjacency matrix
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    N = len(nodes)
    adj_matrix = np.zeros((N, N))

    for edge in edges:
        i, j = node_to_idx[edge[0]], node_to_idx[edge[1]]
        adj_matrix[i, j] = 1

    return adj_matrix, G, nodes


def visualize_pagerank(G, pagerank_scores, nodes, filename="pagerank_graph.png"):
    """
    Visualize the graph with node sizes proportional to PageRank scores.
    """
    plt.figure(figsize=(10, 8))

    # Create position layout
    pos = nx.spring_layout(G, seed=42)

    # Normalize scores for node sizing
    scores_normalized = pagerank_scores / pagerank_scores.max() * 2000

    # Create node size dictionary
    node_sizes = {nodes[i]: scores_normalized[i] for i in range(len(nodes))}

    # Draw the graph
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=[node_sizes[node] for node in G.nodes()],
        node_color="lightblue",
        alpha=0.9,
    )
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20)

    plt.title(
        "PageRank Visualization - Node Size Proportional to Importance", fontsize=14
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Graph visualization saved as '{filename}'")


# =============================================================================
# Task 7.2: Metropolis-Hastings (MCMC)
# =============================================================================


def target_density_double_well(x):
    """
    Double well potential (mixture of two unnormalized Gaussians).
    f(x) ∝ exp(-(x-4)^2/2) + exp(-(x+4)^2/2)
    """
    return np.exp(-((x - 4) ** 2) / 2) + np.exp(-((x + 4) ** 2) / 2)


def metropolis_hastings(target_density, x0, n_samples, proposal_std=2.0):
    """
    Metropolis-Hastings algorithm with Gaussian proposal.

    Parameters:
    -----------
    target_density : function
        Unnormalized target density function
    x0 : float
        Initial state
    n_samples : int
        Number of samples to generate
    proposal_std : float
        Standard deviation of Gaussian proposal

    Returns:
    --------
    samples : np.ndarray
        Array of samples
    acceptance_rate : float
        Acceptance rate of proposals
    """
    samples = np.zeros(n_samples)
    samples[0] = x0

    n_accepted = 0

    for t in range(1, n_samples):
        # Current state
        x_current = samples[t - 1]

        # Propose new state from Gaussian centered at current state
        x_proposed = np.random.normal(x_current, proposal_std)

        # Compute acceptance probability
        # For symmetric proposal, acceptance ratio = target_density(x_proposed) / target_density(x_current)
        # But careful with numerical stability: compare log densities
        log_ratio = np.log(target_density(x_proposed) + 1e-10) - np.log(
            target_density(x_current) + 1e-10
        )
        acceptance_prob = min(1.0, np.exp(log_ratio))

        # Accept or reject
        if np.random.random() < acceptance_prob:
            samples[t] = x_proposed
            n_accepted += 1
        else:
            samples[t] = x_current

    acceptance_rate = n_accepted / (n_samples - 1)

    return samples, acceptance_rate


def plot_mh_results(samples, burnin=500, filename="mh_trace_hist.png"):
    """
    Plot trace plot and histogram for Metropolis-Hastings samples.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Trace plot
    axes[0].plot(samples, alpha=0.7, linewidth=0.5)
    axes[0].axvline(x=burnin, color="red", linestyle="--", label=f"Burn-in ({burnin})")
    axes[0].set_xlabel("Iteration", fontsize=12)
    axes[0].set_ylabel("Value", fontsize=12)
    axes[0].set_title("Trace Plot", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Histogram with true density overlay (after burn-in)
    samples_after_burnin = samples[burnin:]

    axes[1].hist(
        samples_after_burnin,
        bins=50,
        density=True,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
        label="Samples",
    )

    # True density (normalized for comparison)
    x_grid = np.linspace(-10, 10, 1000)
    true_density = target_density_double_well(x_grid)
    # Normalize to integrate to 1 (approximate)
    dx = x_grid[1] - x_grid[0]
    true_density_normalized = true_density / (np.sum(true_density) * dx)

    axes[1].plot(
        x_grid, true_density_normalized, "r-", linewidth=2, label="True density"
    )
    axes[1].set_xlabel("x", fontsize=12)
    axes[1].set_ylabel("Density", fontsize=12)
    axes[1].set_title("Histogram of Samples vs. True Density", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Metropolis-Hastings results saved as '{filename}'")

    return samples_after_burnin


# =============================================================================
# Task 7.3: Gibbs Sampling
# =============================================================================


def gibbs_sampler(rho, n_samples, x0=0, y0=0):
    """
    Gibbs sampler for bivariate Gaussian with correlation rho.

    Parameters:
    -----------
    rho : float
        Correlation coefficient
    n_samples : int
        Number of samples to generate
    x0, y0 : float
        Initial values

    Returns:
    --------
    samples : np.ndarray
        Array of shape (n_samples, 2) containing (x, y) samples
    """
    samples = np.zeros((n_samples, 2))
    samples[0] = [x0, y0]

    # Standard deviations for conditional distributions
    # P(x|y) = N(rho*y, 1-rho^2)
    # P(y|x) = N(rho*x, 1-rho^2)
    cond_std = np.sqrt(1 - rho**2)

    for t in range(1, n_samples):
        # Sample x_t given y_{t-1}
        x_mean = rho * samples[t - 1, 1]
        samples[t, 0] = np.random.normal(x_mean, cond_std)

        # Sample y_t given x_t
        y_mean = rho * samples[t, 0]
        samples[t, 1] = np.random.normal(y_mean, cond_std)

    return samples


def plot_gibbs_results(samples, rho, filename="gibbs_scatter.png"):
    """
    Scatter plot of Gibbs sampler samples.
    """
    plt.figure(figsize=(10, 8))

    # Scatter plot of samples
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10, c="steelblue")

    # Add contour lines of true distribution
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)

    # True bivariate Gaussian
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    rv = multivariate_normal(mean, cov)
    pos = np.dstack((X, Y))
    Z = rv.pdf(pos)

    plt.contour(X, Y, Z, levels=10, colors="red", alpha=0.5, linewidths=1)

    plt.xlabel("X", fontsize=14)
    plt.ylabel("Y", fontsize=14)
    plt.title(f"Gibbs Sampling: Bivariate Gaussian (ρ = {rho})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axis("equal")

    # Add correlation annotation
    plt.annotate(
        f"Correlation: {rho}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Gibbs sampling results saved as '{filename}'")


# =============================================================================
# Bonus B1: Gelman-Rubin Diagnostic (Optional)
# =============================================================================


def gelman_rubin(chains):
    """
    Compute Gelman-Rubin statistic R-hat for convergence diagnostics.

    Parameters:
    -----------
    chains : list of np.ndarray
        List of MCMC chains (each chain is array of samples)

    Returns:
    --------
    R_hat : float
        Potential scale reduction factor (should be close to 1 for convergence)
    """
    len(chains)  # number of chains
    n = len(chains[0])  # length of each chain (assumed equal)

    # Compute within-chain variance
    chain_means = np.array([np.mean(chain) for chain in chains])
    chain_vars = np.array([np.var(chain, ddof=1) for chain in chains])

    W = np.mean(chain_vars)  # within-chain variance

    # Compute between-chain variance
    np.mean(chain_means)
    B = n * np.var(chain_means, ddof=1)

    # Estimate target variance
    var_plus = (n - 1) / n * W + (1 / n) * B

    # Compute R-hat
    R_hat = np.sqrt(var_plus / W)

    return R_hat


def run_gelman_rubin_diagnostic():
    """
    Run multiple chains and compute R-hat over time.
    """
    print("\n" + "=" * 60)
    print("BONUS B1: Gelman-Rubin Convergence Diagnostic")
    print("=" * 60)

    n_samples = 5000
    starting_points = [-5, -2, 2, 5]

    chains = []
    for i, start in enumerate(starting_points):
        chain, _ = metropolis_hastings(target_density_double_well, start, n_samples)
        chains.append(chain)

    # Compute R-hat for increasing chain lengths
    n_points = 20
    r_hat_values = []
    sample_sizes = np.linspace(100, n_samples, n_points, dtype=int)

    for size in sample_sizes:
        chains_subset = [chain[:size] for chain in chains]
        r_hat = gelman_rubin(chains_subset)
        r_hat_values.append(r_hat)

    # Plot R-hat over time
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, r_hat_values, "b-", linewidth=2)
    plt.axhline(y=1.1, color="red", linestyle="--", label="Convergence threshold (1.1)")
    plt.axhline(y=1.0, color="green", linestyle=":", label="Ideal (1.0)")
    plt.xlabel("Number of Samples", fontsize=12)
    plt.ylabel("R-hat", fontsize=12)
    plt.title("Gelman-Rubin Diagnostic for Metropolis-Hastings", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0.95, 2.0)
    plt.tight_layout()
    plt.savefig("gelman_rubin.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Gelman-Rubin diagnostic plot saved as 'gelman_rubin.png'")

    final_r_hat = r_hat_values[-1]
    print(f"Final R-hat value: {final_r_hat:.4f}")
    if final_r_hat < 1.1:
        print("✓ Chains have converged (R-hat < 1.1)")
    else:
        print("✗ Chains may not have converged (R-hat >= 1.1)")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ASSIGNMENT 7: Markov Chains & Sampling Methods")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Task 7.1: PageRank
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TASK 7.1: PageRank (Power Iteration)")
    print("=" * 60)

    # Create graph
    adj_matrix, G, nodes = create_mini_internet_graph()

    # Compute PageRank
    pagerank_scores, n_iter = pagerank_power_iteration(adj_matrix, d=0.85)

    # Display results
    print(f"Converged after {n_iter} iterations")
    print("\nPageRank Scores:")
    for i, node in enumerate(nodes):
        print(f"  Node {node}: {pagerank_scores[i]:.6f}")

    most_important_idx = np.argmax(pagerank_scores)
    print(
        f"\nMost important node: {nodes[most_important_idx]} (score: {pagerank_scores[most_important_idx]:.6f})"
    )

    # Visualize
    visualize_pagerank(G, pagerank_scores, nodes)

    # -------------------------------------------------------------------------
    # Task 7.2: Metropolis-Hastings
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TASK 7.2: Metropolis-Hastings MCMC")
    print("=" * 60)

    # Run Metropolis-Hastings
    np.random.seed(42)  # For reproducibility
    x0 = 0.0
    n_samples = 5000

    samples_mh, acceptance_rate = metropolis_hastings(
        target_density_double_well, x0, n_samples, proposal_std=2.0
    )

    print(f"Acceptance rate: {acceptance_rate:.2%}")

    # Plot results
    samples_after_burnin = plot_mh_results(samples_mh, burnin=500)

    # Analyze mixing
    print("\nMixing Analysis:")
    # Check if chain jumps between modes
    unique_modes = np.unique(np.round(samples_after_burnin / 4) * 4)
    print(f"  Detected modes: {unique_modes}")
    if len(unique_modes) >= 2:
        print("  ✓ Chain successfully jumps between both modes (-4 and +4)")
    else:
        print("  ✗ Chain gets stuck in one mode - proposal variance may be too small")

    # -------------------------------------------------------------------------
    # Task 7.3: Gibbs Sampling
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TASK 7.3: Gibbs Sampling")
    print("=" * 60)

    # Run Gibbs sampler
    np.random.seed(42)
    rho = 0.8
    n_samples_gibbs = 2000

    samples_gibbs = gibbs_sampler(rho, n_samples_gibbs, x0=0, y0=0)

    # Plot results
    plot_gibbs_results(samples_gibbs, rho)

    # Statistics
    sample_corr = np.corrcoef(samples_gibbs[:, 0], samples_gibbs[:, 1])[0, 1]
    print(f"True correlation: {rho}")
    print(f"Sample correlation: {sample_corr:.4f}")

    # -------------------------------------------------------------------------
    # Bonus B1: Gelman-Rubin
    # -------------------------------------------------------------------------

    run_gelman_rubin_diagnostic()
