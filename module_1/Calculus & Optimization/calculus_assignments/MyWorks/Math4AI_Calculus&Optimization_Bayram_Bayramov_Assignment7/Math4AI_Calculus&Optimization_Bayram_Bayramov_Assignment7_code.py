"""
Math4AI - Assignment 7 (Starter)
Constrained Optimization & Portfolio Theory (Bonus: KKT view of SVM support vectors)

What you must implement (see TODOs):
  Part 1 (Equality constraints / Lagrange):
    - solve_portfolio_lagrange(mu, Sigma, target_return)

  Part 2 (Inequality constraints / KKT with w >= 0):
    - solve_portfolio_kkt(mu, Sigma, target_return)

What we provide:
  - helper functions for return/risk
  - efficient frontier loop + plotting helper (so you don't fight matplotlib)

Allowed: numpy, matplotlib
Required for Part 2: scipy.optimize.minimize with method="SLSQP"
Not allowed (unless optional verification): CVXPy / high-level QP solvers

Student: Bayram Bayramov
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None  # Part 2 requires SciPy; we raise a helpful error in that function.


# ============================================================
# Helpers
# ============================================================
def portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    """Expected return: mu^T w."""
    return float(np.dot(mu, w))


def portfolio_variance(w: np.ndarray, Sigma: np.ndarray) -> float:
    """Variance (risk): w^T Sigma w."""
    return float(w.T @ Sigma @ w)


def portfolio_risk_std(w: np.ndarray, Sigma: np.ndarray) -> float:
    """Standard deviation risk: sqrt(w^T Sigma w)."""
    var = portfolio_variance(w, Sigma)
    return float(np.sqrt(max(var, 0.0)))


def feasible_target_grid(
    mu: np.ndarray, num: int = 25, eps: float = 1e-9
) -> np.ndarray:
    """
    Under w >= 0 and sum(w)=1, achievable returns lie in [min(mu), max(mu)].
    We generate a grid in that range.
    """
    lo = float(np.min(mu)) + eps
    hi = float(np.max(mu)) - eps
    if hi <= lo:
        return np.array([float(np.mean(mu))])
    return np.linspace(lo, hi, num=num)


# ============================================================
# Part 1: Equality constraints (Lagrange multipliers)
# ============================================================
def solve_portfolio_lagrange(
    mu: np.ndarray, Sigma: np.ndarray, target_return: float
) -> tuple[np.ndarray, float, float]:
    """
    Solve the equality-constrained Markowitz problem (short selling allowed):

        minimize    0.5 * w^T Sigma w
        subject to  1^T w = 1
                    mu^T w = target_return

    Derivation leads to the block KKT system:

        [ Sigma   mu    1 ] [  w  ]   [ 0 ]
        [ mu^T    0     0 ] [ lambda1  ] = [ R_target ]
        [ 1^T     0     0 ] [ lambda2  ]   [ 1 ]

    Returns:
        w  : (n,) optimal weights
        lambda1 : multiplier for the return constraint (mu^T w = R_target)
        lambda2 : multiplier for the budget constraint (1^T w = 1)
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.size

    # Create column vectors for mu and ones
    mu_col = mu.reshape(-1, 1)
    ones_col = np.ones((n, 1))

    # Construct block matrix: [Sigma, mu, 1] to match [w, lambda1, lambda2]^T
    A_top = np.hstack([Sigma, mu_col, ones_col])
    A_middle = np.hstack([mu_col.T, np.array([[0]]), np.array([[0]])])
    A_bottom = np.hstack([ones_col.T, np.array([[0]]), np.array([[0]])])
    A = np.vstack([A_top, A_middle, A_bottom])

    # RHS vector: [0,...,0, target_return, 1]^T
    b = np.zeros(n + 2)
    b[n] = target_return  # mu^T w = R_target constraint
    b[n + 1] = 1.0  # 1^T w = 1 constraint

    # Solve linear system with fallback for singular matrices
    try:
        z = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        z = np.linalg.lstsq(A, b, rcond=None)[0]

    # Extract solution
    w = z[:n]
    lambda1 = z[n]  # Multiplier for return constraint (mu^T w = R_target)
    lambda2 = z[n + 1]  # Multiplier for budget constraint (1^T w = 1)

    return w, lambda1, lambda2


# ============================================================
# Part 2: Inequality constraints (No short selling, w >= 0)
# ============================================================
def solve_portfolio_kkt(
    mu: np.ndarray,
    Sigma: np.ndarray,
    target_return: float,
    tol: float = 1e-9,
    maxiter: int = 500,
) -> np.ndarray:
    """
    Solve the no-short portfolio problem:

        minimize    0.5 * w^T Sigma w
        subject to  1^T w = 1
                    mu^T w = target_return
                    w >= 0

    This cannot be solved with a single linear solve because of complementarity.
    Use scipy.optimize.minimize with method='SLSQP' and bounds.

    Returns:
        w : (n,) optimal weights
    """
    if minimize is None:
        raise RuntimeError(
            "SciPy is required for Part 2. Install scipy or use the provided environment."
        )

    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.size
    target_return = float(target_return)

    # Objective function: f(w) = 0.5 * w^T Sigma w
    def objective(w):
        return 0.5 * (w.T @ Sigma @ w)

    # Equality constraints: sum(w)=1, mu^T w = target_return
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: w @ mu - target_return},
    ]

    # Bounds: w_i >= 0 (and <= 1 since they sum to 1)
    bounds = [(0.0, 1.0) for _ in range(n)]

    # Initial guess: uniform weights
    w0 = np.ones(n) / n

    # Solve using SLSQP
    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": maxiter, "ftol": tol, "disp": False},
    )

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    w_opt = np.asarray(res.x, dtype=float)

    # Ensure weights sum to exactly 1 (numerical precision)
    if np.abs(np.sum(w_opt) - 1.0) > 1e-8:
        w_opt = w_opt / np.sum(w_opt)

    return w_opt


# ============================================================
# Efficient frontier + plotting (provided)
# ============================================================
def compute_frontier(
    mu: np.ndarray, Sigma: np.ndarray, target_returns: np.ndarray, solver_fn
) -> list[tuple[float, float]]:
    """
    Computes a (risk, return) list for a set of target returns using solver_fn.
    solver_fn signature: solver_fn(mu, Sigma, target_return) -> w
    """
    pts: list[tuple[float, float]] = []
    for R in target_returns:
        try:
            w = solver_fn(mu, Sigma, float(R))
            w = np.asarray(w, dtype=float).reshape(-1)
            risk = portfolio_risk_std(w, Sigma)
            ret = portfolio_return(w, mu)
            pts.append((risk, ret))
        except Exception as e:
            # Skip infeasible/failed targets but keep going
            print(f"[warn] Skipping R={float(R):.6f}: {e}")
            continue
    return pts


def plot_efficient_frontier(
    lagrange_pts: list[tuple[float, float]],
    kkt_pts: list[tuple[float, float]],
    filename: str = "efficient_frontier.png",
) -> None:
    """
    Plots the Efficient Frontier curves and saves to filename.
    lagrange_pts / kkt_pts are lists of (risk_std, return).
    """
    if len(lagrange_pts) == 0 or len(kkt_pts) == 0:
        print("[warn] Not enough points to plot.")
        return

    l_risk, l_ret = zip(*lagrange_pts)
    k_risk, k_ret = zip(*kkt_pts)

    plt.figure(figsize=(10, 6))
    plt.plot(l_risk, l_ret, "b--", linewidth=2, label="With Short Selling (Lagrange)")
    plt.plot(k_risk, k_ret, "r-", linewidth=2, label="No Short Selling (KKT/SciPy)")

    plt.xlabel(r"Risk $\sigma(w)=\sqrt{w^T\Sigma w}$")
    plt.ylabel(r"Return $\mu^T w$")
    plt.title("Efficient Frontier: Impact of the No-Short-Selling Constraint")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()


# ============================================================
# Verification and testing
# ============================================================
def verify_solution_correctness():
    """
    Verify that both solvers produce correct results for a test case.
    """
    print("=" * 60)
    print("Verification of Portfolio Optimization Solvers")
    print("=" * 60)

    mu = np.array([0.05, 0.06, 0.08], dtype=float)
    Sigma = np.array(
        [[0.010, 0.002, 0.001], [0.002, 0.015, 0.005], [0.001, 0.005, 0.020]],
        dtype=float,
    )

    target_return = 0.06

    print(f"\nTest case: Target return = {target_return}")
    print("-" * 40)

    # Test Lagrange solver
    print("\n1. Lagrange Solver (Short Selling Allowed):")
    try:
        w_lagrange, lambda1, lambda2 = solve_portfolio_lagrange(
            mu, Sigma, target_return
        )
        print(
            f"   Optimal weights: [{w_lagrange[0]:.4f}, {w_lagrange[1]:.4f}, {w_lagrange[2]:.4f}]"
        )
        print(f"   Portfolio return: {w_lagrange @ mu:.6f} (target: {target_return})")
        print(f"   Sum of weights: {np.sum(w_lagrange):.6f} (should be 1)")
        print(f"   Portfolio variance: {w_lagrange.T @ Sigma @ w_lagrange:.6f}")
        print(f"   Lagrange multipliers: lambda1={lambda1:.6f}, lambda2={lambda2:.6f}")

        # Verify KKT conditions
        grad_condition = Sigma @ w_lagrange + lambda1 * mu + lambda2 * np.ones(3)
        grad_norm = np.linalg.norm(grad_condition)
        print(f"   Gradient condition norm: {grad_norm:.2e} (should be ~0)")

        if grad_norm > 1e-8:
            print("   WARNING: Gradient condition not satisfied within tolerance")
    except Exception as e:
        print(f"   ERROR: {e}")

    # Test KKT solver
    print("\n2. KKT Solver (No Short Selling):")
    try:
        w_kkt = solve_portfolio_kkt(mu, Sigma, target_return)
        print(f"   Optimal weights: [{w_kkt[0]:.4f}, {w_kkt[1]:.4f}, {w_kkt[2]:.4f}]")
        print(f"   Portfolio return: {w_kkt @ mu:.6f} (target: {target_return})")
        print(f"   Sum of weights: {np.sum(w_kkt):.6f} (should be 1)")
        print(f"   Portfolio variance: {w_kkt.T @ Sigma @ w_kkt:.6f}")
        print(f"   All weights >= 0: {np.all(w_kkt >= -1e-10)}")
    except Exception as e:
        print(f"   ERROR: {e}")

    print("\n" + "=" * 60)


# ============================================================
# Demo / sanity run (you can modify)
# ============================================================
def example_problem() -> tuple[np.ndarray, np.ndarray]:
    """
    Example from the assignment handout (3 assets).
    Returns (mu, Sigma).
    """
    mu = np.array([0.05, 0.06, 0.08], dtype=float)
    Sigma = np.array(
        [
            [0.010, 0.002, 0.001],
            [0.002, 0.015, 0.005],
            [0.001, 0.005, 0.020],
        ],
        dtype=float,
    )
    return mu, Sigma


if __name__ == "__main__":
    # Run verification first
    verify_solution_correctness()

    # Generate efficient frontier plot
    print("\nGenerating Efficient Frontier Plot...")
    print("-" * 40)

    mu, Sigma = example_problem()

    # Target returns grid (feasible for no-short case)
    target_returns = feasible_target_grid(mu, num=25)

    # Part 1 frontier (short selling allowed)
    print("Computing Lagrange frontier (with short selling)...")
    lagrange_pts = compute_frontier(
        mu,
        Sigma,
        target_returns,
        solver_fn=lambda mu_, Sigma_, R_: solve_portfolio_lagrange(mu_, Sigma_, R_)[0],
    )

    # Part 2 frontier (no short selling)
    print("Computing KKT frontier (no short selling)...")
    kkt_pts = compute_frontier(
        mu,
        Sigma,
        target_returns,
        solver_fn=solve_portfolio_kkt,
    )

    print(f"Successfully computed {len(lagrange_pts)} Lagrange points")
    print(f"Successfully computed {len(kkt_pts)} KKT points")

    # Generate and save plot
    plot_efficient_frontier(lagrange_pts, kkt_pts, filename="efficient_frontier.png")

    print("\nAnalysis complete. Plot saved as 'efficient_frontier.png'")
