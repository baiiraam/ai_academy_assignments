# Math4AI: Calculus & Optimization - Assignment 6
# Implementation of BFGS Algorithm

import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# --- Helper Functions ---
# ====================================================================


def backtracking_line_search(f, x, p, grad, alpha0=1.0, c=1e-4, tau=0.5, max_steps=25):
    """Backtracking line search with Armijo condition."""
    alpha = alpha0
    fx = f(x)
    gTp = grad.T @ p

    if gTp >= 0:
        return 0.0

    for _ in range(max_steps):
        if f(x + alpha * p) <= fx + c * alpha * gTp:
            return alpha
        alpha *= tau

    return alpha


# ====================================================================
# Problem 3.2: The BFGS Algorithm
# ====================================================================


def bfgs_algorithm(f, f_grad, x0, max_iter=100, tol=1e-6):
    """
    BFGS (Broyden–Fletcher–Goldfarb–Shanno) algorithm implementation.
    """
    n = len(x0)
    x_k = np.array(x0, dtype=float)
    H_k = np.eye(n)  # Initial inverse Hessian approximation
    path_history = [x_k.copy()]
    grad_k = f_grad(x_k)

    for k in range(max_iter):
        # Check convergence
        if np.linalg.norm(grad_k) < tol:
            break

        # Search direction
        p_k = -H_k @ grad_k

        # Line search
        alpha_k = backtracking_line_search(f, x_k, p_k, grad_k)

        # Update position
        x_k_plus_1 = x_k + alpha_k * p_k
        grad_k_plus_1 = f_grad(x_k_plus_1)

        # BFGS update vectors
        s_k = x_k_plus_1 - x_k
        y_k = grad_k_plus_1 - grad_k
        rho_k = 1.0 / (y_k.T @ s_k)

        # Skip update if denominator is too small
        if y_k.T @ s_k > 1e-8:
            I = np.eye(n)
            H_k = (I - rho_k * np.outer(s_k, y_k)) @ H_k @ (
                I - rho_k * np.outer(y_k, s_k)
            ) + rho_k * np.outer(s_k, s_k)

        # Prepare for next iteration
        x_k = x_k_plus_1
        grad_k = grad_k_plus_1
        path_history.append(x_k.copy())

    return x_k, path_history


# ====================================================================
# Problem 3.3: Conjugate Gradient Method (Fletcher-Reeves)
# ====================================================================


def conjugate_gradient(f, f_grad, x0, max_iter=100, tol=1e-6):
    """
    Fletcher-Reeves Conjugate Gradient algorithm implementation.
    """
    x_k = np.array(x0, dtype=float)
    g_k = f_grad(x_k)
    p_k = -g_k  # Initial search direction
    path_history = [x_k.copy()]

    for k in range(max_iter):
        # Check convergence
        if np.linalg.norm(g_k) < tol:
            break

        # Line search
        def phi(alpha):
            return f(x_k + alpha * p_k)

        # Simple golden section search for step size
        alpha_k = 0.1  # Initial guess, could be improved
        for _ in range(10):  # Simple backtracking
            if phi(alpha_k) < phi(0):
                break
            alpha_k *= 0.5

        # Update position
        x_k_plus_1 = x_k + alpha_k * p_k
        g_k_plus_1 = f_grad(x_k_plus_1)

        # Fletcher-Reeves update
        beta_k_plus_1 = (g_k_plus_1.T @ g_k_plus_1) / (g_k.T @ g_k)
        p_k_plus_1 = -g_k_plus_1 + beta_k_plus_1 * p_k

        # Prepare for next iteration
        x_k = x_k_plus_1
        g_k = g_k_plus_1
        p_k = p_k_plus_1
        path_history.append(x_k.copy())

    return x_k, path_history


# ====================================================================
# Problem 3.4: Trust-Region Visualization
# ====================================================================


def plot_trust_region_concept():
    """
    Create a visualization of the trust-region concept.
    """

    # Simple quadratic function for demonstration
    def f(x, y):
        return x**2 + 10 * y**2

    # Create grid
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1.5, 1.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # Current point and trust region
    x_current = np.array([1.0, 0.8])
    delta = 0.8  # Trust region radius

    plt.figure(figsize=(10, 8))

    # Contour plot
    contour_levels = np.linspace(0, 5, 20)
    plt.contour(X, Y, Z, levels=contour_levels, colors="gray", alpha=0.6)

    # Trust region circle
    circle = plt.Circle(
        (x_current[0], x_current[1]),
        delta,
        color="red",
        fill=False,
        linewidth=3,
        label=f"Trust Region (Δ={delta})",
    )
    plt.gca().add_patch(circle)

    # Current point
    plt.scatter(
        x_current[0],
        x_current[1],
        color="red",
        s=100,
        marker="o",
        label="Current Point",
    )

    # Newton step (would be outside trust region)
    newton_step = np.array([-0.9, -0.6])
    newton_point = x_current + newton_step

    # Plot Newton step
    plt.arrow(
        x_current[0],
        x_current[1],
        newton_step[0],
        newton_step[1],
        head_width=0.05,
        head_length=0.1,
        fc="blue",
        ec="blue",
        linewidth=2,
        label="Unconstrained Newton Step",
    )

    plt.scatter(
        newton_point[0],
        newton_point[1],
        color="blue",
        s=100,
        marker="^",
        label="Newton Point",
    )

    # Scaled step (within trust region)
    newton_norm = np.linalg.norm(newton_step)
    if newton_norm > delta:
        scaled_step = (delta / newton_norm) * newton_step
        trust_point = x_current + scaled_step

        plt.arrow(
            x_current[0],
            x_current[1],
            scaled_step[0],
            scaled_step[1],
            head_width=0.05,
            head_length=0.1,
            fc="green",
            ec="green",
            linewidth=2,
            linestyle="--",
            label="Trust Region Step",
        )

        plt.scatter(
            trust_point[0],
            trust_point[1],
            color="green",
            s=100,
            marker="s",
            label="Trust Region Point",
        )

    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.title("Trust-Region Method Concept", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("trust_region_plot.png", dpi=300)
    plt.show()


# ====================================================================
# --- Verification with Rosenbrock Function ---
# ====================================================================


def rosen(x):
    """Rosenbrock function."""
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosen_grad(x):
    """Gradient of Rosenbrock function."""
    return np.array(
        [-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)]
    )


# ====================================================================
# --- Main Execution ---
# ====================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Math4AI: Assignment 6 - Advanced Optimization Methods")
    print("=" * 50)

    # Test on Rosenbrock function
    x0 = np.array([-1.5, 1.0])

    print("\n--- Problem 3.2: BFGS Algorithm ---")
    bfgs_result, bfgs_path = bfgs_algorithm(rosen, rosen_grad, x0)
    print(f"Starting point: {x0}")
    print(f"BFGS result: {bfgs_result}")
    print(f"Function value: {rosen(bfgs_result):.10e}")
    print(f"Number of iterations: {len(bfgs_path) - 1}")

    print("\n--- Problem 3.3: Conjugate Gradient Method ---")
    cg_result, cg_path = conjugate_gradient(rosen, rosen_grad, x0)
    print(f"CG result: {cg_result}")
    print(f"Function value: {rosen(cg_result):.10e}")
    print(f"Number of iterations: {len(cg_path) - 1}")

    print("\n--- Problem 3.4: Trust-Region Visualization ---")
    print("Generating trust-region visualization plot...")
    plot_trust_region_concept()

    print("\n" + "=" * 50)
    print("Trust-region plot saved as 'trust_region_plot.png'")
    print("=" * 50)
