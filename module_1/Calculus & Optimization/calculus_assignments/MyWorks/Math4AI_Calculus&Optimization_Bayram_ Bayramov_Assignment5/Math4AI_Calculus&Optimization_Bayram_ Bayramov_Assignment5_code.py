# Math4AI: Calculus & Optimization - Assignment 5
# Complete Implementation

import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# --- Problem Setup ---
# ====================================================================

# Generate synthetic data for y = 2*x + 1 + noise
np.random.seed(42)
N = 100  # Number of data points
X = np.random.rand(N, 1) * 10  # X values from 0 to 10
y_true = 2 * X.squeeze() + 1
y = y_true + np.random.randn(N) * 1.5  # Add Gaussian noise

# ====================================================================
# --- Helper Functions ---
# ====================================================================

def compute_loss(X, y, m, b):
    """
    Computes the Mean Squared Error (MSE) loss.
    L(m, b) = (1/N) * sum( (y_i - (m*x_i + b))^2 )
    """
    N = len(y)
    X_flat = X.squeeze()

    # Calculate predictions
    y_pred = m * X_flat + b

    # Calculate squared errors
    squared_errors = (y - y_pred) ** 2

    # Compute mean of squared errors
    loss = np.mean(squared_errors)

    return loss

def compute_gradient(X, y, m, b):
    """
    Computes the gradient of the MSE loss w.r.t. m and b.
    dL/dm = -(2/N) * sum( (y_i - (m*x_i + b)) * x_i )
    dL/db = -(2/N) * sum( (y_i - (m*x_i + b)) )
    """
    N = len(y)
    X_flat = X.squeeze()

    # Calculate error term
    error = y - (m * X_flat + b)

    # Compute gradients
    grad_m = -(2/N) * np.sum(error * X_flat)
    grad_b = -(2/N) * np.sum(error)

    return grad_m, grad_b

def compute_hessian(X, y, m, b):
    """
    Computes the Hessian matrix of the MSE loss.
    H = [[d2L/dm2,  d2L/dmdb],
         [d2L/dbdm, d2L/db2]]
    """
    N = len(y)
    X_flat = X.squeeze()

    # Compute second derivatives
    d2L_dm2 = (2/N) * np.sum(X_flat ** 2)
    d2L_db2 = 2  # Since sum(2/N) = 2
    d2L_dmdb = (2/N) * np.sum(X_flat)

    # Construct Hessian matrix
    hessian = np.array([[d2L_dm2, d2L_dmdb],
                        [d2L_dmdb, d2L_db2]])

    return hessian

# ====================================================================
# Part 2: Gradient-Based Optimization Methods
# ====================================================================

def batch_gradient_descent(X, y, lr, epochs):
    """
    Performs Batch Gradient Descent (BGD).
    """
    m, b = 0.0, 0.0
    loss_history = []
    param_history = [[m, b]]

    for epoch in range(epochs):
        # Compute gradient using entire dataset
        grad_m, grad_b = compute_gradient(X, y, m, b)

        # Update parameters
        m = m - lr * grad_m
        b = b - lr * grad_b

        # Compute and store loss
        loss = compute_loss(X, y, m, b)
        loss_history.append(loss)
        param_history.append([m, b])

    return loss_history, param_history

def stochastic_gradient_descent(X, y, lr, epochs):
    """
    Performs Stochastic Gradient Descent (SGD).
    """
    m, b = 0.0, 0.0
    N = len(y)
    loss_history = []
    param_history = [[m, b]]

    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(N)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Iterate through each data point
        for i in range(N):
            # Get single data point
            X_i = X_shuffled[i:i+1]
            y_i = y_shuffled[i:i+1]

            # Compute gradient for single point
            grad_m, grad_b = compute_gradient(X_i, y_i, m, b)

            # Update parameters
            m = m - lr * grad_m
            b = b - lr * grad_b

        # Compute loss over entire dataset after epoch
        loss = compute_loss(X, y, m, b)
        loss_history.append(loss)
        param_history.append([m, b])

    return loss_history, param_history

def minibatch_gradient_descent(X, y, lr, epochs, batch_size):
    """
    Performs Mini-Batch Gradient Descent.
    """
    m, b = 0.0, 0.0
    N = len(y)
    loss_history = []
    param_history = [[m, b]]

    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(N)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Iterate through mini-batches
        for i in range(0, N, batch_size):
            # Get mini-batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Compute gradient for mini-batch
            grad_m, grad_b = compute_gradient(X_batch, y_batch, m, b)

            # Update parameters
            m = m - lr * grad_m
            b = b - lr * grad_b

        # Compute loss over entire dataset after epoch
        loss = compute_loss(X, y, m, b)
        loss_history.append(loss)
        param_history.append([m, b])

    return loss_history, param_history

def minibatch_gd_with_momentum(X, y, lr, epochs, batch_size, beta):
    """
    Performs Mini-Batch GD with Momentum.
    """
    m, b = 0.0, 0.0
    N = len(y)
    loss_history = []
    param_history = [[m, b]]

    # Initialize velocities
    v_m, v_b = 0.0, 0.0

    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(N)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Iterate through mini-batches
        for i in range(0, N, batch_size):
            # Get mini-batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Compute gradient for mini-batch
            grad_m, grad_b = compute_gradient(X_batch, y_batch, m, b)

            # Update velocities (momentum)
            v_m = beta * v_m + (1 - beta) * grad_m
            v_b = beta * v_b + (1 - beta) * grad_b

            # Update parameters using velocities
            m = m - lr * v_m
            b = b - lr * v_b

        # Compute loss over entire dataset after epoch
        loss = compute_loss(X, y, m, b)
        loss_history.append(loss)
        param_history.append([m, b])

    return loss_history, param_history

def adam_optimizer(X, y, lr, epochs, batch_size, beta1, beta2, epsilon):
    """
    Performs the Adam optimization algorithm.
    """
    m, b = 0.0, 0.0
    N = len(y)
    loss_history = []
    param_history = [[m, b]]

    # Initialize moments
    m_m, m_b = 0.0, 0.0  # First moment
    v_m, v_b = 0.0, 0.0  # Second moment
    t = 0  # Timestep counter

    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(N)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Iterate through mini-batches
        for i in range(0, N, batch_size):
            # Increment timestep
            t += 1

            # Get mini-batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Compute gradient for mini-batch
            grad_m, grad_b = compute_gradient(X_batch, y_batch, m, b)

            # Update first moments
            m_m = beta1 * m_m + (1 - beta1) * grad_m
            m_b = beta1 * m_b + (1 - beta1) * grad_b

            # Update second moments
            v_m = beta2 * v_m + (1 - beta2) * (grad_m ** 2)
            v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)

            # Bias correction
            m_m_hat = m_m / (1 - beta1 ** t)
            m_b_hat = m_b / (1 - beta1 ** t)
            v_m_hat = v_m / (1 - beta2 ** t)
            v_b_hat = v_b / (1 - beta2 ** t)

            # Update parameters
            m = m - lr * m_m_hat / (np.sqrt(v_m_hat) + epsilon)
            b = b - lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

        # Compute loss over entire dataset after epoch
        loss = compute_loss(X, y, m, b)
        loss_history.append(loss)
        param_history.append([m, b])

    return loss_history, param_history

# ====================================================================
# Part 3: Advanced Second-Order Methods
# ====================================================================

def newtons_method(X, y, epochs):
    """
    Performs Newton's method for optimization.
    """
    theta = np.array([0.0, 0.0])  # [m, b]
    loss_history = []
    param_history = [theta.copy()]

    for epoch in range(epochs):
        m, b = theta[0], theta[1]

        # Compute gradient
        grad_m, grad_b = compute_gradient(X, y, m, b)
        gradient = np.array([grad_m, grad_b])

        # Compute Hessian
        hessian = compute_hessian(X, y, m, b)

        # Compute update step (H^{-1} * ∇L)
        hessian_inv = np.linalg.inv(hessian)
        update_step = hessian_inv @ gradient

        # Update parameters
        theta = theta - update_step

        # Compute and store loss
        m_new, b_new = theta[0], theta[1]
        loss = compute_loss(X, y, m_new, b_new)
        loss_history.append(loss)
        param_history.append(theta.copy())

    return loss_history, param_history

# ====================================================================
# --- Main Execution & Verification ---
# ====================================================================

if __name__ == "__main__":

    print("=====================================================")
    print("Math4AI: Assignment 5 Verification")
    print("=====================================================")

    # Define hyperparameters
    LR = 0.01
    LR_SGD = 0.001  # Smaller LR for SGD due to noise
    EPOCHS = 100
    BATCH_SIZE = 16
    BETA = 0.9
    BETA1 = 0.9
    BETA2 = 0.999
    EPSILON = 1e-8

    # Run all optimizers
    print("Running Batch Gradient Descent...")
    loss_bgd, params_bgd = batch_gradient_descent(X, y, LR, EPOCHS)

    print("Running Stochastic Gradient Descent...")
    loss_sgd, params_sgd = stochastic_gradient_descent(X, y, LR_SGD, EPOCHS)

    print("Running Mini-Batch Gradient Descent...")
    loss_mbgd, params_mbgd = minibatch_gradient_descent(X, y, LR, EPOCHS, BATCH_SIZE)

    print("Running Mini-Batch GD with Momentum...")
    loss_momentum, params_momentum = minibatch_gd_with_momentum(
        X, y, LR, EPOCHS, BATCH_SIZE, BETA
    )

    print("Running Adam Optimizer...")
    loss_adam, params_adam = adam_optimizer(
        X, y, LR, EPOCHS, BATCH_SIZE, BETA1, BETA2, EPSILON
    )

    print("Running Newton's Method...")
    loss_newton, params_newton = newtons_method(X, y, EPOCHS)

    # Print final parameters
    print("\nFinal Parameters:")
    print(f"BGD: m = {params_bgd[-1][0]:.4f}, b = {params_bgd[-1][1]:.4f}")
    print(f"SGD: m = {params_sgd[-1][0]:.4f}, b = {params_sgd[-1][1]:.4f}")
    print(f"Mini-Batch: m = {params_mbgd[-1][0]:.4f}, b = {params_mbgd[-1][1]:.4f}")
    print(f"Momentum: m = {params_momentum[-1][0]:.4f}, b = {params_momentum[-1][1]:.4f}")
    print(f"Adam: m = {params_adam[-1][0]:.4f}, b = {params_adam[-1][1]:.4f}")
    print(f"Newton: m = {params_newton[-1][0]:.4f}, b = {params_newton[-1][1]:.4f}")

    # Create the REQUIRED convergence comparison plot
    plt.figure(figsize=(10, 6))

    plt.plot(loss_bgd, label=f'BGD (lr={LR})', linewidth=2)
    plt.plot(loss_sgd, label=f'SGD (lr={LR_SGD})', linewidth=2)
    plt.plot(loss_mbgd, label=f'Mini-Batch (bs={BATCH_SIZE})', linewidth=2)
    plt.plot(loss_momentum, label=f'Momentum (β={BETA})', linewidth=2)
    plt.plot(loss_adam, label=f'Adam (β1={BETA1}, β2={BETA2})', linewidth=2)
    plt.plot(loss_newton, label="Newton's Method", linewidth=3, linestyle='--')

    plt.yscale('log')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title('Convergence Comparison of Optimization Algorithms', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('convergence_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n--- End of Verification ---")
    print("Convergence plot saved as 'convergence_plot.png'")