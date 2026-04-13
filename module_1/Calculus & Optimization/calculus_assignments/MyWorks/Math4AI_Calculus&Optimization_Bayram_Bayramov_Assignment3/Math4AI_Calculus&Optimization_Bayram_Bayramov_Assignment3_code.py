# Math4AI: Calculus & Optimization - Assignment 3
# Starter Code Template
#
# Your file should be named:
# Math4AI_Calculus&Optimization_FirstName_LastName_Assignment3_code.py

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi


# ====================================================================
# --- Helper Function for Pretty Printing ---
# ====================================================================

def print_result(problem, description, value):
    """
    Helper function to print a result with its problem number and description.
    """
    print(f"--- {problem} ---")
    print(f"{description}:")
    if value is None:
        print("None (Function not yet implemented)")
    else:
        # Set print options for numerical results
        if isinstance(value, (int, float, np.number)):
            print(f"{value:.10f}")
        else:
            print(value)
    print("-" * 40)


# ====================================================================
# Part 1: Integral Calculus - Quantifying Accumulated Change
# ====================================================================

# --------------------------------------------------------------------
# Problem 1.1: Numerical Integration using Riemann Sums
# --------------------------------------------------------------------

def riemann_integral(f, a, b, n):
    """
    Calculates the definite integral of a function f from a to b
    using the midpoint Riemann sum with n rectangles.

    Args:
        f (callable): The function to integrate.
        a (float): The lower bound of integration.
        b (float): The upper bound of integration.
        n (int): The number of rectangles (subintervals).

    Returns:
        float: The approximated integral, or None if not implemented.
    """

    # --- YOUR CODE HERE ---

    # TODO 1: Calculate the width of each rectangle (∆x)
    delta_x = (b - a) / n

    # TODO 2: Initialize the sum to 0
    total_sum = 0.0

    # TODO 3: Loop over each rectangle (i from 0 to n-1)
    for i in range(n):
        # TODO 4: Calculate the midpoint of the i-th subinterval
        # x_i* = a + (i + 0.5) * ∆x
        midpoint = a + (i + 0.5) * delta_x

        # TODO 5: Evaluate the function at the midpoint
        f_midpoint = f(midpoint)

        # TODO 6: Add the area of this rectangle to the sum
        # Area = f(x_i*) * ∆x
        total_sum += f_midpoint * delta_x

    return total_sum


# ====================================================================
# --- Main Execution & Verification ---
# ====================================================================

# This block will only run when the script is executed directly
if __name__ == "__main__":
    print("=====================================================")
    print("Math4AI: Assignment 3 Verification")
    print("=====================================================")


    # --- Problem 1.1 Verification ---

    # 1. Define the PDF of the standard normal distribution
    def phi(x):
        """Probability Density Function of the standard normal distribution"""
        return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)


    # 2. Set the integration bounds and parameters
    a_val = -1.0
    b_val = 1.0
    n_val = 1000  # Number of rectangles

    # 3. Call your 'from scratch' implementation
    integral_scratch = riemann_integral(phi, a_val, b_val, n_val)
    print_result("Problem 1.1", f"Integral of phi(x) from {a_val} to {b_val} (Scratch, n={n_val})", integral_scratch)

    # 4. Verify with SciPy

    # TODO:
    # Use scipy.integrate.quad (already imported as spi.quad) to compute
    # a high-precision value for the same integral.
    # spi.quad returns a tuple (value, error_estimate)
    # Extract the value (first element) for comparison.

    # Compute the integral using SciPy's quad function
    integral_scipy, error_estimate = spi.quad(phi, a_val, b_val)

    print_result("Problem 1.1", f"Integral of phi(x) from {a_val} to {b_val} (SciPy)", integral_scipy)

    # Print the error estimate from SciPy
    print(f"SciPy error estimate: {error_estimate:.2e}")

    # Additional comparison
    print("\n--- Comparison for Problem 1.1 ---")
    print(f"Riemann sum (n={n_val}): {integral_scratch:.10f}")
    print(f"SciPy quad: {integral_scipy:.10f}")
    print(f"Absolute difference: {abs(integral_scipy - integral_scratch):.2e}")

    # Known theoretical value
    known_value = 0.682689492137  # Φ(1) - Φ(-1) = 2Φ(1) - 1
    print(f"Theoretical value: {known_value:.10f}")
    print(f"Error from theoretical: {abs(known_value - integral_scratch):.2e}")

    # Additional analysis: Convergence test
    print("\n--- Convergence Test ---")
    print("Testing accuracy with different numbers of rectangles:")
    for test_n in [10, 100, 1000, 10000]:
        test_integral = riemann_integral(phi, a_val, b_val, test_n)
        error = abs(integral_scipy - test_integral)
        print(f"  n = {test_n:6d}: {test_integral:.10f}, error = {error:.2e}")

    print("\n--- End of Verification ---")

    # Optional: Visualization of the Riemann sum
    print("\nGenerating visualization of Riemann sum approximation...")

    # Generate x values for plotting
    x_plot = np.linspace(a_val - 0.5, b_val + 0.5, 1000)
    y_plot = phi(x_plot)

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot the PDF - using raw strings for LaTeX to avoid escape sequence warnings
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label=r'Standard Normal PDF: $\phi(x)$')

    # Shade the area under the curve from -1 to 1
    x_fill = np.linspace(a_val, b_val, 1000)
    y_fill = phi(x_fill)
    plt.fill_between(x_fill, y_fill, alpha=0.3, color='blue', label='Area ≈ 0.6827')

    # Add rectangles for visualization (using n=10 for clarity)
    n_vis = 10
    delta_x_vis = (b_val - a_val) / n_vis
    for i in range(n_vis):
        midpoint = a_val + (i + 0.5) * delta_x_vis
        left = a_val + i * delta_x_vis
        right = a_val + (i + 1) * delta_x_vis
        height = phi(midpoint)

        # Draw rectangle
        rect_x = [left, right, right, left, left]
        rect_y = [0, 0, height, height, 0]
        plt.plot(rect_x, rect_y, 'r-', alpha=0.5, linewidth=0.8)
        plt.fill_between([left, right], [0, 0], [height, height],
                        alpha=0.1, color='red')

    # Add vertical lines at ±1
    plt.axvline(x=-1, color='green', linestyle='--', alpha=0.7, label='x = ±1')
    plt.axvline(x=1, color='green', linestyle='--', alpha=0.7)

    # Add labels and title - using raw strings for LaTeX
    plt.title(r'Riemann Sum Approximation of $\int_{-1}^{1} \phi(x) dx$ (n=' + str(n_vis) + ' rectangles)', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel(r'$\phi(x)$', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add text box with results
    result_text = f'Riemann sum (n={n_val}): {integral_scratch:.6f}\n'
    result_text += f'SciPy quad: {integral_scipy:.6f}\n'
    result_text += f'Error: {abs(integral_scipy - integral_scratch):.2e}'
    plt.text(1.5, 0.35, result_text, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()