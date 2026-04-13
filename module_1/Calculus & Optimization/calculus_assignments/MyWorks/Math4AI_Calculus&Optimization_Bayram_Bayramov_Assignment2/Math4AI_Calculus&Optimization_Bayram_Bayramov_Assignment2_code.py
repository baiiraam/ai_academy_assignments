# Math4AI: Calculus & Optimization - Assignment 2
# Starter Code Template
#
# Your file should be named:
# Math4AI_Calculus&Optimization_FirstName_LastName_Assignment2_code.py

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math


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
# Part 1: Differential Calculus - The Engine of Optimization
# ====================================================================

# --------------------------------------------------------------------
# Problem 1.1: Numerical Differentiation from Scratch
# --------------------------------------------------------------------

def numerical_derivative(f, x, h):
    """
    Calculates the derivative of a single-variable function f at a point x
    using the finite difference method.

    Args:
        f (callable): The function to differentiate.
        x (float): The point at which to compute the derivative.
        h (float): A small step size (e.g., 1e-7).

    Returns:
        float: The approximated derivative, or None if not implemented.
    """

    # --- YOUR CODE HERE ---

    # TODO 1: Evaluate the function at f(x + h)
    f_x_plus_h = f(x + h)

    # TODO 2: Evaluate the function at f(x)
    f_x = f(x)

    # TODO 3: Apply the finite difference formula
    # f'(x) approx (f(x + h) - f(x)) / h
    derivative = (f_x_plus_h - f_x) / h

    return derivative


# --------------------------------------------------------------------
# Problem 1.2: Taylor Series for Function Approximation
# --------------------------------------------------------------------

def approximate_taylor(f_sym, a, order, x_range):
    """
    Computes and visualizes the Taylor series approximation of a function f
    around a point 'a' up to a given 'order'.

    This function will generate and show a plot.

    Args:
        f_sym (sympy.Expr): The symbolic function to approximate (e.g., sp.tanh(x)).
        a (float): The point of expansion.
        order (int): The order n of the Taylor polynomial.
        x_range (np.array): A numpy array of x-values for plotting.

    Returns:
        None (This function should generate and show a plot)
    """

    # --- YOUR CODE HERE ---
    # TODO 1: Define your symbolic variable (must match the one in f_sym).
    # Extract the symbol from the expression
    x = list(f_sym.free_symbols)[0]

    # TODO 2: Initialize the Taylor polynomial sum.
    # Start with a symbolic zero.
    taylor_poly = sp.sympify(0)

    # TODO 3: Loop from k = 0 up to 'order' (inclusive).
    for k in range(order + 1):
        # TODO 4: Inside the loop, find the k-th derivative of f_sym w.r.t. x.
        # Use sympy.diff(f_sym, x, k)
        kth_derivative = sp.diff(f_sym, x, k)

        # TODO 5: Evaluate the k-th derivative at the point 'a'.
        # Use .subs(x, a)
        kth_derivative_at_a = kth_derivative.subs(x, a)

        # TODO 6: Calculate the k-th term of the Taylor series.
        # (f^(k)(a) / k!) * (x - a)^k
        # Remember to use math.factorial() for k!
        term = (kth_derivative_at_a / math.factorial(k)) * ((x - a) ** k)

        # TODO 7: Add the k-th term to your polynomial sum.
        taylor_poly += term

    # --- PLOTTING ---

    # TODO 8: Turn the final symbolic Taylor polynomial into a callable numeric function.
    # Use sp.lambdify(). 'numpy' allows it to work with numpy arrays.
    taylor_func = sp.lambdify(x, taylor_poly, 'numpy')

    # TODO 9: Turn the original symbolic function 'f_sym' into a callable numeric function.
    original_func = sp.lambdify(x, f_sym, 'numpy')

    # TODO 10: Evaluate both callable functions over the given x_range.
    y_original = original_func(x_range)
    y_approx = taylor_func(x_range)

    # TODO 11: Plot the original function and the approximation.
    plt.figure(figsize=(10, 6))

    # Plot the original function
    plt.plot(x_range, y_original, label="Original Function: $f(x) = tanh(x)$", linewidth=2)

    # Plot the Taylor approximation
    plt.plot(x_range, y_approx, label=f"Taylor Approx. (Order {order})", linestyle='--', linewidth=2)

    # Add title, labels, legend, and grid
    plt.title(f"Taylor Approximation of tanh(x) around a={a} (Order {order})", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(-1.5, 1.5)  # Optional: set y-limits for better visualization

    # Mark the expansion point
    plt.axvline(x=a, color='red', linestyle=':', alpha=0.5, label=f'Expansion point a={a}')
    plt.legend(fontsize=12)

    # Show the plot
    plt.show()

    return


# ====================================================================
# --- Main Execution & Verification ---
# ====================================================================

# This block will only run when the script is executed directly
if __name__ == "__main__":
    print("=====================================================")
    print("Math4AI: Assignment 2 Verification")
    print("=====================================================")


    # --- Problem 1.1 Verification ---

    # 1. Define the Loss function L(w)
    def loss_func(w):
        """Loss function L(w) = (w - 5)^2 + 3"""
        return (w - 5) ** 2 + 3


    # 2. Set parameters
    w_val = 10.0
    h_val = 1e-7

    # 3. Call your 'from scratch' implementation
    deriv_scratch = numerical_derivative(loss_func, w_val, h_val)
    print_result("Problem 1.1", f"Derivative of L(w) at w={w_val} (Scratch)", deriv_scratch)

    # 4. Verify with SymPy

    # TODO:
    # 1. Define 'w' as a symbolic variable using sp.symbols().
    w = sp.symbols('w')

    # 2. Define the symbolic function L_sym = (w - 5)**2 + 3.
    L_sym = (w - 5)**2 + 3

    # 3. Compute the derivative of L_sym using sp.diff().
    L_prime_sym = sp.diff(L_sym, w)

    # 4. Substitute (subs) the value w_val into the derivative to get the exact value.
    deriv_sympy = L_prime_sym.subs(w, w_val)

    print_result("Problem 1.1", f"Derivative of L(w) at w={w_val} (SymPy)", deriv_sympy)

    # Additional comparison
    print("\n--- Comparison for Problem 1.1 ---")
    print(f"Numerical derivative: {deriv_scratch:.10f}")
    print(f"Analytical derivative: {deriv_sympy:.10f}")
    print(f"Absolute difference: {abs(float(deriv_sympy) - deriv_scratch):.2e}")

    # --- Problem 1.2 Verification ---
    print("\n" + "-" * 40)
    print("--- Problem 1.2 ---")
    print("Generating Taylor series plots for tanh(x)...")

    # 1. Define the symbolic variable and function for tanh(x)
    x_sym = sp.symbols('x')
    f_sym_tanh = sp.tanh(x_sym)

    # 2. Define the x_range for plotting
    x_vals = np.linspace(-4, 4, 200)  #

    # 3. Define the expansion point
    a_val = 0.0  #

    # 4. Generate plots for orders 1, 3, and 5
    print("Generating plot for Order 1...")
    approximate_taylor(f_sym_tanh, a_val, 1, x_vals)

    print("Generating plot for Order 3...")
    approximate_taylor(f_sym_tanh, a_val, 3, x_vals)

    print("Generating plot for Order 5...")
    approximate_taylor(f_sym_tanh, a_val, 5, x_vals)

    print("Plots for Problem 1.2 have been generated.")
    print("-" * 40)
    print("--- End of Verification ---")