# Math4AI: Calculus & Optimization - Assignment 1
# Starter Code Template
#
# Your file should be named:
# Math4AI_Calculus&Optimization_FirstName_LastName_Assignment1_code.py

import numpy as np
import sympy as sp

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
# Problem 1.1: Approximating Limits
# --------------------------------------------------------------------


def approximate_limit(f, a):
    """
    Numerically approximates the limit of a function f as x approaches a.

    Args:
        f (callable): The function to evaluate.
        a (float): The point to approach.

    Returns:
        float: The approximated limit, or None if not implemented.
    """

    # --- YOUR CODE HERE ---

    # TODO 1: Define a very small value for h.
    h = 1e-7  # As suggested in the assignment

    # TODO 2: Evaluate the function just to the right of 'a'.
    right_value = f(a + h)

    # TODO 3: Evaluate the function just to the left of 'a'.
    left_value = f(a - h)

    # TODO 4: Calculate the average of the left and right values.
    limit_approx = (right_value + left_value) / 2

    return limit_approx


# ====================================================================
# --- Main Execution & Verification ---
# ====================================================================

# This block will only run when the script is executed directly
if __name__ == "__main__":
    print("=====================================================")
    print("Math4AI: Assignment 1 Verification")
    print("=====================================================")

    # --- Problem 1.1 Verification ---

    # 1. Define the function for the famous limit: sin(x) / x
    def func_for_limit(x):
        """Returns sin(x) / x"""
        # Use np.sin for numerical evaluation
        # Handle the case when x is 0 (or very close to 0) to avoid division by zero
        if np.abs(x) < 1e-15:
            return 1.0  # Known limit: sin(x)/x → 1 as x → 0
        return np.sin(x) / x

    # 2. Set the point 'a' to approach
    a_val = 0.0

    # 3. Call your 'from scratch' implementation
    limit_scratch = approximate_limit(func_for_limit, a_val)
    print_result("Problem 1.1", "Limit of sin(x)/x as x->0 (Scratch)", limit_scratch)

    # 4. Verify with SymPy

    # TODO 1:
    # 1. Define 'x' as a symbolic variable using sp.symbols().
    x = sp.symbols("x")

    # 2. Define the symbolic function f_sym = sp.sin(x) / x.
    f_sym = sp.sin(x) / x

    # 3. Use sp.limit() to compute the "exact" analytical value.
    limit_sympy = sp.limit(f_sym, x, 0)

    # 4. Convert to float for comparison
    limit_sympy_float = float(limit_sympy)

    print_result("Problem 1.1", "Limit of sin(x)/x as x->0 (SymPy)", limit_sympy_float)

    # Additional verification: Check the agreement between the two methods
    print("\n--- Comparison ---")
    print(f"Numerical approximation: {limit_scratch:.12f}")
    print(f"SymPy analytical value: {limit_sympy_float:.12f}")
    print(f"Absolute difference: {abs(limit_scratch - limit_sympy_float):.2e}")

    print("\n--- End of Verification ---")
