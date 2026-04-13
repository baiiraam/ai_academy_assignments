# Math4AI: Calculus & Optimization - Assignment 4
# Your file should be named:
# Math4AI_Calculus&Optimization_FirstName_LastName_Assignment4_code.py

# Before starting, I want to mention that
# the image "contour_plot.png" will be generated in the current directory.

import numpy as np
import matplotlib.pyplot as plt
import sympy
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
# Part 1: Multivariable Calculus
# ====================================================================

# --------------------------------------------------------------------
# Problem 1.1: Partial Derivatives and the Gradient Vector
# --------------------------------------------------------------------

def partial_derivative(f, point, var_index, h=1e-7):
    """
    Computes the partial derivative of a multivariable function f
    at a given point w.r.t. the variable at var_index.

    Args:
        f (callable): The multivariable function.
        point (np.ndarray): The point (e.g., [a1, a2, ...]) to evaluate at.
        var_index (int): The index of the variable to differentiate (e.g., 0 for x_0).
        h (float, optional): Step size. Defaults to 1e-7.

    Returns:
        float: The approximated partial derivative, or None.
    """
    # TODO 1: Create a copy of the 'point' array.
    point_plus_h = np.copy(point).astype(float)

    # TODO 2: Modify the copied array at 'var_index' by adding 'h'.
    point_plus_h[var_index] += h

    # TODO 3: Evaluate the function at the modified point.
    f_plus = f(point_plus_h)

    # TODO 4: Evaluate the function at the original, unmodified 'point'.
    f_original = f(point)

    # TODO 5: Apply the finite difference formula.
    partial_deriv = (f_plus - f_original) / h

    return partial_deriv

def compute_gradient(f, point, h=1e-7):
    """
    Computes the full gradient vector of a multivariable function f
    at a given point.

    Args:
        f (callable): The multivariable function.
        point (np.ndarray): The point to evaluate at.
        h (float, optional): Step size to pass to partial_derivative.

    Returns:
        np.ndarray: The gradient vector, or None.
    """
    # Ensure 'point' is a numpy array for easier handling
    point = np.asarray(point)

    # TODO 1: Get the number of variables (n).
    n = len(point)

    # TODO 2: Initialize a zero vector for the gradient.
    gradient_vector = np.zeros(n)

    # TODO 3: Loop through each variable.
    for i in range(n):
        # TODO 4: In the loop, call 'partial_derivative' for the i-th variable.
        gradient_vector[i] = partial_derivative(f, point, i, h)

    return gradient_vector

# --------------------------------------------------------------------
# Problem 1.2: The Hessian Matrix
# --------------------------------------------------------------------

def compute_hessian(f, point, h=1e-5):
    """
    Computes the Hessian matrix of a multivariable function f
    at a given point.

    Args:
        f (callable): The multivariable function.
        point (np.ndarray): The point to evaluate at.
        h (float, optional): Step size.

    Returns:
        np.ndarray: The (n x n) Hessian matrix, or None.
    """
    point = np.asarray(point)

    # TODO 1: Get the number of variables (n).
    n = len(point)

    # TODO 2: Initialize an (n x n) zero matrix for the Hessian.
    hessian_matrix = np.zeros((n, n))

    # TODO 3: Implement a nested loop (for i, for j).
    for i in range(n):
        for j in range(n):
            # TODO 4: Inside the loop, approximate the second partial
            #         derivative H_ij = d/dx_i ( d/dx_j f ).
            #         (Hint: This will likely involve your 'partial_derivative' function).

            # Define a helper function that gives us the partial derivative w.r.t. x_j
            def partial_f_j(p):
                return partial_derivative(f, p, j, h)

            # Now take the partial derivative of partial_f_j w.r.t. x_i
            hessian_matrix[i, j] = partial_derivative(partial_f_j, point, i, h)

    return hessian_matrix

# --------------------------------------------------------------------
# Problem 1.3: Numerical Double Integration
# --------------------------------------------------------------------

def double_integral(f, a, b, c, d, nx, ny):
    """
    Computes the double integral of f(x, y) over [a, b] x [c, d]
    using the nested midpoint rule.

    Args:
        f (callable): The function to integrate (must accept f(x, y)).
        a (float): Lower bound for x.
        b (float): Upper bound for x.
        c (float): Lower bound for y.
        d (float): Upper bound for y.
        nx (int): Number of subintervals for x.
        ny (int): Number of subintervals for y.

    Returns:
        float: The approximated double integral, or None.
    """

    # TODO 1: Calculate delta_x and delta_y.
    delta_x = (b - a) / nx
    delta_y = (d - c) / ny

    # TODO 2: Initialize the total sum.
    total_sum = 0.0

    # TODO 3: Implement a nested loop over nx and ny.
    for i in range(nx):
        for j in range(ny):
            # TODO 4: Inside the loops, calculate the midpoints x_i* and y_j*.
            x_mid = a + (i + 0.5) * delta_x
            y_mid = c + (j + 0.5) * delta_y

            # TODO 5: Evaluate the function at the midpoint f(x_i*, y_j*).
            f_value = f(x_mid, y_mid)

            # TODO 6: Add this value to the total sum.
            total_sum += f_value

    # TODO 7: After the loops, multiply the total sum by the area
    #         of each small rectangle.
    integral_approx = total_sum * delta_x * delta_y

    return integral_approx

# ====================================================================
# --- Main Execution & Verification ---
# ====================================================================

if __name__ == "__main__":

    print("=====================================================")
    print("Math4AI: Assignment 4 Verification")
    print("=====================================================")

    # --- Problem 1.1 Verification ---

    # Define the function f(x, y) = x^2 + 2y^2
    def f_1_1(point):
        x = point[0]
        y = point[1]
        return x**2 + 2*y**2

    test_point_1_1 = np.array([1.0, 1.0])

    # TODO:
    # 1. Call your 'compute_gradient' function.
    # 2. Print the result.
    grad_scratch = compute_gradient(f_1_1, test_point_1_1)
    print_result("Problem 1.1", f"Gradient of x^2+2y^2 at {test_point_1_1} (Scratch)", grad_scratch)

    # TODO: Plotting (for the report)
    # 1. Create a meshgrid for x and y.
    x = np.linspace(-2, 2, 30)
    y = np.linspace(-2, 2, 30)
    X, Y = np.meshgrid(x, y)

    # 2. Plot the contours of f_1_1.
    Z = X**2 + 2*Y**2
    plt.figure(figsize=(10, 8))
    contour = plt.contour(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(contour, label='f(x, y) = x² + 2y²')

    # 3. Define a few test points.
    test_points = np.array([[-1.5, -1.5], [-1, 0], [0, 0], [1, -1], [1.5, 1.5]])

    # 4. Compute the gradient at each point.
    gradients = []
    for point in test_points:
        grad = compute_gradient(f_1_1, point)
        gradients.append(grad)

    # 5. Plot the gradient vectors using plt.quiver.
    test_points_x = test_points[:, 0]
    test_points_y = test_points[:, 1]
    gradients_x = [g[0] for g in gradients]
    gradients_y = [g[1] for g in gradients]

    plt.quiver(test_points_x, test_points_y, gradients_x, gradients_y,
              color='red', scale=15, width=0.005, label='Gradient vectors')

    plt.title('Contour Plot of f(x, y) = x² + 2y² with Gradient Vectors')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')

    # 6. Save and show the plot.
    plt.savefig('gradient_contour_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Problem 1.1: Plot saved as 'gradient_contour_plot.png'")

    # --- Problem 1.2 Verification ---

    # TODO:
    # 1. Define a numerical function f(x, y) = x^2 - y^2.
    def f_1_2(point):
        x = point[0]
        y = point[1]
        return x**2 - y**2

    # 2. Define a test point.
    test_point_1_2 = np.array([1.0, 1.0])

    # 3. Call your 'compute_hessian' function and print the result.
    hessian_scratch = compute_hessian(f_1_2, test_point_1_2)
    print_result("Problem 1.2", "Hessian of x^2-y^2 (Scratch)", hessian_scratch)

    # TODO: SymPy Verification
    # 1. Define symbolic x, y and the symbolic function.
    x_sym, y_sym = sympy.symbols('x y')
    f_sym = x_sym**2 - y_sym**2

    # 2. Use sympy.hessian() to get the symbolic Hessian.
    hessian_sympy = sympy.hessian(f_sym, (x_sym, y_sym))

    # 3. Print the symbolic Hessian.
    print_result("Problem 1.2", "Hessian of x^2-y^2 (SymPy)", hessian_sympy)

    # Also evaluate the symbolic Hessian at the test point
    hessian_sympy_eval = np.array(hessian_sympy.subs({x_sym: 1.0, y_sym: 1.0})).astype(float)
    print_result("Problem 1.2", "Hessian of x^2-y^2 at (1,1) (SymPy evaluated)", hessian_sympy_eval)


    # --- Problem 1.3 Verification ---

    # 1. Define the function to integrate
    def f_1_3(x, y):
        return x * np.sin(y)

    # 2. Set integration parameters
    a, b = 0.0, 1.0  # x-range
    c, d = 0.0, np.pi # y-range
    nx, ny = 100, 100

    # 3. Call your 'from scratch' implementation
    # TODO: Call your 'double_integral' function.
    integral_scratch = double_integral(f_1_3, a, b, c, d, nx, ny)
    print_result("Problem 1.3", "Double Integral of x*sin(y) (Scratch)", integral_scratch)

    # 4. Verify with SciPy
    # TODO:
    # 1. Use spi.dblquad() to get the "exact" value.
    # 2. Assign the value to 'integral_scipy'.
    # 3. Print the result.
    integral_scipy, error = spi.dblquad(f_1_3, c, d, lambda y: a, lambda y: b)
    print_result("Problem 1.3", "Double Integral of x*sin(y) (SciPy)", integral_scipy)

    print("--- End of Verification ---")