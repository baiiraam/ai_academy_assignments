# Math4AI: Linear Algebra - Programming Assignment 5.1
# Starter Code Template

import numpy as np


# --- Helper Functions for Pretty Printing ---
def print_vectors(name, vecs):
    """Prints a list of vectors, handling both lists of arrays and 2D arrays."""
    print(f"{name}:")
    if vecs is None or len(vecs) == 0:
        print("[] (or not implemented)")
    # Check if it's a list of 1D arrays
    elif isinstance(vecs, list) and all(isinstance(v, np.ndarray) for v in vecs):
        for i, v in enumerate(vecs):
            np.set_printoptions(precision=4, suppress=True)
            print(f"  Vector {i + 1}:\n{v.reshape(-1, 1)}")
    else:
        print("Unsupported format for printing vectors.")
    print("-" * 40)


def print_matrix(name, m):
    """Prints a matrix with its name."""
    if m is None:
        print(f"{name}:\nNone (or not implemented)")
    else:
        np.set_printoptions(precision=4, suppress=True)
        print(f"{name}:\n{m}")
    print("-" * 40)


# --- Problem Setup ---
# A set of linearly independent vectors to be orthonormalized
v1 = np.array([1.0, 1.0, 0.0])
v2 = np.array([1.0, 0.0, 1.0])
v3 = np.array([0.0, 1.0, 1.0])

input_vectors = [v1, v2, v3]

print_vectors("Input Vectors (Linearly Independent)", input_vectors)

# ====================================================================
# Part 5.1.1: Gram-Schmidt Process from Scratch
# ====================================================================


def gram_schmidt(vectors):
    """
    Applies the Gram-Schmidt process to a list of linearly independent vectors.

    Args:
        vectors (list): A list of linearly independent numpy arrays.

    Returns:
        list: A list of orthonormal numpy arrays that span the same space.
    """

    orthonormal_basis = []

    # --- YOUR CODE HERE ---
    # Implement the Gram-Schmidt process.

    # Suggested Steps:
    # 1. Iterate through each vector `v` in the input `vectors`.
    # 2. Start with a new vector `u` initialized to be the same as `v`.
    # 3. For each vector `q` already in your `orthonormal_basis`:
    #    a. Calculate the projection of `v` onto `q`. The formula is:
    #       projection = (u . q) * q
    #       (Note: v.dot(q) or np.dot(v, q) calculates the dot product)
    #    b. Subtract this projection from `u`: u = u - projection
    # 4. After subtracting all projections, `u` is now orthogonal to all
    #    vectors currently in `orthonormal_basis`.
    # 5. Normalize `u` to get the new unit vector `q_new`. The formula is:
    #       q_new = u / ||u||
    #       (Note: np.linalg.norm(u) calculates the L2 norm)
    # 6. Append `q_new` to your `orthonormal_basis` list.
    # 7. Repeat for all input vectors.

    for v in vectors:
        u = v.copy().astype(float)

        for q in orthonormal_basis:
            projection = np.dot(u, q) * q
            u = u - projection

        norm_u = np.linalg.norm(u)
        if norm_u > 1e-12:
            q_new = u / norm_u
            orthonormal_basis.append(q_new)

    return orthonormal_basis


# --- Calling the function for Part 2.1 ---
print("--- Part 2.1: Applying the Gram-Schmidt Process ---")
orthonormal_vectors = gram_schmidt(input_vectors)
print_vectors("Orthonormal Basis (from scratch)", orthonormal_vectors)


# ====================================================================
# Part 5.1.2: Verification
# ====================================================================
print("--- Part 2.2: Verification ---")

if not orthonormal_vectors:
    print("Orthonormal basis not implemented, cannot perform verification.")
else:
    # 1. Verification: Orthogonality
    print("--- Verifying Orthogonality ---")
    is_orthogonal = True
    num_vectors = len(orthonormal_vectors)
    for i in range(num_vectors):
        for j in range(i + 1, num_vectors):
            v1 = orthonormal_vectors[i]
            v2 = orthonormal_vectors[j]
            dot_product = np.dot(v1, v2)
            print(
                f"Dot product of Vector {i + 1} and Vector {j + 1}: {dot_product:.6f}"
            )
            if not np.isclose(dot_product, 0):
                is_orthogonal = False
    print(f"Are all distinct pairs orthogonal? {is_orthogonal}\n")

    # 2. Verification: Normalization
    print("--- Verifying Normalization ---")
    is_normalized = True
    for i, v in enumerate(orthonormal_vectors):
        norm = np.linalg.norm(v)
        print(f"Norm of Vector {i + 1}: {norm:.6f}")
        if not np.isclose(norm, 1):
            is_normalized = False
    print(f"Are all vectors normalized (unit vectors)? {is_normalized}\n")

    # 3. Verification: NumPy Comparison using QR Decomposition
    print("--- Verifying with NumPy's QR Decomposition ---")
    # Stack the original vectors as columns of a matrix A
    A = np.column_stack(input_vectors)
    print_matrix("Matrix A (from input vectors)", A)

    # Perform QR decomposition
    Q, R = np.linalg.qr(A)

    print_matrix("Q Matrix from NumPy's np.linalg.qr(A)", Q)
    print("The columns of this Q matrix form NumPy's orthonormal basis.")
    print(
        "Your basis should be equivalent (individual vectors may have opposite signs)."
    )


# Task 5.2

# Math4AI: Linear Algebra - Programming Assignment 5, Part 2
# Starter Code Template

import numpy as np


# --- Helper Functions for Pretty Printing ---
def print_matrix(name, m):
    """Prints a matrix with its name."""
    if m is None:
        print(f"{name}:\nNone (or not implemented)")
    else:
        np.set_printoptions(precision=4, suppress=True)
        print(f"{name}:\n{m}")
    print("-" * 40)


def print_vectors(name, vecs):
    """Prints a list of vectors, handling both lists of arrays and 2D arrays."""
    print(f"{name}:")
    if vecs is None or len(vecs) == 0:
        print("[] (or not implemented)")
    elif isinstance(vecs, list) and all(isinstance(v, np.ndarray) for v in vecs):
        for i, v in enumerate(vecs):
            np.set_printoptions(precision=4, suppress=True)
            print(f"  Vector {i + 1}:\n{v.reshape(-1, 1)}")
    else:
        print("Unsupported format for printing vectors.")
    print("-" * 40)


# --- Setup ---
A2 = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
b2 = np.array([6.0, 0.0, 0.0])
print_matrix("Matrix A for Part 5.2", A2)
print_matrix("Vector b for Part 5.2", b2.reshape(-1, 1))

# --- Reusable Helper Functions (Students must implement or reuse) ---


def transpose_matrix(M):
    """Computes the transpose of a matrix M."""
    rows, cols = M.shape
    transposed = np.zeros((cols, rows))
    for i in range(rows):
        for j in range(cols):
            transposed[j, i] = M[i, j]
    return transposed


def multiply_matrices(M1, M2):
    """Computes the product of two matrices M1 and M2."""
    rows1, cols1 = M1.shape
    rows2, cols2 = M2.shape

    if cols1 != rows2:
        raise ValueError("Matrices dimensions not compatible for multiplication")

    result = np.zeros((rows1, cols2))
    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                result[i, j] += M1[i, k] * M2[k, j]
    return result


# --- 5.2.1: The Projection Matrix ---


def create_projection_matrix(A):
    """
    Creates a projection matrix P that projects onto the column space of A.
    Formula: P = A(A^T A)^-1 A^T
    """
    # Compute A^T using transpose_matrix function
    A_T = transpose_matrix(A)

    # Compute A^T @ A using multiply_matrices function
    A_T_A = multiply_matrices(A_T, A)

    # Compute the inverse using np.linalg.inv()
    A_T_A_inv = np.linalg.inv(A_T_A)

    # Compute A @ (A^T A)^-1
    A_A_T_A_inv = multiply_matrices(A, A_T_A_inv)

    # Compute final product P = A @ (A^T A)^-1 @ A^T
    P = multiply_matrices(A_A_T_A_inv, A_T)

    return P


print("\n--- 5.2.1: Creating the Projection Matrix ---")
P = create_projection_matrix(A2)
print_matrix("Projection Matrix P (from scratch)", P)


# --- 5.2.2: Projecting the Data Vector ---
print("\n--- 5.2.2: Projecting the Vector ---")
if P is not None:
    # p = P @ b2
    p = multiply_matrices(P, b2.reshape(-1, 1)).flatten() if P is not None else None
    print_matrix("Projected vector p = Pb", p.reshape(-1, 1) if p is not None else None)
    if p is not None:
        print("These are the y-values of the best-fit line at t=0, 1, 2.")
else:
    print("Projection matrix not implemented.")


# --- 5.2.3: Decomposing the Vector and Verifying Orthogonality ---
print("\n--- 5.2.3: Verifying Orthogonality ---")
if p is not None:
    # 1. Calculate the error vector e = b - p
    e = b2 - p
    print_matrix("Error vector e = b - p", e.reshape(-1, 1))

    # 2. Verification 1: p and e must be orthogonal
    p_dot_e = np.dot(p, e)
    print(f"Verification 1: Dot product of p and e = {p_dot_e:.4f} (should be 0)")
    print(f"Are they orthogonal? {np.isclose(p_dot_e, 0)}\n")

    # 3. Verification 2: e must be in the left nullspace of A
    #    This means A^T @ e should be the zero vector.
    AT2 = A2.T  # Using numpy's transpose for verification step
    AT_e = AT2 @ e
    print("Verification 2: e must be in the Left Nullspace (A^T @ e = 0)")
    print_matrix("A^T @ e", AT_e.reshape(-1, 1))
    print(f"Is A^T @ e close to zero? {np.allclose(AT_e, 0)}")
else:
    print("Projected vector p not available, cannot verify.")


# Task 5.3
# Math4AI: Linear Algebra - Programming Assignment 5, Part 3 (Multi-feature)
# Complete Implementation

import numpy as np
import matplotlib.pyplot as plt


# --- Helper Functions for Pretty Printing ---
def print_matrix(name, m):
    """Prints a matrix or vector with its name."""
    if m is None:
        print(f"{name}:\nNone (or not implemented)")
    else:
        np.set_printoptions(precision=4, suppress=True)
        print(f"{name}:\n{m}")
    print("-" * 40)


# ====================================================================
# PART 3: LEAST SQUARES, MODEL FITTING, AND VISUALIZATION
# ====================================================================

print("=" * 60)
print("PART 5.3: LEAST SQUARES, MODEL FITTING, AND VISUALIZATION")
print("=" * 60)

# --- Problem Setup: Multi-feature House Price Data ---
# We now have two features: size and age.
np.random.seed(0)  # for reproducibility
num_houses = 20
house_sizes_sq_m = np.linspace(80, 300, num_houses)
house_ages_years = np.linspace(1, 25, num_houses)

# Prices in thousands of dollars, with a more complex trend and some noise
true_prices = (
    80 + 1.8 * house_sizes_sq_m - 2.5 * house_ages_years + 0.005 * house_sizes_sq_m**2
)
noise = np.random.normal(0, 30, house_sizes_sq_m.shape)
observed_prices = true_prices + noise

# Our feature vectors and target vector b
x1_feature_size = house_sizes_sq_m
x2_feature_age = house_ages_years
b_prices = observed_prices

# --- Reusable Helper Functions (Students must implement or reuse) ---


def solve_system(A, b):
    """
    Solves a system of linear equations Ax = b using Gaussian elimination.
    This function is the engine for the least_squares solver.
    """
    # Convert to numpy arrays
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    n = len(b)

    # Create augmented matrix [A | b]
    aug = np.column_stack((A, b))

    # Forward elimination
    for i in range(n):
        # Find pivot row
        max_row = i
        for row in range(i + 1, n):
            if abs(aug[row, i]) > abs(aug[max_row, i]):
                max_row = row

        # Swap rows if necessary
        if max_row != i:
            aug[[i, max_row]] = aug[[max_row, i]]

        # Check for zero pivot
        if abs(aug[i, i]) < 1e-12:
            return None

        # Eliminate below pivot
        for j in range(i + 1, n):
            factor = aug[j, i] / aug[i, i]
            aug[j, i:] -= factor * aug[i, i:]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = aug[i, n]
        for j in range(i + 1, n):
            x[i] -= aug[i, j] * x[j]
        x[i] /= aug[i, i]

    return x


def transpose_matrix(M):
    """Computes the transpose of a matrix M."""
    rows, cols = M.shape
    transposed = np.zeros((cols, rows))
    for i in range(rows):
        for j in range(cols):
            transposed[j, i] = M[i, j]
    return transposed


def multiply_matrices(M1, M2):
    """Computes the product of two matrices M1 and M2."""
    rows1, cols1 = M1.shape
    rows2, cols2 = M2.shape

    if cols1 != rows2:
        raise ValueError("Matrices dimensions not compatible for multiplication")

    result = np.zeros((rows1, cols2))
    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                result[i, j] += M1[i, k] * M2[k, j]
    return result


# ====================================================================
# Part 5.3.1: Least Squares Solver from Scratch
# ====================================================================


def least_squares(A, b):
    """
    Solves for the least-squares solution x_hat to Ax = b using the Normal Equations.
    The Normal Equations are: (A^T * A) * x_hat = (A^T * b)
    """
    # Compute A^T
    A_T = transpose_matrix(A)

    # Compute A^T * A
    A_T_A = multiply_matrices(A_T, A)

    # Compute A^T * b
    A_T_b = multiply_matrices(A_T, b.reshape(-1, 1)).flatten()

    # Solve the system (A^T * A) * x_hat = A^T * b
    x_hat = solve_system(A_T_A, A_T_b)

    return x_hat


# ====================================================================
# Part 5.3.2: Application: Multiple Linear Housing Model
# ====================================================================
print("\n--- Part 5.3.2: Application: Multiple Linear Housing Model ---")

# 1. Construct the feature matrix A_linear for the model y = c0 + c1*size + c2*age
col_ones = np.ones_like(x1_feature_size)
A_linear = np.column_stack([col_ones, x1_feature_size, x2_feature_age])

print_matrix("Linear Feature Matrix A_linear (first 5 rows)", A_linear[:5])

# 2. Use your least_squares function to find the optimal weights
x_hat_linear = least_squares(A_linear, b_prices)
print_matrix("Optimal Weights x_hat_linear [c0, c1, c2] (from scratch)", x_hat_linear)

# 3. Verify your result with numpy.linalg.lstsq
print("--- Verifying with NumPy ---")
x_hat_linear_np = np.linalg.lstsq(A_linear, b_prices, rcond=None)[0]
print_matrix("Optimal Weights x_hat_linear [c0, c1, c2] (from NumPy)", x_hat_linear_np)

# 4. Interpretation
if x_hat_linear is not None:
    c0, c1, c2 = x_hat_linear.flatten()
    size_to_predict = 200
    age_to_predict = 5
    predicted_price = c0 + c1 * size_to_predict + c2 * age_to_predict
    print("\n--- Interpretation ---")
    print(f"The linear model is: price = {c0:.2f} + {c1:.2f}*size + {c2:.2f}*age")
    print(
        f"Predicted price for a {size_to_predict} sq m, {age_to_predict}-year-old house: ${predicted_price:.2f}k"
    )
else:
    print("\nCannot make prediction, weights not calculated.")

# ====================================================================
# Part 5.3.3: Application: Polynomial Fitting with Multiple Features
# ====================================================================
print("\n--- Part 5.3.3: Application: Polynomial Housing Model ---")

# 1. Construct the feature matrix A_poly for y = c0 + c1*size + c2*age + c3*size^2
size_squared = x1_feature_size**2
A_poly = np.column_stack([col_ones, x1_feature_size, x2_feature_age, size_squared])
print_matrix("Polynomial Feature Matrix A_poly (first 5 rows)", A_poly[:5])

# 2. Use your *same* least_squares function to find the new optimal weights
x_hat_poly = least_squares(A_poly, b_prices)
print_matrix("Optimal Weights x_hat_poly [c0, c1, c2, c3] (from scratch)", x_hat_poly)


# ====================================================================
# Part 5.3.4: Visualization and Comparison
# ====================================================================
print("\n--- Part 5.3.4: Visualization and Comparison (3D Plot) ---")

# Create a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot the raw data points
ax.scatter(
    x1_feature_size,
    x2_feature_age,
    b_prices,
    c="r",
    marker="o",
    label="Actual Data Points",
)

# Create a mesh grid to plot the model surfaces
size_surf, age_surf = np.meshgrid(
    np.linspace(x1_feature_size.min(), x1_feature_size.max(), 20),
    np.linspace(x2_feature_age.min(), x2_feature_age.max(), 20),
)

# Plot the linear model fit (a plane)
if x_hat_linear is not None:
    c0_lin, c1_lin, c2_lin = x_hat_linear.flatten()
    price_surf_linear = c0_lin + c1_lin * size_surf + c2_lin * age_surf
    ax.plot_surface(
        size_surf,
        age_surf,
        price_surf_linear,
        color="cyan",
        alpha=0.5,
        label="Linear Fit",
    )

# Plot the polynomial model fit (a curved surface)
if x_hat_poly is not None:
    c0_poly, c1_poly, c2_poly, c3_poly = x_hat_poly.flatten()
    price_surf_poly = (
        c0_poly + c1_poly * size_surf + c2_poly * age_surf + c3_poly * size_surf**2
    )
    ax.plot_surface(
        size_surf,
        age_surf,
        price_surf_poly,
        color="magenta",
        alpha=0.5,
        label="Polynomial Fit",
    )

# Final plot settings
ax.set_xlabel("House Size (sq m)")
ax.set_ylabel("House Age (years)")
ax.set_zlabel("Price (in thousands of $)")
ax.set_title("Housing Price vs. Size and Age: Model Comparison")
# Note: Legends for surfaces are tricky in matplotlib, so we use colors and text.
print("Visualizing models: Cyan surface is Linear, Magenta surface is Polynomial.")
plt.show()

# Analysis Question: Based on the 3D plot, which model's surface appears to fit the cloud of data points
# better and why?
# The polynomial model (magenta curved surface) typically fits the data points better than the linear model (cyan plane).
# This is because the polynomial model includes a quadratic term (size²)
# that allows the surface to curve and better follow the natural curvature in the data.
# The linear model is constrained to a flat plane, which cannot capture non-linear relationships between house size and price.
# The polynomial model has more flexibility to bend and twist to minimize the distance between the surface
# and the actual data points in 3D space.

# Analysis Question:
# What is the potential danger of adding many more features or higher-degree polynomial terms
# (e.g., size^3, age^2, size*age) to fit this small dataset? (This relates to the concept of overfitting
# in higher dimensions).

# The main danger is overfitting.
# When we add too many features (like size³, age², size×age interactions) to a small dataset (only 20 houses), the model may:

# Fit the noise in the data rather than the underlying pattern
# Capture random fluctuations that don't represent true relationships
# Perform poorly on new, unseen data despite fitting the training data well
# Become overly complex and lose interpretability
# Suffer from high variance and low bias
# This is particularly problematic with polynomial terms because they can create
# wildly oscillating surfaces that pass through every data point perfectly but fail to generalize.
# With only 20 data points, adding 4-5 polynomial features means we're estimating nearly
# as many parameters as we have observations, which almost guarantees overfitting.
# The model would essentially "memorize" the training data rather than learning generalizable patterns about housing prices.
