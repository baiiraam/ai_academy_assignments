# Math4AI: Linear Algebra - Programming Assignment 3
# Starter Code Template

import numpy as np
from scipy.linalg import lu as scipy_lu  # Used for verification

# --- Helper Function for Pretty Printing ---
def print_matrix(name, m):
    """
    Helper function to print a matrix with its name.
    Handles None for non-invertible matrices.
    """
    print(f"{name}:")
    if m is None:
        print("None (Matrix is singular or function not implemented)")
    else:
        # Set print options for better readability
        np.set_printoptions(precision=4, suppress=True)
        print(m)
    print("-" * 30)

# --- Problem Setup ---
# The matrix A for this assignment
A = np.array([
    [2., 1., 3.],
    [4., 4., 7.],
    [2., 5., 9.]
])

print_matrix("Original Matrix A", A)

# ====================================================================
# Part 3.1: Matrix Inverse via Gauss-Jordan Elimination
# ====================================================================

def invert_matrix(A):
    """
    Computes the inverse of a square matrix A using Gauss-Jordan elimination.

    Args:
        A (np.ndarray): A square numpy array.

    Returns:
        np.ndarray: The inverse of A, or None if A is singular.
    """
    # Ensure the matrix is a float type for division
    A = A.astype(float)

    # Check if the matrix is square
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("Input matrix must be square.")

    # 1. Create the augmented matrix [A | I]
    identity = np.identity(n)
    augmented_A = np.hstack((A, identity))
    print("Initial Augmented Matrix [A|I]:")
    print(augmented_A)
    print("\nStarting Gauss-Jordan Elimination...")

    # --- YOUR CODE HERE ---
    # Implement the Gauss-Jordan elimination algorithm to transform
    # the left side (A) of the augmented matrix into the identity matrix.
    # The right side will then become the inverse.

    # Suggested Steps:
    # 1. Forward Elimination (getting upper triangular form)
    #    Iterate through each row (pivot row `i` from 0 to n-1):
    #    a. Find the pivot element `augmented_A[i, i]`.
    #    b. If the pivot is zero, the matrix is singular. You may need to
    #       implement pivoting (row swapping) for a more robust solution,
    #       but for now, you can return None.
    #    c. Normalize the pivot row by dividing it by the pivot element.
    #    d. For every other row `j` (where `j != i`):
    #       Subtract `augmented_A[j, i]` times the pivot row from row `j`.

    # 2. After the loop, the left side should be the identity matrix.
    #    If not, something went wrong.

    # 3. Extract the inverse matrix from the right side of the
    #    transformed augmented matrix.

    for i in range(n):
        # Find pivot element
        pivot = augmented_A[i, i]

        # Check if pivot is zero (matrix is singular)
        if abs(pivot) < 1e-12:
            print(f"Zero pivot encountered at position ({i},{i}). Matrix is singular.")
            return None

        # Normalize the pivot row
        augmented_A[i] = augmented_A[i] / pivot

        # Eliminate other rows
        for j in range(n):
            if j != i:
                factor = augmented_A[j, i]
                augmented_A[j] = augmented_A[j] - factor * augmented_A[i]

    # Extract the inverse from the right side
    inverse_A = augmented_A[:, n:]

    # Verify that left side is identity matrix
    left_side = augmented_A[:, :n]
    if np.allclose(left_side, np.identity(n)):
        print("Successfully transformed left side to identity matrix.")
        return inverse_A
    else:
        print("Warning: Left side is not identity matrix. Inverse may be incorrect.")
        return inverse_A

# --- Calling the function for Part 3.1 ---
print("--- Part 3.1: Matrix Inverse from Scratch ---")
A_inv_scratch = invert_matrix(A.copy()) # Use a copy to keep original A intact
print_matrix("Inverse A (from scratch)", A_inv_scratch)


# ====================================================================
# Part 3.2: LU Decomposition from Scratch
# ====================================================================

def lu_decomposition(A):
    """
    Performs LU decomposition of a square matrix A using Doolittle's algorithm.

    Args:
        A (np.ndarray): A square numpy array.

    Returns:
        (np.ndarray, np.ndarray): A tuple of (L, U) matrices.
    """
    # Ensure the matrix is a float type
    A = A.astype(float)

    # Check if the matrix is square
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("Input matrix must be square.")

    # Initialize L as an identity matrix and U as a zero matrix
    L = np.identity(n)
    U = np.zeros((n, n))

    # --- YOUR CODE HERE ---
    # Implement the Doolittle algorithm.
    # Iterate through the matrix to calculate the elements of L and U.

    # Suggested Steps:
    # For each `i` from 0 to n-1:
    #  1. Calculate the i-th row of U:
    #     For `j` from `i` to `n-1`:
    #       U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))
    #
    #  2. Calculate the i-th column of L:
    #     For `j` from `i+1` to `n-1`:
    #       L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]
    #
    # Note: If U[i, i] is zero at any point, the decomposition may fail
    # without pivoting. Assume the given matrix A works without pivoting.

    for i in range(n):
        # Calculate U[i, j] for j = i to n-1
        for j in range(i, n):
            sum_val = 0.0
            for k in range(i):
                sum_val += L[i, k] * U[k, j]
            U[i, j] = A[i, j] - sum_val

        # Calculate L[j, i] for j = i+1 to n-1
        for j in range(i + 1, n):
            sum_val = 0.0
            for k in range(i):
                sum_val += L[j, k] * U[k, i]
            L[j, i] = (A[j, i] - sum_val) / U[i, i]

    return L, U

# --- Calling the function for Part 3.2 ---
print("--- Part 3.2: LU Decomposition from Scratch ---")
L_scratch, U_scratch = lu_decomposition(A.copy())
print_matrix("L (from scratch)", L_scratch)
print_matrix("U (from scratch)", U_scratch)


# ====================================================================
# Part 3.3: NumPy Verification
# ====================================================================
print("--- Part 3.3: NumPy Verification ---")

# 1. Verifying the Matrix Inverse
print("Verifying Matrix Inverse...")
A_inv_numpy = np.linalg.inv(A)
print_matrix("Inverse A (NumPy)", A_inv_numpy)

# 2. Verifying the LU Decomposition
print("Verifying LU Decomposition...")
# We check by multiplying L and U and see if we get back A
if L_scratch is not None and U_scratch is not None:
    product_LU = L_scratch @ U_scratch
    print_matrix("L @ U (from scratch)", product_LU)
    print_matrix("Original A (for comparison)", A)

    # A programmatic check for correctness
    is_correct = np.allclose(A, product_LU)
    print(f"Verification Check (A == L @ U): {is_correct}\n")
else:
    print("LU decomposition not yet implemented.\n")

# Optional: Compare with SciPy's LU decomposition
# Note: SciPy's version may include a permutation matrix P.
# P, L_scipy, U_scipy = scipy_lu(A)
# print_matrix("L (SciPy)", L_scipy)
# print_matrix("U (SciPy)", U_scipy)
# print(f"Verification with SciPy (P@L@U == A): {np.allclose(A, P @ L_scipy @ U_scipy)}")