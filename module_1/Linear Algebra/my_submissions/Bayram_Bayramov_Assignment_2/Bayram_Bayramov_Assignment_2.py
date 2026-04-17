# ============================================
# Math4AI - Programming Assignment 2
# Systems of Linear Equations & Model Fitting
# ============================================

# --- Imports ---
import numpy as np

# --- Example System ---
A = [[2, 1, 3], [4, 4, 7], [2, 5, 9]]

b = [1, 1, 3]

# =====================================================
# PART 2.1: Gaussian Elimination from Scratch
# =====================================================


def gaussian_elimination(A, b):
    """
    Solve Ax = b using Gaussian elimination with partial pivoting.
    Returns:
        - list: solution vector x if unique solution exists
        - str: 'No solution' if inconsistent
        - str: 'Infinite solutions' if system has free variables
    """
    # Convert to numpy arrays
    if isinstance(A, list):
        A = np.array(A, dtype=float)
    if isinstance(b, list):
        b = np.array(b, dtype=float)

    n = len(b)

    # Create augmented matrix [A | b]
    # np.c_ is used to concatenate two arrays column-wise.
    # Since b is a 1D array, we reshape it to a column vector for it to broadcast correctly with A.
    augmented_matrix = np.c_[A, b.reshape(-1, 1)]

    # Forward elimination with partial pivoting
    for i in range(n):
        # Partial pivoting: find row with maximum value in current column
        max_row = i
        for p in range(i + 1, n):
            if abs(augmented_matrix[p, i]) > abs(augmented_matrix[max_row, i]):
                max_row = p

        # Swap rows if necessary
        if max_row != i:
            # Swapping row {i} with max_row
            augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]

        # Check for zero pivot (after pivoting)
        if abs(augmented_matrix[i, i]) < 1e-12:
            # Check if system is inconsistent
            if abs(augmented_matrix[i, n]) > 1e-12:
                return "No solution"
            else:
                return "Infinite solutions"

        # Eliminate entries below the pivot
        for j in range(i + 1, n):
            scaling_factor = augmented_matrix[j, i] / augmented_matrix[i, i]
            # Eliminating row {j} using scaling_factor
            augmented_matrix[j, i:] -= scaling_factor * augmented_matrix[i, i:]

    # Check for zero rows in the final matrix
    for i in range(n):
        if np.all(np.abs(augmented_matrix[i, :n]) < 1e-12):
            if abs(augmented_matrix[i, n]) > 1e-12:
                return "No solution"
            else:
                return "Infinite solutions"

    # Back substitution
    x = np.zeros(n)

    # Start from last row and move upwards
    x[n - 1] = augmented_matrix[n - 1, n] / augmented_matrix[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        sum_ax = 0.0
        for j in range(i + 1, n):
            sum_ax += augmented_matrix[i, j] * x[j]
        x[i] = (augmented_matrix[i, n] - sum_ax) / augmented_matrix[i, i]

    return x.tolist()


# --- Solve using your function ---
solution_scratch = gaussian_elimination(A, b)
print("Solution (from scratch):", solution_scratch)


# =====================================================
# PART 2.2: NumPy Verification
# =====================================================

# Convert to NumPy arrays
np_A = np.array(A, dtype=float)
np_b = np.array(b, dtype=float)

try:
    np_solution = np.linalg.solve(np_A, np_b)
    print("Solution (NumPy):", np_solution)
except np.linalg.LinAlgError as e:
    print("NumPy could not solve the system:", e)


# =====================================================
# Verification
# =====================================================
# TODO: Compare scratch implementation result with NumPy result if unique
if isinstance(solution_scratch, list) or isinstance(solution_scratch, np.ndarray):
    # TODO: Check closeness between the two solutions
    print(np.allclose(solution_scratch, np_solution))
