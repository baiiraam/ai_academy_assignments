# Math4AI: Linear Algebra - Programming Assignment 4
# Starter Code Template

import numpy as np
from scipy.linalg import null_space  # For verification


# --- Helper Functions for Pretty Printing ---
def print_matrix(name, m):
    """Prints a matrix with its name."""
    if m is None:
        print(f"{name}:\nNone")
    else:
        np.set_printoptions(precision=4, suppress=True)
        print(f"{name}:\n{m}")
    print("-" * 40)


def print_vectors(name, vecs):
    """Prints a list of vectors."""
    print(f"{name}:")
    if not vecs:
        print("[]")
    else:
        for i, v in enumerate(vecs):
            # Reshape to ensure it's a column vector for printing
            print(f"  Vector {i+1}:\n{v.reshape(-1, 1)}")
    print("-" * 40)


# --- Problem Setup ---
# The matrix A and vector b for this assignment
A = np.array([[1.0, 2.0, 3.0, 5.0], [2.0, 4.0, 8.0, 12.0], [3.0, 6.0, 7.0, 13.0]])

b = np.array([4.0, 10.0, 10.0])

print_matrix("Original Matrix A", A)
print_matrix("Original Vector b", b.reshape(-1, 1))

# ====================================================================
# This assignment relies heavily on converting a matrix to its
# Reduced Row Echelon Form (RREF). It's highly recommended to
# create a robust helper function for this first.
# ====================================================================


def to_rref(M):
    """
    Converts a matrix M to its Reduced Row Echelon Form (RREF).

    Args:
        M (np.ndarray): The input matrix.

    Returns:
        np.ndarray: The RREF of M.
    """
    # Make a copy of the matrix to avoid modifying the original
    rref = M.copy().astype(float)
    m, n = rref.shape

    pivot_row = 0
    pivot_col = 0

    while pivot_row < m and pivot_col < n:
        # Find the pivot row
        max_row = pivot_row
        for row in range(pivot_row + 1, m):
            if abs(rref[row, pivot_col]) > abs(rref[max_row, pivot_col]):
                max_row = row

        # Swap rows if necessary
        if max_row != pivot_row:
            rref[[pivot_row, max_row]] = rref[[max_row, pivot_row]]

        if abs(rref[pivot_row, pivot_col]) < 0.00005:
            # Pivot is zero, move to next column
            pivot_col += 1
            continue

        # Normalize the pivot row
        pivot_val = rref[pivot_row, pivot_col]
        rref[pivot_row] = rref[pivot_row] / pivot_val

        # Eliminate other rows
        for row in range(m):
            if row != pivot_row:
                factor = rref[row, pivot_col]
                rref[row] = rref[row] - factor * rref[pivot_row]

        pivot_row += 1
        pivot_col += 1

    return rref


# ====================================================================
# Part 4.1: Finding the Nullspace Basis
# ====================================================================


def find_nullspace_basis(A):
    """
    Finds the basis for the nullspace of matrix A.

    Args:
        A (np.ndarray): The input matrix.

    Returns:
        list: A list of numpy arrays, where each array is a basis vector for the nullspace.
    """
    m, n = A.shape

    # 1. Compute the RREF of A
    rref_A = to_rref(A)
    print_matrix("RREF of A", rref_A)

    basis_vectors = []

    # Identify pivot columns
    pivot_cols = []
    row = 0
    for col in range(n):
        if row < m and abs(rref_A[row, col]) > 0.00005:
            pivot_cols.append(col)
            row += 1

    # Identify free columns (non-pivot columns)
    free_cols = [col for col in range(n) if col not in pivot_cols]

    # Create special solutions for each free variable
    for free_col in free_cols:
        # Create a vector of zeros
        special_sol = np.zeros(n)

        # Set the free variable to 1
        special_sol[free_col] = 1

        # Solve for pivot variables
        for i, pivot_col in enumerate(pivot_cols):
            if i < m:
                special_sol[pivot_col] = -rref_A[i, free_col]

        basis_vectors.append(special_sol)

    return basis_vectors


# --- Calling the function for Part 4.1 ---
print("--- Part 4.1: Finding the Nullspace Basis ---")
nullspace_basis = find_nullspace_basis(A.copy())
print_vectors("Nullspace Basis (from scratch)", nullspace_basis)

# ====================================================================
# Part 4.2: Verification of the Nullspace
# ====================================================================
print("--- Part 4.2: Verification of the Nullspace ---")

if not nullspace_basis:
    print("Nullspace basis is empty or not implemented.")
else:
    for i, v in enumerate(nullspace_basis):
        # For each basis vector v, A @ v should be the zero vector
        result = A @ v
        print(f"Verifying basis vector {i+1}: A @ v_{i+1}")
        print_matrix("Result (should be zero vector)", result.reshape(-1, 1))
        # Check if it's close to zero to handle floating point errors
        print(f"Is close to zero? {np.allclose(result, 0)}\n")

# --- Verification using SciPy ---
print("--- Verifying with SciPy ---")
# scipy.linalg.null_space returns an orthonormal basis for the null space
# The basis might look different but spans the same space.
scipy_ns = null_space(A)
print_matrix("Nullspace basis from SciPy (orthonormal)", scipy_ns)


# ====================================================================
# Part 4.3: Finding a Particular Solution
# ====================================================================


def find_particular_solution(A, b):
    """
    Finds one particular solution to the system Ax = b.

    Args:
        A (np.ndarray): The coefficient matrix.
        b (np.ndarray): The target vector.

    Returns:
        np.ndarray: The particular solution vector x_p, or None if no solution exists.
    """
    m, n = A.shape

    # 1. Create the augmented matrix [A | b]
    augmented_matrix = np.hstack((A, b.reshape(-1, 1)))
    print_matrix("Augmented Matrix [A|b]", augmented_matrix)

    # 2. Compute the RREF of the augmented matrix
    rref_augmented = to_rref(augmented_matrix)
    print_matrix("RREF of Augmented Matrix", rref_augmented)

    particular_solution = None

    # Check for inconsistency
    for i in range(m):
        # Check if row is [0, 0, ..., 0 | c] with c != 0
        if (
            np.all(np.abs(rref_augmented[i, :-1]) < 0.00005)
            and abs(rref_augmented[i, -1]) > 0.00005
        ):
            print("System is inconsistent - no solution exists")
            return None

    # Identify pivot columns
    pivot_cols = []
    row = 0
    for col in range(n):
        if row < m and abs(rref_augmented[row, col]) > 0.00005:
            pivot_cols.append(col)
            row += 1

    # Create particular solution (set free variables to 0)
    x_p = np.zeros(n)

    # Solve for pivot variables
    for i, pivot_col in enumerate(pivot_cols):
        if i < m:
            x_p[pivot_col] = rref_augmented[i, -1]

    particular_solution = x_p

    return particular_solution


# --- Calling the function for Part 4.3 ---
print("--- Part 4.3: Finding a Particular Solution ---")
x_p = find_particular_solution(A.copy(), b.copy())
print_matrix(
    "Particular Solution x_p (from scratch)",
    x_p.reshape(-1, 1) if x_p is not None else None,
)


# ====================================================================
# Part 4.4: Constructing and Verifying the Complete Solution
# ====================================================================
print("--- Part 4.4: Constructing and Verifying the Complete Solution ---")

if x_p is None or not nullspace_basis:
    print(
        "Cannot construct complete solution: particular solution or nullspace is missing."
    )
else:
    # 1. Get the components
    print("Particular solution x_p and nullspace basis are available.")

    # 2. Create a vector x_n from the nullspace
    # Example: x_n = 2*s1 - 3*s2
    # Let's use scalars c1=2, c2=-3 if there are at least two basis vectors
    scalars = [2, -3, 1, -1.5]  # Add more if needed
    x_n = np.zeros(A.shape[1])

    print("\nConstructing x_n from nullspace basis:")
    for i, v in enumerate(nullspace_basis):
        if i < len(scalars):
            print(f"Adding {scalars[i]} * v_{i+1}")
            x_n += scalars[i] * v

    print_matrix("Constructed Nullspace Vector x_n", x_n.reshape(-1, 1))

    # 3. Construct the complete solution x_new = x_p + x_n
    x_new = x_p + x_n
    print_matrix("Complete Solution x_new = x_p + x_n", x_new.reshape(-1, 1))

    # 4. The ultimate test: A @ x_new should equal b
    print("--- The Ultimate Test ---")
    final_result = A @ x_new
    print_matrix("A @ x_new", final_result.reshape(-1, 1))
    print_matrix("Original b (for comparison)", b.reshape(-1, 1))

    is_correct = np.allclose(final_result, b)
    print(f"Verification Check (A @ x_new == b): {is_correct}")
