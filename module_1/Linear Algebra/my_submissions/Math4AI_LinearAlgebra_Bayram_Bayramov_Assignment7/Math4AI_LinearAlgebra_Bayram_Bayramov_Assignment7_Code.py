# Math4AI: Linear Algebra - Programming Assignment 7
# Complete Implementation

import numpy as np

# --- Helper Functions for Pretty Printing ---
def print_matrix(name, m):
    """Prints a matrix or vector with its name."""
    if m is None:
        print(f"{name}:\nNone (or not implemented)")
    else:
        np.set_printoptions(precision=4, suppress=True)
        print(f"{name}:\n{m}")
    print("-" * 40)

def print_eigenpairs(name, eigenvalues, eigenvectors):
    """Prints a set of eigenvalues and their corresponding eigenvectors."""
    print(f"{name}:")
    if eigenvalues is None or eigenvectors is None:
        print("None (or not implemented)")
        print("-" * 40)
        return

    for i, val in enumerate(eigenvalues):
        # eigenvectors are columns in the matrix
        vec = eigenvectors[:, i] if isinstance(eigenvectors, np.ndarray) else eigenvectors[i]
        print(f"  Eigenvalue λ_{i+1} = {val:.4f}")
        print(f"  Eigenvector v_{i+1} =\n{vec.reshape(-1, 1)}")
    print("-" * 40)

# ====================================================================
# Problem Setup for All Parts
# ====================================================================

# Matrix for Part 1 and 3
A = np.array([
    [4., -2.],
    [1., 1.]
])

# Matrix for Part 2 (Diagonalizable)
B = np.array([
    [1., 0., 1.],
    [0., 1., 0.],
    [1., 0., 1.]
])

# Matrix for Part 2 (Not Diagonalizable / Defective)
C = np.array([
    [1., 1.],
    [0., 1.]
])

# ====================================================================
# Prerequisite: Nullspace Function (Students must implement or reuse)
# ====================================================================

def find_nullspace_basis(M):
    """
    Finds the basis for the nullspace of matrix M.
    This is required for find_eigenvectors.
    """
    # Convert to numpy array
    M = np.array(M, dtype=float)
    m, n = M.shape

    # Create augmented matrix for RREF
    aug = M.copy()

    # Forward elimination with partial pivoting
    pivot_row = 0
    for col in range(n):
        # Find pivot row
        max_row = pivot_row
        for row in range(pivot_row, m):
            if abs(aug[row, col]) > abs(aug[max_row, col]):
                max_row = row

        # Swap rows if necessary
        if max_row != pivot_row:
            aug[[pivot_row, max_row]] = aug[[max_row, pivot_row]]

        if abs(aug[pivot_row, col]) < 1e-12:
            continue

        # Normalize pivot row
        pivot_val = aug[pivot_row, col]
        aug[pivot_row] = aug[pivot_row] / pivot_val

        # Eliminate other rows
        for row in range(m):
            if row != pivot_row:
                factor = aug[row, col]
                aug[row] = aug[row] - factor * aug[pivot_row]

        pivot_row += 1

    # Identify pivot and free columns
    pivot_cols = []
    free_cols = []
    row = 0
    for col in range(n):
        if row < m and abs(aug[row, col]) > 1e-12:
            pivot_cols.append(col)
            row += 1
        else:
            free_cols.append(col)

    # Create special solutions
    basis_vectors = []
    for free_col in free_cols:
        special_sol = np.zeros(n)
        special_sol[free_col] = 1

        for i, pivot_col in enumerate(pivot_cols):
            if i < m:
                special_sol[pivot_col] = -aug[i, free_col]

        basis_vectors.append(special_sol)

    return basis_vectors

# ====================================================================
# PART 1: FINDING EIGENVALUES AND EIGENVECTORS
# ====================================================================
print("="*60)
print("PART 1: FINDING EIGENVALUES AND EIGENVECTORS")
print("="*60)
print_matrix("Matrix A for Part 1", A)

# --- 1.1: Eigenvalues from the Characteristic Equation ---
def find_eigenvalues_2x2(A):
    """
    Computes the eigenvalues of a 2x2 matrix using the characteristic equation.
    λ^2 - tr(A)λ + det(A) = 0
    """
    # Calculate trace and determinant
    trace = A[0, 0] + A[1, 1]
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

    # Solve quadratic equation: λ^2 - trace*λ + det = 0
    discriminant = trace**2 - 4 * det

    if discriminant >= 0:
        # Real eigenvalues
        sqrt_disc = np.sqrt(discriminant)
        lambda1 = (trace + sqrt_disc) / 2
        lambda2 = (trace - sqrt_disc) / 2
        return [lambda1, lambda2]
    else:
        # Complex eigenvalues
        sqrt_disc = np.sqrt(-discriminant)
        real_part = trace / 2
        imag_part = sqrt_disc / 2
        lambda1 = complex(real_part, imag_part)
        lambda2 = complex(real_part, -imag_part)
        return [lambda1, lambda2]

# --- 1.2: Eigenvectors from the Nullspace ---
def find_eigenvectors(A, eigenvalues):
    """
    Finds the eigenvectors for a matrix given its eigenvalues.
    """
    eigenvectors = []

    for lambda_val in eigenvalues:
        # Construct B = A - λI
        I = np.eye(A.shape[0])
        B = A - lambda_val * I

        # Find nullspace of B
        nullspace_basis = find_nullspace_basis(B)

        # Add eigenvectors to the list
        eigenvectors.extend(nullspace_basis)

    return eigenvectors


print("--- 1.1 & 1.2: Finding Eigenpairs from Scratch ---")
eigvals_scratch = find_eigenvalues_2x2(A)
eigvecs_scratch = find_eigenvectors(A, eigvals_scratch)
# For consistent printing, let's combine the eigenvectors into a matrix
eigvecs_matrix_scratch = np.column_stack(eigvecs_scratch) if eigvecs_scratch else None
print_eigenpairs("Eigenpairs (from scratch)", eigvals_scratch, eigvecs_matrix_scratch)


# --- 1.3: Verification ---
print("--- 1.3: Verification ---")
print("Verifying A*v = λ*v:")
if eigvals_scratch and eigvecs_scratch:
    for i in range(len(eigvals_scratch)):
        l, v = eigvals_scratch[i], eigvecs_scratch[i]
        Av = A @ v
        lv = l * v
        print(f"For λ = {l:.4f}:")
        print_matrix("  A @ v", Av.reshape(-1, 1))
        print_matrix("  λ * v", lv.reshape(-1, 1))
        print(f"  Are they close? {np.allclose(Av, lv)}\n")
else:
    print("Cannot verify, scratch implementation is missing.\n")

print("--- Comparing with NumPy ---")
eigvals_np, eigvecs_np = np.linalg.eig(A)
print_eigenpairs("Eigenpairs (from NumPy)", eigvals_np, eigvecs_np)


# ====================================================================
# PART 2: DIAGONALIZATION
# ====================================================================
print("\n" + "="*60)
print("PART 2: DIAGONALIZATION")
print("="*60)

def diagonalize(A):
    """
    Performs diagonalization of matrix A, if possible.
    Returns S, Lambda, S_inv if diagonalizable, otherwise returns (None, None, None).
    """
    # Find eigenvalues and eigenvectors using NumPy
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Check if matrix is diagonalizable
    # A matrix is diagonalizable if it has n linearly independent eigenvectors
    S = eigenvectors
    if np.linalg.matrix_rank(S) == A.shape[0]:
        # Construct diagonal matrix
        Lambda = np.diag(eigenvalues)

        # Compute inverse of S
        S_inv = np.linalg.inv(S)

        return S, Lambda, S_inv
    else:
        return None, None, None

# --- 2.2: Verification and A Defective Case ---
print("--- 2.2: Verification ---")
print_matrix("Diagonalizable Matrix B", B)
S, L, S_inv = diagonalize(B)
if S is not None:
    print_matrix("Eigenvector Matrix S", S)
    print_matrix("Eigenvalue Matrix Lambda", L)
    print_matrix("Inverse Eigenvector Matrix S^-1", S_inv)

    # Verify by checking if S @ L @ S_inv == B
    B_reconstructed = S @ L @ S_inv
    print_matrix("Reconstructed B = S @ Lambda @ S^-1", B_reconstructed)
    print(f"Is reconstructed B close to original B? {np.allclose(B, B_reconstructed)}")
else:
    print("Diagonalization of B failed or is not implemented.")

print("\n--- Testing a Defective (Non-Diagonalizable) Case ---")
print_matrix("Non-Diagonalizable Matrix C", C)
S_C, L_C, S_inv_C = diagonalize(C)
if S_C is None:
    print("Function correctly identified that C is not diagonalizable.")
else:
    print("Function incorrectly diagonalized C.")

# ====================================================================
# PART 3: THE POWER METHOD
# ====================================================================
print("\n" + "="*60)
print("PART 3: THE POWER METHOD")
print("="*60)

# --- 3.1: Implement the Power Method ---
def power_iteration(A, num_iterations: int):
    """
    Estimates the dominant eigenvector of a matrix A.
    """
    n = A.shape[0]
    # Initialize random vector
    b_k = np.random.rand(n)

    for _ in range(num_iterations):
        # Calculate the matrix-by-vector product Ab
        b_k1 = A @ b_k

        # Calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # Re-normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k

# --- 3.2: Estimating the Eigenvalue with the Rayleigh Quotient ---
print("--- 3.1 & 3.2: Finding Dominant Eigenpair ---")
dominant_eigenvector = power_iteration(A, 100)

if dominant_eigenvector is not None:
    # Rayleigh Quotient: λ = (v.T @ A @ v) / (v.T @ v)
    v = dominant_eigenvector
    numerator = v.T @ A @ v
    denominator = v.T @ v
    dominant_eigenvalue = numerator / denominator
    print(f"Estimated Dominant Eigenvalue (from Rayleigh Quotient): {dominant_eigenvalue:.4f}")
    print_matrix("Estimated Dominant Eigenvector (from Power Iteration)", v.reshape(-1, 1))
else:
    print("Power iteration not implemented.")

# --- 3.3: Verification and Convergence ---
print("\n--- 3.3: Verification and Convergence ---")
if eigvals_scratch:
    # Get the exact dominant eigenvalue (largest in absolute value) from Part 1
    exact_dom_val = max(eigvals_np, key=abs)
    print(f"Exact Dominant Eigenvalue: {exact_dom_val:.4f}\n")

    for iterations in [5, 10, 20, 50]:
        v_est = power_iteration(A, iterations)
        if v_est is not None:
            l_est = (v_est.T @ A @ v_est) / (v_est.T @ v_est)
            print(f"After {iterations} iterations:")
            print(f"  Estimated λ = {l_est:.4f}")
        else:
            print("Power iteration not implemented.")
            break
else:
    print("Cannot verify convergence, Part 1 results missing.")