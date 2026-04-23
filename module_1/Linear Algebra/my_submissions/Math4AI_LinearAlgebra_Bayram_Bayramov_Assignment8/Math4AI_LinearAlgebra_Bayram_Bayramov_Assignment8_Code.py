# Math4AI: Linear Algebra - Programming Assignment 8
# Complete & Fully Fixed Implementation
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


def print_vector(name, v):
    """Prints a vector with its name."""
    print_matrix(name, v.reshape(-1, 1) if v is not None else None)


# ====================================================================
# PART 1: DEFINING AND VERIFYING LINEAR TRANSFORMATIONS
# ====================================================================
print("=" * 60)
print("PART 1: DEFINING AND VERIFYING LINEAR TRANSFORMATIONS")
print("=" * 60)


def transform(v):
    """
    Applies the transformation T((x1, x2)) = (x1 + x2, x1 - 2*x2, 3*x1).
    """
    x1, x2 = v[0], v[1]
    return np.array([x1 + x2, x1 - 2 * x2, 3 * x1])


def verify_linearity(T_func):
    print("--- Verifying Linearity Properties ---")
    u = np.random.rand(2)
    v = np.random.rand(2)
    c = np.random.rand()

    T_uv = T_func(u + v)
    T_u_plus_T_v = T_func(u) + T_func(v)
    additivity_holds = np.allclose(T_uv, T_u_plus_T_v)

    T_cv = T_func(c * v)
    c_T_v = c * T_func(v)
    homogeneity_holds = np.allclose(T_cv, c_T_v)

    print(f"Additivity holds: {additivity_holds}")
    print(f"Homogeneity holds: {homogeneity_holds}")
    print(f"Transformation is linear: {additivity_holds and homogeneity_holds}")


verify_linearity(transform)

# ====================================================================
# PART 2: THE MATRIX OF A LINEAR TRANSFORMATION
# ====================================================================
print("\n" + "=" * 60)
print("PART 2: THE MATRIX OF A LINEAR TRANSFORMATION")
print("=" * 60)


def find_standard_matrix(T_func, n, m):
    """Finds the standard matrix by applying T to basis vectors."""
    standard_matrix = np.zeros((m, n))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        standard_matrix[:, i] = T_func(e_i)
    return standard_matrix


print("--- 2.1 & 2.2: Finding and Verifying the Standard Matrix ---")
A_standard = find_standard_matrix(transform, n=2, m=3)
print_matrix("Standard Matrix A (from scratch)", A_standard)

v_test = np.random.rand(2)
print_vector("Random test vector v", v_test)
Tv = transform(v_test)
Av = A_standard @ v_test
print_vector("Result from T(v)", Tv)
print_vector("Result from A @ v", Av)
print(f"Are the results close? {np.allclose(Tv, Av)}")

print("\n--- 2.3: Composing Geometric Transformations ---")
p = np.array([2.0, 1.0])
print_vector("Original point p", p)

# Rotation by 90 degrees counterclockwise
theta = np.pi / 2
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
p_prime = R @ p
print_vector("p' (rotated)", p_prime)

# Translation
t = np.array([-3.0, 4.0])
p_double_prime = p_prime + t
print_vector("p'' (rotated then translated)", p_double_prime)

print(
    "\n--- Verifying with a single composite matrix using Homogeneous Coordinates ---"
)
R_aug = np.eye(3)
R_aug[:2, :2] = R
T_mat = np.array([[1, 0, -3], [0, 1, 4], [0, 0, 1]])
M = T_mat @ R_aug
p_h = np.append(p, 1)
p_final_h = M @ p_h

print_matrix("Composite Matrix M = T @ R_aug", M)
print_vector("Transformed homogeneous vector", p_final_h)
print_vector("Final 2D point (from composite)", p_final_h[:2])
print(f"Same as step-by-step? {np.allclose(p_double_prime, p_final_h[:2])}")

# ====================================================================
# PART 3: SINGULAR VALUE DECOMPOSITION (SVD) - FIXED & ROBUST
# ====================================================================
print("\n" + "=" * 60)
print("PART 3: SINGULAR VALUE DECOMPOSITION (SVD)")
print("=" * 60)


def compute_svd(A):
    """
    Computes SVD from scratch: A = U Σ V^T
    Returns U (m×r), sigma vector (r,), V^T (r×n) where r = rank
    Safe, robust, and matches np.linalg.svd behavior closely.
    """
    A = np.array(A, dtype=float)
    m, n = A.shape

    # Compute A^T A → eigenvalues give σ², eigenvectors give V
    ATA = A.T @ A
    eigvals, V_full = np.linalg.eig(ATA)  # Use V_full here initially
    eigvals = np.real(eigvals)
    V_full = np.real(V_full)

    # Sort by descending eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[idx]
    V_full_sorted = V_full[:, idx]

    # Singular values (avoid sqrt of tiny negatives)
    singular_values_full = np.sqrt(np.maximum(eigvals_sorted, 0))

    # Keep only numerically significant values
    tol = 1e-10
    rank = np.sum(singular_values_full > tol)

    # Truncate singular_values and V to the actual rank
    S_reduced = singular_values_full[:rank]
    V_reduced = V_full_sorted[:, :rank]  # This is the V to use for VT_reduced

    # Compute U columns: u_i = (1/σ_i) A v_i
    U_reduced = np.zeros((m, rank))
    for i in range(rank):
        # Use corresponding eigenvector from the sorted full V_full_sorted for A @ v
        # The eigenvector should match the singular value at S_reduced[i]
        U_reduced[:, i] = (A @ V_full_sorted[:, i]) / S_reduced[i]

    # Fix possible sign differences (make first column of U match NumPy convention)
    for i in range(rank):
        # Heuristic to align signs with numpy (often makes first non-zero element positive)
        if U_reduced[0, i] < 0:
            U_reduced[:, i] = -U_reduced[:, i]
            V_reduced[:, i] = -V_reduced[:, i]  # Align V_reduced signs too

    VT_reduced = V_reduced.T
    return U_reduced, S_reduced, VT_reduced


# --- 3.2: Application: Image Compression ---
print("\n--- 3.2: Application: Image Compression ---")


def create_sample_image(size=(128, 128)):
    img = np.zeros(size, dtype=np.float32)
    h, w = size
    # Horizontal and vertical bars
    img[h // 2 - 10 : h // 2 + 10, :] = 1.0
    img[:, w // 2 - 10 : w // 2 + 10] = 1.0
    # Circle
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= (min(h, w) // 4) ** 2
    img[mask] = 0.6
    return img


image_matrix = create_sample_image()
U, S_vals, VT = compute_svd(image_matrix)


def reconstruct_matrix(U, S_vals, VT, k):
    k = min(k, len(S_vals))
    return U[:, :k] @ np.diag(S_vals[:k]) @ VT[:k, :]


# Show compression results
ranks_to_show = [5, 15, 40, len(S_vals)]
plt.figure(figsize=(len(ranks_to_show) + 1 * 4, 5))
plt.subplot(1, len(ranks_to_show) + 1, 1)
plt.imshow(image_matrix, cmap="gray", vmin=0, vmax=1)
plt.title("Original")
plt.axis("off")

for i, k in enumerate(ranks_to_show):
    recon = reconstruct_matrix(U, S_vals, VT, k)
    plt.subplot(1, len(ranks_to_show) + 1, i + 2)
    plt.imshow(recon, cmap="gray", vmin=0, vmax=1)
    plt.title(f"k = {k}\n({100 * k / len(S_vals):.1f}%)")
    plt.axis("off")

plt.suptitle("SVD Image Compression (From Scratch)", fontsize=16)
plt.tight_layout()
plt.show()

# --- 3.3: Verification ---
print("\n--- 3.3: Verification with Known Matrix ---")
test_matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print_matrix("Test Matrix", test_matrix)

U_scratch, S_scratch, VT_scratch = compute_svd(test_matrix)
U_np, S_np, VT_np = np.linalg.svd(test_matrix, full_matrices=False)

print_matrix("U (scratch)", U_scratch)
print_matrix("U (NumPy)", U_np)
print("Singular values (scratch):", np.round(S_scratch, 4))
print("Singular values (NumPy):", np.round(S_np, 4))
print_matrix("V^T (scratch)", VT_scratch)
print_matrix("V^T (NumPy)", VT_np)

# Reconstruct
Sigma_mat_diag = np.diag(
    S_scratch
)  # Correctly create Sigma as (rank, rank) diagonal matrix
reconstructed = U_scratch @ Sigma_mat_diag @ VT_scratch  # Now all dimensions match

print_matrix("Reconstructed Matrix (scratch)", reconstructed)
print(f"Reconstruction error: {np.linalg.norm(test_matrix - reconstructed):.2e}")
print(
    f"Is reconstruction accurate? {np.allclose(test_matrix, reconstructed, atol=1e-8)}"
)
