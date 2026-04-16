# ============================================
# Math4AI - Programming Assignment 1
# Vector Operations & Semantic Similarity
# ============================================

# --- Imports ---
import numpy as np

# --- Sample Word Vectors (3D Example) ---
# Note: Values here are just for demonstration
v_king = [0.8, 0.65, 0.0]
v_man = [0.6, 0.4, 0.0]
v_woman = [0.7, 0.3, 0.2]
v_queen = [0.9, 0.55, 0.2]

# =====================================================
# PART 1.1: King - Man + Woman Analogy (From Scratch)
# =====================================================


def vector_add(u, v):
    """Add two vectors u and v (lists)."""
    # TODO: Implement element-wise addition
    # We should check the lists to be the same length.
    # Also, we should return a list, since the result of the addition should be a list.
    if len(u) != len(v):
        raise ValueError("Cannot add vectors that have different lengths")
    result = []
    i = 0
    while i < len(u):
        result.append(u[i] + v[i])
        i += 1
    return result


def vector_sub(u, v):
    """Subtract vector v from u (lists)."""
    # TODO: Implement element-wise subtraction
    if len(u) != len(v):
        raise ValueError("Cannot subtract vectors that have different lengths")
    result = []
    i = 0
    while i < len(u):
        result.append(u[i] - v[i])
        i += 1
    return result


# --- Analogy computation ---
# v_result = king - man + woman
# TODO: Use your vector_sub and vector_add here
# vector_sub(v_king, v_man) -> v_king - v_man
# vector_add(previous, v_woman) -> previous + v_woman
v_result = vector_add(vector_sub(v_king, v_man), v_woman)

print("Analogy result (from scratch):", v_result)


# =====================================================
# PART 1.2: Cosine Similarity (From Scratch)
# =====================================================


def dot_product(u, v):
    """Compute the dot product of u and v."""
    # TODO: Implement dot product using a loop
    # We will use a loop to iterate through elements, multiply them, and sum the results.
    # The lengths of u and v must be the same.
    if len(u) != len(v):
        raise ValueError("Cannot compute dot product of vectors with different lengths")
    result = 0
    i = 0
    while i < len(u):
        result += u[i] * v[i]
        i += 1
    return result


def norm(u):
    """Compute the Euclidean norm (L2 norm) of vector u."""
    # TODO: Implement norm from scratch
    # Euclidean norm of a vector is defined as the sum of squares of that vector's elements.
    sum_of_squares = 0
    i = 0
    while i < len(u):
        sum_of_squares += u[i] ** 2
        i += 1
    return sum_of_squares**0.5  # or math module could be used ---- math.sqrt()


def cosine_similarity(u, v):
    """Compute cosine similarity between u and v."""
    # TODO: Use dot_product() and norm() here
    # Cosine similarity is defined as the dot product of u and v divided by the product of their norms
    # (similarity of two vectors, whether they are looking at the same direction or not).

    # In case either vector has zero norm, we should handle that to avoid division by zero.
    # We can return directly with ternary operator.
    return (
        dot_product(u, v) / (norm(u) * norm(v)) if norm(u) != 0 and norm(v) != 0 else 0
    )


# --- Cosine similarity between analogy result & queen ---
similarity_scratch = cosine_similarity(
    v_result, v_queen
)  # TODO: Compute using your function
print("Cosine similarity (from scratch):", similarity_scratch)


# =====================================================
# PART 1.3: NumPy Verification
# =====================================================

# Convert to numpy arrays
np_king = np.array(v_king)
np_man = np.array(v_man)
np_woman = np.array(v_woman)
np_queen = np.array(v_queen)

# --- Analogy computation with NumPy ---
np_result = np_king - np_man + np_woman  # TODO: Implement with NumPy
print("Analogy result (NumPy):", np_result)

# --- Cosine similarity with NumPy ---
# TODO: Use np.dot() and np.linalg.norm()

# We can return directly with ternary operator, handling zero in denominator.
similarity_numpy = (
    np.dot(np_result, np_queen) / (np.linalg.norm(np_result) * np.linalg.norm(np_queen))
    if np.linalg.norm(np_result) != 0 and np.linalg.norm(np_queen) != 0
    else 0
)

# or, we can do function (no difference, just readability-wise)


def cosine_similarity_using_numpy(u, v):
    """Compute cosine similarity between u and v using NumPy."""
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    zero_flag = norm_u == 0 or norm_v == 0
    if zero_flag:
        return 0
    else:  # In else part, we could have returned directly, but for clarity and readability, we can use variables.
        return dot_product / (norm_u * norm_v)


print("Cosine similarity (NumPy):", similarity_numpy)


# =====================================================
# Verification
# =====================================================
# TODO: Print comparison of scratch vs NumPy results
print("\n" * 3, "--- Verification Results NumPy vs. Python ---")
print(f"From scratch result: {v_result}")
print(f"NumPy result: {np_result}")
print(f"From scratch cosine similarity: {similarity_scratch:.6f}")
print(f"NumPy cosine similarity: {similarity_numpy:.6f}")

# Check if results are approximately equal. Allclose is used for floating point comparison, with a tolerance. \
# We could have used exact equality (==) for integer vectors, but let's assume there might be some errors because of the float specifications.
print(f"Results match: {np.allclose(v_result, np_result, rtol=1e-10)}")
