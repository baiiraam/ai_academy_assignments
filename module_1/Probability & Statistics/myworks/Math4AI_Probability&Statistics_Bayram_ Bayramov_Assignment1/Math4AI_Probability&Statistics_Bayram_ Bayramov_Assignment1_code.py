"""
Math4AI: Probability & Statistics
Assignment 1: The Logic of Uncertainty
Completed Code
"""

import numpy as np
import matplotlib.pyplot as plt
import string

# Reproducibility is key for grading
np.random.seed(42)

# ==========================================
# 1. Probability Spaces: The Birthday Paradox
# ==========================================

def theoretical_birthday_prob(k, days=365):
    """
    Computes the exact probability of at least two people sharing a birthday.
    Formula: 1 - P(all unique)
    """
    if k > days:
        return 1.0

    # Calculate P(unique)
    prob_unique = 1.0
    for i in range(k):
        prob_unique *= (days - i) / days

    return 1.0 - prob_unique

def simulate_birthday_prob(k, num_trials=5000, days=365):
    """
    Estimates probability via Monte Carlo.
    """
    collisions = 0

    for _ in range(num_trials):
        # Generate k random birthdays
        birthdays = np.random.randint(0, days, size=k)

        # Check if there are duplicates
        if len(set(birthdays)) < k:
            collisions += 1

    return collisions / num_trials

def run_section_1():
    print("--- Section 1: Birthday Paradox ---")
    k_values = range(1, 80) # Sufficient range to see the curve
    theory = []
    sim = []

    print("Running simulation (this may take a few seconds)...")
    for k in k_values:
        theory.append(theoretical_birthday_prob(k))
        sim.append(simulate_birthday_prob(k, num_trials=1000))

    # Find k where probability crosses 50%
    cross_k = None
    for i, prob in enumerate(theory):
        if prob >= 0.5:
            cross_k = k_values[i]
            break

    print(f"Theoretical 50% crossover at k = {cross_k}")

    # Plotting for Report
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, theory, label='Theoretical', linewidth=2)
    plt.plot(k_values, sim, 'o', label='Simulated', alpha=0.5, markersize=3)
    plt.axhline(0.5, color='r', linestyle='--', label='50% Threshold')
    plt.axvline(cross_k, color='g', linestyle=':', label=f'k = {cross_k}')
    plt.xlabel('Group Size ($k$)')
    plt.ylabel('P(Collision)')
    plt.title('Birthday Paradox: Theory vs Simulation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('birthday_plot')
    plt.show()

    return cross_k

# ==========================================
# 2. Conditional Probability: Spam Filter
# ==========================================

# Vocabulary and Probabilities from Assignment Table
vocab = {
    "free":    {"P_w_spam": 0.4,  "P_w_ham": 0.05},
    "prize":   {"P_w_spam": 0.2,  "P_w_ham": 0.01},
    "meeting": {"P_w_spam": 0.05, "P_w_ham": 0.5}
}

P_spam_prior = 0.3
P_ham_prior = 0.7

def classify_email(email_words):
    """
    Returns unnormalized scores for Spam and Ham.
    Score = Prior * Product(P(w|class))
    """
    # Initialize scores with Priors
    score_spam = P_spam_prior
    score_ham = P_ham_prior

    for word in email_words:
        if word in vocab:
            score_spam *= vocab[word]["P_w_spam"]
            score_ham *= vocab[word]["P_w_ham"]

    return score_spam, score_ham

def run_section_2():
    print("\n--- Section 2: Spam Classification ---")
    message = ["free", "meeting"]

    spam_score, ham_score = classify_email(message)

    print(f"Message: {message}")
    print(f"Score (Spam): {spam_score:.6f}")
    print(f"Score (Ham):  {ham_score:.6f}")

    winner = "Spam" if spam_score > ham_score else "Ham"
    print(f"Prediction: {winner}")

    return spam_score, ham_score

# ==========================================
# 3. Discrete RVs: Caesar Cipher
# ==========================================

ENGLISH_CORPUS = """
probability theory is the branch of mathematics concerned with probability
although there are several different probability interpretations probability
theory treats the concept in a rigorous mathematical manner by expressing it
through a set of axioms typically these axioms formalize probability in terms
of a probability space which assigns a measure taking values between zero and
one termed the probability measure to a set of outcomes called the sample space
""".replace("\n", " ")

CIPHER_TEXT = "ZNK COXYZ YKV ZU YURBOTM ZNK VXKIOYOUT VXUHRKS OY ZU KROSOTGZK ZNK OSVUYYOHRK"

def get_pmf(text):
    """
    Computes P(Letter) for all A-Z. Returns a vector of length 26.
    """
    # Helper to stick to A-Z
    text = ''.join([c.upper() for c in text if c.isalpha()])
    total = len(text)
    counts = np.zeros(26)

    # Populate counts
    for char in text:
        idx = ord(char) - ord('A')
        if 0 <= idx < 26:
            counts[idx] += 1

    return counts / total if total > 0 else counts

def decrypt_caesar(cipher_text, ground_truth_pmf):
    best_shift = 0
    best_sse = float('inf')
    best_text = ""

    # Try all possible shifts
    for k in range(26):
        # Construct candidate text by shifting cipher_text by -k
        candidate_chars = []
        for char in cipher_text:
            if char.isalpha():
                shifted_idx = (ord(char.upper()) - ord('A') - k) % 26
                candidate_chars.append(chr(shifted_idx + ord('A')))
            else:
                candidate_chars.append(char)

        candidate_text = "".join(candidate_chars)

        # Compute PMF of candidate text
        candidate_pmf = get_pmf(candidate_text)

        # Compute SSE against ground_truth_pmf
        sse = np.sum((candidate_pmf - ground_truth_pmf) ** 2)

        if sse < best_sse:
            best_sse = sse
            best_shift = k
            best_text = candidate_text

    return best_shift, best_text

def run_section_3():
    print("\n--- Section 3: Cryptanalysis ---")
    english_pmf = get_pmf(ENGLISH_CORPUS)

    shift, decrypted = decrypt_caesar(CIPHER_TEXT, english_pmf)
    decrypted_pmf = get_pmf(decrypted)

    print(f"Detected Shift: {shift}")
    print(f"Decrypted Text: {decrypted}")
    print(f"Original Cipher: {CIPHER_TEXT}")

    # Comparison Plot
    letters = list(string.ascii_uppercase)
    x = np.arange(len(letters))

    plt.figure(figsize=(12, 5))
    width = 0.35
    plt.bar(x - width/2, english_pmf, width, label='English Ground Truth')
    plt.bar(x + width/2, decrypted_pmf, width, label='Decrypted Text')
    plt.xticks(x, letters)
    plt.xlabel('Letter')
    plt.ylabel('Probability')
    plt.title('PMF Fingerprint Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('pmf_comparison')
    plt.show()

    return shift, decrypted

if __name__ == "__main__":
    print("Math4AI: Probability & Statistics - Assignment 1")
    print("=" * 50)

    # Run all sections
    k_50 = run_section_1()
    spam_score, ham_score = run_section_2()
    shift, decrypted = run_section_3()