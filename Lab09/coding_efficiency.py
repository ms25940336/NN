# coding_efficeiency.py
import numpy as np
from entropy_utils import compute_entropy

# -----------------------------------------
# Coding Efficiency
# -----------------------------------------

def compare_distributions():
    """
    Compare entropy of:
    - Uniform distribution
    - Skewed distribution
    """

    uniform = np.array([0.25, 0.25, 0.25, 0.25])
    skewed = np.array([0.7, 0.1, 0.1, 0.1])

    # Compute entropy
    H_uniform = compute_entropy(uniform)
    H_skewed = compute_entropy(skewed)

    print("Uniform entropy:", H_uniform, "bits")
    print("Skewed entropy:", H_skewed, "bits")

    if H_uniform > H_skewed:
        print("Uniform distribution is more efficient.")
    else:
        print("Skewed distribution is more efficient.")