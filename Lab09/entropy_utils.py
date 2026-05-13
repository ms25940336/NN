import numpy as np

# -----------------------------------------
# Entropy Computation
# -----------------------------------------

def compute_entropy(probabilities):
    """
    Compute Shannon entropy H(X) in bits.

    Parameters:
        probabilities (array-like): probability distribution (must sum to 1)

    Returns:
        float: entropy in bits
    """

    # TODO 1:
    probabilities = probabilities[probabilities > 0]
    
    # TODO 2:
    # Compute entropy using:
    # H = -sum(p * log2(p))
    
    H = -np.sum(probabilities * np.log2(probabilities))

    return H


def normalize_distribution(counts):
    """
    Convert raw counts into probability distribution.
    """

    # TODO 3:
    # Normalize counts so they sum to 1
    
    counts = np.array(counts)

    # TODO 3: normalize
    probabilities = counts / np.sum(counts)

    return probabilities