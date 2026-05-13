import numpy as np
from entropy_utils import compute_entropy

# -----------------------------------------
# Mutual Information
# -----------------------------------------

def compute_joint_distribution(stimulus, response):
    """
    Compute empirical joint probability P(S, R)
    """

    stimulus_vals = np.unique(stimulus)
    response_vals = np.unique(response)

    joint = np.zeros((len(stimulus_vals), len(response_vals)))

    for i, s in enumerate(stimulus_vals):
        for j, r in enumerate(response_vals):
            joint[i, j] = np.sum((stimulus == s) & (response == r))

    # normalize
    joint /= len(stimulus)

    return joint



def compute_mutual_information(stimulus, response):
    """
    Compute I(S ; R)
    """

    # TODO 2: joint distribution
    P_sr = compute_joint_distribution(stimulus, response)

    # TODO 3: marginals
    P_s = np.sum(P_sr, axis=1)
    P_r = np.sum(P_sr, axis=0)

    # TODO 4: MI formula
    I = 0

    for i in range(len(P_s)):
        for j in range(len(P_r)):
            if P_sr[i, j] > 0:
                I += P_sr[i, j] * np.log2(
                    P_sr[i, j] / (P_s[i] * P_r[j])
                )

    return I