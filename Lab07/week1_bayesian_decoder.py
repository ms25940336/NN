import numpy as np
import matplotlib.pyplot as plt
from config import STIMULI, PRIORS, SPIKE_COUNTS

def gaussian_likelihood(counts, mu, sigma, scale=0.15):
    return scale * np.exp(-0.5 * ((counts - mu) / sigma) ** 2)

# ------------------------------------------------------------
# Provided likelihood curves
# ------------------------------------------------------------
lik_left = gaussian_likelihood(SPIKE_COUNTS, mu=5, sigma=3)
lik_forward = gaussian_likelihood(SPIKE_COUNTS, mu=10, sigma=3)
lik_right = gaussian_likelihood(SPIKE_COUNTS, mu=15, sigma=3)

likelihoods = np.column_stack([lik_left, lik_forward, lik_right])

# ------------------------------------------------------------
# TODO 1: Compute posterior probabilities
# Hint:
# joint = likelihoods[i] * PRIORS
# marginal = sum of joint values
# posterior = joint / marginal
# ------------------------------------------------------------
posteriors = np.zeros_like(likelihoods)

for i in range(len(SPIKE_COUNTS)):
    joint = likelihoods[i] * PRIORS
    marginal = np.sum(joint)
    posteriors[i] = joint / marginal

# ------------------------------------------------------------
# TODO 2: Find MAP decision for each spike count
# Hint:
# - Use np.argmax on each row of posteriors
# - Convert the winning index to the corresponding stimulus label
# ------------------------------------------------------------
map_decisions = [STIMULI[np.argmax(posteriors[i])] for i in range(len(SPIKE_COUNTS))]

# ------------------------------------------------------------
# TODO 3: Print formatted results table
# Show only even spike counts
# Hint:
# You can access posterior values using posteriors[i, 0], posteriors[i, 1], posteriors[i, 2]
# ------------------------------------------------------------
print(f"{'R':>3} {'P(Left|R)':>11} {'P(Fwd|R)':>10} {'P(Right|R)':>11} {'MAP':>8}")
print("-" * 60)

for i, R in enumerate(SPIKE_COUNTS):
    if R % 2 == 0:
        pL = posteriors[i, 0]
        pF = posteriors[i, 1]
        pR = posteriors[i, 2]
        m  = map_decisions[i]
        print(f"{R:3d} {pL:11.3f} {pF:10.3f} {pR:11.3f} {m:>8}")

# ------------------------------------------------------------
# TODO 4: Plot likelihoods and posterior probabilities
# Create two vertically stacked subplots
#
# Top plot:
# - lik_left, lik_forward, lik_right against SPIKE_COUNTS
#
# Bottom plot:
# - posterior curves for Left, Forward, Right
# - add vertical lines where MAP decision changes
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

# Top panel: likelihoods
# axes[0].plot(...)
# axes[0].set_ylabel(...)
# axes[0].set_title(...)
# axes[0].legend()
# axes[0].grid(True)

axes[0].plot(SPIKE_COUNTS, lik_left,    label='Left',    color='tab:blue')
axes[0].plot(SPIKE_COUNTS, lik_forward, label='Forward', color='tab:orange')
axes[0].plot(SPIKE_COUNTS, lik_right,   label='Right',   color='tab:green')
axes[0].set_ylabel('Likelihood P(R|S)')
axes[0].set_title('Likelihood Functions')
axes[0].legend()
axes[0].grid(True)

# Bottom panel: posteriors
# axes[1].plot(...)
# add decision boundary lines
# axes[1].set_xlabel(...)
# axes[1].set_ylabel(...)
# axes[1].set_title(...)
# axes[1].legend()
# axes[1].grid(True)

axes[1].plot(SPIKE_COUNTS, posteriors[:,0], label='Left',    color='tab:blue')
axes[1].plot(SPIKE_COUNTS, posteriors[:,1], label='Forward', color='tab:orange')
axes[1].plot(SPIKE_COUNTS, posteriors[:,2], label='Right',   color='tab:green')

for i in range(1, len(map_decisions)):
    if map_decisions[i] != map_decisions[i-1]:
        boundary = (SPIKE_COUNTS[i-1] + SPIKE_COUNTS[i]) / 2
        axes[1].axvline(boundary, color='gray', linestyle='--', alpha=0.8)

axes[1].set_xlabel('Spike count (R)')
axes[1].set_ylabel('Posterior P(S|R)')
axes[1].set_title('Posterior Probabilities and MAP Boundaries')
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
plt.show()