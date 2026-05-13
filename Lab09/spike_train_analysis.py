import numpy as np
from entropy_utils import compute_entropy

# -----------------------------------------
# Spike Train Binning
# -----------------------------------------

def bin_spike_train(spike_times, duration, bin_size):
    """
    Convert spike times to binned spike count vector.
    """

    bins = np.arange(0, duration + bin_size, bin_size)

    counts, _ = np.histogram(spike_times, bins)

    return counts


def compute_spike_train_entropy(spike_trains, duration, bin_size):
    """
    Compute entropy of spike patterns across trials.
    """

    patterns = []

    for train in spike_trains:
        binned = tuple(
            bin_spike_train(train, duration, bin_size)
        )
        patterns.append(binned)

    # count unique patterns
    unique, counts = np.unique(patterns, axis=0, return_counts=True)

    probabilities = counts / np.sum(counts)

    return compute_entropy(probabilities)