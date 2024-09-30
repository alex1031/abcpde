import numpy as np 
from scipy.stats import rankdata
from scipy.special import kl_div
from scipy.spatial.distance import pdist, squareform, cdist

def wasserstein_distance(simulated_sample: np.ndarray, observed_sample: np.ndarray) -> float:
    # Mean Difference between simulated and observed
    distance = np.mean(np.abs(simulated_sample - observed_sample), axis=0)
    return distance

def maximum_mean_discrepancy(simulated_sample: np.ndarray, observed_sample: np.ndarray, gamma = 1.0) -> float:
    n = simulated_sample.shape[1]
    distances_XX = np.zeros((n, 100, 100))  # Distances within simulated_sample
    distances_YY = np.zeros((n, 100, 100))  # Distances within observed_sample
    distances_XY = np.zeros((n, 100, 100))  # Distances between simulated_sample and observed_sample

    for i in range(n):
        distances_XX[i] = squareform(pdist(simulated_sample[:, i, np.newaxis], metric='euclidean'))
        distances_YY[i] = squareform(pdist(observed_sample[:, i, np.newaxis], metric='euclidean'))
        distances_XY[i] = cdist(simulated_sample[:, i, np.newaxis], observed_sample[:, i, np.newaxis], metric='euclidean')

    # Gaussian kernel
    KXX = np.exp(-distances_XX ** 2 / (2 * gamma ** 2))
    KYY = np.exp(-distances_YY ** 2 / (2 * gamma ** 2))
    KXY = np.exp(-distances_XY ** 2 / (2 * gamma ** 2))

    mean_KXX = np.mean(KXX, axis=(1, 2))  # Mean of kernel within simulated_sample
    mean_KYY = np.mean(KYY, axis=(1, 2))  # Mean of kernel within observed_sample
    mean_KXY = np.mean(KXY, axis=(1, 2))  # Mean of kernel between simulated_sample and observed_sample

    # MMD vectorized for each column
    mmd_values = mean_KXX + mean_KYY - 2 * mean_KXY

    return mmd_values

def cramer_von_mises(simulated_sample: np.ndarray, observed_sample: np.ndarray) -> float:
    if len(simulated_sample) != len(observed_sample):
        return "Size of samples not equal."
    
    nrow = len(simulated_sample) 
    ncol = len(simulated_sample[0]) 
    combined = np.concatenate((simulated_sample, observed_sample))
    # Find corresponding ranks in h associated with simulated/observed
    combined_rank = rankdata(combined, axis=0)
    simulated_rank = combined_rank[:nrow]
    observed_rank = combined_rank[nrow:]

    # Calculate distance
    idx = np.tile(np.arange(1, nrow+1), (ncol, 1)).T
    observed_sum = np.sum((observed_rank - idx)**2, axis=0)
    simulated_sum = np.sum((simulated_rank - idx)**2, axis=0)
    rank_sum = nrow * (observed_sum + simulated_sum)
    distance = rank_sum / (2*nrow**3) - (4*nrow**2 - 1)/(12*nrow)
    return distance

def energy_dist(simulated_sample: np.ndarray, observed_sample: np.ndarray) -> float:
    n = simulated_sample.shape[1]
    distances_XX = np.zeros((n, 100, 100))  # Distances within array1
    distances_YY = np.zeros((n, 100, 100))  # Distances within array2
    distances_XY = np.zeros((n, 100, 100))  # Distances between array1 and array2

    for i in range(n):
        distances_XX[i] = squareform(pdist(simulated_sample[:, i, np.newaxis], metric='euclidean'))
        distances_YY[i] = squareform(pdist(observed_sample[:, i, np.newaxis], metric='euclidean'))
        distances_XY[i] = cdist(simulated_sample[:, i, np.newaxis], observed_sample[:, i, np.newaxis], metric='euclidean')

    mean_dist_XY = np.mean(distances_XY, axis=(1, 2))  # Mean distance between columns
    mean_dist_XX = np.mean(distances_XX, axis=(1, 2))  # Mean distance within array1
    mean_dist_YY = np.mean(distances_YY, axis=(1, 2))  # Mean distance within array2

    # Calculate the energy distances for each column in a vectorized way
    energy_distances = 2 * mean_dist_XY - mean_dist_XX - mean_dist_YY

    # Output the energy distances for each column
    return energy_distances

def kullback_leibler_divergence(simulated_sample: np.ndarray, observed_sample: np.ndarray) -> float:
    # Sum of the divergence of each point
    return sum(kl_div(simulated_sample, observed_sample))

