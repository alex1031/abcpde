import numpy as np 
from sklearn import metrics
from scipy.stats import energy_distance, cramervonmises_2samp
from scipy.special import kl_div
from scipy.spatial.distance import pdist, squareform

def wasserstein_distance(simulated_sample: np.ndarray, observed_sample: np.ndarray) -> float:
    # Mean Difference between simulated and observed
    distance = np.mean(np.abs(simulated_sample - observed_sample), axis=0)
    return distance

def maximum_mean_discrepancy(simulated_sample: np.ndarray, observed_sample: np.ndarray, gamma = 1.0) -> float:
    XX = metrics.pairwise.rbf_kernel(simulated_sample, simulated_sample, gamma)
    YY = metrics.pairwise.rbf_kernel(observed_sample, observed_sample, gamma)
    XY = metrics.pairwise.rbf_kernel(simulated_sample, observed_sample, gamma)

    return XX.mean() + YY.mean() - 2 * XY.mean()

def cramer_von_mises(simulated_sample: np.ndarray, observed_sample: np.ndarray) -> float:
    return cramervonmises_2samp(simulated_sample, observed_sample).statistic

def energy_dist(simulated_sample: np.ndarray, observed_sample: np.ndarray) -> float:
    n = simulated_sample.shape[1]
    distances_XX = np.zeros((n, 100, 100))  # Distances within array1
    distances_YY = np.zeros((n, 100, 100))  # Distances within array2
    distances_XY = np.zeros((n, 100, 100))  # Distances between array1 and array2

    for i in range(n):
        distances_XX[i] = squareform(pdist(simulated_sample[:, i][:, np.newaxis], metric='euclidean'))
        distances_YY[i] = squareform(pdist(observed_sample[:, i][:, np.newaxis], metric='euclidean'))
        distances_XY[i] = np.linalg.norm(simulated_sample[:, i][:, np.newaxis] - observed_sample[:, i], axis=0)

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

