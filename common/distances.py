import numpy as np 
from scipy.stats import rankdata
from scipy.spatial.distance import pdist, squareform, cdist
from line_profiler import profile
LINE_PROFILE = 1

@profile
def wasserstein_distance(simulated_sample: np.ndarray, observed_sample: np.ndarray) -> float:
    # Mean Difference between simulated and observed
    distance = np.mean(np.abs(simulated_sample - observed_sample), axis=0)
    return distance

@profile
def maximum_mean_discrepancy(simulated_sample: np.ndarray, observed_sample: np.ndarray, gamma = 1.0) -> float:
    nrow = simulated_sample.shape[0]
    ncol = simulated_sample.shape[1]
    distances_XX = np.zeros((ncol, nrow, nrow))  # Distances within simulated_sample
    distances_YY = np.zeros((ncol, nrow, nrow))  # Distances within observed_sample
    distances_XY = np.zeros((ncol, nrow, nrow))  # Distances between simulated_sample and observed_sample

    for i in range(ncol):
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

@profile
def cramer_von_mises(simulated_sample: np.ndarray, observed_sample: np.ndarray) -> float:
    if len(simulated_sample) != len(observed_sample):
        return "Size of samples not equal."
    
    nrow = simulated_sample.shape[0]
    ncol = simulated_sample.shape[1]
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

@profile
def energy_dist(simulated_sample: np.ndarray, observed_sample: np.ndarray) -> float:
    nrow = simulated_sample.shape[0]
    ncol = simulated_sample.shape[1]
    distances_XX = np.zeros((ncol, nrow, nrow))  # Distances within array1
    distances_YY = np.zeros((ncol, nrow, nrow))  # Distances within array2
    distances_XY = np.zeros((ncol, nrow, nrow))  # Distances between array1 and array2

    for i in range(ncol):
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

@profile
def kullback_leibler_divergence(simulated_sample: np.ndarray, observed_sample: np.ndarray) -> float:
    nrow = simulated_sample.shape[0]
    ncol = simulated_sample.shape[1]
    distances_XX = np.zeros((ncol, nrow, nrow))
    distances_XY = np.zeros((ncol, nrow, nrow))
    
    # ln(n/(n-1))
    log_term = np.log(nrow/(nrow-1))
    for i in range(ncol):
        # pairwise distance
        distances_XX[i] = squareform(pdist(simulated_sample[:,i, np.newaxis], metric="euclidean"))
        # pairwise distance between Z and Y
        distances_XY[i] = cdist(simulated_sample[:,i, np.newaxis], observed_sample[:,i, np.newaxis], metric="euclidean")
    
    # min_(j!=i)|z_i-z_j|
    XXflat = distances_XX.reshape(ncol, -1)
    XXflat_nonzero = np.where(XXflat==0, np.inf, XXflat)
    nonzero_min_XX = np.min(XXflat_nonzero, axis=1)

    # min_j|z_i-y_j|
    XYflat = distances_XY.reshape(ncol,-1)
    XYflat_nozero = np.where(XYflat==0, np.inf, XYflat)
    nonzero_min_XY = np.min(XYflat_nozero, axis=1)

    kld = np.log(nonzero_min_XY/nonzero_min_XX)/nrow + log_term
    return kld

