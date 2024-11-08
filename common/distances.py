import numpy as np 
from scipy.stats import rankdata
from scipy.spatial.distance import pdist, squareform, cdist

def wasserstein_distance(simulated_sample: np.ndarray, observed_sample: np.ndarray) -> float:
    # Mean Difference between simulated and observed
    distance = np.mean(np.abs(simulated_sample - observed_sample), axis=0)

    return distance

def maximum_mean_discrepancy(simulated_sample: np.ndarray, observed_sample: np.ndarray, gamma=1.0) -> float:
    ncol = simulated_sample.shape[1]

    mean_KXX = np.empty(ncol)  # Mean kernel values within simulated_sample
    mean_KYY = np.empty(ncol)  # Mean kernel values within observed_sample
    mean_KXY = np.empty(ncol)  # Mean kernel values between simulated_sample and observed_sample

    for i in range(ncol):
        # Compute pairwise distances and apply the Gaussian kernel directly
        distances_XX = squareform(pdist(simulated_sample[:, i, np.newaxis], metric='euclidean'))
        distances_YY = squareform(pdist(observed_sample[:, i, np.newaxis], metric='euclidean'))
        distances_XY = cdist(simulated_sample[:, i, np.newaxis], observed_sample[:, i, np.newaxis], metric='euclidean')
        
        # Apply the Gaussian kernel
        KXX = np.exp(-distances_XX ** 2 / (2 * gamma ** 2))
        KYY = np.exp(-distances_YY ** 2 / (2 * gamma ** 2))
        KXY = np.exp(-distances_XY ** 2 / (2 * gamma ** 2))

        # Calculate the means
        mean_KXX[i] = np.mean(KXX)
        mean_KYY[i] = np.mean(KYY)
        mean_KXY[i] = np.mean(KXY)

    # Calculate the MMD for each column
    mmd_values = mean_KXX + mean_KYY - 2 * mean_KXY

    return mmd_values

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

def energy_dist(simulated_sample: np.ndarray, observed_sample: np.ndarray) -> float:
    ncol = simulated_sample.shape[1]

    mean_dist_XY = np.empty(ncol)  # Mean distances between columns
    mean_dist_XX = np.empty(ncol)  # Mean distances within array1
    mean_dist_YY = np.empty(ncol)  # Mean distances within array2

    for i in range(ncol):
        mean_dist_XX[i] = np.mean(squareform(pdist(simulated_sample[:, i, np.newaxis], metric='euclidean')))
        mean_dist_YY[i] = np.mean(squareform(pdist(observed_sample[:, i, np.newaxis], metric='euclidean')))
        mean_dist_XY[i] = np.mean(cdist(simulated_sample[:, i, np.newaxis], observed_sample[:, i, np.newaxis], metric='euclidean'))

    # Calculate the energy distances for each column in a vectorized way
    energy_distances = 2 * mean_dist_XY - mean_dist_XX - mean_dist_YY

    return energy_distances

def kullback_leibler_divergence(simulated_sample: np.ndarray, observed_sample: np.ndarray) -> float:
    nrow = simulated_sample.shape[0]
    ncol = simulated_sample.shape[1]
    
    log_term = np.log(nrow / (nrow - 1))
    kld_values = np.empty(ncol)

    for i in range(ncol):
        # Calculate pairwise distances within simulated_sample
        distances_XX = squareform(pdist(simulated_sample[:, i, np.newaxis], metric="euclidean"))
        np.fill_diagonal(distances_XX, np.inf) # Put inf to avoid counting them in min calculation
        
        # Calculate the minimum non-zero distance for each point
        nonzero_min_XX = np.where(distances_XX>0, distances_XX, np.inf).min(axis=1)

        # Calculate pairwise distances between simulated_sample and observed_sample
        distances_XY = cdist(simulated_sample[:, i, np.newaxis], observed_sample[:, i, np.newaxis], metric="euclidean")
        nonzero_min_XY = np.where(distances_XY>0, distances_XY, np.inf).min(axis=1)

        # Calculate mean of the log ratio for each column
        kld = np.mean(np.log(nonzero_min_XY / nonzero_min_XX)) + log_term
        if kld < 0:
            kld = np.nan
        kld_values[i] = kld

    return kld_values

