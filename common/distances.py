import numpy as np 
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.spatial import cKDTree
from discrete_frechet.distances.discrete import euclidean, FastDiscreteFrechetMatrix
# from numba import njit

def wasserstein_distance(simulated_sample: np.ndarray, observed_sample: np.ndarray) -> float:
    # Mean Difference between simulated and observed
    simulated_sorted = np.sort(simulated_sample, axis=0)
    observed_sorted = np.sort(observed_sample, axis=0)
    distance = np.mean(np.abs(simulated_sorted - observed_sorted), axis=0)

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
    combined_rank = np.argsort(combined, axis=0)+1
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

def wasserstein_distance_3D(simulated_sample: np.ndarray, observed_sample: np.ndarray) -> np.ndarray:
    """
    Compute the Wasserstein distance between two (51, 51, 1200) shaped arrays 
    along the time dimension.

    Sorted along the time axis, we now have a (51, 51, 1200) array, with each 1200-sized array sorted ascendingly 

    Returns a (51, 51) array of distances for each spatial location.
    """
    # Sort along the time axis (axis=2)
    simulated_sorted = np.sort(simulated_sample, axis=2)
    observed_sorted = np.sort(observed_sample, axis=2)

    # Compute the mean absolute difference along the time axis
    distance = np.mean(np.abs(simulated_sorted - observed_sorted), axis=2)

    return distance 

def cramer_von_mises_3d(simulated_sample: np.ndarray, observed_sample: np.ndarray) -> np.ndarray:
    '''
    Outputs a (51, 51), representing CvMD at each grid point.
    '''
    if simulated_sample.shape != observed_sample.shape:
        raise ValueError("Shape of samples not equal.")
    
    x_dim, y_dim, n_samples = simulated_sample.shape
    cvm_matrix = np.zeros((x_dim, y_dim))

    # Comparing the (1200,)-shaped array for each (x, y) grid point
    for i in range(x_dim):
        for j in range(y_dim):
            sim = simulated_sample[i, j, :]
            obs = observed_sample[i, j, :]

            # Combining the two arrays. (2400,)-shaped array
            combined = np.concatenate((sim, obs))
            # np.argsort(combined) gives us indicies that would sort the combined array in ascending order.
            # applying np.argosrt again gives us the rank of each element in the original array.
            combined_rank = np.argsort(np.argsort(combined)) + 1

            # First half of the combined rank is simulated by definition of combined.
            sim_rank = combined_rank[:n_samples]
            obs_rank = combined_rank[n_samples:]

            # Calculation for CvMD
            idx = np.arange(1, n_samples + 1)
            obs_sum = np.sum((obs_rank - idx) ** 2)
            sim_sum = np.sum((sim_rank - idx) ** 2)
            
            rank_sum = n_samples * (obs_sum + sim_sum)
            distance = rank_sum / (2 * n_samples**3) - (4 * n_samples**2 - 1) / (12 * n_samples)
            
            cvm_matrix[i, j] = distance
    
    return cvm_matrix

def directed_hausdorff(A, B):
    """
    Compute the directed Hausdorff distance from set A to set B using cKDTree for efficiency.
    
    Parameters:
    A (numpy array): Set of points (N x d) where N is the number of points, and d is the dimension.
    B (numpy array): Set of points (M x d), where M is the number of points in B.
    
    Returns:
    float: The directed Hausdorff distance from A to B.
    """
    # Reshape (51, 51, 1200) -> (2601, 1200)
    A_flat = A.reshape(-1, A.shape[-1])  # (2601, 1200)
    B_flat = B.reshape(-1, B.shape[-1])  # (2601, 1200)
    
    tree_B = cKDTree(B_flat)  # Build KD-tree for set B
    dists_A_to_B, _ = tree_B.query(A_flat)  # Find nearest neighbor distances for A to B
    cmax_A_to_B = np.max(dists_A_to_B)  # Maximum of minimum distances
    
    tree_A = cKDTree(A_flat)  # Build KD-tree for set A
    dists_B_to_A, _ = tree_A.query(B_flat)  # Find nearest neighbor distances for B to A
    cmax_B_to_A = np.max(dists_B_to_A)  # Maximum of minimum distances
    
    return max(cmax_A_to_B, cmax_B_to_A)

fdfdm = FastDiscreteFrechetMatrix(euclidean)
def frechet_distance(P, Q):
    return fdfdm.distance(P, Q)