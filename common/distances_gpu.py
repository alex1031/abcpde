import torch
from torch.nn.functional import pairwise_distance

def wasserstein_distance(simulated_sample: torch.Tensor, observed_sample: torch.Tensor) -> float:
    # Mean Difference between simulated and observed
    distance = torch.mean(torch.abs(simulated_sample - observed_sample), dim=0)

    return distance.cpu()

def maximum_mean_discrepancy(simulated_sample: torch.Tensor, observed_sample: torch.Tensor, gamma = 1.0) -> float:
    nrow = simulated_sample.shape[0]
    ncol = simulated_sample.shape[1]
    distances_XX = torch.zeros((ncol, nrow, nrow), dtype=torch.float32).cuda()  # Distances within simulated_sample
    distances_YY = torch.zeros((ncol, nrow, nrow), dtype=torch.float32).cuda()  # Distances within observed_sample
    distances_XY = torch.zeros((ncol, nrow, nrow), dtype=torch.float32).cuda()  # Distances between simulated_sample and observed_sample

    for i in range(ncol):
        distances_XX[i] = torch.cdist(simulated_sample[:, i].unsqueeze(1), simulated_sample[:, i].unsqueeze(1))
        distances_YY[i] = torch.cdist(observed_sample[:, i].unsqueeze(1), observed_sample[:, i].unsqueeze(1))
        distances_XY[i] = torch.cdist(simulated_sample[:, i].unsqueeze(1), observed_sample[:, i].unsqueeze(1))

    # Gaussian kernel
    KXX = torch.exp(-distances_XX ** 2 / (2 * gamma ** 2))
    KYY = torch.exp(-distances_YY ** 2 / (2 * gamma ** 2))
    KXY = torch.exp(-distances_XY ** 2 / (2 * gamma ** 2))

    mean_KXX = torch.mean(KXX, dim=(1, 2))  # Mean of kernel within simulated_sample
    mean_KYY = torch.mean(KYY, dim=(1, 2))  # Mean of kernel within observed_sample
    mean_KXY = torch.mean(KXY, dim=(1, 2))  # Mean of kernel between simulated_sample and observed_sample

    # MMD vectorized for each column
    mmd_values = mean_KXX + mean_KYY - 2 * mean_KXY

    return mmd_values.cpu()

def cramer_von_mises(simulated_sample: torch.Tensor, observed_sample: torch.Tensor) -> float:
    if len(simulated_sample) != len(observed_sample):
        return "Size of samples not equal."
    
    nrow = simulated_sample.shape[0]
    ncol = simulated_sample.shape[1]
    combined = torch.cat((simulated_sample, observed_sample))
    # Find corresponding ranks in h associated with simulated/observed
    combined_rank = torch.argsort(torch.argsort(combined, dim=0), dim=0).float() + 1
    simulated_rank = combined_rank[:nrow]
    observed_rank = combined_rank[nrow:]

    # Calculate distance
    idx = torch.arange(1, nrow + 1, device=simulated_sample.device).unsqueeze(1).repeat(1, ncol)
    observed_sum = torch.sum((observed_rank - idx) ** 2, dim=0)
    simulated_sum = torch.sum((simulated_rank - idx) ** 2, dim=0)
    rank_sum = nrow * (observed_sum + simulated_sum)
    distance = rank_sum / (2*nrow**3) - (4*nrow**2 - 1)/(12*nrow)

    return distance.cpu()

def energy_dist(simulated_sample: torch.Tensor, observed_sample: torch.Tensor) -> float:
    nrow = simulated_sample.shape[0]
    ncol = simulated_sample.shape[1]
    distances_XX = torch.zeros((ncol, nrow, nrow), dtype=torch.float32).cuda()  # Distances within array1
    distances_YY = torch.zeros((ncol, nrow, nrow), dtype=torch.float32).cuda()  # Distances within array2
    distances_XY = torch.zeros((ncol, nrow, nrow), dtype=torch.float32).cuda()  # Distances between array1 and array2

    for i in range(ncol):
        distances_XX[i] = torch.cdist(simulated_sample[:, i].unsqueeze(1), simulated_sample[:, i].unsqueeze(1))
        distances_YY[i] = torch.cdist(observed_sample[:, i].unsqueeze(1), observed_sample[:, i].unsqueeze(1))
        distances_XY[i] = torch.cdist(simulated_sample[:, i].unsqueeze(1), observed_sample[:, i].unsqueeze(1))

    mean_dist_XY = torch.mean(distances_XY, dim=(1, 2))  # Mean distance between columns
    mean_dist_XX = torch.mean(distances_XX, dim=(1, 2))  # Mean distance within array1
    mean_dist_YY = torch.mean(distances_YY, dim=(1, 2))  # Mean distance within array2

    # Calculate the energy distances for each column in a vectorized way
    energy_distances = 2 * mean_dist_XY - mean_dist_XX - mean_dist_YY

    return energy_distances.cpu()

def kullback_leibler_divergence(simulated_sample: torch.Tensor, observed_sample: torch.Tensor) -> float:
    nrow = simulated_sample.shape[0]
    ncol = simulated_sample.shape[1]
    distances_XX = torch.zeros((ncol, nrow, nrow), dtype=torch.float32).cuda()
    distances_XY = torch.zeros((ncol, nrow, nrow), dtype=torch.float32).cuda()
    
    # ln(n/(n-1))
    log_term = torch.log(torch.tensor(nrow / (nrow - 1)).cuda())
    for i in range(ncol):
        # pairwise distance
        distances_XX[i] = torch.cdist(simulated_sample[:, i].unsqueeze(1), simulated_sample[:, i].unsqueeze(1))
        # pairwise distance between Z and Y
        distances_XY[i] = torch.cdist(simulated_sample[:, i].unsqueeze(1), observed_sample[:, i].unsqueeze(1))
    
    # min_(j!=i)|z_i-z_j|
    XXflat_nonzero = torch.where(distances_XX.view(ncol, -1) == 0, float('inf'), distances_XX.view(ncol, -1))
    nonzero_min_XX = torch.min(XXflat_nonzero, dim=1).values

    # min_j|z_i-y_j|
    XYflat_nonzero = torch.where(distances_XY.view(ncol, -1) == 0, float('inf'), distances_XY.view(ncol, -1))
    nonzero_min_XY = torch.min(XYflat_nonzero, dim=1).values

    kld = torch.log(nonzero_min_XY / nonzero_min_XX) / nrow + log_term
    return kld.cpu()

