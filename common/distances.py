import numpy as np 
from sklearn import metrics
from scipy.stats import energy_distance, cramervonmises_2samp
from scipy.special import kl_div

def wasserstein_distance(simulated_sample: np.ndarray, observed_sample: np.ndarray) -> float:
    # Mean Difference between simulated and observed
    distance = np.mean(np.abs(simulated_sample - observed_sample))
    return distance

def maximum_mean_discrepancy(simulated_sample: np.ndarray, observed_sample: np.ndarray, gamma = 1.0) -> float:
    XX = metrics.pairwise.rbf_kernel(simulated_sample, simulated_sample, gamma)
    YY = metrics.pairwise.rbf_kernel(observed_sample, observed_sample, gamma)
    XY = metrics.pairwise.rbf_kernel(simulated_sample, observed_sample, gamma)

    return XX.mean() + YY.mean() - 2 * XY.mean()

def cramer_von_mises(simulated_sample: np.ndarray, observed_sample: np.ndarray) -> float:
    return cramervonmises_2samp(simulated_sample, observed_sample).statistic

def energy_dist(simulated_sample: np.ndarray, observed_sample: np.ndarray) -> float:
    return energy_distance(simulated_sample, observed_sample)

def kullback_leibler_divergence(simulated_sample: np.ndarray, observed_sample: np.ndarray) -> float:
    # Sum of the divergence of each point
    return sum(kl_div(simulated_sample, observed_sample))

