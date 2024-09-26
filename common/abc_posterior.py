import numpy as np

# Format for distance = [alpha, beta, wasserstein, energy, mmd, cvmd, kullback-leibler]

def abc_posterior(nparams: int, distances: np.ndarray, distance_quantile: float, distance_metric: str) -> np.ndarray:
    if distance_metric == "Wasserstein Distance":
        index = 2
    elif distance_metric == "Energy Distance":
        index = 3
    elif distance_metric == "Maxmimum Mean Discrepancy":
        index = 4
    elif distance_metric == "Cramer-von Mises Distance":
        index = 5
    elif distance_metric == "Kullback-Leibler Divergence":
        index = 6

    # Calculate quantile from given quantile
    threshold = np.quantile(distances[:,index], distance_quantile)

    # Calculate posterior probability for each metric
    posterior = np.zeros(nparams)
    ## Identify Alpha and Beta after filtering
    posterior_params = distances[distances[:,index] <= threshold][:,0:nparams]
    for i in range(nparams):
        posterior[i] = np.mean(posterior_params[:,i])

    return posterior



