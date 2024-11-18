import numpy as np
TRUE_VAL = 1

# Format for distance = [alpha, beta, wasserstein, energy, mmd, cvmd, kullback-leibler]

def abc_posterior(nparams: int, distances: np.ndarray, distance_quantile: float, distance_metric: str) -> np.ndarray:
    if distance_metric == "Wasserstein Distance":
        index = 2
    elif distance_metric == "Energy Distance":
        index = 3
    elif distance_metric == "Maximum Mean Discrepancy":
        index = 4
    elif distance_metric == "Cramer-von Mises Distance":
        index = 5
    elif distance_metric == "Kullback-Leibler Divergence":
        index = 6

    # Calculate quantile from given quantile
    threshold = np.nanquantile(distances[:,index], distance_quantile)

    # Calculate posterior mean, median, lower bound and upper bound for each metric
    posterior_mean = np.zeros(nparams)
    posterior_median = np.zeros(nparams)
    posterior_lower_bound = np.zeros(nparams)
    posterior_upper_bound = np.zeros(nparams)
    posterior_std = np.zeros(nparams)
    posterior_sqerr = np.zeros(nparams)

    ## Identify Alpha and Beta after filtering
    posterior_params = distances[distances[:,index] <= threshold][:,0:nparams]
    for i in range(nparams):
        posterior_mean[i] = np.mean(posterior_params[:,i])
        posterior_median[i] = np.nanquantile(posterior_params[:,i], 0.5)
        posterior_lower_bound[i] = np.nanquantile(posterior_params[:,i], 0.025)
        posterior_upper_bound[i] = np.nanquantile(posterior_params[:,i], 0.975)
        posterior_std[i] = np.std(posterior_params[:,i])
        posterior_sqerr[i] = (TRUE_VAL - posterior_median[i])**2

    posterior = np.array([posterior_mean, posterior_median, posterior_std, posterior_lower_bound, posterior_upper_bound, posterior_sqerr])
    posterior = posterior.T

    # Format is [[alpha posterior], [beta posterior]]
    return posterior



