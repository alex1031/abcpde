import numpy as np
import os
from common.abc_posterior import gaussian_abc_posterior

RUN_DIR = "./gaussian_plume/runs"
SAVE_DIR = "./gaussian_plume/results"
NUM_RUNS = 2
NPARAMS = 3 # cx, cy, s
DISTANCE_METRIC = ["Wasserstein Distance", "Cramer-von Mises Distance", "Frechet Distance", "Hausdorff Distance"]
QUANTILES = [0.05, 0.01, 0.001] # 5%, 1%, 0.1%

models = os.listdir(RUN_DIR)

# For each model
for model in models:
    
    # We generate result for each distance metric
    for metric in DISTANCE_METRIC:
        metric_path = os.path.join(SAVE_DIR, metric) # e.g. results/"Wasserstein Distance"
        
        # If the path doesn't exist 
        if not os.path.isdir(metric_path):
            os.mkdir(metric_path)
        
        # Separate results are needed for each quantile for threshold analysis.
        for quantile in QUANTILES: 
            posterior = np.zeros((NUM_RUNS, NPARAMS, 6)) # 6 - Median, Mean, Lower & Upper Bound, StDev, RMSE
            
            # Analyse results from each run
            for i in range(NUM_RUNS):
                run_path = os.path.join(RUN_DIR, model, f"run{i+1}.npy")
                run_data = np.load(run_path)
                posterior[i] = gaussian_abc_posterior(NPARAMS, run_data, quantile, metric)
                
            posterior_path = os.path.join(metric_path, f"{quantile}posterior.npy")
            np.save(posterior_path, posterior)