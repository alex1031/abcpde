import numpy as np
import os, sys
from common.abc_posterior import abc_posterior

NUM_RUNS = 5
NPARAMS = 2 # Alpha and Beta
SAVE_DIR = "./lotka_volterra/results"
RUN_DIR = "./lotka_volterra/runs"
DISTANCE_METRIC = ["Wasserstein Distance", "Energy Distance"]
DISTANCE_QUANTILES = [0.05, 0.01, 0.001] # 5% 1% and 0.1%

if __name__ == "__main__":
    
    for simulations in os.listdir(RUN_DIR): # For each simulation, we calculate posterior for each metric

        sim_path = os.path.join(SAVE_DIR, simulations)
        if not os.path.isdir(sim_path):
            os.mkdir(sim_path)
        
        # want to store as: [[wdistalpha0.05, wdistbeta0.05, edistalpha0.05, edistbeta0.05, ...],
        # [wdistalpha0.01, wdistbeta0.01, edistalpha0.01, edistbeta0.01, ...],
        # ...]

        for metric in DISTANCE_METRIC:
            metric_path = os.path.join(sim_path, metric)
            for quantile in DISTANCE_QUANTILES:
                posteriors = np.zeros((NUM_RUNS, NPARAMS)) 
                for run in simulations:
                    for i in range(NUM_RUNS):
                        run_path = os.path.join(RUN_DIR, simulations, f"run{i+1}.npy")
                        run = np.load(run_path)
                        posteriors[i] = abc_posterior(NPARAMS, run, quantile, metric)

            
            posterior_path = os.path.join(SAVE_DIR, run)
