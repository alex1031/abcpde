import numpy as np
import pandas as pd
import os, sys
from common.abc_posterior import abc_posterior

NUM_RUNS = 100
NPARAMS = 2 # Alpha and Beta
SAVE_DIR = "./lotka_volterra/results"
RUN_DIR = "./lotka_volterra/runs"
DATAFRAME_DIR = "./lotka_volterra/dataframe"
DISTANCE_METRIC = ["Wasserstein Distance", "Energy Distance", "Maximum Mean Discrepancy", "Cramer-von Mises Distance", "Kullback-Leibler Divergence"]
DISTANCE_QUANTILES = [0.05, 0.01, 0.001] # 5% 1% and 0.1%

if __name__ == "__main__":
    
    sim_time_dict = {}

    for simulations in os.listdir(RUN_DIR): # For each simulation, we calculate posterior for each metric

        sim_path = os.path.join(SAVE_DIR, simulations)
        if not os.path.isdir(sim_path):
            os.mkdir(sim_path)

        # To handle the sim times
        sim_times = []

        for i in range(NUM_RUNS):
            sim_time_path = os.path.join(RUN_DIR, simulations, f"run{i+1}_sim_time.npy")
            sim_time = np.load(sim_time_path)
            sim_times.append(sim_time)
        
        avg_sim_times = np.mean(sim_times, axis=0)

        sim_time_dict[simulations] = avg_sim_times[0] 

        df = pd.DataFrame.from_dict(sim_time_dict, orient="index",
                                    columns=["Wasserstein Distance", "Energy Distance", "Maximum Mean Discrepancy", "Cramer-von Mises Distance", "Kullback-Leibler Divergence"])
        
        df = df.reset_index().rename(columns={"index": "model"})
        
        df.to_csv(DATAFRAME_DIR + "/sim_times.csv", index=False)

        for metric in enumerate(DISTANCE_METRIC):

            metric_path = os.path.join(sim_path, metric)
            if not os.path.isdir(metric_path):
                os.mkdir(metric_path)

            for quantile in DISTANCE_QUANTILES:
                posteriors = np.zeros((NUM_RUNS, NPARAMS, 6)) 

                for run in simulations:
                    for i in range(NUM_RUNS):
                        run_path = os.path.join(RUN_DIR, simulations, f"run{i+1}.npy")
                        run = np.load(run_path)
                        posteriors[i] = abc_posterior(NPARAMS, run, quantile, metric)
            
                posterior_path = os.path.join(metric_path,f"{quantile}posterior.npy")
                np.save(posterior_path, posteriors)