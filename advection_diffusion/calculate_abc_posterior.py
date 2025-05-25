import numpy as np
import os
from common.abc_posterior import gaussian_abc_posterior
import pandas as pd

RUN_DIR = "./gaussian_plume/runs"
SAVE_DIR = "./gaussian_plume/results"
DATAFRAME_DIR = "./gaussian_plume/dataframe"
NUM_RUNS = 2
NPARAMS = 3 # cx, cy, s
DISTANCE_METRIC = ["Wasserstein Distance", "Cramer-von Mises Distance", "Frechet Distance", "Hausdorff Distance"]
QUANTILES = [0.05, 0.01, 0.001, 0.0001] # 5%, 1%, 0.1%, 0.01%

if __name__ == "__main__":

    models = os.listdir(RUN_DIR)
    sim_time_dict = {}

    # For each model
    for model in models:
        
        sim_times = []

        # We store sim times in a seperate csv file
        for i in range(NUM_RUNS):
            sim_time_path = os.path.join(RUN_DIR, model, f"run{i+1}_sim_time.npy")
            sim_time = np.load(sim_time_path)
            sim_times.append(sim_time)
        
        avg_sim_times = np.mean(sim_times, axis=0)

        sim_time_dict[model] = avg_sim_times 

        df = pd.DataFrame.from_dict(sim_time_dict, orient="index",
                                    columns=["Wasserstein Distance", "Cramer-von Mises Distance", "Frechet Distance", "Hausdorff Distance"])
        
        df = df.reset_index().rename(columns={"index": "model"})
        
        df.to_csv(DATAFRAME_DIR + "/sim_times.csv", index=False)

        model_path = os.path.join(SAVE_DIR, model) # e.g. results/no_noise
            
        # If the path doesn't exist 
        if not os.path.isdir(model_path):
            os.mkdir(model_path)

        # We generate result for each distance metric
        for metric in DISTANCE_METRIC:
            metric_model_path = os.path.join(model_path, metric) # e.g. results/no_noise/Wasserstein

            if not os.path.isdir(metric_model_path):
                os.mkdir(metric_model_path)
            
            # Separate results are needed for each quantile for threshold analysis.
            for quantile in QUANTILES: 
                posterior = np.zeros((NUM_RUNS, NPARAMS, 6)) # 6 - Median, Mean, Lower & Upper Bound, StDev, RMSE
                
                # Analyse results from each run
                for i in range(NUM_RUNS):
                    run_path = os.path.join(RUN_DIR, model, f"run{i+1}.npy")
                    run_data = np.load(run_path)
                    if model == "no_noise_calm_air":
                        posterior[i] = gaussian_abc_posterior(NPARAMS, run_data, quantile, metric, calm_air=True)
                    else:
                        posterior[i] = gaussian_abc_posterior(NPARAMS, run_data, quantile, metric)
                    
                posterior_path = os.path.join(metric_model_path, f"{quantile}posterior.npy")
                np.save(posterior_path, posterior)