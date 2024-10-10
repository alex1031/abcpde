import numpy as np
import pandas as pd
from pandas.plotting import table
import os
import itertools

# Assume the directories are correct

# Results in form [[a_mean, a_median, a_lowerbound, a_upperbound], [b_mean, b_median, b_lowerbound, b_upperbound]]
RESULTS_DIR = "./lotka_volterra/results"
SAVE_DIR = "./lotka_volterra/dataframe"
SUMMARY_STATISTICS = ["Mean", "Median", "Lower Bound", "Upper Bound"]
QUANTILE = [0.05, 0.01, 0.001]
QUANTILE_NAME = ["5%", "1%", "0.1%"]
PARAMS = ["a", "b"]
TRUE_VALUE = 1

if __name__ == "__main__":
    
    models = os.listdir(RESULTS_DIR) # Different models
    model_path = os.path.join(RESULTS_DIR, models[0])
    distances = os.listdir(model_path) # All metrics - Used as columns

    permutations = list(itertools.product(models, QUANTILE_NAME, PARAMS, SUMMARY_STATISTICS))
    index_perms = np.array([list(item) for item in permutations])

    distance_val = {} 

    for result in models:
        result_path = os.path.join(RESULTS_DIR, result)

        for metric in distances:
            metric_path = os.path.join(result_path, metric)
            posteriors = np.zeros((2, 4))
        
            for q in QUANTILE:
                posterior_path = os.path.join(metric_path, f"{q}posterior.npy")
                post = np.load(posterior_path)
                posterior_avg = np.mean(post, axis=0)
                if metric not in distance_val:
                    distance_val[metric] = posterior_avg
                else:
                    distance_val[metric] = np.concatenate((distance_val[metric], posterior_avg))

    for metric in distance_val:
        distance_val[metric] = distance_val[metric].flatten()

    df = pd.DataFrame(columns=["model", "quantile", "param", "summary_statistic"], data=index_perms)

    distance_df = pd.DataFrame.from_dict(distance_val)

    final_df = pd.concat([df, distance_df], axis=1)

    df_path = os.path.join(SAVE_DIR, "all_summary_statistics.csv")
    final_df.to_csv(df_path, index=False)
    
                   