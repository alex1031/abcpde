import numpy as np
import pandas as pd
import os
import itertools

# Assume the directories are correct

# Results in form [[cx_mean, cx_median, cx_lowerbound, cx_upperbound, cx_std, cx_sqerr], same for cy and s]
# cx_sqerr used to calculate the RMSE

RESULTS_DIR = "./gaussian_plume/results"
SAVE_DIR = "./gaussian_plume/dataframe"
SUMMARY_STATISTICS = ["Mean", "Median", "StDev", "Lower Bound", "Upper Bound", "RMSE"]
QUANTILE = [0.05, 0.01, 0.001, 0.0001]
QUANTILE_NAME = ["5%", "1%", "0.1%", "0.01%"]
PARAMS = ["cx", "cy", "s"]
TRUE_CX = 0.5
TRUE_CY = 0.5
TRUE_S = 5e-5

if __name__ == "__main__":
    
    models = os.listdir(RESULTS_DIR) # Different models
    model_path = os.path.join(RESULTS_DIR, models[0])
    distances = os.listdir(model_path) # All metrics - Used as columns

    permutations = list(itertools.product(models, QUANTILE_NAME, PARAMS, SUMMARY_STATISTICS)) # All the different combination of permutations
    index_perms = np.array([list(item) for item in permutations])

    distance_val = {} 

    for result in models:
        result_path = os.path.join(RESULTS_DIR, result) # results/Wasserstein

        for metric in distances:
            metric_path = os.path.join(result_path, metric) 
            # posteriors = np.zeros((2, 4))
        
            for q in QUANTILE:
                posterior_path = os.path.join(metric_path, f"{q}posterior.npy")
                post = np.load(posterior_path)
                posterior_avg = np.mean(post[:,:,:5], axis=0) # Exclude squared error axis
                posterior_rmse = np.sqrt(np.sum(post[:,:,5], axis=0)/post.shape[0]).reshape((3, 1)) # RMSE formula
                posterior_avg = np.concatenate((posterior_avg, posterior_rmse), axis=1)
                
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
    
                   