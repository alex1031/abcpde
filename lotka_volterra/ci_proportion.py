import numpy as np
import pandas as pd
import os

RESULT_PATH = "./lotka_volterra/results"
DATAFRAME_PATH = "./lotka_volterra/dataframe"
TRUE_VALUE = 1

models = os.listdir(RESULT_PATH)
df_dict = {"Metric": [], "Model":[], "alpha_proportion": [], "beta_proportion": []}

for model in models:
    model_path = os.path.join(RESULT_PATH, model)
    distance = os.listdir(model_path)
    for metric in distance:
        distance_path = os.path.join(model_path, metric)
        quantile = os.listdir(distance_path)
        quantile_path = os.path.join(distance_path, quantile[0]) # Only want 0.01 quantile

        posterior = np.load(quantile_path)
        lower_bound, upper_bound = posterior[:,:,3], posterior[:,:,4]
        alpha_lb, alpha_up = lower_bound[:,0], upper_bound[:,0]
        beta_lb, beta_up = lower_bound[:,1], upper_bound[:,1]
        
        alpha_bound = np.column_stack((alpha_lb, alpha_up))
        beta_bound = np.column_stack((beta_lb, beta_up))

        alpha_contain_true = np.sum((alpha_lb <= TRUE_VALUE) & (TRUE_VALUE <= alpha_up))
        beta_contain_true = np.sum((beta_lb <= TRUE_VALUE) & (TRUE_VALUE <= beta_up))

        alpha_proportion = alpha_contain_true/alpha_bound.shape[0]
        beta_proportion = beta_contain_true/beta_bound.shape[0]

        df_dict["Metric"].append(metric)
        df_dict["Model"].append(model)
        df_dict["alpha_proportion"].append(alpha_proportion)
        df_dict["beta_proportion"].append(beta_proportion)

df = pd.DataFrame.from_dict(df_dict)
save_path = os.path.join(DATAFRAME_PATH, "ci_proportion.csv")
df.to_csv(save_path, index=False)
