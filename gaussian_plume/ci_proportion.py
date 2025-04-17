import numpy as np
import pandas as pd
import os

RESULT_PATH = "./gaussian_plume/results"
DATAFRAME_PATH = "./gaussian_plume/dataframe"
TRUE_CX = 0.5
TRUE_CY = 0.5
TRUE_S = 5e-5

models = os.listdir(RESULT_PATH)
df_dict = {"Metric": [], "Model":[], "cx_proportion": [], "cy_proportion": [], "s_proportion": []}

for model in models:
    model_path = os.path.join(RESULT_PATH, model)
    distance = os.listdir(model_path)
    for metric in distance:
        distance_path = os.path.join(model_path, metric)
        quantile = os.listdir(distance_path)
        quantile_path = os.path.join(distance_path, quantile[0]) # 0 for 0.001 quantile, 1 for 0.0001 quantile

        posterior = np.load(quantile_path)
        lower_bound, upper_bound = posterior[:,:,3], posterior[:,:,4]
        cx_lb, cx_up = lower_bound[:,0], upper_bound[:,0]
        cy_lb, cy_up = lower_bound[:,1], upper_bound[:,1]
        s_lb, s_up = lower_bound[:,2], upper_bound[:,2]
        
        cx_bound = np.column_stack((cx_lb, cx_up))
        cy_bound = np.column_stack((cy_lb, cy_up))
        s_bound = np.column_stack((s_lb, s_up))

        cx_contain_true = np.sum((cx_lb <= TRUE_CX) & (TRUE_CX <= cx_up))
        cy_contain_true = np.sum((cy_lb <= TRUE_CY) & (TRUE_CY <= cy_up))
        s_contain_true = np.sum((s_lb <= TRUE_S) & (TRUE_S <= s_up))

        cx_proportion = cx_contain_true/cx_bound.shape[0]
        cy_proportion = cy_contain_true/cy_bound.shape[0]
        s_proportion = s_contain_true/s_bound.shape[0]

        df_dict["Metric"].append(metric)
        df_dict["Model"].append(model)
        df_dict["cx_proportion"].append(cx_proportion)
        df_dict["cy_proportion"].append(cy_proportion)
        df_dict["s_proportion"].append(s_proportion)

df = pd.DataFrame.from_dict(df_dict)
title = quantile[0].split(".")[:2]
title = ".".join(title)
save_path = os.path.join(DATAFRAME_PATH, f"{title}_ci_proportion.csv")
df.to_csv(save_path, index=False)