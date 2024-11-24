import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataframe_image as dfi
import os

DATAFRAME_PATH = "./lotka_volterra/dataframe/all_summary_statistics.csv"
PLOT_PATH = "./lotka_volterra/plots"

model_dict = {
    "n0_no_smoothing": "ε ~ N(0, 0) No Smoothing",
    "n0.25_no_smoothing": "ε ~ N(0, 0.25^2) No Smoothing",
    "n0.25_smoothing": "ε ~ N(0, 0.25^2) With Smoothing",
    "n0.5_no_smoothing": "ε ~ N(0, 0.5^2) No Smoothing",
    "n0.5_smoothing": "ε ~ N(0, 0.5^2) With Smoothing",
    "n0.75_no_smoothing": "ε ~ N(0, 0.75^2) No Smoothing",
    "n0.75_smoothing": "ε ~ N(0, 0.75^2) With Smoothing",
    "nlinear_no_smoothing": "ε ~ N(0, t^2) No Smoothing",
    "nlinear_smoothing": "ε ~ N(0, t^2) Smoothing",
    "a": "ɑ",
    "b": "β"
}
df = pd.read_csv(DATAFRAME_PATH)
df.rename(columns={"model": "Model", "param": "Parameter"}, inplace=True)
df["Model"].replace(model_dict, inplace=True, regex=True)
df["Parameter"].replace(model_dict, inplace=True, regex=True)
median = df[(df["summary_statistic"] == "Median") & (df["quantile"] == "0.1%")].reset_index(drop=True)

keep_cols = ["Model", "Parameter", "Wasserstein Distance", "Energy Distance", "Maximum Mean Discrepancy", "Cramer-von Mises Distance", "Kullback-Leibler Divergence"]

median_table = median[keep_cols]

# Table 1 - Add Column for True Value + Median Values

median_table["True Value"] = np.ones(len(median_table))
piv_table = pd.pivot_table(median_table, values=["True Value", "Wasserstein Distance", "Energy Distance", "Maximum Mean Discrepancy",
                                          "Cramer-von Mises Distance", "Kullback-Leibler Divergence"], index = ["Model", "Parameter"])

piv_table = piv_table[["True Value", "Wasserstein Distance", "Energy Distance", "Maximum Mean Discrepancy",
                       "Cramer-von Mises Distance", "Kullback-Leibler Divergence"]]

df_styled = piv_table.style.background_gradient(axis=None, vmin=1, vmax=10, cmap="YlOrRd") #adding a gradient based on values in cell
df_styled.format(precision=2)

save_path_t1 = os.path.join(PLOT_PATH, "median_table.png")
dfi.export(df_styled, save_path_t1)

# Table 2 - RMSE table
rmse = df[(df["summary_statistic"] == "RMSE") & (df["quantile"] == "0.1%")].reset_index(drop=True)
rmse = rmse[keep_cols]

rmse_table = pd.pivot_table(rmse, values=["Wasserstein Distance", "Energy Distance", "Maximum Mean Discrepancy",
                                          "Cramer-von Mises Distance", "Kullback-Leibler Divergence"], index = ["Model", "Parameter"])

rmse_styled = rmse_table.style.background_gradient(axis=None, vmin=0, vmax=10, cmap="YlOrRd") #adding a gradient based on values in cell
rmse_styled.format(precision=2)

save_path_t2 = os.path.join(PLOT_PATH, "rmse_table.png")
dfi.export(rmse_styled, save_path_t2)

# Table 3 - Standard Deviation table
stdev_table = df[(df["summary_statistic"] == "StDev") & (df["quantile"] == "0.1%")].reset_index(drop=True)
piv_table = pd.pivot_table(stdev_table, values=["Wasserstein Distance", "Energy Distance", "Maximum Mean Discrepancy",
                                          "Cramer-von Mises Distance", "Kullback-Leibler Divergence"], index = ["Model", "Parameter"])

piv_table = piv_table[["Wasserstein Distance", "Energy Distance", "Maximum Mean Discrepancy",
                       "Cramer-von Mises Distance", "Kullback-Leibler Divergence"]]

stdev_styled = piv_table.style.background_gradient(axis=None, vmin=0, vmax=5, cmap="YlOrRd") #adding a gradient based on values in cell
stdev_styled.format(precision=2)

save_path_t3 = os.path.join(PLOT_PATH, "stdev_table.png")
dfi.export(stdev_styled, save_path_t3)

# Table 4 - 95% Credible Interval Table
ci_table = df[(df["quantile"] == "0.1%") & df["summary_statistic"].isin(["Lower Bound", "Upper Bound"])].reset_index(drop=True)

pivoted_data = ci_table.pivot_table(
    index=['Model', 'Parameter'],
    columns=["summary_statistic"],
    values = ["Wasserstein Distance", "Energy Distance", "Maximum Mean Discrepancy", "Cramer-von Mises Distance", "Kullback-Leibler Divergence"]
)

for metric in pivoted_data.columns.levels[0]:
    pivoted_data[(metric, '')] = "(" + round(pivoted_data[(metric, 'Lower Bound')], 2).astype(str) + ", " + round(pivoted_data[(metric, 'Upper Bound')], 2).astype(str) + ")"

pivoted_data = pivoted_data.drop(columns=["Lower Bound", "Upper Bound"], level=1)
pivoted_data.columns = [f"{metric}" for metric, _ in pivoted_data.columns]
pivoted_data = pivoted_data.reset_index()

save_path_t4 = os.path.join(PLOT_PATH, "ci_table.png")
dfi.export(pivoted_data, save_path_t4)

# Table 4 - 95% Credible Interval Table for just Wasserstein and Energy

pivoted_data = ci_table.pivot_table(
    index=['Model', 'Parameter'],
    columns=["summary_statistic"],
    values = ["Wasserstein Distance", "Energy Distance"]
)

for metric in pivoted_data.columns.levels[0]:
    pivoted_data[(metric, '')] = "(" + round(pivoted_data[(metric, 'Lower Bound')], 2).astype(str) + ", " + round(pivoted_data[(metric, 'Upper Bound')], 2).astype(str) + ")"

pivoted_data = pivoted_data.drop(columns=["Lower Bound", "Upper Bound"], level=1)
pivoted_data.columns = [f"{metric}" for metric, _ in pivoted_data.columns]
pivoted_data = pivoted_data.reset_index()

save_path_t5 = os.path.join(PLOT_PATH, "ci_table_energy_wasserstein.png")
dfi.export(pivoted_data, save_path_t5)