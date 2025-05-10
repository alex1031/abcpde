import pandas as pd
import dataframe_image as dfi
import os

DATAFRAME_PATH = "./gaussian_plume/dataframe/all_summary_statistics.csv"
THRESHOLD = 0.0001
CI_PROPORTION_PATH = f"./gaussian_plume/dataframe/{THRESHOLD}posterior_ci_proportion.csv"
PLOT_PATH = "./gaussian_plume/plots"

if __name__ == "__main__":
    quantile = str(THRESHOLD*100) + "%"
    df = pd.read_csv(DATAFRAME_PATH)
    df.rename(columns={"model": "Model", "param": "Parameter"}, inplace=True)
    keep_cols = ["Model", "Parameter", "Wasserstein Distance", "Cramer-von Mises Distance", "Frechet Distance", "Hausdorff Distance"]
    metric_col = ["Wasserstein Distance", "Cramer-von Mises Distance", "Frechet Distance", "Hausdorff Distance"]

    # Table 2 - RMSE table
    rmse = df[(df["summary_statistic"] == "RMSE") & (df["quantile"] == quantile)].reset_index(drop=True)
    rmse = rmse[keep_cols]

    rmse_table = pd.pivot_table(rmse, values=metric_col, index = ["Model", "Parameter"])

    rmse_styled = rmse_table.style.background_gradient(axis=None, vmin=0, vmax=1, cmap="YlOrRd") #adding a gradient based on values in cell

    save_path_t2 = os.path.join(PLOT_PATH, f"{THRESHOLD}rmse_table.png")
    dfi.export(rmse_styled, save_path_t2, table_conversion="celenium")

    # Table 3 - Standard Deviation table
    stdev_table = df[(df["summary_statistic"] == "StDev") & (df["quantile"] == quantile)].reset_index(drop=True)
    piv_table = pd.pivot_table(stdev_table, values=metric_col, index = ["Model", "Parameter"])

    piv_table = piv_table[metric_col]

    stdev_styled = piv_table.style.background_gradient(axis=None, vmin=0, vmax=5, cmap="YlOrRd") #adding a gradient based on values in cell

    save_path_t3 = os.path.join(PLOT_PATH, f"{THRESHOLD}stdev_table.png")
    dfi.export(stdev_styled, save_path_t3, table_conversion="celenium")

    # Table 4i - 95% Credible Interval Table
    ci_table = df[(df["quantile"] == quantile) & df["summary_statistic"].isin(["Lower Bound", "Upper Bound"])].reset_index(drop=True)

    pivoted_data = ci_table.pivot_table(
        index=['Model', 'Parameter'],
        columns=["summary_statistic"],
        values = metric_col
    )

    for metric in pivoted_data.columns.levels[0]:
        pivoted_data[(metric, '')] = "(" + round(pivoted_data[(metric, 'Lower Bound')], 2).astype(str) + ", " + round(pivoted_data[(metric, 'Upper Bound')], 2).astype(str) + ")"

    pivoted_data = pivoted_data.drop(columns=["Lower Bound", "Upper Bound"], level=1)
    pivoted_data.columns = [f"{metric}" for metric, _ in pivoted_data.columns]
    pivoted_data = pivoted_data.reset_index()

    save_path_t4 = os.path.join(PLOT_PATH, f"{THRESHOLD}ci_table.png")
    dfi.export(pivoted_data, save_path_t4, table_conversion="celenium")

    # Table 5 - 95% CI Proportion table
    ci_proportion = pd.read_csv(CI_PROPORTION_PATH)

    melted_ci_proportion = ci_proportion.melt(id_vars=["Metric", "Model"],
                                              value_vars = ["cx_proportion", "cy_proportion", "s_proportion"],
                                              var_name="Parameter", value_name="Proportion")

    melted_ci_proportion.replace({"cx_proportion": "cx", "cy_proportion": "cy", "s_proportion": "s"}, inplace=True)

    ci_pivot_table = pd.pivot(melted_ci_proportion,
                              values=["Proportion"], index=["Model", "Parameter"], columns=["Metric"])

    ci_pivot_table = ci_pivot_table.style.format(precision=2)
    save_path_t6 = os.path.join(PLOT_PATH, f"{THRESHOLD}ci_proportion.png")
    dfi.export(ci_pivot_table, save_path_t6, table_conversion="celenium")