import pandas as pd
import dataframe_image as dfi
import os

DATAFRAME_PATH = "./gaussian_plume/dataframe/all_summary_statistics.csv"
CI_PROPORTION_PATH = "./gaussian_plume/dataframe/ci_proportion.csv"
PLOT_PATH = "./gaussian_plume/plots"

if __name__ == "__main__":
    model_dict = {
        "no_noise": "ε ~ N(0, 0)",
        "0.025_noise": "ε ~ N(0, 0.025^2)",
        "0.05_noise": "ε ~ N(0, 0.05^2)",
        "0.075_noise": "ε ~ N(0, 0.075^2)",
        "linear_noise": "ε ~ N(0, t^2)"
    }

    df = pd.read_csv(DATAFRAME_PATH)
    df.rename(columns={"model": "Model", "param": "Parameter"}, inplace=True)
    df["Model"] = df["Model"].replace(model_dict, regex=True)
    median = df[(df["summary_statistic"] == "Median") & (df["quantile"] == "0.1%")].reset_index(drop=True)

    keep_cols = ["Model", "Parameter", "Wasserstein Distance", "Cramer-von Mises Distance", "Frechet Distance", "Hausdorff Distance"]

    median_table = median[keep_cols]
    metric_col = ["Wasserstein Distance", "Cramer-von Mises Distance", "Frechet Distance", "Hausdorff Distance"]

    # Table 1 - Add Column for True Value + Median Values

    # median_table["True Value"] = np.ones(len(median_table))
    # piv_table = pd.pivot_table(median_table, values=["True Value", "Wasserstein Distance", "Energy Distance", "Maximum Mean Discrepancy",
    #                                           "Cramer-von Mises Distance", "Kullback-Leibler Divergence"], index = ["Model", "Parameter"])

    # piv_table = piv_table[["True Value", "Wasserstein Distance", "Energy Distance", "Maximum Mean Discrepancy",
    #                        "Cramer-von Mises Distance", "Kullback-Leibler Divergence"]]

    # df_styled = piv_table.style.background_gradient(axis=None, vmin=1, vmax=10, cmap="YlOrRd") #adding a gradient based on values in cell
    # df_styled.format(precision=2)

    # save_path_t1 = os.path.join(PLOT_PATH, "median_table.png")
    # dfi.export(df_styled, save_path_t1)

    # # Table 2 - RMSE table
    rmse = df[(df["summary_statistic"] == "RMSE") & (df["quantile"] == "0.1%")].reset_index(drop=True)
    rmse = rmse[keep_cols]

    rmse_table = pd.pivot_table(rmse, values=metric_col, index = ["Model", "Parameter"])

    rmse_styled = rmse_table.style.background_gradient(axis=None, vmin=0, vmax=1, cmap="YlOrRd") #adding a gradient based on values in cell
    # rmse_styled.format(precision=2)

    save_path_t2 = os.path.join(PLOT_PATH, "rmse_table.png")
    dfi.export(rmse_styled, save_path_t2, table_conversion="celenium")

    # Table 3 - Standard Deviation table
    stdev_table = df[(df["summary_statistic"] == "StDev") & (df["quantile"] == "0.1%")].reset_index(drop=True)
    piv_table = pd.pivot_table(stdev_table, values=metric_col, index = ["Model", "Parameter"])

    piv_table = piv_table[metric_col]

    stdev_styled = piv_table.style.background_gradient(axis=None, vmin=0, vmax=5, cmap="YlOrRd") #adding a gradient based on values in cell
    # stdev_styled.format(precision=2)

    save_path_t3 = os.path.join(PLOT_PATH, "stdev_table.png")
    dfi.export(stdev_styled, save_path_t3, table_conversion="celenium")

    # # Table 4i - 95% Credible Interval Table
    ci_table = df[(df["quantile"] == "0.1%") & df["summary_statistic"].isin(["Lower Bound", "Upper Bound"])].reset_index(drop=True)

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

    save_path_t4 = os.path.join(PLOT_PATH, "ci_table.png")
    dfi.export(pivoted_data, save_path_t4, table_conversion="celenium")

    # # Table 4ii - 95% Credible Interval Table for just Wasserstein and Energy

    # pivoted_data = ci_table.pivot_table(
    #     index=['Model', 'Parameter'],
    #     columns=["summary_statistic"],
    #     values = ["Wasserstein Distance", "Energy Distance"]
    # )

    # for metric in pivoted_data.columns.levels[0]:
    #     pivoted_data[(metric, '')] = "(" + round(pivoted_data[(metric, 'Lower Bound')], 2).astype(str) + ", " + round(pivoted_data[(metric, 'Upper Bound')], 2).astype(str) + ")"

    # pivoted_data = pivoted_data.drop(columns=["Lower Bound", "Upper Bound"], level=1)
    # pivoted_data.columns = [f"{metric}" for metric, _ in pivoted_data.columns]
    # pivoted_data = pivoted_data.reset_index()

    # save_path_t5 = os.path.join(PLOT_PATH, "ci_table_energy_wasserstein.png")
    # dfi.export(pivoted_data, save_path_t5)

    # Table 5 - 95% CI Proportion table
    ci_proportion = pd.read_csv(CI_PROPORTION_PATH)

    melted_ci_proportion = ci_proportion.melt(id_vars=["Metric", "Model"],
                                              value_vars = ["cx_proportion", "cy_proportion", "s_proportion"],
                                              var_name="Parameter", value_name="Proportion")

    melted_ci_proportion.replace({"cx_proportion": "cx", "cy_proportion": "cy", "s_proportion": "s"}, inplace=True)

    ci_pivot_table = pd.pivot(melted_ci_proportion,
                              values=["Proportion"], index=["Model", "Parameter"], columns=["Metric"])

    ci_pivot_table.index = pd.MultiIndex.from_tuples([('ε ~ N(0, 0)', 'cx'),
                                                    ('ε ~ N(0, 0)',  'cy'),
                                                    ('ε ~ N(0, 0)',  's'),
                                                    ('ε ~ N(0, 0.025^2)', 'cx'),
                                                    ('ε ~ N(0, 0.025^2)',  'cy'),
                                                    ('ε ~ N(0, 0.025^2)', 's'),
                                                    ('ε ~ N(0, 0.05^2)',  'cx'),
                                                    ('ε ~ N(0, 0.05^2)', 'cy'),
                                                    ('ε ~ N(0, 0.05^2)', 's'),
                                                    ('ε ~ N(0, 0.075^2)',  'cx'),
                                                    ('ε ~ N(0, 0.075^2)',  'cy'),
                                                    ('ε ~ N(0, 0.075^2)',  's'),
                                                    ('ε ~ N(0, t^2)', 'cx'),
                                                    ('ε ~ N(0, t^2)',  'cy'),
                                                    ('ε ~ N(0, t^2)', 's')], names=["Model", "Parameter"])

    ci_pivot_table = ci_pivot_table.style.format(precision=2)
    save_path_t6 = os.path.join(PLOT_PATH, "ci_proportion.png")
    dfi.export(ci_pivot_table, save_path_t6, table_conversion="celenium")