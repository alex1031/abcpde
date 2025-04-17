import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

PLOT_PATH = "./lotka_volterra/plots"
DATAFRAME_PATH = "./lotka_volterra/dataframe/all_summary_statistics.csv"
DISTANCE = ["Cramer-von Mises Distance", "Energy Distance", "Kullback-Leibler Divergence", "Maximum Mean Discrepancy",  "Wasserstein Distance"]
MODELS = ["n0_no_smoothing", "n0.25_no_smoothing", "n0.25_smoothing", "n0.5_no_smoothing", "n0.5_smoothing", "n0.75_no_smoothing", "n0.75_smoothing",
          "nlinear_no_smoothing", "nlinear_smoothing"]
NOISE = [0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, "linear", "linear"]

models = os.listdir(PLOT_PATH)
df = pd.read_csv(DATAFRAME_PATH)

for j, model in enumerate(MODELS):
    # Filter + store dataframe for RMSE and StDev
    model_rmse = df[(df["model"] == model) & (df["summary_statistic"] == "RMSE")].reset_index(drop=True)
    model_stdev = df[(df["model"] == model) & (df["summary_statistic"] == "StDev")].reset_index(drop=True)

    # Environment for RMSE and StDev plot
    fig_rmse, axes_rmse = plt.subplots(2, 3, sharey=True, figsize=(10, 10))
    axes_rmse[1, 2].axis("off")
    ax_rmse = axes_rmse.flatten()

    fig_stdev, axes_stdev = plt.subplots(2, 3, sharey=True, figsize=(10, 10))
    axes_stdev[1, 2].axis("off")
    ax_stdev = axes_stdev.flatten()

    for i, d in enumerate(DISTANCE):
        # Barplot for RMSE
        rmse = sns.barplot(data=model_rmse, x="quantile", y=d, hue="param", palette="Set1", ax=ax_rmse[i])

        # Barplot for StDev
        stdev = sns.barplot(data=model_stdev, x="quantile", y=d, hue="param", palette="Set1", ax=ax_stdev[i])

        # Only Keep Legend for one of them + change legend title
        if i != 0:
            rmse.legend_.remove()
            stdev.legend_.remove()
        else:
            rmse.legend_.set_title("Parameter")
            new_labels = ["$\\alpha$", "$\\beta$"]
            for t, l in zip(rmse.legend_.texts, new_labels):
                t.set_text(l)
            
            stdev.legend_.set_title("Parameter")
            for t, l in zip(stdev.legend_.texts, new_labels):
                t.set_text(l)

        # Change xlabel and ylabel
        ax_rmse[i].set_xlabel("Quantile")
        ax_rmse[i].set_ylabel(" ")

        ax_stdev[i].set_xlabel("Quantile")
        ax_stdev[i].set_ylabel(" ")
        # Add plot title
        ax_rmse[i].set_title(d)
        ax_stdev[i].set_title(d)
    
    if NOISE[j] == "linear":
        title = "$\\varepsilon\\sim N(0, t^2)$"
    else:
        title = f"$\\varepsilon\\sim N(0, {NOISE[j]}^2)$"
    if "no_smoothing" in model:
        title += " No Smoothing"
    else:
        title += " Smoothed"

    # Add suptitle and supxlabel
    fig_rmse.suptitle("RMSE for Each Threshold " + title)
    fig_rmse.supylabel("RMSE", fontsize=12)
    fig_rmse.tight_layout()

    fig_stdev.suptitle("Standard Deviation for Each Threshold " + title)
    fig_stdev.supylabel("Standard Deviation", fontsize=12)
    fig_stdev.tight_layout()

    fig_rmse.savefig(os.path.join(PLOT_PATH, model + "/rmse_threshold_bar_plot.png"))
    fig_stdev.savefig(os.path.join(PLOT_PATH, model + "/stdev_threshold_bar_plot.png"))