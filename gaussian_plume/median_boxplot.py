import numpy as np
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "./gaussian_plume/results"
SAVE_DIR = "./gaussian_plume/plots"
QUANTILE = 0.001 # Change to 0.001 later 

if __name__ == "__main__":

    # Get model names
    models = os.listdir(RESULTS_DIR)

    # Loop through each model and generate a separate figure
    for model in models:
        model_path = os.path.join(RESULTS_DIR, model)
        distances = os.listdir(model_path)  # Get available distance metrics for this model
        save_path = os.path.join(SAVE_DIR, model)
        
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        # Prepare subplots for this model
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns (cx, cy, s)
        axes = axes.flatten()

        # Store data for boxplots
        cx_data, cy_data, s_data = [], [], []
        labels = []  # To store metric names for labeling

        # Loop through distance metrics
        for metric in distances:
            metric_path = os.path.join(model_path, metric)
            posterior_path = os.path.join(metric_path, f"{QUANTILE}posterior.npy")

            if os.path.exists(posterior_path):
                posterior = np.load(posterior_path)  # Shape: (NUM_RUNS, NPARAMS, 6)

                # Extract the 1st statistical value (e.g., Mean or Median)
                cx_values = posterior[:, 0, 1]  # cx (parameter index 0, statistic index 1)
                cy_values = posterior[:, 1, 1]  # cy (parameter index 1, statistic index 1)
                s_values = posterior[:, 2, 1]   # s  (parameter index 2, statistic index 1)

                # Store extracted values
                cx_data.append(cx_values)
                cy_data.append(cy_values)
                s_data.append(s_values)
                if metric == "Cramer-von Mises Distance":
                    lab = "CvMD"
                elif metric == "Wasserstein Distance":
                    lab = "WD"
                elif metric == "Hausdorff Distance":
                    lab = "HD"
                elif metric == "Frechet Distance":
                    lab = "FD"
                labels.append(lab)

        # Plot boxplots for cx, cy, and s
        axes[0].boxplot(cx_data, labels=labels)
        axes[0].set_title("$c_x$")
        axes[0].set_ylabel("Value")

        axes[1].boxplot(cy_data, labels=labels)
        axes[1].set_title("$c_y$")


        axes[2].boxplot(s_data, labels=labels)
        axes[2].set_title("$s$")

        model_str = model.split("_")
        if model_str[0] == "linear":
            title = "Boxplot for Distribution of Each Parameter with $\\varepsilon\\sim N(0, t^2)$"
        elif model_str[0] == "no":
            title = "Boxplot for Distribution of Each Parameter with No Noise"
        else:
            title = f"Boxplot for Distribution of Each Parameter with $\\varepsilon\\sim N(0, {model_str[0]}^2)$"

        fig.suptitle(title)
        # Adjust layout
        plt.tight_layout()

        plot_save_path = os.path.join(save_path, "median_boxplot.png")
        plt.savefig(plot_save_path)