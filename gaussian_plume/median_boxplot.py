import numpy as np
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "./gaussian_plume/results"
SAVE_DIR = "./gaussian_plume/plots"
QUANTILE = 0.001

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
                    lab = "Wasserstein"
                elif metric == "Hausdorff Distance":
                    lab = "Hausdorff"
                elif metric == "Frechet Distance":
                    lab = "Frechet"
                labels.append(lab)

        # Plot boxplots for cx, cy, and s
        axes[0].boxplot(cx_data, labels=labels)
        axes[0].set_title("$c_y$")
        axes[0].set_ylabel("Value")
        if model == "no_noise_diffusion":
            axes[0].axhline(0, linestyle="--", label="True Value",c="r")
        elif model == "no_noise_5e-3_advection":
            axes[0].axhline(5e-3, linestyle="--", label="True Value",c="r")
        elif model == "no_noise_calm_air":
            axes[0].axhline(0.05, linestyle="--", label="True Value",c="r")
        else:
            if "case_study" not in model:
                axes[0].axhline(0.5, linestyle="--", label="True Value",c="r")

        axes[1].boxplot(cy_data, labels=labels)
        axes[1].set_title("$c_x$")
        if model == "no_noise_diffusion":
            axes[1].axhline(0, linestyle="--", c="r")
        elif model == "no_noise_5e-3_advection":
            axes[1].axhline(5e-3, linestyle="--", c="r")
        elif model == "no_noise_calm_air":
            axes[1].axhline(0.05, linestyle="--", c="r")
        else:
            if "case_study" not in model:
                axes[1].axhline(0.5, linestyle="--", c="r")

        axes[2].boxplot(s_data, labels=labels)
        axes[2].set_title("$D$")
        if "case_study" not in model:
            axes[2].axhline(5e-5, linestyle="--", c="r")

        model_str = model.split("_")
        if model_str[0] == "linear":
            title = "Boxplot for Each Parameter with $\\varepsilon\\sim N(0, t^2)$"
        elif model_str[0] == "no":
            title = "Boxplot for Each Parameter with No Noise"
        elif model_str[0] == "case":
            title = "Boxplot for Each Parameter in Case Study Dataset"
        else:
            title = f"Boxplot for Each Parameter with $\\varepsilon\\sim N(0, {model_str[0]}^2)$"

        fig.suptitle(title)
        fig.supylabel("Value")
        fig.supxlabel("Distance Metric")
        fig.legend()
        # Adjust layout
        plt.tight_layout()

        plot_save_path = os.path.join(save_path, "median_boxplot.png")
        plt.savefig(plot_save_path)