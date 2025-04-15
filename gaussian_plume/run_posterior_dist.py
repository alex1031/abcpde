import numpy as np 
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from common.abc_posterior import gaussian_abc_posterior_data

MODELS_PATH = "./gaussian_plume/runs"
PLOT_PATH = "./gaussian_plume/plots"

PARAMS = ["$c_x$", "$c_y$", "$s$"]
TRUE_VALUES = [0.5, 0.5, 5e-5]
BINS = [7, 7, 10]
DISTANCES = ["Wasserstein Distance", "Energy Distance", "Maximum Mean Discrepancy", "Cramer-von Mises Distance", "Kullback-Leibler Divergence"]
NPARAM = 2

if __name__ == "__main__":
    models = os.listdir(MODELS_PATH)

    for model in models:
        fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=False)
        cx_values, cy_values, s_values = {}, {}, {}
        metric_path = os.path.join(MODELS_PATH, model)
        plot_path = os.path.join(PLOT_PATH, model)
        metrics = os.listdir(metric_path) 

        for metric in metrics:
            posterior_path = os.path.join(metric_path, metric)
            posteriors = os.listdir(posterior_path)
            path = os.path.join(posterior_path, posteriors[0])
            p = np.load(path)
            cx_values[metric] = p[:, :, 1][:, 0]
            cy_values[metric] = p[:, :, 1][:, 1]
            s_values[metric]  = p[:, :, 1][:, 2]

        for i, (param, true_val, bins) in enumerate(zip([cx_values, cy_values, s_values], TRUE_VALUES, BINS)):
            for metric, values in param.items():
                sns.histplot(ax=ax[i], data=values, stat="probability", element="bars", fill=True, label=metric, bins=bins)
            ax[i].axvline(true_val, color='red', linestyle='--', label="True Value" if i == 0 else "")
            ax[i].set_xlabel(PARAMS[i])
            ax[i].set_ylabel("")
            ax[i].grid(True)
            if i == 2:
                ax[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax[i].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        # Title
        model_name = model.split("_")
        if model_name[0] == "no":
            title = "Distribution of Posterior Medians for $\\varepsilon\\sim N(0, 0)$"
        elif model_name[0] == "linear":
            title = "Distribution of Posterior Medians for $\\varepsilon\\sim N(0, t^2)$"
        else:
            title = f"Distribution of Posterior Medians for $\\varepsilon\\sim N(0, {model_name[0]}^2)$"
        fig.suptitle(title, fontsize=14)
        fig.supylabel("Probability")

        # Only one legend, placed smartly
        ax[0].legend(loc="upper right", bbox_to_anchor=(1.3, 1), title="Distance Metric", title_fontsize='medium')

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leaves space for suptitle
        os.makedirs(plot_path, exist_ok=True)
        save_path = os.path.join(plot_path, "posterior_median_distribution.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
