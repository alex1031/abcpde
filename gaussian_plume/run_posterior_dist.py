import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from common.abc_posterior import gaussian_abc_posterior_data

RUN_PATH = "./gaussian_plume/runs"
SAVE_PATH = "./gaussian_plume/plots"
OBSERVED_PATH = "./gaussian_plume/observed_data/no_noise/no_noise.npy"
METRICS = ["Cramer-von Mises Distance", "Frechet Distance", "Hausdorff Distance", "Wasserstein Distance"]
MODELS = ["no_noise", "linear_noise", "0.025_noise", "0.05_noise", "0.075_noise", "no_noise_diffusion", "no_noise_5e-3_advection", "no_noise_calm_air"]
NPARAMS = 3
PARAMS = ["$c_x$", "$c_y$", "$s$"]
THRESHOLD = 0.0001
TRUE_VALUES = [0.5, 0.5, 5e-5]
TRUE_VALUES_DIFFUSION = [0, 0, 5e-5]
TRUE_VALUES_ADVECTION = [5e-3, 5e-3, 5e-5]
TRUE_VALUES_CALM_AIR = [0.05, 0.05, 5e-5]
BINS = [10, 10, 20]

palette = sns.color_palette("colorblind", len(METRICS))
linestyles = {
    "Cramer-von Mises Distance": "-",
    "Frechet Distance": "--",
    "Hausdorff Distance": "-.",
    "Wasserstein Distance": ":"
}

if __name__ == "__main__":

    for model in MODELS:

        fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=False)

        run_path = os.path.join(RUN_PATH, model + "/run1.npy")
        run = np.load(run_path)
        cx_values, cy_values, s_values = {}, {}, {}
        plot_path = os.path.join(SAVE_PATH, model)

        for metric in METRICS:
            threshold_data = gaussian_abc_posterior_data(NPARAMS, run, THRESHOLD, metric)
            cx, cy, s = threshold_data[:,0], threshold_data[:,1], threshold_data[:,2]
            cx_values[metric] = cx
            cy_values[metric] = cy
            s_values[metric] = s
        
        if model == "no_noise_diffusion":
            true_values = TRUE_VALUES_DIFFUSION
        elif model == "no_noise_5e-3_advection":
            true_values = TRUE_VALUES_ADVECTION
        elif model == "no_noise_calm_air":
            true_values = TRUE_VALUES_CALM_AIR
        else:
            true_values = TRUE_VALUES

        for i, (param, true_val, bins) in enumerate(zip([cx_values, cy_values, s_values], true_values, BINS)):
            all_vals = np.concatenate(list(param.values()))
            xmin, xmax = all_vals.min(), all_vals.max()

            for j, (metric, values) in enumerate(param.items()):
                sns.histplot(
                    ax=ax[i],
                    data=values,
                    stat="probability",
                    element="step",
                    fill=False,
                    bins=bins,
                    alpha=0.9,
                    linewidth=1.5,
                    label=metric,
                    color=palette[j],
                    linestyle=linestyles[metric]
                )

            ax[i].axvline(true_val, color='red', linestyle='--', linewidth=2.0, label="True Value" if i == 0 else "")
            if i == 2:
                ax[i].set_xlim(0, 1e-4)
            else:
                if model == "no_noise_5e-3_advection":
                    ax[i].set_xlim(0, 1e-2)
                elif model == "no_noise_calm_air":
                    ax[i].set_xlim(0, 0.1)
                else:
                    ax[i].set_xlim(0, 1)
            ax[i].set_xlabel(PARAMS[i])
            ax[i].set_ylabel("Probability")
            ax[i].grid(True)
            if i == 2:
                ax[i].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax[i].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        model_name = model.split("_")
        if model_name[0] == "no":
            title = "Posterior Distribution for $\\varepsilon\\sim N(0, 0)$"
        elif model_name[0] == "linear":
            title = "Posterior Distribution for $\\varepsilon\\sim N(0, t^2)$"
        else:
            title = f"Posterior Distribution for $\\varepsilon\\sim N(0, {model_name[0]}^2)$"
        fig.suptitle(title, fontsize=14)

        handles, labels = ax[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax[0].legend(by_label.values(), by_label.keys(), loc="upper right", bbox_to_anchor=(1.3, 1), title="Distance Metric", title_fontsize='medium')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        os.makedirs(plot_path, exist_ok=True)
        save_path = os.path.join(plot_path, f"{THRESHOLD}run_posterior_distribution.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)