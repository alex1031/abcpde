import numpy as np
import matplotlib.pyplot as plt
import os
from common.abc_posterior import abc_posterior_data

RUN_PATH = "./lotka_volterra/runs"
PLOT_PATH = "./lotka_volterra/plots"
DISTANCE_METRIC = ["Wasserstein Distance", "Energy Distance"]
QUANTILE = ["5%", "1%", "0.1%"]
NPARAM = 2

models = os.listdir(RUN_PATH)

for model in models:
    model_path = os.path.join(RUN_PATH, model)
    run_path = os.path.join(model_path, "run1.npy")
    run = np.load(run_path)
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax = ax.flatten()
    i = 0
    for metric in DISTANCE_METRIC:
        # 5% threshold (High)
        high_thresh = abc_posterior_data(NPARAM, run, 0.05, metric)
        # 1% threshold (Medium)
        medium_thresh = abc_posterior_data(NPARAM, run, 0.01, metric)
        # 0.1% threshold (Low)
        low_thresh = abc_posterior_data(NPARAM, run, 0.001, metric)
        
        ax[i].scatter(high_thresh[:,0], high_thresh[:,1], c="r", label="5%")
        ax[i].scatter(medium_thresh[:,0], medium_thresh[:,1], c="g", label="1%")
        ax[i].scatter(low_thresh[:,0], low_thresh[:,1], c="y", label="0.01%")
        ax[i].set_title(metric, fontsize=6)
        ax[i].axvline(1)
        ax[i].axhline(1)
        i += 1

    model_name = model.split("_")
    model_noise = model_name[0][1:len(model_name[0])]
    if len(model_name) == 3:
        model_smoothing = "No Smoothing"
    else:
        model_smoothing = "With Smoothing"
    if model_noise == "linear":
        model_noise = "t"
    plot_title = f"Parameter Distribution of $\epsilon\sim N(0, {model_noise}^2)$ {model_smoothing}"
    fig.suptitle(plot_title)
    fig.supxlabel(r'$\alpha$')
    fig.supylabel(r'$\beta$', rotation=0)
    fig.legend(QUANTILE, title="Threshold", title_fontproperties={'weight':'bold'}, bbox_to_anchor=(0.9, 0.4))
    fig.tight_layout()
    
    plot_model_path = os.path.join(PLOT_PATH, model)
    model_save_path = os.path.join(PLOT_PATH, model)
    save_path = os.path.join(model_save_path, "run_scatter_plot.png")
    fig.savefig(save_path)
        

