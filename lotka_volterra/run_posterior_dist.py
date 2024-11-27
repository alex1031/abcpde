import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from common.abc_posterior import abc_posterior_data

RUN_PATH = "./lotka_volterra/runs"
PLOT_PATH = "./lotka_volterra/plots"
DISTANCES = ["Wasserstein Distance", "Energy Distance", "Maximum Mean Discrepancy", "Cramer-von Mises Distance", "Kullback-Leibler Divergence"]
NPARAM = 2

models = os.listdir(RUN_PATH)

for model in models:
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    run_model_path = os.path.join(RUN_PATH, model)
    run = os.path.join(RUN_PATH, model, "run1.npy")
    distances = np.load(run)

    for i in range(NPARAM, distances.shape[1]):
        temp_dist = distances[:,i]
        threshold_data = abc_posterior_data(NPARAM, distances, 0.001, DISTANCES[i-2])
        threshold_data = np.where(threshold_data != np.inf, threshold_data, np.nan)
        alpha, beta = threshold_data[:,0], threshold_data[:,1]
        print(DISTANCES[i-2], threshold_data.shape)
        sns.histplot(ax=ax[0], data=alpha, element="step", fill=True, label=DISTANCES[i-2], binwidth=0.1)
        sns.histplot(ax=ax[1], data=beta, element="step", fill=True, label=DISTANCES[i-2], binwidth=0.1)
        ax[0].axvline(1, c='r')
        ax[1].axvline(1, c='r')
        ax[0].set_xlabel(r"$\alpha$")
        ax[1].set_xlabel(r"$\beta$")
        ax[0].set_ylabel("")
        ax[1].set_ylabel("")
    
    model_name = model.split("_")
    model_noise = model_name[0][1:len(model_name[0])]
    if len(model_name) == 3:
        model_smoothing = "No Smoothing"
    else:
        model_smoothing = "With Smoothing"
    
    if model_noise == "linear":
        model_noise = "t"
    plot_title = f"Posterior Distribution for $\epsilon\sim N(0, {model_noise}^2)$ {model_smoothing}"
    fig.suptitle(plot_title)
    fig.supylabel("Density")
    fig.tight_layout()
    plt.legend()

    model_path = os.path.join(PLOT_PATH, model)
    save_path = os.path.join(model_path, "run_posterior_distribution.png")
    plt.savefig(save_path)
