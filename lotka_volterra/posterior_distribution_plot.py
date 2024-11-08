import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import os
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

MODELS_PATH = "./lotka_volterra/results"
PLOT_PATH = "./lotka_volterra/plots"
COLORS = ['b', 'g', 'r', 'c', 'm']

models = os.listdir(MODELS_PATH)

fig, ax = plt.subplots(2, 3, sharex='row', sharey='row')
for idx, model in enumerate(models):
    metric_path = os.path.join(MODELS_PATH, model)
    metrics = os.listdir(metric_path)
    for metric_idx, metric in enumerate(metrics):
        # Want all metrics in one plot
        posterior_path = os.path.join(metric_path, metric)
        posteriors = os.listdir(posterior_path)
        # We only want the 0.1% quantiles
        path = os.path.join(posterior_path, posteriors[0])
        p = np.load(path)
        # Want median values only
        alpha_median = p[:,:,1][:,0]
        beta_median = p[:,:,1][:,1]
        sns.histplot(ax=ax[0, idx], data=alpha_median, element="poly", fill=False, color=COLORS[metric_idx], label=metric)
        sns.histplot(ax=ax[1, idx], data=beta_median, element="poly", fill=False, color=COLORS[metric_idx], label=metric)
        ax[0, idx].set_xlabel(r"$\alpha$")
        ax[1, idx].set_xlabel(r"$\beta$")
        ax[0, idx].set_ylabel("density")
        ax[1, idx].set_ylabel("density")
        ax[0, idx].set_title(model, fontsize=7)

handles, labels = ax[0,0].get_legend_handles_labels()  # Collect labels from the first subplot
plt.subplots_adjust(bottom=0.15)
fig.legend(handles, labels, loc='lower center', ncol=len(metric), bbox_to_anchor=(0.5, -0.05), fontsize=7)  # Adjust location and number of columns in the legend
fig.tight_layout()
save_path = os.path.join(PLOT_PATH, "posterior_distribution.png")
plt.savefig(save_path)