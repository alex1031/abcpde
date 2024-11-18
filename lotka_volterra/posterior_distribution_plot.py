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

# fig, ax = plt.subplots(2, 9, sharex='row', sharey='row', figsize=(20, 20))
# fig, ax = plt.subplots(2, 9, figsize=(20, 20))

# for idx, model in enumerate(models):
#     metric_path = os.path.join(MODELS_PATH, model)
#     metrics = os.listdir(metric_path)
#     for metric_idx, metric in enumerate(metrics):
#         # Want all metrics in one plot
#         posterior_path = os.path.join(metric_path, metric)
#         posteriors = os.listdir(posterior_path)
#         # We only want the 0.1% quantiles
#         path = os.path.join(posterior_path, posteriors[0])
#         p = np.load(path)
#         # Want median values only
#         alpha_median = p[:,:,1][:,0]
#         beta_median = p[:,:,1][:,1]
#         sns.histplot(ax=ax[0, idx], data=alpha_median, element="poly", fill=False, color=COLORS[metric_idx], label=metric)
#         sns.histplot(ax=ax[1, idx], data=beta_median, element="poly", fill=False, color=COLORS[metric_idx], label=metric)
#         ax[0, idx].set_xlabel(r"$\alpha$")
#         ax[1, idx].set_xlabel(r"$\beta$")
#         ax[0, idx].set_ylabel("density")
#         ax[1, idx].set_ylabel("density")
#         ax[0, idx].set_title(model, fontsize=7)

# handles, labels = ax[0,0].get_legend_handles_labels()  # Collect labels from the first subplot
# plt.subplots_adjust(bottom=0.15)
# fig.legend(handles, labels, loc='lower center', ncol=len(metric), bbox_to_anchor=(0.5, -0.05), fontsize=7)  # Adjust location and number of columns in the legend
# fig.tight_layout()
# save_path = os.path.join(PLOT_PATH, "posterior_distribution.png")
# plt.savefig(save_path)

for model in models:
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    alpha_values, beta_values = {}, {}
    metric_path = os.path.join(MODELS_PATH, model)
    plot_path = os.path.join(PLOT_PATH, model)
    metrics = os.listdir(metric_path) 
    for metric in metrics:
        posterior_path = os.path.join(metric_path, metric)
        posteriors = os.listdir(posterior_path)
        # We only want the 0.1% quantiles
        path = os.path.join(posterior_path, posteriors[0])
        p = np.load(path)
        # Want median values only
        alpha_median = p[:,:,1][:,0]
        beta_median = p[:,:,1][:,1]
        alpha_values[metric] = alpha_median
        beta_values[metric] = beta_median 

    for val in alpha_values:
        dist_metric = val
        alpha_med = alpha_values[val]
        beta_med = beta_values[val]
        sns.histplot(ax=ax[0], data=alpha_med, element="poly", fill=False, label=dist_metric)
        sns.histplot(ax=ax[1], data=beta_med, element="poly", fill=False, label=dist_metric)
        ax[0].set_xlabel(r"$\alpha$")
        ax[1].set_xlabel(r"$\beta$")
        ax[0].set_ylabel("")
        ax[1].set_ylabel("")
        # ax[0].set_title(model, fontsize=7)
    
    # Title for plot
    model_name = model.split("_")
    model_noise = model_name[0][1:len(model_name[0])]
    if len(model_name) == 3:
        model_smoothing = "No Smoothing"
    else:
        model_smoothing = "With Smoothing"
    
    if model_noise == "linear":
        model_noise = "t"
    plot_title = f"$\epsilon\sim N(0, {model_noise}^2)$ {model_smoothing}"
    fig.suptitle(plot_title)
    fig.supylabel("density")
    plt.subplots_adjust(bottom=0.15)
    plt.legend(title="Distance Metric", title_fontproperties={'weight':'bold'})
    fig.tight_layout()
    save_path = os.path.join(plot_path, "posterior_distribution.png")
    plt.savefig(save_path)