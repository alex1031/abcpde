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
        sns.histplot(ax=ax[0], data=alpha_med, stat="probability", element="bars", fill=True, label=dist_metric, bins=7)
        sns.histplot(ax=ax[1], data=beta_med, stat="probability", element="bars", fill=True, label=dist_metric, bins=7)
        ax[0].set_xlabel(r"$\alpha$")
        ax[1].set_xlabel(r"$\beta$")
        ax[0].set_ylabel("")
        ax[1].set_ylabel("")
    
    # Title for plot
    model_name = model.split("_")
    model_noise = model_name[0][1:len(model_name[0])]
    if len(model_name) == 3:
        model_smoothing = "No Smoothing"
    else:
        model_smoothing = "With Smoothing"
    
    if model_noise == "linear":
        model_noise = "t"
    plot_title = f"Distribution for Posterior Medians in $\epsilon\sim N(0, {model_noise}^2)$ {model_smoothing}"
    fig.suptitle(plot_title)
    fig.supylabel("density")
    plt.subplots_adjust(bottom=0.15)
    plt.legend(title="Distance Metric", title_fontproperties={'weight':'bold'})
    fig.tight_layout()
    save_path = os.path.join(plot_path, "posterior_median_distribution.png")
    plt.savefig(save_path)