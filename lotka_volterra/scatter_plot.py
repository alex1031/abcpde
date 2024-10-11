import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

DISTANCE_METRIC = ["Wasserstein Distance"]
RESULT_PATH = "./lotka_volterra/results"
PLOT_PATH = "./lotka_volterra/plots"
QUANTILE = ["0.1%", "1%", "5%"]
EDGECOLORS = ["r", "g", "y"]

if __name__ == "__main__":
    
    models = os.listdir(RESULT_PATH)
    metric_path = os.path.join(RESULT_PATH, models[0])
    metrics = os.listdir(metric_path)

    for model in models:
        fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
        ax = ax.flatten()
        i = 0
        for idx, metric in enumerate(metrics):
            metric_path_plot = os.path.join(RESULT_PATH, model)
            posterior_path = os.path.join(metric_path_plot, metric)
            posteriors = os.listdir(posterior_path)
            for posterior in posteriors:
                path = os.path.join(posterior_path, posterior)
                p = np.load(path)
                ax[idx].scatter(p[:,:,1][:,0],  p[:,:,1][:,1], facecolors="none", edgecolor=EDGECOLORS[i%3])
                i += 1
            ax[idx].set_title(metric, fontsize=6)
            ax[idx].axvline(1)
            ax[idx].axhline(1)
        
        fig.legend(QUANTILE, loc="lower right")
        plot_model_path = os.path.join(PLOT_PATH, model)
        if not os.path.isdir(plot_model_path):
            os.mkdir(plot_model_path)
        
        fig_path = os.path.join(plot_model_path, "scatter_plot.png")
        fig.savefig(fig_path)
