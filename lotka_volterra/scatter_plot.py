import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

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
        ax[-1].axis('off')
        
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
        fig.supxlabel(r'$\alpha$')
        fig.supylabel(r'$\beta$', rotation=0)
        fig.legend(QUANTILE, loc="lower right", title="Threshold", title_fontproperties={'weight':'bold'}, bbox_to_anchor=(0.85, 0.2))
        plot_model_path = os.path.join(PLOT_PATH, model)
        if not os.path.isdir(plot_model_path):
            os.mkdir(plot_model_path)
        
        fig_path = os.path.join(plot_model_path, "scatter_plot.png")
        fig.savefig(fig_path)
