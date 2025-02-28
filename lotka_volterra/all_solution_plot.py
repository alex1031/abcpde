import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

DATAFRAME_PATH = "./lotka_volterra/dataframe/all_summary_statistics.csv"
OBSERVED_PATH = "./lotka_volterra/observed_data"
PLOT_PATH = "./lotka_volterra/plots"
DISTANCE_METRICS = ["Cramer-von Mises Distance", "Energy Distance", "Kullback-Leibler Divergence", "Maximum Mean Discrepancy", "Wasserstein Distance"]
NOISE = [0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 0, "linear", "linear"]

def dUdt(U, t, a, b):
    x, y = U

    return [a*x - x*y, b*x*y - y]

t = np.linspace(0, 10, 100)
true_curve = np.load(os.path.join(OBSERVED_PATH, "n0_no_smoothing/n0_no_smoothing.npy"))
summary = pd.read_csv(DATAFRAME_PATH)
relevantss = summary[(summary['quantile'] == "0.1%") & (summary["summary_statistic"] == "Median")].reset_index(drop=True) # Want posterior median of 0.1% quantile

j = 0
for model in os.listdir(OBSERVED_PATH):
    # Identify relevant rows in dataframe
    current = relevantss[relevantss["model"] == model].reset_index(drop=True)
    # Load observed data
    observed = np.load(os.path.join(OBSERVED_PATH, model+"/"+model+".npy"))

    # Plot component
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 10))
    axes[1, 2].axis("off")
    ax = axes.flatten()

    for i, metric in enumerate(DISTANCE_METRICS):
        params = current[metric]
        alpha = params[0]
        beta = params[1]
        S0 = (0.5, 0.5) # Initial Conditions
        sol = odeint(dUdt, S0, t, args=(alpha, beta)) # Solving the ODEs
        ax[i].scatter(t, observed[:,0], s=5, c="blue", label="Observed Prey", alpha=0.4)
        ax[i].scatter(t, observed[:,1], s=5, c="red", label="Observed Predator", alpha=0.4)
        ax[i].plot(t, true_curve[:,0], c="blue", label="True Prey", alpha=0.7)
        ax[i].plot(t, true_curve[:,1], c="red", label="True Predator", alpha=0.7)
        ax[i].set_title(metric, fontsize=10)
        ax[i].plot(t, sol[:,0], c="blue", linestyle = "--", label="Simulated Prey")
        ax[i].plot(t, sol[:,1], c="red", linestyle = "--", label="Simulated Predator")
        ax[i].set_ylim(-7.5, 14)
        

    if j == 6:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.7, 0.3))

    if NOISE[j] == "linear":
        title = "Solution of $\\varepsilon\\sim N(0, t^2)$"
    else:
        title = f"Solution of $\\varepsilon\\sim N(0, {NOISE[j]}^2)$"
    
    if "no_smoothing" in model:
        title += " No Smoothing"
    else:
        title += " Smoothed"
    fig.suptitle(title)
    fig.supxlabel("Time (t)", fontsize=12)
    fig.supylabel("Population", fontsize=12)
    fig.tight_layout()

    j += 1

    fig.savefig(os.path.join(PLOT_PATH, model + "/all_solution_plot.png"))