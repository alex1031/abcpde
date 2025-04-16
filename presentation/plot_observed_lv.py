import os
import numpy as np
import matplotlib.pyplot as plt

OBSERVED_PATH = "./lotka_volterra/observed_data"
SAVE_PATH = "./presentation/plots"
SMOOTHING = ["smoothing", "no_smoothing"]
MODELS = ["n0.25", "n0.5", "n0.75", "nlinear"]

if __name__ == "__main__":

    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 10))
    x = np.linspace(0, 10, 100)
    axes = ax.flatten()
    axes[-1].axis("off")
    true_val = np.load("./lotka_volterra/observed_data/n0_no_smoothing/n0_no_smoothing.npy")
    true_prey = true_val[:,0]
    true_pred = true_val[:,1]
    # Plot observed
    axes[0].plot(x, true_prey, color="blue", label="Observed Prey")
    axes[0].plot(x, true_pred, color="red", label="Observed Predator")

    for i, model in enumerate(MODELS):
        # Need to always plot the true solution
        axes[i+1].plot(x, true_prey)
        axes[i+1].plot(x, true_pred)
        for smoothing in SMOOTHING:
            obs_path = os.path.join(OBSERVED_PATH, model+"_"+smoothing+f"/{model}_{smoothing}.npy")
            obs = np.load(obs_path)
            # Plot scatter if no smoothing
            if smoothing == "no_smoothing":
                axes[i+1].scatter(x, obs[:,0], color="blue", s=10, label="Prey w/Noise")
                axes[i+1].scatter(x, obs[:,1], color="red", s=10, label="Predator w/Noise")
            # Plot dotted line for smoothing
            else:
                axes[i+1].plot(x, obs[:,0], color="blue", linestyle='--', label="Prey Spline")
                axes[i+1].plot(x, obs[:,1], color="red", linestyle='--', label="Predator Spline")
    
    save_path = os.path.join(SAVE_PATH, "lv_observed.png")
    fig.savefig(save_path)
