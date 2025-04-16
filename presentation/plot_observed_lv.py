import os
import numpy as np
import maptplotlib.pyplot as plt

OBSERVED_PATH = "./lotka_volterra/observed_data"
SMOOTHING = ["smoothing", "no_smoothing"]
MODELS = ["n0", "n0.25", "n0.5", "n0.75", "nlinear"]

if __name__ == "__main__":

    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 10))
    axes = ax.flatten()
    axes[-1].axis("off")
    true_val = np.load("./lotka_volterra")

    for i, model in enumerate(MODELS):
        # Need to always plot the true solution (Can probably exclude n0 in models)
        if model != "n0":
            for smoothing in SMOOTHING:
                obs_path = os.path.join(OBSERVED_PATH, model+"_"+smoothing+f"{model}_{smoothing}.npy")
                obs = np.load(obs_path)
                # Plot scatter if no smoothing
                if smoothing == "no_smoothing":
                    axes[i].scatter(obs[:,0], color="blue", s=10, label="Observed Prey")
                    axes[i].scatter(obs[:,1], color="red", s=10, label="Observed Predator")\
                # Plot dotted line for smoothing
                else:
                    axes[i].plot()
