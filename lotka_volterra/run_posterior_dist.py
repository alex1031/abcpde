# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from common.abc_posterior import abc_posterior_data

# RUN_PATH = "./lotka_volterra/runs"
# PLOT_PATH = "./lotka_volterra/plots"
# DISTANCES = ["Wasserstein Distance", "Energy Distance", "Maximum Mean Discrepancy", "Cramer-von Mises Distance", "Kullback-Leibler Divergence"]
# NPARAM = 2

# models = os.listdir(RUN_PATH)

# for model in models:
#     fig, ax = plt.subplots(2, 1, figsize=(10, 5))
#     run_model_path = os.path.join(RUN_PATH, model)
#     run = os.path.join(RUN_PATH, model, "run1.npy")
#     distances = np.load(run)

#     for i in range(NPARAM, distances.shape[1]):
#         temp_dist = distances[:,i]
#         threshold_data = abc_posterior_data(NPARAM, distances, 0.001, DISTANCES[i-2])
#         threshold_data = np.where(threshold_data != np.inf, threshold_data, np.nan)
#         alpha, beta = threshold_data[:,0], threshold_data[:,1]
#         print(DISTANCES[i-2], threshold_data.shape)
#         sns.histplot(ax=ax[0], data=alpha, element="step", fill=True, label=DISTANCES[i-2], binwidth=0.1)
#         sns.histplot(ax=ax[1], data=beta, element="step", fill=True, label=DISTANCES[i-2], binwidth=0.1)
#         ax[0].axvline(1, c='r')
#         ax[1].axvline(1, c='r')
#         ax[0].set_xlabel(r"$\alpha$")
#         ax[1].set_xlabel(r"$\beta$")
#         ax[0].set_ylabel("")
#         ax[1].set_ylabel("")
    
#     model_name = model.split("_")
#     model_noise = model_name[0][1:len(model_name[0])]
#     if len(model_name) == 3:
#         model_smoothing = "No Smoothing"
#     else:
#         model_smoothing = "With Smoothing"
    
#     if model_noise == "linear":
#         model_noise = "t"
#     plot_title = f"Posterior Distribution for $\epsilon\sim N(0, {model_noise}^2)$ {model_smoothing}"
#     fig.suptitle(plot_title)
#     fig.supylabel("Density")
#     fig.tight_layout()
#     plt.legend()

#     model_path = os.path.join(PLOT_PATH, model)
#     save_path = os.path.join(model_path, "run_posterior_distribution.png")
#     plt.savefig(save_path)

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
n_models = len(models)

# Determine the number of rows and columns (max columns = 3)
max_columns = 3
n_rows = int(np.ceil(n_models / max_columns)) * 2  # Two rows (alpha and beta) per model group

# Set up the figure
fig, axes = plt.subplots(n_rows, max_columns, figsize=(20, 2 * n_rows), sharex=True, sharey=False)

# Flatten the axes array for easier indexing
axes = axes.reshape(n_rows, max_columns)

# Iterate through models and plot
for idx, model in enumerate(models):
    col = idx % max_columns
    row = (idx // max_columns) * 2  # Two rows per model (alpha, beta)
    
    ax_alpha = axes[row, col]       # Top row for alpha
    ax_beta = axes[row + 1, col]    # Bottom row for beta

    run_model_path = os.path.join(RUN_PATH, model)
    run = os.path.join(RUN_PATH, model, "run1.npy")
    distances = np.load(run)

    for i in range(NPARAM, distances.shape[1]):
        temp_dist = distances[:, i]
        threshold_data = abc_posterior_data(NPARAM, distances, 0.001, DISTANCES[i - 2])
        threshold_data = np.where(threshold_data != np.inf, threshold_data, np.nan)
        alpha, beta = threshold_data[:, 0], threshold_data[:, 1]
        
        # Plot alpha and beta
        sns.histplot(ax=ax_alpha, data=alpha, element="step", fill=True, label=DISTANCES[i - 2], binwidth=0.1)
        sns.histplot(ax=ax_beta, data=beta, element="step", fill=True, label=DISTANCES[i - 2], binwidth=0.1)
        
        # Add vertical lines
        ax_alpha.axvline(1, c='r')
        ax_beta.axvline(1, c='r')
        
    # Set titles and labels for individual plots
    model_name = model.split("_")
    model_noise = model_name[0][1:len(model_name[0])]
    if len(model_name) == 3:
        model_smoothing = "No Smoothing"
    else:
        model_smoothing = "With Smoothing"
    
    if model_noise == "linear":
        model_noise = "t"
    # plot_title = f"Posterior Distribution for $\epsilon\sim N(0, {model_noise}^2)$ {model_smoothing}"
    plot_title = f"$\epsilon\sim N(0, {model_noise}^2)$ {model_smoothing}"
    ax_alpha.set_title(plot_title)
    ax_alpha.set_ylabel("Density ($\\alpha$)")
    ax_beta.set_ylabel("Density ($\\beta$)")
    ax_beta.set_xlabel("Parameter Value")

# Remove unused subplots
for r in range(n_rows):
    for c in range(max_columns):
        if r * max_columns + c >= n_models * 2:  # Beyond total models
            fig.delaxes(axes[r, c])

# Add a single legend for the entire figure
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.85, 0.5), title="Distances")

# Adjust layout to leave space for the legend
fig.suptitle("Posterior Distribution of $\\alpha$ and $\\beta$ for All Models", fontsize=16)
fig.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on the right for the legend
save_path = os.path.join(PLOT_PATH, "combined_posterior_distribution.png")
plt.savefig(save_path)
plt.close(fig)

