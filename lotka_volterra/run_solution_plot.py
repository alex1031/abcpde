# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from torchdiffeq_mod._impl import odeint
# from common.abc_posterior import abc_posterior_data

# OBSERVED_PATH = "./lotka_volterra/observed_data"
# RUN_PATH = "./lotka_volterra/runs"
# PLOT_PATH = "./lotka_volterra/plots"
# NOISE = [0, 0.25, 0.5, 0.75, "linear"]
# DISTANCE_METRIC = ["Wasserstein Distance", "Energy Distance"]
# NPARAM = 2
# THRESHOLD = 0.001

# def dUdt(t, state, theta_a, theta_b):
#     x, y = state[..., 0], state[..., 1]
#     dxdt = theta_a * x - x * y
#     dydt = theta_b * x * y - y
#     return torch.stack((dxdt, dydt), dim=-1)

# def solve_ode(alpha, beta):
#     return odeint(lambda t, state: dUdt(t, state, alpha, beta), ic, t, method='rk4')

# def extract_population(solution):
#     return np.nan_to_num(np.array(solution.cpu()[:, :, 0])), np.nan_to_num(np.array(solution.cpu()[:, :, 1]))

# def load_data(noise, smoothing=True):
#     smoothing_path = "smoothing" if smoothing else "no_smoothing"
#     run_path = os.path.join(RUN_PATH, f"n{noise}_{smoothing_path}/run1.npy")
#     return np.load(run_path)

# def plot_results(ax, x, observed, true_prey, true_predator, prey, predator, metric, scatter=True):
#     if scatter:
#         ax.scatter(x, observed[:, 0], color='blue', s=10, label='Observed Prey')
#         ax.scatter(x, observed[:, 1], color='red', s=10, label='Observed Predator')
#     ax.plot(x, true_prey, color='blue', label='True Prey')
#     ax.plot(x, true_predator, color='red', label='True Predator')
#     ax.plot(x, prey, color='blue', linestyle='--', label='Simulated Prey')
#     ax.plot(x, predator, color='red', linestyle='--', label='Simulated Predator')
#     ax.set_title(metric)

# ic = torch.full((1, 2), 0.5).cuda()
# t = torch.linspace(0, 10, 100).cuda()
# true_solution = solve_ode(1, 1)
# true_prey, true_predator = extract_population(true_solution)
# x = np.linspace(0, 10, 100)

# for n in NOISE:
#     observed_path = os.path.join(OBSERVED_PATH, f"n{n}_no_smoothing/n{n}_no_smoothing.npy")
#     observed = np.load(observed_path)

#     fig_no_smoothing, ax_no_smoothing = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
#     fig_smoothing, ax_smoothing = None, None

#     run_no_smoothing = load_data(n, smoothing=False)
#     run_smoothing = None if n == 0 else load_data(n, smoothing=True)

#     if n != 0:
#         fig_smoothing, ax_smoothing = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))

#     for i, metric in enumerate(DISTANCE_METRIC):
#         low_thresh_no_smoothing = abc_posterior_data(NPARAM, run_no_smoothing, THRESHOLD, metric)
#         alpha_no, beta_no = np.median(low_thresh_no_smoothing[:, 0]), np.median(low_thresh_no_smoothing[:, 1])
#         prey_no, predator_no = extract_population(solve_ode(alpha_no, beta_no))

#         plot_results(ax_no_smoothing[i], x, observed, true_prey, true_predator, prey_no, predator_no, metric, scatter=(n != 0))

#         if n != 0 and run_smoothing is not None:
#             low_thresh_smoothing = abc_posterior_data(NPARAM, run_smoothing, THRESHOLD, metric)
#             alpha_s, beta_s = np.median(low_thresh_smoothing[:, 0]), np.median(low_thresh_smoothing[:, 1])
#             prey_s, predator_s = extract_population(solve_ode(alpha_s, beta_s))
#             plot_results(ax_smoothing[i], x, observed, true_prey, true_predator, prey_s, predator_s, metric, scatter=True)

#     # Single Legend for No Smoothing
#     fig_no_smoothing.legend(["Observed Prey w/Noise", "Observed Predator w/Noise", "Observed Prey", "Observed Predator", "Simulated Prey", "Simulated Prey"])
#     if n == "linear":
#         fig_no_smoothing.suptitle(f"$\\epsilon\\sim N(0, t^2)$ Without Smoothing")
#     else:
#         fig_no_smoothing.suptitle(f"$\\epsilon\\sim N(0, {n}^2)$ Without Smoothing")
#     fig_no_smoothing.supxlabel("t")
#     fig_no_smoothing.supylabel("Population")
#     fig_no_smoothing.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for legend space
#     fig_no_smoothing.savefig(os.path.join(PLOT_PATH, f"n{n}_no_smoothing/sim_wd_ed_sol.png"))

#     # Single Legend for Smoothing if applicable
#     if n != 0 and fig_smoothing:
#         fig_smoothing.legend(["Observed Prey w/Noise", "Observed Predator w/Noise", "Observed Prey", "Observed Predator", "Simulated Prey", "Simulated Prey"])
#         if n == "linear":
#             fig_smoothing.suptitle(f"$\\epsilon\\sim N(0, t^2)$ With Smoothing")
#         else:
#             fig_smoothing.suptitle(f"$\\epsilon\\sim N(0, {n}^2)$ With Smoothing")
#         fig_smoothing.supxlabel("t")
#         fig_smoothing.supylabel("Population")
#         fig_smoothing.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for legend space
#         fig_smoothing.savefig(os.path.join(PLOT_PATH, f"n{n}_smoothing/sim_wd_ed_sol.png"))

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchdiffeq_mod._impl import odeint
from common.abc_posterior import abc_posterior_data

OBSERVED_PATH = "./lotka_volterra/observed_data"
RUN_PATH = "./lotka_volterra/runs"
PLOT_PATH = "./lotka_volterra/plots"
NOISE = [0, 0.25, 0.5, 0.75, "linear"]
DISTANCE_METRIC = ["Wasserstein Distance", "Energy Distance"]
NPARAM = 2
THRESHOLD = 0.001

def dUdt(t, state, theta_a, theta_b):
    x, y = state[..., 0], state[..., 1]
    dxdt = theta_a * x - x * y
    dydt = theta_b * x * y - y
    return torch.stack((dxdt, dydt), dim=-1)

def solve_ode(alpha, beta):
    return odeint(lambda t, state: dUdt(t, state, alpha, beta), ic, t, method='rk4')

def extract_population(solution):
    return np.nan_to_num(np.array(solution.cpu()[:, :, 0])), np.nan_to_num(np.array(solution.cpu()[:, :, 1]))

def load_data(noise, smoothing=True):
    smoothing_path = "smoothing" if smoothing else "no_smoothing"
    run_path = os.path.join(RUN_PATH, f"n{noise}_{smoothing_path}/run1.npy")
    return np.load(run_path)

def plot_results(ax, x, observed, true_prey, true_predator, prey, predator, metric, scatter=True):
    if scatter:
        ax.scatter(x, observed[:, 0], color='blue', s=10, label='Observed Prey')
        ax.scatter(x, observed[:, 1], color='red', s=10, label='Observed Predator')
    ax.plot(x, true_prey, color='blue', label="True Prey")
    ax.plot(x, true_predator, color='red', label="True Predator")
    ax.plot(x, prey, color='blue', linestyle='--', label='Simulated Prey')
    ax.plot(x, predator, color='red', linestyle='--', label='Simulated Predator')
    ax.set_title(metric)

# Prepare initial conditions
ic = torch.full((1, 2), 0.5).cuda()
t = torch.linspace(0, 10, 100).cuda()
true_solution = solve_ode(1, 1)
true_prey, true_predator = extract_population(true_solution)
x = np.linspace(0, 10, 100)

# Create grids for "no smoothing" and "with smoothing"
fig, axes = plt.subplots(2, len(NOISE), figsize=(40, 10), sharex=True, sharey=True)
axes_no_smoothing = axes[0]  # First row for "no smoothing"
axes_with_smoothing = axes[1]  # Second row for "with smoothing"

for idx, n in enumerate(NOISE):
    observed_path = os.path.join(OBSERVED_PATH, f"n{n}_no_smoothing/n{n}_no_smoothing.npy")
    observed = np.load(observed_path)

    run_no_smoothing = load_data(n, smoothing=False)
    if n != 0:
        run_smoothing = load_data(n, smoothing=True)
    else:
        run_smoothing = None

    # Create subplots within each grid cell
    for smoothing, ax_main in zip(["No Smoothing", "With Smoothing"], [axes_no_smoothing[idx], axes_with_smoothing[idx]]):
        if smoothing == "With Smoothing" and n == 0:
            continue  # Skip "with smoothing" for noise level 0

        run = run_no_smoothing if smoothing == "No Smoothing" else run_smoothing
        for i, metric in enumerate(DISTANCE_METRIC):
            low_thresh = abc_posterior_data(NPARAM, run, THRESHOLD, metric)
            alpha, beta = np.median(low_thresh[:, 0]), np.median(low_thresh[:, 1])
            prey, predator = extract_population(solve_ode(alpha, beta))

            ax = ax_main.inset_axes([0.05 + i * 0.5, 0.1, 0.4, 0.8])  # Create insets for two subplots
            plot_results(ax, x, observed, true_prey, true_predator, prey, predator, f"{metric}", scatter=(n != 0))
            ax_main.axis("off")

            if n == "linear":
                ax_main.set_title(f"$\\epsilon\\sim N(0, t^2)$ {smoothing}")
            else:
                ax_main.set_title(f"$\\epsilon\\sim N(0, {n}^2)$, {smoothing}")


# Add legends
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.76, 1), ncol=2, title="$\\bf{Legend}$")

# Add suptitles
fig.suptitle("Combined Posterior Distributions for Different Noise Levels", fontsize=20)
fig.supxlabel("Time (t)", fontsize=15)
fig.supylabel("Population", fontsize=15)

axes_no_smoothing[0].axis("off")
axes_with_smoothing[0].axis("off")
# Adjust layout and save the figure
fig.tight_layout(rect=[0.01, 0, 1, 0.95])  # Leave space for the legend
fig.savefig(os.path.join(PLOT_PATH, "sim_sol_wd_ed_combined.png"))
plt.close(fig)
