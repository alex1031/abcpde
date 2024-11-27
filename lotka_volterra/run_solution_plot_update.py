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
    dxdt = theta_a*x - x*y
    dydt = theta_b*x*y - y
    return torch.stack((dxdt, dydt), dim=-1)

ic = torch.full((1, 2), 0.5).cuda()
t = torch.linspace(0, 10, 100).cuda()
true_solution = odeint(lambda t, state: dUdt(t, state, 1, 1), ic, t, method='rk4')
true_prey = true_solution[:,0]
true_predator = true_solution[:,1]

x = np.linspace(0, 10, 100)
for n in NOISE:
    fig_no_smoothing, ax_no_smoothing = plt.subplots(1, 2, sharex=True, sharey=True)
    fig_smoothing, ax_smoothing = plt.subplots(1, 2, sharex=True, sharey=True)
    observed_no_smoothing_path = os.path.join(OBSERVED_PATH, f"n{n}_no_smoothing/n{n}_no_smoothing.npy")
    observed_no_smoothing = np.load(observed_no_smoothing_path)

    i = 0
    if n == 0: # We only have non-smoothing for 0 noise
        run_no_smoothing_path = os.path.join(RUN_PATH, f"n{n}_no_smoothing/run1.npy")
        run_no_smoothing = np.load(run_no_smoothing_path)
        for metric in DISTANCE_METRIC:

            low_thresh_no_smoothing = abc_posterior_data(NPARAM, run_no_smoothing, THRESHOLD, metric)
            alpha_no_smoothing_median, beta_no_smoothing_median = np.median(low_thresh_no_smoothing[:,0]), np.median(low_thresh_no_smoothing[:,1])
            no_smoothing_sol = odeint(lambda t, state: dUdt(t, state, alpha_no_smoothing_median, beta_no_smoothing_median), ic, t, method='rk4')
            no_smoothing_prey_sol = np.nan_to_num(np.array(no_smoothing_sol.cpu()[:, :, 0]))
            no_smoothing_predator_sol = np.nan_to_num(np.array(no_smoothing_sol.cpu()[:, :, 1]))

            ax_no_smoothing[i].plot(x, true_prey, label="True Prey")
            ax_no_smoothing[i].plot(x, true_predator, label="True Predator")
            ax_no_smoothing[i].plot(x, no_smoothing_prey_sol, label="Simulated Predator")
            ax_no_smoothing[i].plot(x, no_smoothing_prey_sol, label="Simulated Predator")
            
            ax_no_smoothing[i].set_title(metric)
            i += 1        

        fig_no_smoothing.suptitle(f"$\epsilon\sim N(0, {n}^2)$ smoothing")
        fig_no_smoothing.supxlabel("t")
        fig_no_smoothing.supylabel("Population")
        save_path = os.path.join(PLOT_PATH, f"n{n}_no_smoothing/sim_wd_ed_sol.png")
        fig_no_smoothing.savefig(save_path)

    # else:
    #     observed_no_smoothing_path = os.path.join(OBSERVED_PATH, f"n{n}_no_smoothing/n{n}_no_smoothing.npy")
    #     observed_no_smoothing = np.load(observed_no_smoothing_path)
        
    #     run_smoothing_path = os.path.join(RUN_PATH, f"n{n}_smoothing/run1.npy")
    #     run_no_smoothing_path = os.path.join(RUN_PATH, f"n{n}_no_smoothing/run1.npy")
    #     run_smoothing = np.load(run_smoothing_path)
    #     run_no_smoothing = np.load(run_no_smoothing_path)