import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchdiffeq_mod._impl import odeint
from common.abc_posterior import abc_posterior_data

OBSERVED_PATH = "./lotka_volterra/observed_data"
RUN_PATH = "./lotka_volterra/runs"
PLOT_PATH = "./lotka_volterra/plots"
NOISE = [0.25, 0.5, 0.75, "linear"]
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

for n in NOISE:
    observed_smoothing_path = os.path.join(OBSERVED_PATH, f"n{n}_smoothing/n{n}_smoothing.npy")
    observed_no_smoothing_path = os.path.join(OBSERVED_PATH, f"n{n}_no_smoothing/n{n}_no_smoothing.npy")
    observed_smoothing = np.load(observed_smoothing_path)
    observed_no_smoothing = np.load(observed_no_smoothing_path)
    
    run_smoothing_path = os.path.join(RUN_PATH, f"n{n}_smoothing/run1.npy")
    run_no_smoothing_path = os.path.join(RUN_PATH, f"n{n}_no_smoothing/run1.npy")
    run_smoothing = np.load(run_smoothing_path)
    run_no_smoothing = np.load(run_no_smoothing_path)

    # Need to draw plot for smoothing and no smoothing separately
    for metric in DISTANCE_METRIC:
        low_thresh_smoothing = abc_posterior_data(NPARAM, run_smoothing, THRESHOLD, metric)
        low_thresh_no_smoothing = abc_posterior_data(NPARAM, run_no_smoothing, THRESHOLD, metric)
        alpha_smoothing_median, beta_smoothing_median = np.median(low_thresh_smoothing[:,0]), np.median(low_thresh_smoothing[:,1])
        # alpha_smoothing_lower, beta_smoothing_lower = np.quantile(low_thresh_smoothing[:,0], 0.025), np.quantile(low_thresh_smoothing[:,1], 0.025)
        # alpha_smoothing_upper, beta_smoothing_upper = np.quantile(low_thresh_smoothing[:,0], 0.975), np.quantile(low_thresh_smoothing[:,1], 0.975)
        alpha_no_smoothing_median, beta_no_smoothing_median = np.median(low_thresh_no_smoothing[:,0]), np.median(low_thresh_no_smoothing[:,1])
        # alpha_no_smoothing_lower, beta_no_smoothing_lower = np.quantile(low_thresh_no_smoothing[:,0], 0.025), np.quantile(low_thresh_no_smoothing[:,1], 0.025)
        # alpha_no_smoothing_upper, beta_no_smoothing_upper = np.quantile(low_thresh_no_smoothing[:,0], 0.975), np.quantile(low_thresh_no_smoothing[:,1], 0.975)

        smoothing_sol = odeint(lambda t, state: dUdt(t, state, alpha_smoothing_median, beta_smoothing_median), ic, t, method='rk4')
        # smoothing_lower_sol = odeint(lambda t, state: dUdt(t, state, alpha_smoothing_lower, beta_smoothing_lower), ic, t, method='rk4')
        # smoothing_upper_sol = odeint(lambda t, state: dUdt(t, state, alpha_smoothing_upper, beta_smoothing_upper), ic, t, method='rk4')
        smoothing_prey_sol = np.nan_to_num(np.array(smoothing_sol.cpu()[:, :, 0]))
        smoothing_predator_sol = np.nan_to_num(np.array(smoothing_sol.cpu()[:, :, 1]))
        # smoothing_prey_lower_sol = np.nan_to_num(np.array(smoothing_lower_sol.cpu()[:, :, 0]))
        # smoothing_predator_lower_sol = np.nan_to_num(np.array(smoothing_lower_sol.cpu()[:, :, 1]))
        # smoothing_prey_upper_sol = np.nan_to_num(np.array(smoothing_upper_sol.cpu()[:, :, 0]))
        # smoothing_predator_upper_sol = np.nan_to_num(np.array(smoothing_upper_sol.cpu()[:, :, 1]))

        no_smoothing_sol = odeint(lambda t, state: dUdt(t, state, alpha_no_smoothing_median, beta_no_smoothing_median), ic, t, method='rk4')
        # no_smoothing_lower_sol = odeint(lambda t, state: dUdt(t, state, alpha_no_smoothing_lower, beta_no_smoothing_lower), ic, t, method='rk4')
        # no_smoothing_upper_sol = odeint(lambda t, state: dUdt(t, state, alpha_no_smoothing_upper, beta_no_smoothing_upper), ic, t, method='rk4')
        no_smoothing_prey_sol = np.nan_to_num(np.array(no_smoothing_sol.cpu()[:, :, 0]))
        no_smoothing_predator_sol = np.nan_to_num(np.array(no_smoothing_sol.cpu()[:, :, 1]))
        # no_smoothing_prey_lower_sol = np.nan_to_num(np.array(no_smoothing_lower_sol.cpu()[:, :, 0]))
        # no_smoothing_predator_lower_sol = np.nan_to_num(np.array(no_smoothing_lower_sol.cpu()[:, :, 1]))
        # no_smoothing_prey_upper_sol = np.nan_to_num(np.array(no_smoothing_upper_sol.cpu()[:, :, 0]))
        # no_smoothing_predator_upper_sol = np.nan_to_num(np.array(no_smoothing_upper_sol.cpu()[:, :, 1]))

        if metric == "Wasserstein Distance":
            save_name = "wd"
        else:
            save_name = "ed"
        x = np.linspace(0, 10, 100)

        # Smoothing Plot
        plt.plot(x, observed_smoothing[:,0], label="Prey")
        plt.plot(x, observed_smoothing[:,1], label="Predator")
        plt.scatter(x, observed_no_smoothing[:,0])
        plt.scatter(x, observed_no_smoothing[:,1])
        plt.plot(x, smoothing_prey_sol, label="Simulated Prey")
        plt.plot(x, smoothing_predator_sol, label="Simulated Predator")
        # plt.plot(x, smoothing_prey_lower_sol, linestyle="--")
        # plt.plot(x, smoothing_predator_lower_sol, linestyle="--")
        plt.legend()
        plt.xlabel("t")
        plt.ylabel("Population")
        if n == "linear":
            plt.title(f"$\epsilon\sim N(0, t^2)$ smoothing {metric}")
        else:
            plt.title(f"$\epsilon\sim N(0, {n}^2)$ smoothing {metric}")
        
        save_path = os.path.join(PLOT_PATH, f"n{n}_smoothing/sim_{save_name}_sol.png")
        plt.savefig(save_path)
        plt.close()

        # No Smoothing
        plt.plot(x, observed_smoothing[:,0], label="Prey")
        plt.plot(x, observed_smoothing[:,1], label="Predator")
        plt.scatter(x, observed_no_smoothing[:,0])
        plt.scatter(x, observed_no_smoothing[:,1])
        plt.plot(x, no_smoothing_prey_sol, label="Simulated Prey")
        plt.plot(x, no_smoothing_predator_sol, label="Simulated Predator")
        # plt.plot(x, smoothing_prey_lower_sol, linestyle="--")
        # plt.plot(x, smoothing_predator_lower_sol, linestyle="--")
        plt.legend()
        plt.xlabel("t")
        plt.ylabel("Population")
        if n == "linear":
            plt.title(f"$\epsilon\sim N(0, t^2)$ no smoothing {metric}")
        else:
            plt.title(f"$\epsilon\sim N(0, {n}^2)$ no smoothing {metric}")
        
        save_path = os.path.join(PLOT_PATH, f"n{n}_no_smoothing/sim_{save_name}_sol.png")
        plt.savefig(save_path)
        plt.close()