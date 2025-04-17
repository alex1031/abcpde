import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from common.abc_posterior import abc_posterior

OBSERVED_PATH = "./lotka_volterra/observed_data"
RUN_PATH = "./lotka_volterra/runs"
PLOT_PATH = "./lotka_volterra/plots"
METRICS = ["Cramer-von Mises Distance", "Energy Distance", "Kullback-Leibler Divergence", "Maximum Mean Discrepancy", "Wasserstein Distance"]
NOISE = [0, 0.25, 0.5, 0.75, "linear"]
SMOOTHING = ["no_smoothing", "smoothing"]
NPARAM = 2
THRESHOLD = 0.001


def dUdt(U, t, a, b):
    x, y = U

    return [a*x - x*y, b*x*y - y]

if __name__ == "__main__":
    t = np.linspace(0, 10, 100)
    true_curve = np.load(os.path.join(OBSERVED_PATH, "n0_no_smoothing/n0_no_smoothing.npy"))
    true_prey = true_curve[:,0]
    true_pred = true_curve[:,1]

    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 10))
    ax = axes.flatten()
    ax[-1].axis("off")

    for noise in NOISE:
        # No noise is a special condition because it only has no smoothing.
        for i, metric in enumerate(METRICS):
            if noise != 0:
                for smoothing in SMOOTHING:
                    run_path = os.path.join(RUN_PATH, f"n{noise}_{smoothing}/run1.npy")
                    run = np.load(run_path)
                    # We want the 0.1% threshold data
                    posterior = abc_posterior(NPARAM, run, THRESHOLD, metric)

                    # a - 0, b - 1
                    a_mean = posterior[0][0]
                    b_mean = posterior[1][0]

                    # To also get upper and lower bound
                    a_lb = posterior[0][3]
                    a_ub = posterior[0][4]
                    b_lb = posterior[1][3]
                    b_ub = posterior[1][4]

                    S0 = (0.5, 0.5) # Initial Conditions
                    sol = odeint(dUdt, S0, t, args=(a_mean, b_mean)) # Solving the ODEs
                    sol_upper = odeint(dUdt, S0, t, args=(a_lb, b_lb))
                    sol_lower = odeint(dUdt, S0, t, args=(a_ub, b_ub))

                    # Plotting everything
                    ax[i].plot(t, true_prey, c="blue", label="True Prey", alpha=0.7)
                    ax[i].plot(t, true_pred, c="red", label="True Predator", alpha=0.7)
                    ax[i].set_title(metric, fontsize=10)
                    ax[i].plot(t, sol[:,0], c="orange", label="Simulated Prey")
                    ax[i].plot(t, sol[:,1], c="green", label="Simulated Predator")
                    ax[i].plot(t, sol_upper[:,0], c="orange", linestyle = "--", label="Simulated Prey w/Upper Bound", alpha=0.7)
                    ax[i].plot(t, sol_upper[:,1], c="green", linestyle = "--", label="Simulated Predator w/Upper Bound", alpha=0.7)
                    ax[i].plot(t, sol_lower[:,0], c="orange", linestyle = ":", label="Simulated Prey w/Lower Bound", alpha=0.7)
                    ax[i].plot(t, sol_lower[:,1], c="green", linestyle = ":", label="Simulated Predator w/Lower Bound", alpha=0.7)
                    ax[i].set_ylim(-7.5, 14)

                    if noise == "linear":
                        title = "Solution of $\\varepsilon\\sim N(0, t^2)$"
                    else:
                        title = f"Solution of $\\varepsilon\\sim N(0, {noise}^2)$"
                    
                    if smoothing == "no_smoothing":
                        title += " No Smoothing"
                    else:
                        title += " Smoothed"
                    fig.suptitle(title)
                    fig.supxlabel("Time (t)", fontsize=12)
                    fig.supylabel("Population", fontsize=12)
                    fig.tight_layout()
                    fig.savefig(os.path.join(PLOT_PATH, f"n{noise}_{smoothing}" + "/run_solution_plot.png"))
            
            # If noise is 0 then we do everything without the smoothing loop
            else:
                run_path = os.path.join(RUN_PATH, f"n{noise}_no_smoothing/run1.npy")
                run = np.load(run_path)
                # We want the 0.1% threshold data
                posterior = abc_posterior(NPARAM, run, THRESHOLD, metric)

                # a - 0, b - 1
                a_mean = posterior[0][0]
                b_mean = posterior[1][0]

                # To also get upper and lower bound
                a_lb = posterior[0][3]
                a_ub = posterior[0][4]
                b_lb = posterior[1][3]
                b_ub = posterior[1][4]

                S0 = (0.5, 0.5) # Initial Conditions
                sol = odeint(dUdt, S0, t, args=(a_mean, b_mean)) # Solving the ODEs
                sol_upper = odeint(dUdt, S0, t, args=(a_lb, b_lb))
                sol_lower = odeint(dUdt, S0, t, args=(a_ub, b_ub))

                # Plotting everything
                ax[i].plot(t, true_prey, c="blue", label="True Prey", alpha=0.7)
                ax[i].plot(t, true_pred, c="red", label="True Predator", alpha=0.7)
                ax[i].set_title(metric, fontsize=10)
                ax[i].plot(t, sol[:,0], c="orange", label="Simulated Prey")
                ax[i].plot(t, sol[:,1], c="green", label="Simulated Predator")
                ax[i].plot(t, sol_upper[:,0], c="orange", linestyle = "--", label="Simulated Prey w/Upper Bound", alpha=0.7)
                ax[i].plot(t, sol_upper[:,1], c="green", linestyle = "--", label="Simulated Predator w/Upper Bound", alpha=0.7)
                ax[i].plot(t, sol_lower[:,0], c="orange", linestyle = ":", label="Simulated Prey w/Lower Bound", alpha=0.7)
                ax[i].plot(t, sol_lower[:,1], c="green", linestyle = ":", label="Simulated Predator w/Lower Bound", alpha=0.7)
                ax[i].set_ylim(-7.5, 14)
                
                title = "Solution of $\\varepsilon\\sim N(0, 0)$ No Smoothing"

                fig.suptitle(title)
                fig.supxlabel("Time (t)", fontsize=12)
                fig.supylabel("Population", fontsize=12)
                fig.tight_layout()
                fig.savefig(os.path.join(PLOT_PATH, f"n{noise}_no_smoothing" + "/run_solution_plot.png"))
