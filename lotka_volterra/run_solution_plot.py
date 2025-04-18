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

# Centralized style dictionary
styles = {
    "observed_prey": {"color": "navy", "linestyle": "-", "linewidth": 2.0, "alpha": 0.9},
    "observed_pred": {"color": "firebrick", "linestyle": "-", "linewidth": 2.0, "alpha": 0.9},
    "sim_prey": {"color": "darkorange", "linestyle": "-", "linewidth": 2.0, "alpha": 0.9},
    "sim_pred": {"color": "forestgreen", "linestyle": "-", "linewidth": 2.0, "alpha": 0.9},
    "upper_prey": {"color": "darkorange", "linestyle": "--", "linewidth": 1.5, "alpha": 0.5},
    "lower_prey": {"color": "darkorange", "linestyle": ":", "linewidth": 1.5, "alpha": 0.5},
    "upper_pred": {"color": "forestgreen", "linestyle": "--", "linewidth": 1.5, "alpha": 0.5},
    "lower_pred": {"color": "forestgreen", "linestyle": ":", "linewidth": 1.5, "alpha": 0.5}
}

def dUdt(U, t, a, b):
    x, y = U
    return [a*x - x*y, b*x*y - y]

if __name__ == "__main__":
    t = np.linspace(0, 10, 100)
    true_curve = np.load(os.path.join(OBSERVED_PATH, "n0_no_smoothing/n0_no_smoothing.npy"))
    true_prey = true_curve[:, 0]
    true_pred = true_curve[:, 1]

    for noise in NOISE:
        smoothing_options = ["no_smoothing"] if noise == 0 else SMOOTHING

        for smoothing in smoothing_options:
            fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(14, 10))
            ax = axes.flatten()

            for i, metric in enumerate(METRICS):
                run_dir = f"n{noise}_{smoothing}"
                run_path = os.path.join(RUN_PATH, run_dir, "run1.npy")

                if not os.path.exists(run_path):
                    print(f"Warning: {run_path} does not exist. Skipping metric {metric}.")
                    continue

                run = np.load(run_path)
                posterior = abc_posterior(NPARAM, run, THRESHOLD, metric)

                a_mean, a_lb, a_ub = posterior[0][0], posterior[0][3], posterior[0][4]
                b_mean, b_lb, b_ub = posterior[1][0], posterior[1][3], posterior[1][4]

                S0 = (0.5, 0.5)
                sol = odeint(dUdt, S0, t, args=(a_mean, b_mean))
                sol_upper = odeint(dUdt, S0, t, args=(a_lb, b_lb))
                sol_lower = odeint(dUdt, S0, t, args=(a_ub, b_ub))

                # Plot Observed
                ax[i].plot(t, true_prey, label="Observed Prey", **styles["observed_prey"])
                ax[i].plot(t, true_pred, label="Observed Predator", **styles["observed_pred"])

                # Plot Simulated Mean
                ax[i].plot(t, sol[:, 0], label="Simulated Prey", **styles["sim_prey"])
                ax[i].plot(t, sol[:, 1], label="Simulated Predator", **styles["sim_pred"])

                # Plot Upper and Lower Bounds
                ax[i].plot(t, sol_upper[:, 0], label="Upper Bound Prey", **styles["upper_prey"])
                ax[i].plot(t, sol_upper[:, 1], label="Upper Bound Predator", **styles["upper_pred"])
                ax[i].plot(t, sol_lower[:, 0], label="Lower Bound Prey", **styles["lower_prey"])
                ax[i].plot(t, sol_lower[:, 1], label="Lower Bound Predator", **styles["lower_pred"])

                ax[i].set_title(metric, fontsize=10)
                ax[i].set_ylim(-7.5, 14)

                if i == 0:
                    ax[i].legend(fontsize=8)

            # Remove unused 6th subplot
            ax[-1].axis("off")

            if noise == "linear":
                noise_desc = "$\\varepsilon\\sim N(0, t^2)$"
            else:
                noise_desc = f"$\\varepsilon\\sim N(0, {noise}^2)$"

            fig.suptitle(f"{noise_desc} - {smoothing.replace('_', ' ').title()}", fontsize=14)
            fig.supxlabel("Time (t)", fontsize=12)
            fig.supylabel("Population", fontsize=12)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            plot_dir = os.path.join(PLOT_PATH, f"n{noise}_{smoothing}")
            os.makedirs(plot_dir, exist_ok=True)
            fig.savefig(os.path.join(plot_dir, "run_solution_plot.png"))
            plt.close()
