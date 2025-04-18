import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from common.abc_posterior import abc_posterior

# Paths and constants
OBSERVED_PATH = "./lotka_volterra/observed_data"
RUN_PATH = "./lotka_volterra/runs"
PLOT_PATH = "./lotka_volterra/plots"
METRICS = ["Cramer-von Mises Distance", "Energy Distance", "Kullback-Leibler Divergence", "Maximum Mean Discrepancy", "Wasserstein Distance"]
NOISE = [0, 0.25, 0.5, 0.75, "linear"]
SMOOTHING = ["no_smoothing", "smoothing"]
NPARAM = 2
THRESHOLD = 0.001

# Styles dictionary
styles = {
    "true_prey": {"color": "navy", "linestyle": "-", "linewidth": 2.0, "alpha": 0.9},
    "true_pred": {"color": "firebrick", "linestyle": "-", "linewidth": 2.0, "alpha": 0.9},
    "sim_prey": {"color": "darkorange", "linestyle": "-", "linewidth": 2.0, "alpha": 0.9},
    "sim_pred": {"color": "forestgreen", "linestyle": "-", "linewidth": 2.0, "alpha": 0.9},
    "upper_prey": {"color": "darkorange", "linestyle": "--", "linewidth": 1.5, "alpha": 0.5},
    "lower_prey": {"color": "darkorange", "linestyle": ":", "linewidth": 1.5, "alpha": 0.5},
    "upper_pred": {"color": "forestgreen", "linestyle": "--", "linewidth": 1.5, "alpha": 0.5},
    "lower_pred": {"color": "forestgreen", "linestyle": ":", "linewidth": 1.5, "alpha": 0.5},
}

# Lotka-Volterra model
def dUdt(U, t, a, b):
    x, y = U
    return [a * x - x * y, b * x * y - y]

# Main script
if __name__ == "__main__":
    t = np.linspace(0, 10, 100)
    true_curve = np.load(os.path.join(OBSERVED_PATH, "n0_no_smoothing/n0_no_smoothing.npy"))
    true_prey = true_curve[:, 0]
    true_pred = true_curve[:, 1]

    for noise in NOISE:
        smoothing_options = ["no_smoothing"] if noise == 0 else SMOOTHING

        for smoothing in smoothing_options:
            for i, metric in enumerate(METRICS):
                fig, ax = plt.subplots(figsize=(8, 6))

                run_dir = f"n{noise}_{smoothing}"
                run_path = os.path.join(RUN_PATH, run_dir, "run1.npy")

                if not os.path.exists(run_path):
                    print(f"Warning: {run_path} does not exist. Skipping.")
                    continue

                run = np.load(run_path)
                posterior = abc_posterior(NPARAM, run, THRESHOLD, metric)

                a_mean, a_lb, a_ub = posterior[0][0], posterior[0][3], posterior[0][4]
                b_mean, b_lb, b_ub = posterior[1][0], posterior[1][3], posterior[1][4]

                S0 = (0.5, 0.5)
                sol = odeint(dUdt, S0, t, args=(a_mean, b_mean))
                sol_upper = odeint(dUdt, S0, t, args=(a_lb, b_lb))
                sol_lower = odeint(dUdt, S0, t, args=(a_ub, b_ub))

                # Plotting observed
                ax.plot(t, true_prey, label="Observed Prey", **styles["true_prey"])
                ax.plot(t, true_pred, label="Observed Predator", **styles["true_pred"])

                # Plotting simulated mean
                ax.plot(t, sol[:, 0], label="Simulated Prey", **styles["sim_prey"])
                ax.plot(t, sol[:, 1], label="Simulated Predator", **styles["sim_pred"])

                # Plotting bounds
                ax.plot(t, sol_upper[:, 0], label="Upper Bound Prey", **styles["upper_prey"])
                ax.plot(t, sol_upper[:, 1], label="Upper Bound Predator", **styles["upper_pred"])
                ax.plot(t, sol_lower[:, 0], label="Lower Bound Prey", **styles["lower_prey"])
                ax.plot(t, sol_lower[:, 1], label="Lower Bound Predator", **styles["lower_pred"])

                # Labels and title
                ax.set_title(metric, fontsize=12)
                ax.set_xlabel("Time (t)")
                ax.set_ylabel("Population")
                ax.legend()

                # Noise descriptor for title
                if noise == "linear":
                    noise_desc = "$\\varepsilon\\sim N(0, t^2)$"
                else:
                    noise_desc = f"$\\varepsilon\\sim N(0, {noise}^2)$"

                fig.suptitle(f"{noise_desc} - {smoothing.replace('_', ' ').title()} - {metric}", fontsize=14)

                # Save plot
                plot_dir = os.path.join(PLOT_PATH, run_dir)
                os.makedirs(plot_dir, exist_ok=True)
                fig.tight_layout()
                fig.savefig(os.path.join(plot_dir, f"{metric.replace(' ', '_')}_solution_plot.png"))
                plt.close()
