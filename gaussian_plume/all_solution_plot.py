import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

METRICS = ["Cramer-von Mises Distance", "Frechet Distance", "Hausdorff Distance", "Wasserstein Distance"]
MODELS = ["no_noise", "linear_noise", "0.025_noise", "0.05_noise", "0.075_noise"]
DF_PATH = "./gaussian_plume/dataframe/all_summary_statistics.csv"
SAVE_PATH = "./gaussian_plume/plots"
OBSERVED_PATH = "./gaussian_plume/observed_data/no_noise/no_noise.npy"
TEND = 0.1
DT = 0.001

def generate_solution(nx, ny, Lx, Ly, cx, cy, s):
    dx, dy = Lx/(nx-1), Ly/(ny-1)
    dt = DT
    tend = TEND
    t = 0

    cfl_x, cfl_y = cx * dt/dx, cy * dt/dy
    diff_x, diff_y = s * dt/dx**2, s * dt/dy**2

    u = np.zeros((nx+2, ny+2))
    sol = []
    source_x, source_y = nx // 2, ny // 2
    u[source_x, source_y] = 1.0 # Cocentration starts from the central peak
    
    while t < tend:
        unew = u.copy()
        sol.append(u[1:-1, 1:-1])

         # Advection (Upwind Scheme)
        unew[1:-1, 1:-1] -= cfl_x * (u[1:-1, 1:-1] - u[1:-1, :-2])
        unew[1:-1, 1:-1] -= cfl_y * (u[1:-1, 1:-1] - u[:-2, 1:-1])
    
        # Diffusion (Central Differencing)
        unew[1:-1, 1:-1] += diff_x * (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2])
        unew[1:-1, 1:-1] += diff_y * (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1])

        u = unew
        t += dt

    '''
    We transpose the axis the solution. Such that:
    - Axis 0: x
    - Axis 1: y
    - Axis 2: time
    Interpretation: For each x-grid, we have the concentration of each y-grid over time.
    Essentially, the (ny, 1200) array represents the concentration at each y over the time.
    So we have an array of size 1200 for each y. (A curve)
    '''
    sol = np.transpose(sol, (1, 2, 0))
    return np.array(sol)

if __name__ == "__main__":
    Nx, Ny= 50, 50  # Grid points
    Lx, Ly = 1.0, 1.0  # Domain size in meters
    x, y = np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny)  # Centered at (0,0)
    cx, cy = 1, 1
    s = 5e-5

    df = pd.read_csv(DF_PATH)
    df_01 = df[df["quantile"] == "0.1%"]
    cx = df_01[(df_01["param"] == "cx") & (df_01["summary_statistic"] == "Median")].reset_index(drop=True)
    cy = df_01[(df_01["param"] == "cy") & (df_01["summary_statistic"] == "Median")].reset_index(drop=True)
    s = df_01[(df_01["param"] == "s") & (df_01["summary_statistic"] == "Median")].reset_index(drop=True)

    for model in MODELS:
        # To determine the global color range
        # Load observed data
        observed = np.load(OBSERVED_PATH)
        observed_mean = np.mean(observed, axis=2)

        # Compute global min and max to fix color range
        all_means = [observed_mean]

        for i, metric in enumerate(METRICS):
            cx_metric = list(cx[(cx["model"] == model)][metric])[0]
            cy_metric = list(cy[(cy["model"] == model)][metric])[0]
            s_metric = list(s[(s["model"] == model)][metric])[0]

            sol = generate_solution(Nx, Ny, Lx, Ly, cx_metric, cy_metric, s_metric)
            sol_mean = np.mean(sol[23:31, 23:31, :], axis=2)
            all_means.append(sol_mean)

        # Stack all mean fields and find min/max
        all_means_stack = np.stack(all_means)
        vmin, vmax = all_means_stack.min(), all_means_stack.max()

        # Create the figure and subplots
        fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(15, 10))
        axes = axes.flatten()
        axes[-1].axis("off")
        save_path = os.path.join(SAVE_PATH, model)

        # Initialize a variable to store the mappable for colorbar
        mappable = None
        X, Y = np.meshgrid(x[23:31], y[23:31])
        pcm = axes[0].pcolor(X, Y, observed_mean, cmap="jet", shading="auto", vmin=vmin, vmax=vmax)
        axes[0].set_xlabel("x (m)")
        axes[0].set_ylabel("y (m)")
        axes[0].set_title("Observed", fontsize=12)
        mappable = pcm

        for i, metric in enumerate(METRICS):
            cx_metric = list(cx[(cx["model"] == model)][metric])[0]
            cy_metric = list(cy[(cy["model"] == model)][metric])[0]
            s_metric = list(s[(s["model"] == model)][metric])[0]

            sol = generate_solution(Nx, Ny, Lx, Ly, cx_metric, cy_metric, s_metric)
            
            # Store the output of pcolor to use for colorbar
            pcm = axes[i+1].pcolor(X, Y, np.mean(sol[23:31, 23:31, :], axis=2), cmap='jet', shading='auto', vmin=vmin, vmax=vmax)

            axes[i+1].set_xlabel("x (m)")
            axes[i+1].set_ylabel("y (m)")
            if metric == "Cramer-von Mises Distance":
                subtitle = "CvMD"
            elif metric == "Frechet Distance":
                subtitle = "Frechet"
            elif metric == "Hausdorff Distance":
                subtitle = "Hausdorff"
            elif metric == "Wasserstein Distance":
                subtitle = "Wasserstein"
            axes[i+1].set_title(subtitle, fontsize=12)


        # Construct the title
        model_name = model.split("_")
        if model_name[0] == "no":
            title = "Time-Average Concentration Field for $\\varepsilon\\sim N(0, 0)$"
        elif model_name[0] == "linear":
            title = "Time-Average Concentration Field for $\\varepsilon\\sim N(0, t^2)$"
        else:
            title = f"Time-Average Concentration Field for $\\varepsilon\\sim N(0, {model_name[0]}^2)$"
        fig.suptitle(title)

        # Add a single colorbar to the figure
        cbar = fig.colorbar(mappable, ax=axes, orientation='vertical', fraction=0.02, pad=0.05)
        cbar.set_label("Concentration per unit volume (kg/$m^3$)")

        # Save the figure
        image_save = os.path.join(save_path, "solution.png")
        fig.savefig(image_save, bbox_inches='tight')