import numpy as np
import matplotlib.pyplot as plt
import os
from common.abc_posterior import gaussian_abc_posterior

RUN_PATH = "./gaussian_plume/runs"
SAVE_PATH = "./gaussian_plume/plots"
OBSERVED_PATH = "./gaussian_plume/observed_data/no_noise/no_noise.npy"
OBSERVED_DIFFUSION_PATH = "./gaussian_plume/observed_data/no_noise_diffusion/no_noise_diffusion.npy"
OBSERVED_ADVECTION_PATH = "./gaussian_plume/observed_data/no_noise_5e-3_advection/no_noise_5e-3_advection.npy"
METRICS = ["Cramer-von Mises Distance", "Frechet Distance", "Hausdorff Distance", "Wasserstein Distance"]
MODELS = ["no_noise", "linear_noise", "0.025_noise", "0.05_noise", "0.075_noise", "no_noise_diffusion", "no_noise_5e-3_advection"]
TEND = 0.1
DT = 0.001
NPARAMS = 3
THRESHOLD = 0.001

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

def generate_solution_2tend(nx, ny, Lx, Ly, cx, cy, s):
    dx, dy = Lx/(nx-1), Ly/(ny-1)
    dt = 0.02
    tend = 2
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
    x, y = np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny)
    # Load observed data
    observed = np.load(OBSERVED_PATH)
    observed_mean = np.mean(observed, axis=2)

    observed_diffusion = np.load(OBSERVED_DIFFUSION_PATH)
    observed_diffusion_mean = np.mean(observed_diffusion, axis=2)

    observed_advection = np.load(OBSERVED_ADVECTION_PATH)
    observed_advection_mean = np.mean(observed_advection, axis=2)

    for model in MODELS:
        # Path to load run data
        run_path = os.path.join(RUN_PATH, model + "/run1.npy")
        run = np.load(run_path)

        # Compute global min and max to fix color range
        # all_means = [observed_mean]

        # for i, metric in enumerate(METRICS):
        #     posterior = gaussian_abc_posterior(NPARAMS, run, THRESHOLD, metric)

        #     # We only want the mean: 0 - cx, 1 - cy, 2 - s
        #     cx_mean = posterior[0][0]
        #     cy_mean = posterior[1][0]
        #     s_mean = posterior[2][0]

        #     sol = generate_solution(Nx, Ny, Lx, Ly, cx_mean, cy_mean, s_mean)
        #     sol_mean = np.mean(sol[23:31, 23:31, :], axis=2)
        #     all_means.append(sol_mean)

        # # Stack all mean fields and find min/max
        # all_means_stack = np.stack(all_means)
        # vmin, vmax = all_means_stack.min(), all_means_stack.max()

        # Plot set up
        fig, ax = fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(15, 10))
        axes = axes.flatten()
        axes[-1].axis("off")
        save_path = os.path.join(SAVE_PATH, model)
        if model == "no_noise_diffusion":
            X, Y = np.meshgrid(x[22:27], y[22:27])
            pcm = axes[0].pcolor(X, Y, observed_diffusion_mean, cmap="jet", shading="auto", vmin=0, vmax=0.25)
        elif model == "no_noise_5e-3_advection":
            X, Y = np.meshgrid(x[22:28], y[22:28])
            pcm = axes[0].pcolor(X, Y, observed_advection_mean, cmap="jet", shading="auto", vmin=0, vmax=0.25)
        else:
            X, Y = np.meshgrid(x[23:31], y[23:31])
            pcm = axes[0].pcolor(X, Y, observed_mean, cmap="jet", shading="auto", vmin=0, vmax=0.25)
        axes[0].set_xlabel("x (m)")
        axes[0].set_ylabel("y (m)")
        axes[0].set_title("Observed", fontsize=12)
        textstr = f"$c_x$ = 0.5\n$c_y$ = 0.5\n$s$ = $5\\times10^{{-5}}$"
        axes[0].text(
                0.9, 0.95, textstr,
                transform=axes[0].transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
        mappable = pcm
        
        # Find threshold data for each distance metric
        for i, metric in enumerate(METRICS):
            posterior = gaussian_abc_posterior(NPARAMS, run, THRESHOLD, metric)

            # We only want the mean: 0 - cx, 1 - cy, 2 - s
            cx_mean = posterior[0][0]
            cy_mean = posterior[1][0]
            s_mean = posterior[2][0]
            
            # Use these parameters to generate solution
            if (model == "no_noise_diffusion") or (model == "no_noise_5e-3_advection"):
                plume_sim = generate_solution_2tend(Nx, Ny, Lx, Ly, cx_mean, cy_mean, s_mean)
            else:
                plume_sim = generate_solution(Nx, Ny, Lx, Ly, cx_mean, cy_mean, s_mean)

            # Plot the plume
            if model == "no_noise_diffusion":
                pcm = axes[i+1].pcolor(X, Y, np.mean(plume_sim[22:27, 22:27, :], axis=2), cmap='jet', shading='auto', vmin=0, vmax=0.25)
            elif model == "no_noise_5e-3_advection":
                pcm = axes[i+1].pcolor(X, Y, np.mean(plume_sim[22:28, 22:28, :], axis=2), cmap='jet', shading='auto', vmin=0, vmax=0.25)
            else:
                pcm = axes[i+1].pcolor(X, Y, np.mean(plume_sim[23:31, 23:31, :], axis=2), cmap='jet', shading='auto', vmin=0, vmax=0.25)
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

            # Add annotation box with mean parameters, we can put in the upper and lower bound values as well (index 3 and 4)
            cx_lb = posterior[0][3]
            cx_ub = posterior[0][4]
            cy_lb = posterior[1][3]
            cy_ub = posterior[1][4]
            s_lb = posterior[2][3]
            s_ub = posterior[2][4]
            textstr = f"$c_x$={cx_mean:.3f} CI=({cx_lb:.3f}, {cx_ub:.3f})\n$c_y$={cy_mean:.3f} CI=({cy_lb:.3f}, {cy_ub:.3f}) \n$s$=${s_mean*1e5:.1f}\\times10^{{-5}}$ CI=(${s_lb*1e5:.1f}\\times10^{{-5}}$, ${s_ub*1e5:.1f}\\times10^{{-5}}$)"
        
            axes[i+1].text(
                0.1, 0.98, textstr,
                transform=axes[i+1].transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5)
            )

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
        image_save = os.path.join(save_path, f"{THRESHOLD}run_solution.png")
        fig.savefig(image_save, bbox_inches='tight')
