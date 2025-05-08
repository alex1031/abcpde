import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from common.abc_posterior import gaussian_abc_posterior

RUN_PATH = "./gaussian_plume/runs"
OBSERVED_PATH = "./gaussian_plume/observed_data/case_study/case_study.npy"
DF_PATH = "./gaussian_plume/dataframe"
PLOT_PATH = "./gaussian_plume/plots"
MODELS = ["case_study_no_advection", "case_study_with_advection", "case_study_with_advection_U(0,0.014)"]
DISTANCE_METRICS = ["Frechet Distance", "Hausdorff Distance", "Wasserstein Distance"]
RUNS = 1
THRESHOLD = 0.0001
NPARAMS = 3

def generate_solution_case_study(nx, ny, Lx, Ly, cx, cy, s): 
    dx, dy = Lx/(nx-1), Ly/(ny-1)
    dt = 3
    tend = 300
    t = 0

    cfl_x, cfl_y = cx * dt/dx, cy * dt/dy
    diff_x, diff_y = s * dt/dx**2, s * dt/dy**2

    u = np.zeros((nx+2, ny+2))
    sol = []
    source_x, source_y = 0, ny // 2 # Source point comes from x=0 and half of y
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

    return np.array(sol) 


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

if __name__ == "__main__":

    # Create dictionary to store result for models
    predicted_rmse_all = {}

    # Parameters to solve the equation
    case_nx, case_ny = 11, 18
    case_Lx, case_Ly = 1200, 800
    Lx, Ly = 1.0, 1.0
    case_x, case_y = np.linspace(200, case_Lx, case_nx), np.linspace(-1000, case_Ly, case_ny)

    # List to store the dataframe
    rmse_df_ls = []

    for model in MODELS:
        # Set path for run data and load observed data
        runs_path = os.path.join(RUN_PATH, model)
        observed = np.load(OBSERVED_PATH)
        # Initialise dictionary to store predicted rmse for each run
        predicted_rmse = {}

        # Need to iterate for each distance metric
        for metric in DISTANCE_METRICS:
            predicted_rmse_metric = np.zeros(RUNS)

            for i in range(RUNS):
                # Load run data
                run_path = os.path.join(runs_path, f"run{i+1}.npy")
                run_data = np.load(run_path)

                posterior = gaussian_abc_posterior(NPARAMS, run_data, THRESHOLD, metric)
                # We only want the mean: 0 - cx, 1 - cy, 2 - s
                cx_mean = posterior[0][0]
                cy_mean = posterior[1][0]
                s_mean = posterior[2][0]
                # Generate the solution
                plume_sim = generate_solution_case_study(case_nx, case_ny, Lx, Ly, cx_mean, cy_mean, s_mean)
                # Calculate RMSE
                predicted_rmse_metric[i] = rmse(plume_sim[-1].T, observed)
            
            # Store everything in the dictionary for each model
            predicted_rmse[metric] = predicted_rmse_metric

        # Store everything in a pandas dataframe
        predicted_rmse_df = pd.DataFrame.from_dict(predicted_rmse)
        # Add column to be identified easier
        predicted_rmse_df["Model"] = [model] * RUNS

        rmse_df_ls.append(predicted_rmse_df)
    
    # Put all dataframes together
    rmse_df = pd.concat(rmse_df_ls).reset_index(drop=True)
    
    # Store the data in the dataframe
    rmse_df.to_csv(DF_PATH + "/predicted_rmse.csv", index=False)

    # Plot boxplot for distribution of RMSE values
    plot_path = os.path.join(PLOT_PATH)

    # Loop through each distance metric and plot boxplots
    for model in MODELS:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter dataframe for the model
        model_df = rmse_df[rmse_df["Model"] == model]

        # Gather data per metric
        data = [model_df[metric].dropna() for metric in DISTANCE_METRICS]

        # Set the title
        title = f"Predicted RMSE Distribution for {model.replace('_', ' ').title()}"

        # Boxplot
        ax.boxplot(data, labels=DISTANCE_METRICS)
        ax.set_title(title)
        ax.set_xlabel('Distance Metric')
        ax.set_ylabel('RMSE')
        ax.grid(True)

        # Save the figure
        fig.tight_layout()
        fig.savefig(os.path.join(plot_path, f"{model}_{THRESHOLD}boxplot.png"))
        plt.close(fig) # Close to avoid overlapping figures
