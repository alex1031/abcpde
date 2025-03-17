import numpy as np
import time as time
import os
from common.distances import wasserstein_distance_3D, cramer_von_mises_3d, directed_hausdorff, frechet_distance_dp
from scipy.signal import resample

RUN_PATH = "./gaussian_plume/runs"
NX = 51
NY = 51
LX = 5000
LY = 5000

def generate_solution(nx, ny, Lx, Ly, cx, cy, s):
    dx, dy = Lx/(nx-1), Ly/(ny-1)
    dt = 1
    tend = 1200
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

def downsample_trajectory(traj, new_size=100):
    """Resamples a 1D trajectory to a smaller number of points."""
    return resample(traj, new_size)

def abc_simulation(observed, n=5): # ** params stored in a dictionary 
    # To store overall results and all the sim times
    results, sim_time = [], [] 

    cx, cy = np.random.RandomState().uniform(-10, 10, n), np.random.RandomState().uniform(-10, 10, n)
    s = np.random.RandomState().uniform(-10, 10, n)

    for i in range(n):
        simulated = generate_solution(NX, NY, LX, LY, cx[i], cy[i], s[i])

        # Applying Distance Metrics
        ## Wasserstein Distance
        wass_dist_start = time.time()
        wass = np.max(wasserstein_distance_3D(simulated, observed))
        wass_dist_end = time.time()
        wass_dist_run = wass_dist_end - wass_dist_start
        print("Wasserstein Done")

        ## CvMD
        cvmd_dist_start = time.time()
        cvmd = np.max(cramer_von_mises_3d(simulated, observed))
        cvmd_dist_end = time.time()
        cvmd_dist_run = cvmd_dist_end - cvmd_dist_start
        print("CvMD Done")

        ## Functional Frechet - We have to calculate for each grid's concentration over time.
        def downsample_trajectory(traj, new_size=10): # Resample to make computation easier, should consider downsampling for all?
            """Resamples a 1D trajectory to a smaller number of points."""
            return resample(traj, new_size)
        
        sim_ds = np.apply_along_axis(downsample_trajectory, 2, simulated)
        obs_ds = np.apply_along_axis(downsample_trajectory, 2, observed)
        
        frechet_start = time.time()
        frechet_distances = np.zeros((51, 51))
        for j in range(NX):
            for m in range(NY):
                frechet_distances[j, m] = frechet_distance_dp(sim_ds[j, m, :], obs_ds[j, m, :])
        frechet = np.max(frechet_distances)
        frechet_end = time.time()
        frechet_run = frechet_end - frechet_start
        print("Frechet Done")

        ## Hausdorff
        hausdorff_start = time.time()
        hausdorff = np.max(directed_hausdorff(sim_ds, obs_ds))
        hausdorff_end = time.time()
        hausdorff_run = hausdorff_end - hausdorff_start
        print("Hausdorff Done")

        dist_sim = [wass_dist_run, cvmd_dist_run, frechet_run, hausdorff_run]
        dist_results = [cx[i], cy[i], s[i], wass, cvmd, frechet, hausdorff]
        sim_time.append(dist_sim)
        results.append(dist_results)

        print(f"Iteration {i} Done")

    return np.vstack(results), sim_time


if __name__ == "__main__":
    observed_sol = np.load("./gaussian_plume/test.npy")
    run, run_time = abc_simulation(observed_sol)
    np.save("./gaussian_plume/runs/test_run.npy", run)
    np.save("./gaussian_plume/runs/test_run_time.npy", run_time)


