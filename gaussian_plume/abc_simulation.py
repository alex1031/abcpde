import numpy as np
import time
import logging
import os
from common.distances import wasserstein_distance_3D, cramer_von_mises_3d, directed_hausdorff, frechet_distance
from scipy.signal import resample
from numba import njit, prange

NX = 51
NY = 51
LX = 5000
LY = 5000
TEND = 1200
DT = 1

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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

def downsample_trajectory(traj, new_size=10):
    """Resamples a 1D trajectory to a smaller number of points."""
    return resample(traj, new_size)

# @njit(parallel=True)
def compute_frechet_distance(sim, obs):
    """Computes Frechet distance in a parallelized manner."""
    frechet_distances = np.zeros((NX, NY))
    for j in prange(NX):
        for m in prange(NY):
            frechet_distances[j, m] = frechet_distance(sim[j, m, :], obs[j, m, :])
    return np.max(frechet_distances)

def abc_simulation(observed, n=5): # Performs Approximate Bayesian Computation (ABC) simulation. Returns results and timing information
    # To store overall results and all the sim times
    results, sim_time = [], [] 

    # cx, cy = np.random.RandomState().uniform(-10, 10, n), np.random.RandomState().uniform(-10, 10, n)
    # s = np.random.RandomState().uniform(-10, 10, n)

    rng = np.random.default_rng()
    cx, cy, s = rng.uniform(-10, 10, (3, n))

    for i in range(n):
        start_time = time.time()
        simulated = generate_solution(NX, NY, LX, LY, cx[i], cy[i], s[i])
        logging.info(f"Iteration {i+1}/{n}: Simulation complete.")

        # Applying Distance Metrics
        ## Wasserstein Distance
        wass_start = time.time()
        wass = np.max(wasserstein_distance_3D(simulated, observed))
        wass_time = time.time() - wass_start

        ## CvMD
        cvmd_start = time.time()
        cvmd = np.max(cramer_von_mises_3d(simulated, observed))
        cvmd_time = time.time() - cvmd_start

        ## Functional Frechet - We have to calculate for each grid's concentration over time.
        sim_ds = np.apply_along_axis(downsample_trajectory, 2, simulated)
        obs_ds = np.apply_along_axis(downsample_trajectory, 2, observed)
        
        frechet_start = time.time()
        frechet = compute_frechet_distance(sim_ds, obs_ds)
        frechet_time = time.time() - frechet_start

        ## Hausdorff
        hausdorff_start = time.time()
        hausdorff = np.max(directed_hausdorff(sim_ds, obs_ds))
        hausdorff_time = time.time() - hausdorff_start

        results.append([cx[i], cy[i], s[i], wass, cvmd, frechet, hausdorff])
        sim_time.append([wass_time, cvmd_time, frechet_time, hausdorff_time])

        logging.info(f"Iteration {i+1}/{n}: Distances computed in {time.time() - start_time:.2f}s.")


    return np.array(results), sim_time

def main(observed_path: str, save_path: str, save_path_sim:str) -> None:
    if os.path.exists(save_path):
        logging.info("Results already exist. Skipping computation.")
        return
    
    observed_data = np.load(observed_path)
    start_time = time.time()
    results, sim_times = abc_simulation(observed_data)
    logging.info(f"Total run time: {time.time() - start_time:.2f}s.")
    # np.save(save_path, results)
    # np.save(save_path_sim, sim_times)
    np.savez_compressed(save_path, results=results)
    np.savez_compressed(save_path_sim, sim_times=sim_times)



