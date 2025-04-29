import numpy as np
import time
import logging
import os
from common.distances import wasserstein_distance_3D, cramer_von_mises_3d, directed_hausdorff, frechet_distance
from scipy.spatial.distance import directed_hausdorff

NX = 11
NY = 18
LX = 1.0
LY = 1.0
TEND = 2.0
DT = 0.02

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

    '''
    We transpose the axis the solution. Such that:
    - Axis 0: x
    - Axis 1: y
    - Axis 2: time
    Interpretation: For each x-grid, we have the concentration of each y-grid over time.
    Essentially, the (ny, 1200) array represents the concentration at each y over the time.
    So we have an array of size 1200 for each y. (A curve)
    '''
    sol = np.transpose(sol, (1, 2, 0)) # We no longer need to filter for the relevant parts.
    return np.array(sol)

def compute_frechet_distance(sim, obs):
    return np.mean([frechet_distance(sim[i, :, :], obs[i, :, :]) for i in range(sim.shape[0])])

def compute_directed_hausdorff(sim, obs):
    return np.mean([max(directed_hausdorff(sim[i, :, :], obs[i, :, :])[0], directed_hausdorff(obs[i, :, :], sim[i, :, :])[0]) for i in range(sim.shape[0])])

def abc_simulation(observed, n=100): # Performs Approximate Bayesian Computation (ABC) simulation. Returns results and timing information
    # To store overall results and all the sim times
    results, sim_time = [], [] 

    rng = np.random.default_rng()
    cx, cy = rng.uniform(0, 0.1, n), rng.uniform(0, 0.1, n)
    s = rng.uniform(0, 1e-4, n)

    for i in range(n):
        start_time = time.time()
        simulated = generate_solution(NX, NY, LX, LY, cx[i], cy[i], s[i])

        if i % 10000 == 0 or i == n-1:
            logging.info(f"Iteration {i+1}/{n}: Simulation completed in {time.time() - start_time:.2f}s.")

        # Applying Distance Metrics (Need to be changed depending on how the final result is computed, can possibly extend all the other distance metrics?)
        ## Wasserstein Distance
        wass_start = time.time()
        wass = np.mean(wasserstein_distance_3D(simulated, observed))
        wass_time = time.time() - wass_start

        ## CvMD
        cvmd_start = time.time()
        cvmd = np.mean(cramer_von_mises_3d(simulated, observed))
        cvmd_time = time.time() - cvmd_start

        ## Functional Frechet - We have to calculate for each grid's concentration over time.
        frechet_start = time.time()
        frechet = compute_frechet_distance(simulated, observed)
        frechet_time = time.time() - frechet_start

        ## Hausdorff
        hausdorff_start = time.time()
        hausdorff = compute_directed_hausdorff(simulated, observed)
        hausdorff_time = time.time() - hausdorff_start

        results.append([cx[i], cy[i], s[i], wass, cvmd, frechet, hausdorff])
        sim_time.append([wass_time, cvmd_time, frechet_time, hausdorff_time])

        if i % 10000 == 0 or i == n-1:
            logging.info(f"Iteration {i+1}/{n}: Distances computed in {time.time() - start_time:.2f}s.")

    '''
        np.mean(sim_time, axis=0): Gives us the average computation time for each distance metric 
    '''
    return np.array(results), np.mean(sim_time, axis=0) 

def main(observed_path: str, save_path: str, save_path_sim:str) -> None:
    if os.path.exists(save_path):
        logging.info("Results already exist. Skipping computation.")
        return
    
    observed_data = np.load(observed_path)
    start_time = time.time()
    results, sim_times = abc_simulation(observed_data)
    logging.info(f"Total run time: {time.time() - start_time:.2f}s.")
    np.save(save_path, results)
    np.save(save_path_sim, sim_times)



