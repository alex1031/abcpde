import numpy as np
import os
from common.distances import *
from scipy.integrate import odeint
import time

def simulation_uniform(observed_prey: np.ndarray, observed_predator: np.ndarray, niter: int=1000000) -> np.ndarray:
    def dUdt(U, t, a, b):
        x = U[:len(a)]
        y = U[len(a):]

        dxdt = a*x - x*y
        dydt = b*x*y - y

        return np.concatenate([dxdt, dydt]) 
    
    # Initial conditions
    theta_a, theta_b = np.random.RandomState().uniform(-10, 10, niter), np.random.RandomState().uniform(-10, 10, niter)

    sim_start = time.time()
    # Within each iteration: generate sample and then calculate distance
    x0, y0 = np.full(niter, 0.5), np.full(niter, 0.5)
    S0 = np.concatenate([x0, y0])
    tspan = np.linspace(0, 10, 100)
    sim_sol = odeint(dUdt, S0, tspan, args=(theta_a, theta_b))
    prey_sol = sim_sol[:, :niter]
    predator_sol = sim_sol[:, niter:]
    sim_end = time.time()
    print(f"Time taken to simulate {niter}: {sim_end - sim_start}")

    dist_start = time.time()
    wasserstein_prey = wasserstein_distance(prey_sol, observed_prey)
    wasserstein_predator = wasserstein_distance(predator_sol, observed_predator)
    wasserstein = (wasserstein_prey + wasserstein_predator)/2

    energy_prey = energy_dist(prey_sol, observed_prey)
    energy_predator = energy_dist(predator_sol, observed_predator)
    energy = (energy_prey + energy_predator)/2

    mmd_prey = maximum_mean_discrepancy(prey_sol, observed_prey)
    mmd_predator = maximum_mean_discrepancy(predator_sol, observed_predator)
    mmd = (mmd_prey + mmd_predator)/2

    cvmd_prey = cramer_von_mises(prey_sol, observed_prey)
    cvmd_predator = cramer_von_mises(predator_sol, observed_predator)
    cramer = (cvmd_prey + cvmd_predator)/2

    kld_prey = kullback_leibler_divergence(prey_sol, observed_prey)
    kld_predator = kullback_leibler_divergence(predator_sol, observed_predator)
    kld = (kld_prey + kld_predator)/2
    dist_end = time.time()
    print(f"Time taken to calculate distances: {dist_end-dist_start}")

    results = np.column_stack((theta_a, theta_b, wasserstein, energy, mmd, cramer, kld))
    
    return results

def main(observed_path: str, save_path: str) -> None:
    if os.path.exists(save_path):
        return
    
    observed_data = np.load(observed_path)
    observed_prey = np.tile(observed_data[:,0], (1000000, 1)).T
    observed_predator = np.tile(observed_data[:,1], (1000000, 1)).T
    start_time = time.time()
    results = simulation_uniform(observed_prey, observed_predator)
    end_time = time.time()
    print("Run time:", end_time - start_time)
    np.save(save_path, results)