import numpy as np
import os
from common.distances import *
from scipy.integrate import odeint
import time
import torch
from torchdiffeq_mod._impl import odeint

def simulation_uniform(observed_prey: np.ndarray, observed_predator: np.ndarray, niter: int=10000, batch_size: int=10000) -> np.ndarray:
    # Initial conditions
    results = []
    sim_time = []
    num_batches = niter//batch_size

    def dUdt(t, state, theta_a, theta_b):
        x, y = state[..., 0], state[..., 1]
        dxdt = theta_a*x - x*y
        dydt = theta_b*x*y - y
        return torch.stack((dxdt, dydt), dim=-1)

    for i in range(num_batches):
        theta_a, theta_b = np.random.RandomState().uniform(-10, 10, batch_size), np.random.RandomState().uniform(-10, 10, batch_size)
        theta_a, theta_b = torch.from_numpy(theta_a).cuda(), torch.from_numpy(theta_b).cuda() 
        sim_start = time.time()
        # Within each iteration: generate sample and then calculate distance
        ic = torch.full((batch_size, 2), 0.5).cuda() # initial conditions
        t = torch.linspace(0, 10, 100).cuda()
        sim_sol = odeint(lambda t, state: dUdt(t, state, theta_a, theta_b), ic, t, method='rk4')
        prey_sol = np.nan_to_num(np.array(sim_sol.cpu()[:, :, 0]))
        predator_sol = np.nan_to_num(np.array(sim_sol.cpu()[:, :, 1]))
        sim_end = time.time()
        print(f"Time taken to simulate {batch_size}: {sim_end - sim_start}")

        wass_dist_start = time.time()
        wasserstein_prey = wasserstein_distance(prey_sol, observed_prey)
        wasserstein_predator = wasserstein_distance(predator_sol, observed_predator)
        wasserstein = (wasserstein_prey + wasserstein_predator)/2
        wass_dist_end = time.time()
        wass_dist_sim = wass_dist_end - wass_dist_start
        
        energy_dist_start = time.time()
        energy_prey = energy_dist(prey_sol, observed_prey)
        energy_predator = energy_dist(predator_sol, observed_predator)
        energy = (energy_prey + energy_predator)/2
        energy_dist_end = time.time()
        energy_dist_sim = energy_dist_end - energy_dist_start
        
        mmd_start = time.time()
        mmd_prey = maximum_mean_discrepancy(prey_sol, observed_prey)
        mmd_predator = maximum_mean_discrepancy(predator_sol, observed_predator)
        mmd = (mmd_prey + mmd_predator)/2
        mmd_end = time.time()
        mmd_sim = mmd_end - mmd_start
        
        cvmd_start = time.time()
        cvmd_prey = cramer_von_mises(prey_sol, observed_prey)
        cvmd_predator = cramer_von_mises(predator_sol, observed_predator)
        cramer = (cvmd_prey + cvmd_predator)/2
        cvmd_end = time.time()
        cvmd_sim = cvmd_end - cvmd_start
        
        kld_start = time.time()
        kld_prey = kullback_leibler_divergence(prey_sol, observed_prey)
        kld_predator = kullback_leibler_divergence(predator_sol, observed_predator)
        kld = (kld_prey + kld_predator)/2
        kld_end = time.time()
        kld_sim = kld_end - kld_start
        
        dist_sim = np.array([wass_dist_sim, energy_dist_sim, mmd_sim, cvmd_sim, kld_sim])
        batch_results = np.column_stack((theta_a.cpu().T, theta_b.cpu().T, wasserstein, energy.T, mmd.T, cramer, kld.T))
        results.append(batch_results)
        sim_time.append(dist_sim)
    
    return np.vstack(results), sim_time

def main(observed_path: str, save_path: str, save_path_sim: str) -> None:
    if os.path.exists(save_path):
        return
    
    observed_data = np.load(observed_path)
    observed_prey = np.tile(observed_data[:,0], (10000, 1)).T
    observed_predator = np.tile(observed_data[:,1], (10000, 1)).T
    start_time = time.time()
    results, dist_sim_time = simulation_uniform(observed_prey, observed_predator)
    end_time = time.time()
    print("Run time:", end_time - start_time)
    np.save(save_path, results)
    np.save(save_path_sim, dist_sim_time)