import numpy as np
import os
from common.distances import *
from scipy.integrate import odeint
import time
import torch
from torchdiffeq import odeint

def simulation_uniform(observed_prey: np.ndarray, observed_predator: np.ndarray, niter: int=1000000, batch_size: int=1000000) -> np.ndarray:
    # Initial conditions
    results = []
    num_batches = niter//batch_size

    def dUdt(t, state, theta_a, theta_b):
        x, y = state[..., 0], state[..., 1]
        dxdt = theta_a*x - x*y
        dydt = theta_b*x*y - y
        return torch.stack((dxdt, dydt), dim=-1)

    for i in range(num_batches):
        # theta_a, theta_b = torch.randint(-10, 10, size=(1,batch_size)).cuda(), torch.randint(-10, 10, size=(1,batch_size)).cuda()
        theta_a, theta_b = torch.distributions.Uniform(-10, 10).sample((1,batch_size)).cuda(), torch.distributions.Uniform(-10, 10).sample((1,batch_size)).cuda()
        sim_start = time.time()
        # Within each iteration: generate sample and then calculate distance
        ic = torch.full((batch_size, 2), 0.5).cuda() # initial conditions
        t = torch.linspace(0, 10, 100).cuda()
        sim_sol = odeint(lambda t, state: dUdt(t, state, theta_a, theta_b), ic, t, method='rk4')
        prey_sol = np.nan_to_num(np.array(sim_sol.cpu()[:, :, 0]))
        # prey_sol = np.round(prey_sol, 4)
        predator_sol = np.nan_to_num(np.array(sim_sol.cpu()[:, :, 1]))
        # predator_sol = np.round(predator_sol, 4)
        # prey_sol = torch.tensor(prey_sol, dtype=torch.float32).cuda()
        # predator_sol = torch.tensor(predator_sol, dtype=torch.float32).cuda()
        # prey_sol = prey_sol.astype(np.float32)
        # predator_sol = predator_sol.astype(np.float32)
        sim_end = time.time()
        print(f"Time taken to simulate {batch_size}: {sim_end - sim_start}")

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

        batch_results = np.column_stack((theta_a.cpu().T, theta_b.cpu().T, wasserstein, energy.T, mmd.T, cramer, kld.T))
        results.append(batch_results)
    
    return np.vstack(results)

def main(observed_path: str, save_path: str) -> None:
    if os.path.exists(save_path):
        return
    
    observed_data = np.load(observed_path)
    observed_prey = np.tile(observed_data[:,0], (1000000, 1)).T
    observed_predator = np.tile(observed_data[:,1], (1000000, 1)).T
    # observed_prey = torch.tensor(observed_prey, dtype=torch.float32).cuda()
    # observed_predator = torch.tensor(observed_predator, dtype=torch.float32).cuda()
    # observed_prey = observed_prey.astype(np.float32)
    # observed_predator = observed_predator.astype(np.float32)
    start_time = time.time()
    results = simulation_uniform(observed_prey, observed_predator)
    end_time = time.time()
    print("Run time:", end_time - start_time)
    np.save(save_path, results)