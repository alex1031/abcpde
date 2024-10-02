import numpy as np
import os
from common.distances import *
# from joblib import Parallel, delayed
from scipy.integrate import odeint
# from lotka_volterra.lv_funcs import generate_sample
import time

def simulation_uniform(observed_prey: np.ndarray, observed_predator: np.ndarray, niter: int=1000) -> np.ndarray:
    def dUdt(U, t, a, b):
        x = U[:len(a)]
        y = U[len(a):]

        dxdt = a*x - x*y
        dydt = b*x*y - y

        return np.concatenate([dxdt, dydt]) 
    
    # Initial conditions
    theta_a, theta_b = np.random.uniform(-10, 10, niter), np.random.uniform(-10, 10, niter)

    sim_start = time.time()
    # Within each iteration: generate sample and then calculate distance
    x0, y0 = np.array([0.5] * niter), np.array([0.5] * niter)
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
    observed_prey = np.tile(observed_data[:,0], (1000, 1)).T
    observed_predator = np.tile(observed_data[:,1], (1000, 1)).T
    start_time = time.time()
    results = simulation_uniform(observed_prey, observed_predator)
    end_time = time.time()
    print("Run time:", end_time - start_time)
    np.save(save_path, results)


# def simulation_uniform(observed_prey: np.ndarray, observed_predator: np.ndarray) -> np.ndarray:
        
#     # Generate required parameters
#     theta_a = np.random.uniform(-10, 10)
#     theta_b = np.random.uniform(-10, 10)

#     # Using uniformly generated alpha and beta to solve ode
#     tspan = np.linspace(0, 10, 1000)
#     simulated_dUdt = generate_sample((theta_a, theta_b, 0), tspan)
#     simulated_prey = simulated_dUdt[:,0]
#     simulated_predator = simulated_dUdt[:,1]

#     # Compare by summary statistics - Calculate distances
#     ## Wasserstein Distance
#     wdist_prey = wasserstein_distance(simulated_prey, observed_prey)
#     wdist_predator = wasserstein_distance(simulated_predator, observed_predator)
#     wdist = (wdist_prey + wdist_predator)/2

#     ## Energy Distance
#     edist_prey = energy_distance(simulated_prey, observed_prey)
#     edist_predator = energy_distance(simulated_predator, observed_predator)
#     edist = (edist_prey + edist_predator)/2

#     ## Maxmimum Mean Discrepancy
#     mmd_prey = maximum_mean_discrepancy(simulated_prey.reshape(-1, 1), observed_prey.reshape(-1, 1))
#     mmd_predator = maximum_mean_discrepancy(simulated_predator.reshape(-1, 1), observed_predator.reshape(-1, 1))
#     mmd = (mmd_prey + mmd_predator)/2
    
#     ## Cramer-von Mises Distance
#     cvmd_prey = cramer_von_mises(simulated_prey, observed_prey)
#     cvmd_predator = cramer_von_mises(simulated_predator, observed_predator)
#     cvmd = (cvmd_prey + cvmd_predator)/2

#     ## Kullback-Leibler Divergence
#     kld_prey = kullback_leibler_divergence(simulated_prey, observed_prey)
#     kld_predator = kullback_leibler_divergence(simulated_predator, observed_predator)
#     kld = (kld_prey + kld_predator)/2

#     # Add to results
#     results = np.array([theta_a, theta_b, wdist, edist, mmd, cvmd, kld])
#     return results

# def main(observed_path: str, save_path: str, niter: int=10000, n_jobs: int=-1) -> None:
#     if os.path.exists(save_path):
#         return
    
#     observed_data = np.load(observed_path)
#     observed_prey = observed_data[:,0]
#     observed_predator = observed_data[:,1]

#     results = Parallel(n_jobs=n_jobs)(
#         delayed(simulation_uniform)(observed_prey, observed_predator) for i in range(niter)
#     )

#     results = np.array(results)
#     np.save(save_path, results)

# def simulation_uniform(observed_prey: np.ndarray, observed_predator: np.ndarray, niter: int=1000) -> np.ndarray:
#     # Initial conditions
#     theta_a, theta_b = np.zeros(niter), np.zeros(niter)
#     wdist_ls, edist_ls, cvmd_ls, mmd_ls, kld_ls = np.zeros(niter), np.zeros(niter), np.zeros(niter), np.zeros(niter), np.zeros(niter) 

#     # Within each iteration: generate sample and then calculate distance
#     for i in range(niter):
#         # Generate required parameters
#         theta_i_a = np.random.uniform(-10, 10)
#         theta_i_b = np.random.uniform(-10, 10)
#         theta_a[i] = theta_i_a
#         theta_b[i] = theta_i_b

#         # Using uniformly generated alpha and beta to solve ode
#         tspan = np.linspace(0, 10, 100)
#         simulated_dUdt = generate_sample((theta_i_a, theta_i_a, 0), tspan)
#         simulated_prey = simulated_dUdt[:,0]
#         simulated_predator = simulated_dUdt[:,1]

#         # Compare by summary statistics - Calculate distances
#         ## Wasserstein Distance
#         wdist_prey = wasserstein_distance(simulated_prey, observed_prey)
#         wdist_predator = wasserstein_distance(simulated_predator, observed_predator)
#         wdist = (wdist_prey + wdist_predator)/2
#         wdist_ls[i] = wdist

#         ## Energy Distance
#         edist_prey = energy_distance(simulated_prey, observed_prey)
#         edist_predator = energy_distance(simulated_predator, observed_predator)
#         edist = (edist_prey + edist_predator)/2
#         edist_ls[i] = edist

#         ## Maxmimum Mean Discrepancy
#         mmd_prey = maximum_mean_discrepancy(simulated_prey.reshape(-1, 1), observed_prey.reshape(-1, 1))
#         mmd_predator = maximum_mean_discrepancy(simulated_predator.reshape(-1, 1), observed_predator.reshape(-1, 1))
#         mmd = (mmd_prey + mmd_predator)/2
#         mmd_ls[i] = mmd
        
#         ## Cramer-von Mises Distance
#         cvmd_prey = cramer_von_mises(simulated_prey, observed_prey)
#         cvmd_predator = cramer_von_mises(simulated_predator, observed_predator)
#         cvmd = (cvmd_prey + cvmd_predator)/2
#         cvmd_ls[i] = cvmd

#         ## Kullback-Leibler Divergence
#         kld_prey = kullback_leibler_divergence(simulated_prey, observed_prey)
#         kld_predator = kullback_leibler_divergence(simulated_predator, observed_predator)
#         kld = (kld_prey + kld_predator)/2
#         kld_ls[i] = kld 

#     # Add to results
#     results = np.column_stack((theta_a, theta_b, wdist_ls, edist_ls, mmd_ls, cvmd_ls, kld_ls))
#     return results