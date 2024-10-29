import os
import numpy as np
from multiprocessing import Pool, Manager, cpu_count, Queue, Lock
from typing import List, Iterator, Tuple
from lotka_volterra.abc_simulation import main
import time
from mpi4py import MPI

SAVE_DIR = "./lotka_volterra/runs"
OBSERVED_DIR = "./lotka_volterra/observed_data"
NUM_RUNS = 5 # Change this later - Number of times simulation is repeated

# MPI Initialisation
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def generate_path(fit_spline: List[bool], noise: List[float]) -> Iterator[Tuple[str, str]]:
    # Sorting out the paths to load and save data
    observed_args, output_args = [], []
    params = list(zip(fit_spline, noise))

    for p in params:
        smoothing = p[0]
        sigma = p[1] # variance of the noise

        if smoothing:
            observed_path = os.path.join(OBSERVED_DIR, f"n{sigma}_smoothing/n{sigma}_smoothing.npy")
            output_dir = os.path.join(SAVE_DIR, f"n{sigma}_smoothing")
        else:
            observed_path = os.path.join(OBSERVED_DIR, f"n{sigma}_no_smoothing/n{sigma}_no_smoothing.npy")
            output_dir = os.path.join(SAVE_DIR, f"n{sigma}_no_smoothing")

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        for i in range(NUM_RUNS):
            output_path = os.path.join(output_dir, f"run{i+1}.npy")
            output_args.append(output_path)
            observed_args.append(observed_path)

    return list(zip(observed_args, output_args))

if __name__ == "__main__":
    fit_spline = [False, False, True]
    noise = [0, 0.5, 0.5]
    path = generate_path(fit_spline, noise)
    total_tasks = len(path)
    # Scatter tasks to all processes
    if rank == 0:
        # Split list of paths into roughly equal chunks for each process
        tasks = [path[i::size] for i in range(size)]
    else:
        tasks = None
    # Scatter tasks to each process
    tasks = comm.scatter(tasks, root=0)    

    start_time = time.time()

    for task in tasks:
        main(*task)

    comm.Barrier()

    end_time = time.time()
    print("Execution time:", end_time - start_time)
    
        



