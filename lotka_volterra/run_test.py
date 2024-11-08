import os
import numpy as np
from multiprocessing import Pool, Manager, cpu_count, Queue, Lock
from typing import List, Iterator, Tuple
from lotka_volterra.abc_simulation_gpu import main
import time

SAVE_DIR = "./lotka_volterra/runs"
OBSERVED_DIR = "./lotka_volterra/observed_data"
NUM_WORKERS = 5 # Needs to be changed by the number of jobs being parallelised
NUM_RUNS = 10 # Change this later - Number of times simulation is repeated

def generate_path(fit_spline: List[bool], noise: List) -> Iterator[Tuple[str, str]]:
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

    return zip(observed_args, output_args)

def worker_process(queue: Queue, lock: Lock) -> None:
    while True:
        with lock:
            if queue.empty():
                return
            args = queue.get()
        
        main(*args)

if __name__ == "__main__":
    # fit_spline = [False, False, True]
    # noise = [0, 0.5, 0.5]
    fit_spline = [True, False]
    noise = [0.25, "linear"]
    path = generate_path(fit_spline, noise)

    start_time = time.time()
    with Manager() as manager:
        task_queue = manager.Queue()
        task_lock = manager.Lock()
        for p in path:
            task_queue.put(p)

        with Pool() as pool:
            for _ in range(NUM_WORKERS):
                pool.apply_async(worker_process, (task_queue, task_lock))
                # pool.apply(worker_process, (task_queue, task_lock))

            pool.close()
            pool.join()
            time.sleep(0.1)
    end_time = time.time()
    print("Execution time:", end_time - start_time)
        



