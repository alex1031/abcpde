import os
import numpy as np
from multiprocessing import Pool, Manager, cpu_count, Queue, Lock
from typing import List, Iterator, Tuple
from gaussian_plume.abc_simulation import main
import time

SAVE_DIR = "./gaussian_plume/runs"
OBSERVED_DIR = "./gaussian_plume/observed_data"
NUM_WORKERS = 5 # Needs to be changed by the number of jobs being parallelised
NUM_RUNS = 2 # Change this later - Number of times simulation is repeated

def generate_path(noise: str) -> Iterator[Tuple[str, str]]:
    # Generating paths to load and save data

    observed_args, output_args, time_output_args = [], [], []
    observed_path = os.path.join(OBSERVED_DIR, noise + f"/{noise}.npy")
    output_dir = os.path.join(SAVE_DIR, noise)

    for i in range(NUM_RUNS):
        output_path = os.path.join(output_dir, f"run{i+1}.npy")
        time_output_path = os.path.join(output_dir, f"run{i+1}_sim_time.npy")
        output_args.append(output_path)
        observed_args.append(observed_path)
        time_output_args.append(time_output_path)
    
    return zip(observed_args, output_args, time_output_args)


def worker_process(queue: Queue, lock: Lock) -> None:
    while True:
        with lock:
            if queue.empty():
                return
            args = queue.get()
        
        main(*args)

if __name__ == "__main__":

    noise = "linear"
    path = generate_path(noise)

    start_time = time.time()
    with Manager() as manager:
        task_queue = manager.Queue()
        task_lock = manager.Lock()
        for p in path:
            task_queue.put(p)

        with Pool() as pool:
            for _ in range(NUM_WORKERS):
                pool.apply_async(worker_process, (task_queue, task_lock))

            pool.close()
            pool.join()
            time.sleep(0.1)
    end_time = time.time()
    print("Execution time:", end_time - start_time)
        



