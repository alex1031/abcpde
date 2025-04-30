import os
import numpy as np
import logging
import traceback
from multiprocessing import Pool, cpu_count
from typing import List, Iterator, Tuple
from gaussian_plume.abc_simulation_case_study_no_advection import main
import time

SAVE_DIR = "./gaussian_plume/runs"
OBSERVED_DIR = "./gaussian_plume/observed_data"
NUM_WORKERS = 5 # Needs to be changed by the number of jobs being parallelised
NUM_RUNS = 2 # Change this later - Number of times simulation is repeated

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def generate_paths() -> List[Tuple[str, str, str]]:
    """
    Generates file paths for observed data and output files for multiple runs.
    """
    observed_path = os.path.join(OBSERVED_DIR, "case_study", "case_study.npy")
    output_dir = os.path.join(SAVE_DIR, "case_study_no_advection")

    return [
        (observed_path, os.path.join(output_dir, f"run{i+1}"), os.path.join(output_dir, f"run{i+1}_sim_time"))
        for i in range(NUM_RUNS)
    ]

def run_simulation(args: Tuple[str, str, str]) -> None:
    """
    Wrapper function for `main()` to execute ABC simulation.
    """

    observed_path, output_path, time_output_path = args
    try:
        logging.info(f"Starting simulation for: {output_path}")
        main(observed_path, output_path, time_output_path)
        logging.info(f"Completed simulation: {output_path}")
    except Exception as e:
        error_message = f"Error in simulation for {output_path}: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_message)

def task_callback(result: str) -> None:
    """Logs successful task completion."""
    logging.info(result)

def error_callback(error: Exception) -> None:
    """Logs failed task errors."""
    logging.error(f"Task failed with error: {error}")

if __name__ == "__main__":

    task_args = generate_paths()

    start_time = time.time()
    
    num_workers = min(cpu_count(), NUM_WORKERS)
    with Pool(num_workers) as pool:
        # pool.starmap_async(run_simulation, task_args, callback=custom_callback)
        for args in task_args:
            pool.apply_async(run_simulation, args=(args,), callback=task_callback, error_callback=error_callback)
        pool.close()
        pool.join()

    logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds")
        



