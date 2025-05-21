# Parameter Inference using Approximate Bayesian Computation (ABC) Rejection-Sampling

This repository includes all the code used for the Honours Thesis 'Approximate Bayesian Computation in Partial Differential Equations'.

## Experiment 1:

All the files related to experiment 1 are stored in the `lotka_volterra` folder.

`run_test.py` is the file that runs all the simulations. To run this, do `python -m run_test.py` in the command prompt

Change `NUM_WORKERS` for how many CPUs to distribute the processes to. `NUM_RUNS` control how many times the simulation is ran.

To determine which observed data to use, add or remove values in `fit_spline` and `noise`. From the thesis, these values for noise are available: 0, 0.25, 0.5, 0.75, 'linear'. For each noise level (except for 0 noise), the option to choose smoothing or not is also present. To do so, put `True` or `False` in the respective index within the `fit_spline` list.

To change how many samples are generated, change `niter` within the `simulation_uniform` function in the `abc_simulation_gpu.py` file. 

## Experiment 2:

Related files can be found in the `gaussian_plume` folder.

Similar to experiment 1, `run_test_diffusion.py` and `run_test_calm_air.py` corresponds to experiment 1 and experiment 2 in the thesis respectively. `run_test_case_study_no_advection.py` and `run_test_case_study_with_advection.py` then corresponds to scenario 1 and scenario 2 in the case study section. `NUM_WORKERS` and `NUM_RUNS` can be changed accordingly similar to above.

The values for the priors can be changed in the respective `abc_simulation_xxx.py` files,  with `xxx` replace with what follows `run_test` above.

## Processing Results

After the simulations are ran, they would be stored in the `runs` folder in the respective experiment. To process the runs, run the `calculate_abc_posterior.py` file in the respective files. This generates the relevant summary statistics, which is stored in the `results` folder.

`NUM_RUNS` should be changed to match how many runs are in the `runs` folder. `NPARAMS` should be changed to match how many parameters are being estimated. Different acceptance threshold values can be added to the `QUANTILES` list. The strings in `DISTANCE_METRICS` correspond to which distance metrics are being tested (They matchup with the index in the runs file, structured as `[param_1, param_2, ... param_n, metric_1, ..., metric_n]`).

For both experiments, `results_dataframe.py` should be ran first to generate the relevant summary statistics across all runs. Afterwards the relevant visualisations can be generated in any order.
