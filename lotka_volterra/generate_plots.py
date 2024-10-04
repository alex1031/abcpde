import numpy as np
import os, sys
import matplotlib.pyplot as plt

# Assume the directories are correct

# Results in form [[a_mean, a_median, a_lowerbound, a_upperbound], [b_mean, b_median, b_lowerbound, b_upperbound]]
RESULTS_DIR = "./lotka_volterra/results"
PLOTS_DIR = "./lotka_volterra/plots"
DISTANCE_METRICS = ["Wasserstein Distance", "Energy Distance", "Maximum Mean Discrepancy", "Cramer-von Mises Distance", "Kullback-Leibler Divergence"]
SUB_COLUMNS = ["Mean", "Median", "Lower Bound", "Upper Bound"]
QUANTILE = [0.05, 0.01, 0.001]
PARAM = ["a", "b"]