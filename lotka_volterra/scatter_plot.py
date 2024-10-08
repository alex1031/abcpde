import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DISTANCE_METRIC = ["Wasserstein Distance"]
DF_PATH = "./lotka_volterra/dataframe/all_results.csv"

if __name__ == "__main__":
    
    results = pd.read_csv(DF_PATH)
    medians = results[results["summary_statistic"] == "Median"]