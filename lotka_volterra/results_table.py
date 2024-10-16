import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataframe_image as dfi

DATAFRAME_PATH = "./lotka_volterra/dataframe/all_summary_statistics.csv"

df = pd.read_csV(DATAFRAME_PATH)
median = df[(df["summary_statistic"] == "Median") & (df["quantile"] == "0.1%")].reset_index(drop=True)

keep_cols = ["model", "param", "Cramer-von Mises Distance", "Energy Distance", "Kullback-Leibler Divergence", "Maximum Mean Discrepancy", "Wasserstein Distance"]

median_table = median[keep_cols]