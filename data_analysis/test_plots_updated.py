
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.getcwd())

from src.viz import plots
from src.runner.lyapunov_lib import plotting as lyap_plotting
from config import CONFIG

# Dummy data
dist_data = np.random.rand(100)
matrix_data = np.random.rand(10, 10)
times = np.arange(100)

os.makedirs("test_plots", exist_ok=True)

print("Testing src.viz.plots...")
try:
    plots.plot_pairwise_distance_distribution(
        {"pairs": {"p1": {"cos": {"mean": dist_data}}}}, 
        "test_plots/dist_hist.png", 
        metric_name="cos", 
        aggregate_type="mean"
    )
    print(" - plot_pairwise_distance_distribution: OK")
except Exception as e:
    print(f" - plot_pairwise_distance_distribution: FAILED {e}")

try:
    plots.plot_mean_log_distance_vs_time(dist_data, "test_plots/mean_log.png")
    print(" - plot_mean_log_distance_vs_time: OK")
except Exception as e:
    print(f" - plot_mean_log_distance_vs_time: FAILED {e}")

try:
    plots.plot_time_series_for_pair(dist_data, "test_plots/timeseries.png")
    print(" - plot_time_series_for_pair: OK")
except Exception as e:
    print(f" - plot_time_series_for_pair: FAILED {e}")

print("\nTesting src.runner.lyapunov_lib.plotting...")
try:
    lyap_plotting.plot_curves(times, [dist_data, dist_data*0.5], "test_plots/lyap_curves.png")
    print(" - plot_curves: OK")
except Exception as e:
    print(f" - plot_curves: FAILED {e}")

print("\nChecking for PDF files...")
expected_pdfs = [
    "test_plots/dist_hist.pdf",
    "test_plots/mean_log.pdf",
    "test_plots/timeseries.pdf",
    "test_plots/lyap_curves.pdf"
]
for f in expected_pdfs:
    if os.path.exists(f):
        print(f" - {f}: FOUND")
    else:
        print(f" - {f}: MISSING")

print("\nDone.")
