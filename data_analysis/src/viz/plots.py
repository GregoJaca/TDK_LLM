# STATUS: PARTIAL
import matplotlib.pyplot as plt
import numpy as np
from config import CONFIG

def plot_pairwise_distance_distribution(aggregates, outpath, metric_name="cos", aggregate_type="mean"):
    """Plots a histogram of the distances for a specific metric and aggregate type."""
    # Respect global plot saving configuration
    if not CONFIG.get("plots", {}).get("save_histograms", True):
        return

    distances = []
    for pair, data in aggregates["pairs"].items():
        if metric_name in data and aggregate_type in data[metric_name]:
            distances.append(data[metric_name][aggregate_type])

    if not distances:
        # print(f"No data to plot for pairwise distance distribution for metric '{metric_name}' and aggregate type '{aggregate_type}'.")
        return

    plt.figure(figsize=(6, 4))
    plt.hist(distances, bins=20, alpha=0.75, edgecolor='k', linewidth=0.5)
    plt.title(f"Distribution of {aggregate_type.capitalize()} {metric_name.capitalize()} Distances")
    plt.xlabel(f"{aggregate_type.capitalize()} {metric_name.capitalize()} Distance")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_mean_log_distance_vs_time(mean_log_array, outpath, window=None, slope=None, r2=None):
    """Plots the mean log distance vs time, with optional fitted line."""
    plt.figure()
    plt.plot(mean_log_array, label="Mean Log Distance")
    plt.title("Mean Log Distance vs Time")
    plt.xlabel("Time Step") # More generic than "Time"
    plt.ylabel("Mean Log Distance")
    
    if window and slope is not None:
        plt.axvspan(window[0], window[1], color='red', alpha=0.2, label=f"Linear Fit Window")
        t = np.arange(window[0], window[1])
        fit_line = slope * t + (mean_log_array[window[0]] - slope * window[0])
        plt.plot(t, fit_line, 'r--', label=f"Fit (Slope={slope:.4f}, R2={r2:.2f})")
        
        # Add text box with slope and R2
        props = dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5)
        plt.text(0.05, 0.95, f'Slope: {slope:.4f}\nR2: {r2:.2f}', transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

    plt.legend()
    plt.savefig(outpath)
    plt.close()

def plot_pca_explained_variance(pca_model, outpath):
    """Plots the explained variance ratio of PCA components."""
    if "explained_variance_ratio" not in pca_model:
        print("PCA model does not contain explained_variance_ratio. Cannot plot.")
        return

    explained_variance_ratio = pca_model["explained_variance_ratio"]
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5, align='center', label='Individual Explained Variance')
    plt.step(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance, where='mid', label='Cumulative Explained Variance')
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Number of Principal Components')
    plt.title('PCA Explained Variance and Cumulative Sum')
    plt.legend(loc='best')
    plt.grid(True)

    # Add a line for 95% cumulative explained variance
    target_variance = 0.95
    idx_95_percent = np.argmax(cumulative_explained_variance >= target_variance) + 1
    plt.axvline(x=idx_95_percent, color='r', linestyle='--', label=f'{target_variance*100:.0f}% Variance Explained')
    plt.text(idx_95_percent + 0.5, 0.5, f'{idx_95_percent} components for {target_variance*100:.0f}% variance', rotation=90, verticalalignment='center')

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_hyperparam_sweep(sweep_results_df, outpath_prefix):
    """Plots the results of a hyperparameter sweep, including various metrics and heatmaps."""

    metrics_to_plot = [
        ("mean_lyap", "Mean Lyapunov Exponent", "Mean Lyapunov Exponent"),
        ("r2_lyap", "Lyapunov R2", "R2 Value"),
        ("mean_cos", "Mean Cosine Distance", "Mean Cosine Distance"),
        ("std_cos", "Std Dev Cosine Distance", "Std Dev Cosine Distance"),
        ("mean_dtw_fast", "Mean DTW Distance", "Mean DTW Distance"),
        ("std_dtw_fast", "Std Dev DTW Distance", "Std Dev DTW Distance"),
        ("mean_hausdorff", "Mean Hausdorff Distance", "Mean Hausdorff Distance"),
        ("std_hausdorff", "Std Dev Hausdorff Distance", "Std Dev Hausdorff Distance"),
        ("mean_frechet", "Mean Frechet Distance", "Mean Frechet Distance"),
        ("std_frechet", "Std Dev Frechet Distance", "Std Dev Frechet Distance"),
    ]

    r_values = sorted(sweep_results_df['r'].unique())
    shift_values = sorted(sweep_results_df['shift'].unique())

    # Plot each metric vs r, colored by shift
    for col_name, title_prefix, ylabel in metrics_to_plot:
        if col_name in sweep_results_df.columns:
            plt.figure(figsize=(10, 7))
            for shift_val in shift_values:
                subset = sweep_results_df[sweep_results_df['shift'] == shift_val]
                plt.plot(subset['r'], subset[col_name], marker='o', label=f'Shift: {shift_val}')
            plt.xlabel('PCA Dimension (r)')
            plt.ylabel(ylabel)
            plt.title(f'{title_prefix} vs. PCA Dimension (r)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{outpath_prefix}_{col_name}_vs_r.png')
            plt.close()

    # Plot each metric vs shift, colored by r (if multiple shifts)
    if len(shift_values) > 1:
        for col_name, title_prefix, ylabel in metrics_to_plot:
            if col_name in sweep_results_df.columns:
                plt.figure(figsize=(10, 7))
                for r_val in r_values:
                    subset = sweep_results_df[sweep_results_df['r'] == r_val]
                    plt.plot(subset['shift'], subset[col_name], marker='o', label=f'r: {r_val}')
                plt.xlabel('Shift')
                plt.ylabel(ylabel)
                plt.title(f'{title_prefix} vs. Shift')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'{outpath_prefix}_{col_name}_vs_shift.png')
                plt.close()

    # Heatmaps for 2D sweeps (r vs shift)
    for col_name, title_prefix, ylabel in metrics_to_plot:
        if col_name in sweep_results_df.columns:
            # Pivot the DataFrame to get a matrix for heatmap
            pivot_table = sweep_results_df.pivot(index='shift', columns='r', values=col_name)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(pivot_table, cmap='viridis', aspect='auto', origin='lower',
                       extent=[min(r_values), max(r_values), min(shift_values), max(shift_values)])
            plt.colorbar(label=ylabel)
            plt.xlabel('PCA Dimension (r)')
            plt.ylabel('Shift')
            plt.title(f'Heatmap of {title_prefix}')
            plt.xticks(r_values)
            plt.yticks(shift_values)
            plt.tight_layout()
            plt.savefig(f'{outpath_prefix}_{col_name}_heatmap.png')
            plt.close()

def plot_time_series_for_pair(pair_timeseries, outpath):
    """Plots a distance timeseries for a single pair."""
    plt.figure(figsize=(8, 4))
    # Thinner lines, smaller markers, and slight transparency to reduce visual collision
    plt.plot(pair_timeseries, marker='o', markersize=3, linewidth=1.0, alpha=0.75)
    plt.title("Distance Timeseries for a Pair")
    plt.xlabel("Time")
    plt.ylabel("Distance")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()