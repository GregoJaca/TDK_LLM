# STATUS: PARTIAL
import matplotlib.pyplot as plt
import numpy as np
from config import CONFIG

def plot_pairwise_distance_distribution(aggregates, outpath, metric_name="cos", aggregate_type="mean"):
    """Plots a histogram of the distances for a specific metric and aggregate type."""
    # Respect global plot saving configuration
    if not CONFIG["plots"].get("save_histograms", True):
        return

    # Collect numeric values robustly: some entries are scalars, others arrays
    distances = []
    for pair, data in aggregates["pairs"].items():
        if metric_name in data and aggregate_type in data[metric_name]:
            val = data[metric_name][aggregate_type]
            if val is None:
                continue
            if isinstance(val, (list, tuple, np.ndarray)):
                try:
                    flat = np.ravel(val)
                    for v in flat:
                        distances.append(float(v))
                except Exception:
                    # fallback: skip malformed entries
                    continue
            else:
                try:
                    distances.append(float(val))
                except Exception:
                    continue

    if not distances:
        # print(f"No data to plot for pairwise distance distribution for metric '{metric_name}' and aggregate type '{aggregate_type}'.")
        return

    arr = np.array(distances, dtype=float)
    if arr.size == 0:
        return

    plt.style.use('ggplot')
    plt.figure(figsize=(6, 4))

    # Robust bin selection
    if arr.size < 10:
        # very small sample: plot exact values as counts of unique values
        unique_vals, counts = np.unique(arr, return_counts=True)
        if unique_vals.size == 0:
            return
        # width relative to data span or small fixed width
        span = float(arr.max() - arr.min()) if arr.max() != arr.min() else 1.0
        width = max(1e-6, span * 0.03)
        plt.bar(unique_vals, counts, width=width, alpha=0.75, edgecolor='k')
        # add small jittered markers to show each real data point
        jitter = (np.random.RandomState(0).rand(arr.size) - 0.5) * width * 0.6
        plt.scatter(arr + jitter, np.zeros_like(arr) + -0.03 * counts.max(), marker='|', color='k')
        plt.ylabel(f'Count ({metric_name.capitalize()} Distance)')
        plt.ylim(bottom=-0.06 * counts.max(), top=counts.max() * 1.15)
    else:
        # Freedman–Diaconis style with limits
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = max(1e-6, q75 - q25)
        bin_width = 2 * iqr * (arr.size ** (-1/3))
        if bin_width <= 0:
            bins = min(60, int(np.sqrt(arr.size)))
        else:
            bins = int(max(10, min(100, np.ceil((arr.max() - arr.min()) / bin_width))))

        counts, bin_edges = np.histogram(arr, bins=bins, density=False)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        widths = np.diff(bin_edges)
        plt.bar(centers, counts, width=widths, alpha=0.75, edgecolor='k', align='center')
        plt.ylabel(f'Count ({metric_name.capitalize()} Distance)')
        plt.ylim(bottom=0, top=max(1, counts.max()) * 1.15)

    plt.title(f"Distribution of {aggregate_type.capitalize()} {metric_name.capitalize()} Distances (n={arr.size})")
    plt.xlabel(f"{aggregate_type.capitalize()} {metric_name.capitalize()} Distance")
    plt.ylabel(f"Density ({metric_name.capitalize()} Distance)")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
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

def plot_time_series_for_pair(pair_timeseries, outpath, title="Distance Timeseries for a Pair", ylabel="Distance", x=None):
    """Plots a distance timeseries for a single pair.

    Parameters:
    - pair_timeseries: 1D array of values
    - outpath: file path to save the figure
    - title: plot title
    - ylabel: label for the y-axis
    - x: optional x-values to plot against (same length as pair_timeseries)
    """
    plt.figure(figsize=(8, 4))
    # Use consistent thin line and small markers across all time-series plots
    if x is None:
        plt.plot(pair_timeseries, marker='o', markersize=3, linewidth=1.0, alpha=0.75)
    else:
        plt.plot(x, pair_timeseries, marker='o', markersize=3, linewidth=1.0, alpha=0.75)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_rank_eigen_full(closest_ranks, outpath, traj_indices=None):
    """Scatter plot of eigenvector rank mapping for full trajectories."""
    plt.figure(figsize=(6, 6))
    plt.scatter(range(1, len(closest_ranks) + 1), closest_ranks, marker='o')
    plt.plot([1, len(closest_ranks)], [1, len(closest_ranks)], 'k--', label="Perfect Match")
    xlabel = f"Eigenvector Rank"
    ylabel = f"Rank of Closest Match"
    if traj_indices:
        xlabel = f"Eigenvector Rank (pair {traj_indices})"
        ylabel = f"Rank of Closest Match (pair {traj_indices})"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Eigenvector ranking using Cosine similarity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_rank_eigen_sliding(positions, deviations, outpath, metric_cfg=None):
    """Line plot for sliding-window rank deviation reusing the standard timeseries style."""
    ylabel = 'Deviation from perfect rank'
    if metric_cfg and metric_cfg.get('deviation_metric', 'rms').lower() == 'rms':
        ylabel += ' (RMS)'
    else:
        ylabel += ' (Mean abs)'
    title = 'Sliding-window rank deviation'
    # Reuse the standard time-series plotting function so style (markersize, linewidth) stays consistent
    plot_time_series_for_pair(deviations, outpath, title=title, ylabel=ylabel, x=positions)