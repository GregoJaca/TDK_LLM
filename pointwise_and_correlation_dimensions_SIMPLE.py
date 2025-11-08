import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import json
import torch

def calculate_correlation_dimension(distance_matrix: np.ndarray, 
                                   num_thresholds: int = 20,
                                   min_percent: float = 0.05,
                                   max_percent: float = 0.95,
                                   first_n_exclude: int = 3,
                                   last_n_exclude: int = 3):
    """
    Calculate correlation dimension from a Distance matrix.
    
    Args:
        distance_matrix: NxN matrix of pairwise cosine similarities between points
        num_thresholds: Number of threshold values to use
        min_percent: Minimum Distance threshold
        max_percent: Maximum Distance threshold
        first_n_exclude: Number of points to exclude from beginning when fitting
        last_n_exclude: Number of points to exclude from end when fitting
    
    Returns:
        Correlation dimension value
    """
    N = distance_matrix.shape[0]
    assert distance_matrix.shape == (N, N), "Distance matrix must be square"
    
    thresholds = np.logspace(np.log10(min_percent), np.log10(max_percent), num_thresholds)
    
    correlation_sums = []
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    for eps in thresholds:
        recurrence_matrix = distance_matrix <= eps # 
        # Calculate correlation sum (sum of all recurrence points divided by N²)
        correlation_sum = (torch.sum(recurrence_matrix).item()-N) / (N * (N-1)) # -N is to not count the diagonal (so dont count a point with itself) 
        correlation_sums.append(correlation_sum)
    
    correlation_sums = np.array(correlation_sums)
    
    # Plot correlation sum vs threshold (log-log)
    with np.errstate(divide='ignore', invalid='ignore'):  # Handle log(0) cases
        log_thresholds = np.log(thresholds)
        log_correlation_sums = np.log(correlation_sums)
    # Remove any inf or nan values for the plot
    valid_indices = np.isfinite(log_thresholds) & np.isfinite(log_correlation_sums)
    plot_thresholds = thresholds[valid_indices]
    plot_log_corr_sums = log_correlation_sums[valid_indices]
    ax.loglog(plot_thresholds, np.exp(plot_log_corr_sums), 'go-', linewidth=2)
    ax.set_xlabel('log(Distance Threshold)')
    ax.set_ylabel('log(Correlation Sum)')
    ax.set_title('Correlation Dimension (log-log scale)')
    # Add padding to y-axis for log-log plot
    log_y_min_corr = min(np.exp(plot_log_corr_sums)) * 0.3
    log_y_max_corr = max(np.exp(plot_log_corr_sums)) * 1.5
    ax.set_ylim(log_y_min_corr, log_y_max_corr) # GGG
    
    # Line fitting removed per request. Return NaN for the correlation slope.
    correlation_slope = float('nan')
    
    # with open('thresholds.npy', 'wb') as f:
    #     np.save(f, thresholds)
    # with open('correlation_sums.npy', 'wb') as f:
    #     np.save(f, correlation_sums)


    # results = {
    #     "correlation_dimension": correlation_slope,
    #     "r_value": r_value,
    #     "p_value": p_value,
    #     "std_err": std_err,
    #     "intercept": correlation_intercept
    # }
    # with open('dimension_results_L2.json', 'w') as f:
    #     json.dump(results, f, indent=4)

        
    # (Line fitting removed) finalize plot and save
    plt.tight_layout()
    plt.savefig("cosine_sim_first_first_T06.png")
    
    return correlation_slope


if __name__ == "__main__":

# temperature
    # distance_matrix = 1-torch.load("C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/code_LLM/results_pecs_temperature_0-6/interstellar_propulsion_review/cosine_sim_last_last.pt")
    # correlation_dim = calculate_correlation_dimension(
    #     distance_matrix,
    #     num_thresholds=60,
    #     min_percent=0.00008,
    #     max_percent=torch.max(distance_matrix)*1.1,
    #     first_n_exclude=0,
    #     last_n_exclude=27
    # )

    distance_matrix = 1-torch.load("C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/code_LLM/results_pecs/childhood_personality_development/cosine_sim_first_first.pt")
    correlation_dim = calculate_correlation_dimension(
        distance_matrix,
        num_thresholds=40,
        min_percent=0.09,
        max_percent=torch.max(distance_matrix)*1.1,
        first_n_exclude=1,
        last_n_exclude=1
    )

    
    # distance_matrix = 1-torch.load("C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/code_LLM/results_pecs/quantum_consciousness_hallucination/cosine_sim_last_last.pt")
    # correlation_dim = calculate_correlation_dimension(
    #     distance_matrix,
    #     num_thresholds=50,
    #     min_percent=0.006,
    #     max_percent=torch.max(distance_matrix)*1.8,
    #     first_n_exclude=3,
    #     last_n_exclude=9
    # )

    # distance_matrix = 1-torch.load("C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/code_LLM/results_pecs/quantum_consciousness_hallucination/cosine_sim_first_first.pt")
    # correlation_dim = calculate_correlation_dimension(
    #     distance_matrix,
    #     num_thresholds=40,
    #     min_percent=0.2,
    #     max_percent=torch.max(distance_matrix)*1.1,
    #     first_n_exclude=19,
    #     last_n_exclude=10
    # )

    # # random cos
    # N = 100000
    # x = torch.randn(N, 1536)
    # x = torch.nn.functional.normalize(x, p=2, dim=1)
    # cos_sim = torch.matmul(x.T, x)
    # distance_matrix = 1-cos_sim
    # correlation_dim = calculate_correlation_dimension(
    #     distance_matrix,
    #     num_thresholds=30,
    #     min_percent=0.01,
    #     max_percent=2,#torch.max(distance_matrix)*1.1,
    #     first_n_exclude=12,
    #     last_n_exclude=5
    # ) 

    # # random cos
    # N = 2000
    # x = torch.randn(N, 153)
    # x = torch.nn.functional.normalize(x, p=2, dim=1)
    # cos_sim = torch.matmul(x, x.T)
    # distance_matrix = 1-cos_sim
    # print(cos_sim.shape)
    # print(torch.min(distance_matrix), torch.max(distance_matrix))
    # correlation_dim = calculate_correlation_dimension(
    #     distance_matrix,
    #     num_thresholds=30,
    #     min_percent=0.9,
    #     max_percent=1.25,#torch.max(distance_matrix)*1.1,
    #     first_n_exclude=0,
    #     last_n_exclude=25
    # ) 

    #     # random cos
    # N = 10000
    # # x = torch.randn(N, 536)
    # x = torch.randn(N, 1536)
    # x = torch.nn.functional.normalize(x, p=2, dim=1)
    # cos_sim = torch.matmul(x, x.T)
    # distance_matrix = 1-cos_sim
    # print(cos_sim.shape)
    # print(torch.min(distance_matrix), torch.max(distance_matrix))
    # correlation_dim = calculate_correlation_dimension(
    #     distance_matrix,
    #     num_thresholds=60,
    #     min_percent=0.3,
    #     max_percent=1.15,#torch.max(distance_matrix)*1.1,
    #     first_n_exclude=0,
    #     last_n_exclude=10
    # ) 

    # # random euclidean
    # N = 2000
    # x = torch.randn(N, 53)
    # x = torch.nn.functional.normalize(x, p=2, dim=1)
    # distance_matrix = torch.cdist(x, x, p=2)
    # correlation_dim = calculate_correlation_dimension(
    #     distance_matrix,
    #     num_thresholds=20,
    #     min_percent=1,
    #     max_percent=torch.max(distance_matrix)*1.5,
    #     first_n_exclude=0,
    #     last_n_exclude=16
    # )

    # # random euclidean
    # N = 2000
    # x = torch.randn(N, 53)
    # x = torch.nn.functional.normalize(x, p=2, dim=1)
    # distance_matrix = torch.cdist(x, x, p=2)
    # correlation_dim = calculate_correlation_dimension(
    #     distance_matrix,
    #     num_thresholds=60,
    #     min_percent=0.1,
    #     max_percent=torch.max(distance_matrix)*1.3,
    #     first_n_exclude=0,
    #     last_n_exclude=5
    # )

    # # random euclidean
    # N = 10000
    # x = torch.randn(N, 1536)
    # x = torch.nn.functional.normalize(x, p=2, dim=1)
    # distance_matrix = torch.cdist(x, x, p=2)
    # correlation_dim = calculate_correlation_dimension(
    #     distance_matrix,
    #     num_thresholds=20,
    #     min_percent=0.5,
    #     max_percent=torch.max(distance_matrix)*5,
    #     first_n_exclude=0,
    #     last_n_exclude=0
    # )

    # distance_matrix = 1-torch.load("C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/code_LLM/results_pecs/childhood_personality_development/cosine_sim_last_last.pt")
    # correlation_dim = calculate_correlation_dimension(
    #     distance_matrix,
    #     num_thresholds=50,
    #     min_percent=0.006,
    #     max_percent=torch.max(distance_matrix)*1.6,
    #     first_n_exclude=4,
    #     last_n_exclude=10
    # )

    # distance_matrix = 1-torch.load("C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/code_LLM/results_pecs/childhood_personality_development/cosine_sim_first_first.pt")
    # correlation_dim = calculate_correlation_dimension(
    #     distance_matrix,
    #     num_thresholds=40,
    #     min_percent=0.2,
    #     max_percent=torch.max(distance_matrix)*1.1,
    #     first_n_exclude=18,
    #     last_n_exclude=9
    # )

    # ---------------------

    # distance_matrix = torch.load("./l2_dist_last_last.pt")

    # distance_matrix = 1-torch.load("C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/code_LLM/results_pecs/interstellar_propulsion_review/cosine_sim_last_last.pt")
    # correlation_dim = calculate_correlation_dimension(
    #     distance_matrix,
    #     num_thresholds=40,
    #     min_percent=0.008,
    #     max_percent=0.05, #torch.max(distance_matrix)*1.1,
    #     first_n_exclude=10,
    #     last_n_exclude=15
    # )

    # distance_matrix = 1-torch.load("C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/code_LLM/results_pecs/interstellar_propulsion_review/cosine_sim_first_first.pt")
    # correlation_dim = calculate_correlation_dimension(
    #     distance_matrix,
    #     num_thresholds=40,
    #     min_percent=0.2,
    #     max_percent=torch.max(distance_matrix)*1.1,
    #     first_n_exclude=18,
    #     last_n_exclude=10
    # )
















    # distance_matrix = torch.load("C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/code_LLM/results_pecs/interstellar_propulsion_review/l2_dist_last_last.pt")
    # correlation_dim = calculate_correlation_dimension(
    #     distance_matrix,
    #     num_thresholds=40,
    #     min_percent=10,
    #     max_percent=torch.max(distance_matrix)*1.1,
    #     first_n_exclude=1,
    #     last_n_exclude=1
    # )

    # distance_matrix = torch.load("C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/code_LLM/results_pecs/interstellar_propulsion_review/l2_dist_first_first.pt")
    # correlation_dim = calculate_correlation_dimension(
    #     distance_matrix,
    #     num_thresholds=30,
    #     min_percent=0.6,
    #     max_percent=torch.max(distance_matrix),
    #     first_n_exclude=15,
    #     last_n_exclude=6
    # )

    
