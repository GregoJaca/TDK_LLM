# config.py (only file with parameters; no other hard-coded values)
from datetime import timedelta
from dataclasses import dataclass

### /home/grego/LLM/launch_sep/interstellar_propulsion_review_0_0.00035

window = 16

CONFIG = {
    # IO
    "run_id_format": "%Y%m%d-%H%M%S",        # used with random suffix

    # Reduction
    "reduction": {
        "methods": ["none"],  # ["pca", "whiten", "autoencoder"]
        "default_method": "pca",
        "pca": {
            "enabled": False,
            "r_values": [16, 64], # AA now only the first [0] is used
            "explained_variance_thresholds": [0.90, 0.95],
            "use_gpu": False,
        },
        "whiten": {
            "eps": 1e-10,
            "compute_on": "all",   # "all" or "per_trajectory" XX
        },
        "autoencoder": {
            "contract": "module_state_dict",  # A: nn.Module's state_dict
            "latent_fixed": True,
        },
        "compute_on": "per_trajectory"  # "all" or "per_trajectory" XX
    },

    # Metrics
    "metrics": {
    "save_plots": True,
        "rank_eigen": {
            "enabled": False,
            "deviation_metric": 'sum_cos_dist', # 'sum_cos_dist', # "rms", # JJ
            "run_rank_eigenvectors": True,
            "window_size": window, # embed
            "window_size": window, # hidden
            "threshold": 1.2, # XX window normalization currently is wrong, this is for window_size 4
        },
        # "default_pairing": "ref0",   # "all" or "ref0" # AA
        "cos": {
            "enabled": True,
            "aggregate": ["mean", "median", "std"],
            "shifts": [0],   # absolute steps to sweep; included in sweep script
            "shift_aggregation": "mean", # "min", "mean"
            # "default_max_shift": 5,
            "window_size": window, # embed
            "window_size": window, # hidden
            "threshold": 0.25, # 
        },
        
        
        "cross_corr": { # only works with window_size > 1 and is a bit noisy + window_size adds some diagonal distortion to time series distance. rp are cool across different window_size too.
            "enabled": False,
            "correlation_type": "pearson", # "pearson" or "spearman" # JJ almost the same
            "window_size": window, # embed (but for rp 4 is better)
            # "window_size": window1, # hidden
            "threshold": 0.25, # 
        },
        
        

        # Best without sliding window
        "hausdorff": {
            "enabled": False,
            # "symmetric": True, # AA
            "aggregation": "mean_of_max", # "max_of_mean", "mean_of_max" # JJ
            "window_size": window, # embed
            "window_size": window, # hidden
            "threshold": 0.4, # 
        },
        "frechet": {
            "enabled": False,
            # "discrete": True # AA
            "window_size": window, # embed
            "window_size": window, # hidden
            "threshold": 0.4, # 
        },
        "dtw": {
            "enabled": False,
            # "use_fastdtw": True, # AA  this not being used anywhere 
            "window_size": window, # embed
            "window_size": window, # hidden
            "threshold": 0.4, # 
        },
        
        # not so good
        "cross_cos": { # rossz
            "enabled": False,
        },
        "wasserstein": { # maybe the diverging region could be ok, but the saturated region is too noisy. also window size normalization is wrong-. looked at embed only.  
            "enabled": False,
        },
        "cos_sim": { # okaish with window size 1. but then i guess it's same as cos
            "enabled": False,
            "window_agg": "min",  # "mean" or "min" aggregation across vectors in window # JJ
            "gaussian_weight": True,
            "centric_mode": "both",    # "a", "b", or "both"
            "centric_agg": "min",  # when both: "mean" or "min" 
        },
    },

    # Unified sliding-window parameters used by metrics that support sliding analysis
    "sliding_window": {
        "use_window": True,
        "window_size": window,
        "displacement": 1
    },

    # Pair computations
    "pairwise": {
        "compute_all_pairs": True, # AA XX doesnt work
        # Single, simple toggle: when True, save one aggregated per-metric
        # timeseries file containing all pairs. No per-pair files are written.
        "save_pairwise_aggregated": False,
        "reference_index": None,
        # If compute_all_pairs is False and reference_index is None,
        # use this explicit list of index pairs for computation and plotting.
        # "pairs_to_plot": [[0, 1]],
        # "pairs_to_plot": [[0, 1], [0,2]], #[ [i, j] for i in range(5) for j in range(i,10) ],
        "pairs_to_plot": [], 
        # "pairs_to_plot": [ [i, j] for i in range(100) for j in range(i+1,100) ],
        # "pairs_to_plot": [ [i, j] for i in range(4) for j in range(i+1,4) ],
    },

    # Lyapunov (fast pairwise slope)
    "lyapunov": {
    "enabled": True,
        "method": "pairwise_slope",
        "source_metric": "cos",
        "operation_mode": "average_first",  # "fit_first" or "average_first"
        "exclude_saturation": {
            "mode": "exclude_half",  # "none", "exclude_full", "exclude_half"
            "detect_in_log_scale": True,
            "log_eps": 1e-12,
            "debug_plot": False,
            "save_midpoint_distribution": True,
            "baseline_frac": 0.05,  # fraction from start to estimate baseline
            "plateau_frac": 0.2,    # fraction from end to estimate plateau
            "midpoint_frac": 0.5,   # midpoint between baseline and plateau
            "min_points": 5,
        },
        "time_dependent": {
            "enabled": True,
            "mode": "ftle",  # finite-time Lyapunov exponent: lambda(t)=ln(d(t)/d0)/t
            "eps": 1e-12,
            "save_csv": False,
            "align_distance_on_midpoint": False,
            "align_lyapunov_on_midpoint": False,
        },
        "fit": {
            "type": "exponential",  # default fitter type
            "log_offset_eps": 1e-12,
        },
        "linear_window": {
            "auto_detect": True,
            "min_window_len": 5,
            "r2_threshold": 0.85,   # heuristic, not strict
        },
        "initial_time_cutoff_frac": 0.15,  # default region to examine for slope-fitting
        "averaging": {
            "method": "aligned_mean",  # "mean", "median", "aligned_mean"
            "outlier_detection": {
                "method": "no_rise",  # "no_rise" or "threshold"
                "max_baseline_frac": 0.1, # fraction of max to consider baseline
            },
            "align_on": "rise_index", # alignment strategy
            "truncate_saturation": True,
            "saturation_trim_frac": 0.1
        },
        "plot": {
            "save": True,
            "log_plot": True,
            "save_source_distance_timeseries": True,
            "show_fit": False
        }
    },

    # Parallel & performance
    "parallel": {
        "max_workers": 16,
        "use_multiprocessing": False
    },

    # GPU usage
    "gpu": {
        "use_gpu_for_ae": False,
        "use_gpu_for_pca": False,
    },

    # Outputs / saving
    "save": {
        "tensor_names": {
            "pca": "pca_reduced_hidden_states",
            "whiten": "whitened_hidden_states",
            "autoencoder": "ae_reduced_hidden_states"
        },
        "save_format": "pt",
        "save_reduced_tensors": False,
        "tensor_save_format": "pt",
    },
    "save_metrics_json": False,

    # Plotting options
    "plots": {
        "save_histograms": False,
        "save_timeseries": False,
        "plot_rp_threshold": False,
        "rp_threshold": 0.3, # XX GG different for different metric. reuse that for the outlier detector
        "log_plot": True,
        "output_formats": ["png"],  # choose from ["png"], ["pdf"], ["png", "pdf"]
    },

    # Logging
    "logging": {
        "level": "INFO",
        "quiet_during_tasks": True,  # no per-iteration progress bars; print summary after tasks
        "enabled": False
    },

    # Misc
    "random_seed": 42,
    "default_dtype": "float32"
}

# Embedding / input selection: supports two modes and lives inside CONFIG under 'EMBEDDING_CONFIG'
CONFIG['EMBEDDING_CONFIG'] = {
    "input_mode": "per_trajectory",
    "embedding_methods": [
        # examples: "sentence-transformers/all-mpnet-base-v2",
        # For per-trajectory mode provide file-stem templates like
        # "hidden_states_cond_{i}_layer_-1",
        "sentence-transformers_all-mpnet-base-v2_traj{i}",
    ],
    # When in per_trajectory mode, input_template is used to build filenames relative to run folder.
    # Example: "{embed}.pt" or "hidden_states_cond_{i}_layer_-1.pt"
    "input_template": "hidden_states_cond_{i}_layer_-1.pt"
    # "input_template": "sentence-transformers_all-mpnet-base-v2_traj{i}.pt"
    
}
