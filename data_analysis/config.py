# config.py (only file with parameters; no other hard-coded values)
from datetime import timedelta
from dataclasses import dataclass

CONFIG = {
    # IO
    # "input_path": "C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/git_repo/TDK_LLM/runs_aug/run_0_0.0001/sentence-transformers_all-mpnet-base-v2.pt",   # AAA delete
    "run_id_format": "%Y%m%d-%H%M%S",        # used with random suffix
    # "results_root": "results",               # AAA delete

    # Reduction
    "reduction": {
        "methods": ["none"],  # ["pca", "whiten", "autoencoder"]
        "default_method": "pca",
        "pca": {
            "enabled": False,
            "r_values": [16, 64],     # sweep list
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
    # Metric selection now uses per-metric 'enabled' flags in the
    # CONFIG['metrics'][<metric>] entries. Do not rely on an
    # explicit 'available' whitelist.
    "save_plots": True,
        "rank_eigen": {
            "enabled": True,
            "deviation_metric": 'rms', # 'sum_cos_dist', # "rms",
            "run_rank_eigenvectors": True
        },
        "default_pairing": "ref0",   # "all" or "ref0"
        "cos": {
            "enabled": False,
            "aggregate": ["mean", "median", "std"],
            "shifts": [0],   # absolute steps to sweep; included in sweep script
            "shift_aggregation": "min", # "min", "mean"
            "default_max_shift": 5,
        },
        "cos_sim": {
            "enabled": False,
            "window_agg": "mean",  # "mean" or "min" aggregation across vectors in window
            "gaussian_weight": False,
            "centric_mode": "a",    # "a", "b", or "both"
            "centric_agg": "mean",  # when both: "mean" or "min"
        },
        "dtw": {
            "enabled": False,
            "use_fastdtw": True, # why is this not being used anywhere ??
        },
        "hausdorff": {
            "enabled": False,
            "symmetric": True, # XX
            "aggregation": "max_of_mean" # "max_of_mean", "mean_of_max"
        },
        "frechet": {
            "enabled": False,
            "discrete": True # XX
        },
        "cross_corr": {
            "enabled": False,
            "correlation_type": "pearson" # "pearson" or "spearman"
        },
        "cross_cos": {
            "enabled": False,
        },
        "wasserstein": {
            "enabled": False
        }
    },

    # Unified sliding-window parameters used by metrics that support sliding analysis
    "sliding_window": {
        "use_window": True,
        "window_size": 16,
        "displacement": 16
    },

    # Pair computations
    "pairwise": {
        "compute_all_pairs": False,
        # Single, simple toggle: when True, save one aggregated per-metric
        # timeseries file containing all pairs. No per-pair files are written.
        "save_pairwise_aggregated": True,
        "reference_index": None,
        # If compute_all_pairs is False and reference_index is None,
        # use this explicit list of index pairs for computation and plotting.
        # "pairs_to_plot": [[0, 1]],
        "pairs_to_plot": [[0, 1], [0,2], [2,3]],
    },

    # Lyapunov (fast pairwise slope)
    "lyapunov": {
    "enabled": False,
        "method": "pairwise_slope",
        "operation_mode": "fit_first",  # "fit_first" or "average_first"
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
            "log_plot": False,
            "show_fit": True
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
        "save_timeseries": True,
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
        "hidden_states_cond_{i}_layer_-1",
    ],
    # When in per_trajectory mode, input_template is used to build filenames relative to run folder.
    # Example: "{embed}.pt" or "hidden_states_cond_{i}_layer_-1.pt"
    "input_template": "hidden_states_cond_{i}_layer_-1.pt"
}
