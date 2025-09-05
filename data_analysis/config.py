# config.py (only file with parameters; no other hard-coded values)
from datetime import timedelta
from dataclasses import dataclass


@dataclass
class CrossCosParams:
    use_window: bool = False
    window_size: int = 5
    window_stride: int = 1

CONFIG = {
    # IO
    # "input_path": "C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/code_LLM/new/embedder/outputs/sentence-transformers_all-mpnet-base-v2.pt", # 
    "input_path": "C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/git_repo/TDK_LLM/runs_aug/run_0_0.001/hidden_states_layer_-1.pt",   # path to input tensor (n, T, D)
    "tensor_save_format": "pt",              # "pt" expected
    "run_id_format": "%Y%m%d-%H%M%S",        # used with random suffix
    "results_root": "results",               # root folder for outputs

    # Data shape expectations (informational)
    "expected_shape_min": [16, 3000, 1536],
    "expected_shape_max": [100, 10000, 1536],

    # Reduction
    "reduction": {
        # "methods": ["pca", "whiten", "autoencoder"],  # order of supported techniques
        "methods": ["none"],  # order of supported techniques
        "default_method": "pca",
        "pca": {
            "enabled": False,
            "r_values": [16, 64],     # sweep list
            "explained_variance_thresholds": [0.90, 0.95],
            "use_gpu": False,
        },
        "whiten": {
            "eps": 1e-10,
            "compute_on": "all",   # "all" or "per_trajectory"
        },
        "autoencoder": {
            "contract": "module_state_dict",  # A: nn.Module's state_dict
            "latent_fixed": True,
        },
        "compute_on": "per_trajectory"  # "all" or "per_trajectory" - user choice
    },

    # Metrics
    "metrics": {
        "available": ["cos", "dtw_fast", "hausdorff", "frechet", "cross_cos"],
        "default_pairing": "ref0",   # "all" or "ref0"
        "cos": {
            "enabled": False,
            "aggregate": ["mean", "median", "std"],
            "shifts": [0, 5],   # absolute steps to sweep; included in sweep script
            "default_max_shift": 5,
        },
        "dtw": {
            "enabled": False,
            "use_fastdtw": True,
            "window_sizes": [None, 0.05, 0.20],  # proportion of T; optional exact DTW
        },
        "hausdorff": {
            "enabled": False,
            "symmetric": True
        },
        "frechet": {
            "enabled": False,
            "discrete": True
        }
    },
    # Cross cosine similarity metric settings
    "cross_cos": {
        "enabled": True,
        "use_window": True,
        "window_size": 64,
        "window_stride": 8,
        "params": CrossCosParams(),
    },

    # Pair computations
    "pairwise": {
        "compute_all_pairs": False,  
        "save_all_pair_timeseries": False,
        "reference_index": 0,
        "save_pairwise_timeseries_for": ["ref0"],  # or list of explicit pairs
    },

    # Lyapunov (fast pairwise slope)
    "lyapunov": {
    "enabled": False,
        "method": "pairwise_slope",
        "linear_window": {
            "auto_detect": True,
            "min_window_len": 5,
            "r2_threshold": 0.9,   # heuristic, not strict
        },
        "initial_time_cutoff_frac": 0.25,  # default region to examine for slope-fitting
    },

    


    # Parallel & performance
    "parallel": {
        "max_workers": 8,
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
        "save_reduced_tensors": False
    },
    # Whether to write metrics.json after metric computation
    "save_metrics_json": False,

    # Plotting options
    "plots": {
        # If False, histogram plots for distributions will not be saved
        "save_histograms": False,
    },

    # Logging
    "logging": {
        "level": "INFO",
        "quiet_during_tasks": True  # no per-iteration progress bars; print summary after tasks
    },

    # Misc
    "random_seed": 42,
    "default_dtype": "float32"
}
