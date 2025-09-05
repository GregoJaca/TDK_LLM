# STATUS: PARTIAL
import numpy as np
import torch

# Import the metric functions
from src.metrics import cos, dtw_fast, hausdorff, frechet

def generate_test_data(n_traj=2, n_steps=50, n_dims=8):
    """Generates a batch of random walk trajectories."""
    return torch.cumsum(torch.randn(n_traj, n_steps, n_dims), dim=1)

def run_checks():
    """Runs simple checks on the metric functions."""
    print("Generating test data...")
    test_data = generate_test_data()
    traj_a = test_data[0].numpy()
    traj_b = test_data[1].numpy()

    print("\n--- Testing Cosine Distance ---")
    ts, agg = cos.compare_trajectories(traj_a, traj_b)
    print(f"Aggregates: {agg}")
    assert isinstance(agg["mean"], float), "Cosine mean is not a float"
    if ts is not None:
        print(f"Timeseries shape: {ts.shape}")
        assert ts.shape == (traj_a.shape[0],), "Cosine timeseries shape is incorrect"

    print("\n--- Testing DTW Distance ---")
    try:
        ts, agg = dtw_fast.compare_trajectories(traj_a, traj_b)
        print(f"Aggregates: {agg}")
        assert isinstance(agg["dtw_distance"], float), "DTW distance is not a float"
        if ts is not None:
            print(f"Timeseries shape: {ts.shape}")
    except ImportError as e:
        print(f"Skipping DTW test: {e}")

    print("\n--- Testing Hausdorff Distance ---")
    ts, agg = hausdorff.compare_trajectories(traj_a, traj_b)
    print(f"Aggregates: {agg}")
    assert isinstance(agg["hausdorff_distance"], float), "Hausdorff distance is not a float"
    assert ts is None, "Hausdorff should not return a timeseries"

    print("\n--- Testing Fréchet Distance ---")
    ts, agg = frechet.compare_trajectories(traj_a, traj_b)
    print(f"Aggregates: {agg}")
    assert isinstance(agg["frechet_distance"], float), "Fréchet distance is not a float"
    assert ts is None, "Fréchet should not return a timeseries"
    
    print("\nAll checks passed!")

if __name__ == "__main__":
    run_checks()
