import os
import glob
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_perturbation_distances(results_dir):
    """
    Reads the .pt hidden state tensors from the given results directory,
    calculates the L2 distance between the base unperturbed state and all
    perturbed states, and generates clear plots.
    """
    config_path = os.path.join(results_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json found in {results_dir}")
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # Find all layer files
    layer_files = glob.glob(os.path.join(results_dir, "*.pt"))
    layer_indices = []
    
    # Extract layer numbers
    for f in layer_files:
        basename = os.path.basename(f)
        # e.g., single_token_perturb_all_r0.00035_layer_12.pt
        try:
            layer_str = basename.split("_layer_")[-1].replace(".pt", "")
            layer_indices.append(int(layer_str))
        except Exception:
            pass
            
    layer_indices = sorted(layer_indices)
    if not layer_indices:
        print(f"No .pt files found in {results_dir}")
        return
        
    num_layers = len(layer_indices)
    
    # We will compute the average L2 norm (Euclidean distance) between the unperturbed
    # trajectory (index 0) and all perturbed trajectories (indices 1 to N).
    mean_distances = []
    std_distances = []
    
    for layer in layer_indices:
        # Load file
        # Format is {run_name}_layer_{layer}.pt, but we can just use glob matching
        file_matches = glob.glob(os.path.join(results_dir, f"*_layer_{layer}.pt"))
        if not file_matches:
            continue
            
        states = torch.load(file_matches[0], weights_only=True) # shape: [n_conditions, seq_len, hidden_size]
        
        # states[0] is the baseline unperturbed state
        base_state = states[0] # [seq_len, hidden_size]
        perturbed_states = states[1:] # [n_conditions-1, seq_len, hidden_size]
        
        if perturbed_states.shape[0] == 0:
            print("Only one condition found (no perturbations). Cannot compute distances.")
            return
            
        # Calculate L2 distance across the hidden size dimension
        # Shape becomes [n_conditions-1, seq_len]
        diffs = perturbed_states - base_state.unsqueeze(0)
        distances = torch.norm(diffs, p=2, dim=-1)
        
        # Average distance across sequence length and perturbed conditions
        # To get a single scalar representing the spread at this layer
        mean_dist = distances.mean().item()
        std_dist = distances.std().item()
        
        mean_distances.append(mean_dist)
        std_distances.append(std_dist)
        
    # Ensure plots directory exists
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Distance vs Layer
    plt.figure(figsize=(10, 6))
    
    layer_arr = np.array(layer_indices)
    mean_arr = np.array(mean_distances)
    std_arr = np.array(std_distances)
    
    plt.plot(layer_arr, mean_arr, marker='o', color='purple', label="Mean L2 Distance")
    plt.fill_between(layer_arr, mean_arr - std_arr, mean_arr + std_arr, color='purple', alpha=0.2)
    
    plt.title(f"Perturbation Divergence across Layers\n{config['setup']['name']} (Radius: {config['radius']})", fontsize=14)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("L2 Distance from Base State", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(plots_dir, "divergence_over_layers.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to: {plot_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot perturbation data from saved states.")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to the directory containing .pt files and config.json")
    
    args = parser.parse_args()
    plot_perturbation_distances(args.results_dir)

if __name__ == "__main__":
    main()
