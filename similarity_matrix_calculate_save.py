import matplotlib.pyplot as plt
import torch
import os
import pickle




# Simple version: only compute and save cosine similarity matrices for selected trajectories
import torch
import os
import pickle

trajs_to_compute = [1, 6]
RESULTS_DIR = "./launch_pentek_0_0.02/"  # Adjust as needed
OUTPUT_DIR = "./launch_pentek_0_0.02/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_pickle(name):
    with open(os.path.join(RESULTS_DIR, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)

trajectories = load_pickle("trajectories")

for idx in trajs_to_compute:
    traj = trajectories[idx]
    traj_normalized = traj / torch.norm(traj, dim=1, keepdim=True)
    cosine_sim = traj_normalized @ traj_normalized.T  # (n, n)
    # Save the matrix
    out_path = os.path.join(OUTPUT_DIR, f"cos_similarity_traj_{idx}.pt")
    torch.save(cosine_sim, out_path)


