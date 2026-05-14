import os
import glob
import torch

results_dir = "./results_perturbations"

# Find one valid subdirectory
subdirs = [d for d in glob.glob(os.path.join(results_dir, "*")) 
           if os.path.isdir(d) and os.path.basename(d) != "aggregated_plots"]

if not subdirs:
    print("No subdirectories found.")
    exit()

# Pick the first one
subdir = subdirs[0]
print(f"Checking files in: {subdir}")

# Get all .pt files
files = glob.glob(os.path.join(subdir, "*.pt"))
layer_files = {}
for f in files:
    try:
        layer_idx = int(os.path.basename(f).split("_layer_")[-1].replace(".pt", ""))
        layer_files[layer_idx] = f
    except ValueError:
        pass

# Print the shape of each layer's tensor
for layer_idx in sorted(layer_files.keys()):
    f = layer_files[layer_idx]
    tensor = torch.load(f, weights_only=True)
    print(f"Layer {layer_idx:>2}: Shape = {tensor.shape} | dtype = {tensor.dtype}")
