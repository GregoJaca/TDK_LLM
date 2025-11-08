import os
import json
import torch
import matplotlib.pyplot as plt

# Configuration
RESULTS_ROOT = "results_pecs"
PLOTS_ROOT = os.path.join(RESULTS_ROOT, "plots_UJ")
SELECTED_LAYERS = ["first", "middle", "last"]
THRESHOLD = 0.3

# Ensure plots directory exists
os.makedirs(PLOTS_ROOT, exist_ok=True)

# Process each prompt folder
for prompt_name in os.listdir(RESULTS_ROOT):
    prompt_dir = os.path.join(RESULTS_ROOT, prompt_name)
    if not os.path.isdir(prompt_dir) or prompt_name == "plots":
        continue

    # # Load result.json
    # result_file = os.path.join(prompt_dir, "result.json")
    # with open(result_file) as f:
    #     result = json.load(f)
    # prompt_text = result['prompt']
    # generated_text = result['generated_text']

    # (No Word output) -- processing prompt metadata available in variables if needed

    # For each selected layer, only compare the layer with itself and use cosine distance
    for layer in SELECTED_LAYERS:
        cos_name = f"cosine_sim_{layer}_{layer}.pt"
        cos_path = os.path.join(prompt_dir, cos_name)

        if not os.path.exists(cos_path):
            print(f"Missing cosine similarity file: {cos_path}, skipping")
            continue

        # Load cosine similarity and compute cosine distance
        cos_sim = torch.load(cos_path)
        # Ensure tensor is float
        cos_sim = cos_sim.float()
        cos_distance = 1.0 - cos_sim

        # Normalize distances to [0,1] if possible
        maxval = float(cos_distance.max().item()) if cos_distance.numel() > 0 else 0.0
        if maxval > 0:
            distances = cos_distance / maxval
        else:
            distances = cos_distance.clone()

        # Save the distance matrix
        # dist_fname = f"{prompt_name}_cos_distance_{layer}.pt"
        # dist_path = os.path.join(PLOTS_ROOT, dist_fname)
        # torch.save(distances.cpu(), dist_path)

        # Plot distance matrix
        plt.figure(figsize=(6, 6))
        plt.title(f"Cosine Distance")
        plt.xlabel("Token Index")
        plt.ylabel("Token Index")
        im = plt.imshow(distances.cpu().numpy(), origin='lower', aspect='equal')
        plt.colorbar(im, label="Cosina Distance")
        dist_plot = os.path.join(PLOTS_ROOT, f"{prompt_name}_cos_distance_{layer}_{prompt_name}.png")
        plt.savefig(dist_plot, dpi=300)
        plt.close()

        # Compute recurrence matrix using threshold
        recurrence = distances < THRESHOLD

        # Save recurrence matrix
        # rec_fname = f"{prompt_name}_recurrence_{layer}_thr_{THRESHOLD}.pt"
        # rec_path = os.path.join(PLOTS_ROOT, rec_fname)
        # torch.save(recurrence.cpu(), rec_path)

        # Plot recurrence matrix
        plt.figure(figsize=(6, 6))
        plt.title(f"Recurrence Plot - threshold={THRESHOLD}")
        plt.xlabel("Token Index")
        plt.ylabel("Token Index")
        plt.imshow(recurrence.cpu().numpy(), cmap='binary', origin='lower')
        plt.colorbar(label="Recurrence")
        rec_plot = os.path.join(PLOTS_ROOT, f"{prompt_name}_recurrence_{layer}_thr_{THRESHOLD}_{prompt_name}.png")
        plt.savefig(rec_plot, dpi=300)
        plt.close()

    print(f"Plots and matrices saved to: {PLOTS_ROOT}")

