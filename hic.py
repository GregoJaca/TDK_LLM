import os
import pickle
import torch
import matplotlib.pyplot as plt

# Configuration (matches the style used in plotter.py for outputs)
radiuses = [0.02, 0.04]
temps = [0]
THRESHOLDS = [0.3]
MAX_TRAJ_SAVE = 10  # process up to first 10 trajectories (matches example)


def load_pickle(results_dir, name):
    p = os.path.join(results_dir, f"{name}.pkl")
    with open(p, "rb") as f:
        return pickle.load(f)


def safe_normalize(x, dim=1, eps=1e-12):
    # x: (N, D) -> normalize rows
    norms = torch.norm(x, dim=dim, keepdim=True)
    norms = torch.where(norms <= eps, torch.ones_like(norms), norms)
    return x / norms


def process_trajectories(trajectories, output_dir, thresholds=THRESHOLDS):
    os.makedirs(output_dir, exist_ok=True)

    # limit number of trajectories for plotting (same as your example)
    trajectories = trajectories[: MAX_TRAJ_SAVE + 1]

    for traj_idx, traj in enumerate(trajectories[:MAX_TRAJ_SAVE]):
        # Expect traj shaped (T, D)
        if not isinstance(traj, torch.Tensor):
            traj = torch.tensor(traj)

        if traj.ndim != 2:
            print(f"Skipping traj {traj_idx}: expected 2D tensor, got shape {traj.shape}")
            continue

        # Compute cosine similarity (self vs self)
        traj_norm = safe_normalize(traj, dim=1)
        cosine_sim = traj_norm @ traj_norm.T  # (T, T)

        # Compute cosine distance: 1 - similarity
        cos_distance = 1.0 - cosine_sim

        # Normalize distances to [0,1] if possible
        if cos_distance.numel() > 0:
            maxval = float(cos_distance.max().item())
        else:
            maxval = 0.0

        if maxval > 0:
            distances = cos_distance / maxval
        else:
            distances = cos_distance.clone()

        # # Plot distance matrix (style similar to plotter.py)
        # plt.figure(figsize=(6, 6))
        # plt.title(f"Cosine Distance (Trajectory {traj_idx})")
        # plt.xlabel("Token Index")
        # plt.ylabel("Token Index")
        # im = plt.imshow(distances.cpu().numpy(), origin='lower', aspect='equal')
        # plt.colorbar(im, label="Cosina Distance")
        # dist_plot = os.path.join(output_dir, f"traj_{traj_idx}_cos_distance.png")
        # plt.savefig(dist_plot, dpi=300)
        # plt.close()

        # Compute and plot recurrence matrices for each threshold
        for thr in thresholds:
            recurrence = distances < thr

            # Plot recurrence (use same labels as plotter.py)
            plt.figure(figsize=(6, 6))
            plt.title(f"Recurrence Plot - threshold={thr}")
            plt.xlabel("Token Index")
            plt.ylabel("Token Index")
            plt.imshow(recurrence.cpu().numpy(), cmap='binary', origin='lower')
            plt.colorbar(label="Recurrence")
            rec_plot = os.path.join(output_dir, f"traj_{traj_idx}_recurrence_thr_{thr}.png")
            plt.savefig(rec_plot, dpi=300)
            plt.close()

        print(f"Saved traj {traj_idx} plots in {output_dir}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for rrr in radiuses:
        for ttt in range(len(temps)):
            TEMPERATURE = temps[ttt]
            RADIUS_INITIAL_CONDITIONS = rrr
            RESULTS_DIR = f"./launch_pentek_{TEMPERATURE}_{RADIUS_INITIAL_CONDITIONS}/"
            PLOTS_DIR = f"./recurrence_plots/launch_pentek_{TEMPERATURE}_{RADIUS_INITIAL_CONDITIONS}/"
            os.makedirs(PLOTS_DIR, exist_ok=True)

            # load trajectories.pkl from the results directory
            try:
                trajectories = load_pickle(RESULTS_DIR, "trajectories")
            except FileNotFoundError:
                print(f"No trajectories.pkl in {RESULTS_DIR}, skipping")
                continue

            # If trajectories are Python lists, convert items to tensors lazily in process function
            # Restrict to first 11 to mirror your example
            trajectories = trajectories[:11]
            process_trajectories(trajectories, output_dir=PLOTS_DIR, thresholds=THRESHOLDS)
