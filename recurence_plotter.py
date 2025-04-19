import matplotlib.pyplot as plt
import torch
import os
import pickle



radiuses = [0.01, 0.02, 0.04]
temps = [0, 0.6]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pickle(name):
    with open(os.path.join(RESULTS_DIR, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)

def recurrence_plot_with_threshold(trajectories, output_dir="./"):
    
    for traj_idx, traj in enumerate(trajectories[:10]):
        
        ### DISTANCE WITH ITSELF (typical recurrence plot)

        # cosine 
        traj_normalized = traj / torch.norm(traj, dim=1, keepdim=True)
        cosine_sim = traj_normalized @ traj_normalized.T  # (n, n)
        distances = 1 - cosine_sim

        distances = distances / torch.max(distances)

        plt.figure(figsize=(10, 8))
        plt.title(f"Distance Plot (Trajectory {traj_idx})")
        plt.xlabel("Time Index")
        plt.ylabel("Time Index")
        plt.imshow(distances.cpu().numpy(), cmap='coolwarm', aspect='equal', vmin=-1, vmax=1, origin='lower')
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, f"recurrence_traj_{traj_idx}_cos_distance.png"))
        plt.close()

        for threshold in [0.3]:
            recurrence = distances < threshold
            # Plot recurrence plot
            plt.figure(figsize=(10, 8))
            plt.imshow(recurrence.cpu().numpy(), cmap='binary', origin='lower')
            plt.title(f"Recurrence Plot (Trajectory {traj_idx})\n Threshold: {threshold}")
            plt.xlabel("Time Index")
            plt.ylabel("Time Index")
            plt.colorbar(label="Recurrence")
            plt.savefig(os.path.join(output_dir, f"recurrence_traj_{traj_idx}_cos_threshold_{threshold}.png"))
            plt.close()

            
        ### DISTANCE WITH THE NEXT TRAJECTORY (NOT a recurrence plot)

        # cosine 
        traj_normalized = traj / torch.norm(traj, dim=1, keepdim=True)
        traj_normalized1 = trajectories[traj_idx] / torch.norm(trajectories[traj_idx], dim=1, keepdim=True)
        traj_normalized2 = trajectories[traj_idx+1] / torch.norm(trajectories[traj_idx+1], dim=1, keepdim=True)
        cosine_sim = traj_normalized1 @ traj_normalized2.T  
        distances = 1 - cosine_sim

        plt.figure(figsize=(10, 8))
        plt.title(f"Distance Plot (Trajectory {traj_idx} vs {traj_idx+1})")
        plt.xlabel("Time Index")
        plt.ylabel("Time Index")
        plt.imshow(distances.cpu().numpy(), cmap='coolwarm', aspect='equal', vmin=-1, vmax=1, origin='lower')
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, f"consecutive_recurrence_traj_{traj_idx}_{traj_idx+1}_cos_distance.png"))
        plt.close()

        for threshold in [0.4]:
            recurrence = distances < threshold
            # Plot recurrence plot
            plt.figure(figsize=(10, 8))
            plt.imshow(recurrence.cpu().numpy(), cmap='binary', origin='lower')
            plt.title(f"Recurrence Plot (Trajectory {traj_idx} vs {traj_idx+1})\n Threshold: {threshold}")
            plt.xlabel("Time Index")
            plt.ylabel("Time Index")
            plt.colorbar(label="Recurrence")
            plt.savefig(os.path.join(output_dir, f"consecutive_recurrence_traj_{traj_idx}_{traj_idx+1}_cos_threshold_{threshold}.png"))
            plt.close()




for rrr in radiuses:
    for ttt in range(len(temps)):
        TEMPERATURE = temps[ttt]
        RADIUS_INITIAL_CONDITIONS = rrr
        RESULTS_DIR = f"./launch_pentek_{TEMPERATURE}_{RADIUS_INITIAL_CONDITIONS}/"
        PLOTS_DIR = f"./recurrence_plots/launch_pentek_{TEMPERATURE}_{RADIUS_INITIAL_CONDITIONS}/"
        os.makedirs(PLOTS_DIR, exist_ok=True)

        trajectories = load_pickle("trajectories")  
        trajectories = trajectories[:11]
        recurrence_plot_with_threshold(trajectories, output_dir=PLOTS_DIR)


