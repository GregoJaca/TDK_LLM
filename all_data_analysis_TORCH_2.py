import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import pickle
# import json
from scipy import stats
import torch
from collections import defaultdict



# =============================================================================
# PARAMETERS
# =============================================================================

Save_plots = True  # Toggle this to True to save plots, or False to just show them

# For large plots (entropy, distances, etc.)
PLOT_LIMIT = None  # Set to None to show all, or an integer to limit how many curves to plot XX GG

# Parameters
n_components = 2 # for PCA. If you make it different from 2, then you have to change some of the plots, bocsi. XX
pairs_to_plot = [[1, 2], [1,4], [3,5]] # for some of the plots, we compare pairs of trajectories, but it's a huge mess if you plot all of them XX GG
individuals_to_plot = [0,1,2,3]

SPLIT_K = 0  # Set >0 for split PCA. 
# This is bc I thought that the PCA could be significantly different between the first couple tokens and the rest.
# I'm still not 100 percent sure that I implemented it well for the "Local Dimensionality via SVD" part

# Later we'll look at how many components are needed in the PCA to explain minimum_variance_explanation of the variance.
# this is done with a sliding window. My idea is that towards the end of the trajectory, you 
sliding_window_size = 32
sliding_window_displacement = 32
minimum_variance_explanation = 0.9


radiuses = [0.01, 0.02, 0.04]
temps = [0, 0.6]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pickle(name):
    with open(os.path.join(RESULTS_DIR, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)

def printg(file_ptr, text):
    file_ptr.write(text)
    file_ptr.write("\n")
    
def calculate_displacement_components(trajectory: torch.Tensor):
    """
    trajectory: Tensor of shape [T, D]
    Returns:
        parallel_mags: Tensor of shape [T-1]
        perpendicular_mags: Tensor of shape [T-1]
    """
    displacement = trajectory[1:] - trajectory[:-1]  # [T-1, D]
    current_states = trajectory[1:]  # [T-1, D]

    state_norms = torch.norm(current_states, dim=1) + 1e-8  # [T-1]

    # Dot product between current_state and displacement
    dot_products = torch.sum(current_states * displacement, dim=1)  # [T-1]

    parallel_mags = dot_products / state_norms  # [T-1]
    disp_mags = torch.norm(displacement, dim=1)  # [T-1]
    perpendicular_mags = disp_mags - parallel_mags

    return disp_mags, parallel_mags, perpendicular_mags

def torch_pca_fit(X, n_components):
    X_mean = X.mean(dim=0)
    X_centered = X - X_mean
    U, S, V = torch.pca_lowrank(X_centered, q=n_components)
    explained_var = (S ** 2) / (X.size(0) - 1)
    total_var = explained_var.sum()
    explained_var_ratio = explained_var / total_var
    return {
        "mean": X_mean,
        "components": V[:, :n_components].T,
        "explained_var_ratio": explained_var_ratio[:n_components]
    }

def torch_pca_transform(X, pca_dict):
    return (X - pca_dict["mean"]) @ pca_dict["components"].T

def cosine_similarity_matrix(A, B):
    A_norm = F.normalize(A, p=2, dim=1)
    B_norm = F.normalize(B, p=2, dim=1)
    return A_norm @ B_norm.T

def cosine_similarity(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

### Exponential fitting of axis lengths
def compute_exponential_slopes_with_stats(axis_lengths, n_axis):
    """
    For each of the first `n_axis` components in `axis_lengths`, fit a line to the log-transformed values.
    Returns a list of [slope, 95% CI half-width, R²] for each component.
    """
    axis_array = np.array(axis_lengths.cpu(), dtype=np.float64)
    axis_array = np.nan_to_num(axis_array, nan=1e-8, posinf=1e-8, neginf=1e-8)
    axis_array[axis_array <= 0] = 1e-8  # avoid log(0) and negative values

    logs = np.log(axis_array[:, :n_axis])
    t = np.arange(logs.shape[0])
    results = []

    for i in range(n_axis):
        y = logs[:, i]
        slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
        ci_half_width = 1.96 * std_err  # 95% CI half-width
        r_squared = r_value ** 2
        results.append([slope, ci_half_width, r_squared])

    return results

def compute_pca_eigenvectors(X, k=None):
    X_centered = X - X.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    # Return eigenvectors of covariance matrix
    if k is not None:
        Vh = Vh[:k]
    return Vh  # shape: (k or D, D)

def cosine_sim_matrix(A, B):
    A_norm = A / A.norm(dim=1, keepdim=True)
    B_norm = B / B.norm(dim=1, keepdim=True)
    return A_norm @ B_norm.T  # shape: (len(A), len(B))

def compute_deltas(trajectories, step):
    deltas = []
    for traj in trajectories:
        if traj.shape[0] > step:
            delta = traj[step] - traj[0]  # h_step - h_0
            deltas.append(delta)
    return torch.stack(deltas)  # shape: [N, D]

def cosine_sim_matrix_auto(A):
    A_norm = A / A.norm(dim=1, keepdim=True)
    return A_norm @ A_norm.T  # shape: (N, N)

def euclidean_dist_matrix_auto(A):
    # Efficient Euclidean distance computation using broadcasting
    norm_A = A.norm(dim=1, keepdim=True)
    dist_matrix = norm_A + norm_A.T - 2 * torch.mm(A, A.T)
    return dist_matrix  # shape: (N, N)

# =============================================================================
# MAIN
# =============================================================================

for rrr in radiuses:
    for ttt in range(len(temps)):
        TEMPERATURE = temps[ttt]
        RADIUS_INITIAL_CONDITIONS = rrr
        RESULTS_DIR = f"./launch_pentek_{TEMPERATURE}_{RADIUS_INITIAL_CONDITIONS}/"
        PLOTS_DIR = f"./plots/launch_pentek_{TEMPERATURE}_{RADIUS_INITIAL_CONDITIONS}/"
        os.makedirs(PLOTS_DIR, exist_ok=True)

        f_out_name = os.path.join(PLOTS_DIR, "out.txt")
        f_out_text = open(f_out_name, "w+")
        f_out_text.write("Starting...\n")

        # =============================================================================
        # LOAD DATA
        # =============================================================================


        # the hidden states corresponding to different initial conditions at a given state form a sort of blob.
        # we look at the hypervolume and the largest axes.
        # you can think of it as a 3d ellipsoid and we look at how it deforms, but in higher dimensions.
        hypervolumes = load_pickle("hypervolume")  # tensor of tensor (that contain a single float) hypervolumes enclosed per step or trajectory. 
        axis_lengths = load_pickle("axis_lengths")  # tensor of tensors that contains the length of the k largest axes for each step. 
        # k is a parameter that I pick when I run all the cases. Normally 3 or 4

        ### These below are all lists and every element of the list corresponds to a different initial condtion

        # List of [seq_len, hidden_dim] tensors for each generation. Each tensor contains the hidden state for every generated token at the last layer.
        trajectories = load_pickle("trajectories")  
        # trajectories = trajectories[:10] # XX GG fontos
        minimum_trajectory_len = min( [len(t) for t in trajectories] )




        # =============================================================================
        # PLOT and SAVE
        # =============================================================================

        ### Plot: Hypervolume and Axis Lengths
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(hypervolumes.cpu())
        plt.title("Hypervolumes")
        plt.xlabel("Time step")
        plt.ylabel("Volume")

        plt.subplot(1, 2, 2)
        axis_lengths_np = np.array(axis_lengths.cpu())
        if axis_lengths_np.ndim == 2:
            for i in range(axis_lengths_np.shape[1]):
                plt.plot(axis_lengths_np[:, i], label=f"Axis {i+1}")
            plt.title("Axis Lengths")
            plt.xlabel("Time step")
            plt.ylabel("Length")
            plt.legend()
        else:
            plt.title("Axis Lengths - Format Issue")

        if Save_plots:
            plt.savefig(os.path.join(PLOTS_DIR, "hypervolume_and_axes.png"))
            plt.clf()
        else:
            plt.show()



        # Later we'll look at how many components are needed in the PCA to explain minimum_variance_explanation of the variance.
        # this is done with a sliding window. My idea is that towards the end of the trajectory, you 

        projected_trajectories = []
        projected_first = []
        projected_after = []

        if SPLIT_K == 0:
            # --- Single PCA ---
            all_states = torch.cat(trajectories, dim=0)
            pca = torch_pca_fit(all_states, n_components)

            for traj in trajectories:
                projected_trajectories.append(torch_pca_transform(traj, pca))

            explained_var_ratio = pca["explained_var_ratio"].cpu().numpy()
            printg(f_out_text, f"PCA: Explained var {explained_var_ratio}")


        else:
            # --- Split PCA ---
            all_first_k = torch.cat([t[:SPLIT_K] for t in trajectories if t.size(0) > SPLIT_K], dim=0)
            all_after_k = torch.cat([t[SPLIT_K:] for t in trajectories if t.size(0) > SPLIT_K], dim=0)

            pca_first = torch_pca_fit(all_first_k, n_components)
            pca_after = torch_pca_fit(all_after_k, n_components)

            for traj in trajectories:
                if traj.size(0) <= SPLIT_K:
                    first_part = torch_pca_transform(traj, pca_first)
                    projected_first.append(first_part)
                    projected_trajectories.append(first_part)
                else:
                    first_part = torch_pca_transform(traj[:SPLIT_K], pca_first)
                    second_part = torch_pca_transform(traj[SPLIT_K:], pca_after)
                    full_proj = torch.cat([first_part, second_part], dim=0)
                    projected_first.append(first_part)
                    projected_after.append(second_part)
                    projected_trajectories.append(full_proj)



        # --- Plot projected trajectories ---
        # It's still hard to see the direction of the trajectory XX
        if SPLIT_K == 0:
            plt.figure(figsize=(10, 8))
            for proj in projected_trajectories:
                p0 = proj[:, 0].cpu()
                p1 = proj[:, 1].cpu()
                plt.scatter(p0, p1, s=10, alpha=0.4)
                plt.plot(p0, p1, alpha=0.05)
            
            explained_var_ratio = pca["explained_var_ratio"].cpu().numpy()
            plt.title(f"PCA Projected Trajectories\nExplained var: {explained_var_ratio}")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.grid(True)
            plt.tight_layout()
            if Save_plots:
                plt.savefig(os.path.join(PLOTS_DIR, "pca_no_split.png"))
                plt.clf()
            else:
                plt.show()


            # Plotting individual 
            for ii in individuals_to_plot:
                plt.figure(figsize=(10, 8))
                for proj in projected_trajectories[ii]:
                    p0 = proj[0].cpu()
                    p1 = proj[1].cpu()
                    plt.scatter(p0, p1, s=10, alpha=0.7)
                    # plt.plot(p0, p1, alpha=0.2)
                plt.title(f"PCA Projected Trajectory {ii}")
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plt.grid(True)
                plt.tight_layout()
                if Save_plots:
                    plt.savefig(os.path.join(PLOTS_DIR, f"pca_individual_{ii}.png"))
                    plt.clf()
                else:
                    plt.show()

        else:
            # Plot both projections
            # Note: the two plots are done with different prijections corresponding to different PCAs, so they are in different subspaces
            # Just don't think that one is the continuation of the other
            plt.figure(figsize=(14, 6))

            plt.subplot(1, 2, 1)
            for proj in projected_first:
                p0 = proj[:, 0].cpu()
                p1 = proj[:, 1].cpu()
                plt.scatter(p0, p1, s=10, alpha=0.5)
                plt.plot(p0, p1, alpha=0.1)
            plt.title(f"PCA on First {SPLIT_K} Steps")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.grid(True)

            plt.subplot(1, 2, 2)
            for proj in projected_after:
                p0 = proj[:, 0].cpu()
                p1 = proj[:, 1].cpu()
                plt.scatter(p0, p1, s=10, alpha=0.6)
                plt.plot(p0, p1, alpha=0.5)
            plt.title(f"PCA on Steps After {SPLIT_K}")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.grid(True)

            plt.tight_layout()
            if Save_plots:
                plt.savefig(os.path.join(PLOTS_DIR, f"pca_split_{SPLIT_K}.png"))
                plt.clf()
            else:
                plt.show()






        # --- Local Dimensionality via SVD --- (Basically a PCA on a sliding window)
        # measuring the number of dimensions needed to account for minimum_variance_explanation for each trajectory
        local_dims = []
        for traj_idx, traj in enumerate(trajectories):
            dim_over_time = []
            for i in range(0, traj.size(0) - sliding_window_size + 1, sliding_window_displacement):
                window = traj[i:i + sliding_window_size]  # shape: (W, D)
                window = window - window.mean(dim=0, keepdim=True)
                
                # SVD
                U, S, Vh = torch.linalg.svd(window, full_matrices=False)  # S: (min(W, D),)

                # Variance explained by each component
                var_explained = S**2 / (sliding_window_size - 1)
                cumulative_variance = torch.cumsum(var_explained, dim=0)
                total_variance = cumulative_variance[-1]
                cumulative_ratio = cumulative_variance / total_variance

                # Smallest number of components explaining desired variance
                num_components = torch.searchsorted(cumulative_ratio, minimum_variance_explanation) + 1
                dim_over_time.append(num_components.item())

            local_dims.append(dim_over_time)

        # Plotting local dimensionality
        plt.figure(figsize=(8, 5))

        # If you want to plot the dimension for each individual trajectory GG
        # for idx, dims in enumerate(local_dims):
        #     plt.plot(np.arange(len(dims)) * sliding_window_displacement, dims, label=f"Trajectory {idx}")

        # If you want to average the dimension needed for different trajectories
        time_buckets = defaultdict(list)
        for traj in local_dims:
            for t, val in enumerate(traj):
                time_buckets[t].append(val)
        local_dims_avg = [np.mean(time_buckets[t]) for t in sorted(time_buckets)]

        plt.plot(np.arange(len(local_dims_avg)) * sliding_window_displacement, local_dims_avg, label=f"Average for all trajectories")

        plt.title(f"Local Dimensionality (≥{minimum_variance_explanation * 100:.0f}% Variance)")
        plt.xlabel("Time step (center of window)")
        plt.ylabel("# of Components")
        plt.legend(fontsize="small", loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        if Save_plots:
            plt.savefig(os.path.join(PLOTS_DIR, "local_dimensionality.png"))
            plt.clf()
        else:
            plt.show()



        # --- Rank Eigenvectors of pairs of trajectories ---
        for i, j in pairs_to_plot:
            t1, t2 = trajectories[i], trajectories[j]

            # Clip to same length
            min_len = min(t1.shape[0], t2.shape[0])
            t1, t2 = t1[:min_len], t2[:min_len]

            # Compute PCA eigenvectors
            v1 = compute_pca_eigenvectors(t1)
            v2 = compute_pca_eigenvectors(t2)

            # Plot cross-rank
            # Compute cosine similarity between eigenvectors
            sim_matrix = cosine_sim_matrix(v1, v2)  # [D1, D2]
            closest_ranks = []

            for ii in range(sim_matrix.shape[0]):
                # Rank of trajectory j's vectors by similarity to i-th of traj1
                _, indices = torch.sort(sim_matrix[ii], descending=True)
                # Index of the most similar vector
                closest_idx = indices[0].item()
                closest_ranks.append(closest_idx + 1)  # 1-based indexing

            plt.figure(figsize=(6, 6))
            plt.scatter(range(1, len(closest_ranks) + 1), closest_ranks, marker='o')
            plt.plot([1, len(closest_ranks)], [1, len(closest_ranks)], 'k--', label="Perfect Match")
            plt.xlabel(f"Eigenvector Rank (Trajectory {i})")
            plt.ylabel(f"Rank of Closest Match (Trajectory {j})")
            plt.title(f"Trajectory {i} vs {j}\n Eigenvector ranking using Cosine similarity")
            plt.grid(True)
            plt.legend()

            if Save_plots:
                plt.savefig(os.path.join(PLOTS_DIR, f"rank_eigen_pca_{i}_{j}.png"))
                plt.clf()
            else:
                plt.show()



        # ------ Similarity between first displacement -------

        steps_to_look_at = [1,5,20]
        for ii in steps_to_look_at:
            # --- Compute delta vectors ---
            delta1 = compute_deltas(trajectories, step=ii)
            # --- Cosine similarity matrices ---
            cos_sim_delta1 = cosine_sim_matrix_auto(delta1)
            plt.figure(figsize=(6, 5))
            plt.imshow(cos_sim_delta1.cpu().numpy(), cmap='coolwarm', aspect='equal', vmin=-1, vmax=1)
            plt.colorbar()
            plt.title(f"Cosine Similarity of displacement \nfrom the first hidden state: Δ{ii} = h{ii} - h₀")
            plt.xlabel("Trajectory Index")
            plt.ylabel("Trajectory Index")
            plt.tight_layout()
            if Save_plots:
                plt.savefig(os.path.join(PLOTS_DIR, f"displacement_{ii}_similarity.png"))
                plt.clf()
            else:
                plt.show()




        # ------ Recurrence plot ---------

        # this is not technically a recurrence plot as recurrence plot is binary either it came back or not
        # I plot the distance. Meaning wise, it's equivalent

        for idx in individuals_to_plot:
            auto_distance_matrix = cosine_sim_matrix_auto(trajectories[idx])

            plt.figure(figsize=(6, 5))
            plt.imshow(auto_distance_matrix.cpu().numpy(), cmap='coolwarm', aspect='equal', vmin=-1, vmax=1)
            plt.colorbar()
            plt.title(f"Recurrence plot trajectory {idx}\nCosine Distance")
            plt.xlabel("Generation Step")
            plt.ylabel("Generation Step")
            plt.tight_layout()
            if Save_plots:
                plt.savefig(os.path.join(PLOTS_DIR, f"recurrence_plot_cos_traj_{idx}.png"))
                plt.clf()
            else:
                plt.show()

    
        f_out_text.write("\nFinished.")
        f_out_text.close()




        
        # Below is a graveyard of things I thought could be interesting but weren't. No clear trends, just noise.
        # Hopefully once we reduce the dimensionality of the data, some of these can yield interesting results.




        # ### Uninteresting
        # List (for different initial conditions) of lists (for different steps) that contain the norm of the hidden state at that step. 
        # trajectory_hidden_lengths = load_pickle("trajectory_hidden_lengths")  # List[List[[length]]] per trajectory per step. 
        # entropy_values = load_pickle("entropy_values")  # List of lists of float: entropy per token per trajectory

        # ### Plot: Entropy and Perplexity
        # plt.figure(figsize=(12, 5))
        # plt.subplot(1, 2, 1)
        # for idx, e in enumerate(entropy_values):
        #     if PLOT_LIMIT is not None and idx >= PLOT_LIMIT:
        #         break
        #     plt.plot(e)
        # plt.title("Entropy per Token")
        # plt.xlabel("Token index")
        # plt.ylabel("Entropy")

        # plt.subplot(1, 2, 2)
        # for idx, e in enumerate(entropy_values):
        #     if PLOT_LIMIT is not None and idx >= PLOT_LIMIT:
        #         break
        #     perplexity = [np.exp(val) if val > 0 else 1.0 for val in e]
        #     plt.plot(perplexity)
        # plt.title("Perplexity (exp(Entropy))")
        # plt.xlabel("Token index")
        # plt.ylabel("Perplexity")
        # if Save_plots:
        #     plt.savefig(os.path.join(PLOTS_DIR, "entropy_perplexity.png"))
        #     plt.clf()
        # else:
        #     plt.show()

        # ### Plot: Hidden state vector lengths over time
        # plt.figure(figsize=(8, 5))
        # for idx, traj in enumerate(trajectory_hidden_lengths):
        #     if PLOT_LIMIT is not None and idx >= PLOT_LIMIT:
        #         break
        #     plt.plot([x[0] if isinstance(x, (list, tuple)) else x.item() for x in traj])
        # plt.title("Magnitude of Hidden States")
        # plt.xlabel("Time step")
        # plt.ylabel("L2 Norm")

        # if Save_plots:
        #     plt.savefig(os.path.join(PLOTS_DIR, "trajectory_hidden_lengths.png"))
        #     plt.clf()
        # else:
        #     plt.show()


        # ### Summary results output
        # RESULTS_FILE = os.path.join(RESULTS_DIR, "results_summary.json")
        # with open(RESULTS_FILE, "r") as f:
        #     results = json.load(f)

        # printg(f_out_text, "Lyapunov calculation: ")
        # printg(f_out_text, str(results["lyapunov_exponents"])) # Here I added the lyapunov exponent you get from using the euclidean  and cosine norm. Next time I'll do them separately

        # alternative_lyap = compute_exponential_slopes_with_stats(axis_lengths, 2)
        # printg(f_out_text, "Alternative Lyapunov calculation: ")
        # printg(f_out_text, " [value], [interval_half_width (+-) ], [r2] ")
        # printg(f_out_text, str(alternative_lyap))

        # printg(f_out_text, "Fractal dimension calculation (mean): ")
        # printg(f_out_text, str(np.mean(results["fractal_dimensions"]))) # XX GG mean

        # printg(f_out_text, "Fractal dimension calculation (max): ")
        # printg(f_out_text, str(max(results["fractal_dimensions"]))) # XX GG max






        # # Distances between trajectories (different trajectories correspond to different initial conditions)
        # distances = load_pickle("distances")  # Dict["i-j"] -> { "euclidean": np.array, "cosine": np.array }. 
        # ### Plot: Pairwise Distances (Euclidean + Cosine)
        # plt.figure(figsize=(12, 5))
        # plt.subplot(1, 2, 1)
        # for idx, (key, val) in enumerate(distances.items()):
        #     if PLOT_LIMIT is not None and idx >= PLOT_LIMIT:
        #         break
        #     plt.plot(val["euclidean"], label=key)
        # plt.title("Euclidean Distances Between Trajectories")
        # plt.xlabel("Time step")
        # plt.ylabel("Distance")
        # plt.legend(fontsize="xx-small", loc="upper left")

        # plt.subplot(1, 2, 2)
        # for idx, (key, val) in enumerate(distances.items()):
        #     if PLOT_LIMIT is not None and idx >= PLOT_LIMIT:
        #         break
        #     plt.plot(val["cosine"], label=key)
        # plt.title("Cosine Distances Between Trajectories")
        # plt.xlabel("Time step")
        # plt.ylabel("Distance")
        # plt.legend(fontsize="xx-small", loc="upper left")

        # if Save_plots:
        #     plt.savefig(os.path.join(PLOTS_DIR, "distances_euclidean_cosine.png"))
        #     plt.clf()
        # else:
        #     plt.show()



        ### untested yet SPLIT_K > 0
        # # --- Compare PCA Bases for SPLIT_K ---
        # if SPLIT_K > 0:
        #     similarities = cosine_similarity_matrix(pca_first["components"], pca_after["components"])

        #     explained_var_first = pca_first["explained_var_ratio"]
        #     explained_var_after = pca_after["explained_var_ratio"]

        #     printg(f_out_text, "\n--- PCA Component Cosine Similarities (First vs. After SPLIT_K) ---")
        #     for i in range(n_components):
        #         for j in range(n_components):
        #             printg(f_out_text, f"Cosine Sim PC{i+1}_first vs PC{j+1}_after: {similarities[i, j].item():.4f}")

        #     printg(f_out_text, "\n--- Explained Variance Ratios ---")
        #     for i in range(n_components):
        #         printg(f_out_text, f"PC{i+1}: First = {explained_var_first[i].item():.4f}, After = {explained_var_after[i].item():.4f}")




        # # --- Metrics between pairs in projection ---
        # for pair in pairs_to_plot:
        #     i, j = pair
        #     t1, t2 = projected_trajectories[i], projected_trajectories[j]
        #     min_len = minimum_trajectory_len # XX

        #     cosine_sim = [
        #         F.cosine_similarity(t1[t].unsqueeze(0), t2[t].unsqueeze(0)).item()
        #         for t in range(min_len)
        #     ]

        #     euclidean_dist = torch.norm(t1[:min_len] - t2[:min_len], dim=1).cpu()
        #     curvature = []
        #     for t in range(1, min_len - 1):
        #         a, b, c = t1[t - 1], t1[t], t1[t + 1]
        #         ba = a - b
        #         bc = c - b
        #         cosine = torch.dot(ba, bc) / (torch.norm(ba) * torch.norm(bc) + 1e-8)
        #         cosine = torch.clamp(cosine, -1.0, 1.0)
        #         angle = torch.acos(cosine)
        #         curvature.append(angle.item())
        #     curvature = [0.0] + curvature + [0.0]

        #     plt.figure(figsize=(12, 4))
        #     ax1 = plt.gca()

        #     # Cosine similarity on primary y-axis
        #     line1, = ax1.plot(cosine_sim, label=f"Cosine Sim: {pair}", color='tab:blue', alpha=0.7)
        #     ax1.set_ylabel("Cosine Similarity", color='tab:blue')
        #     ax1.tick_params(axis='y', labelcolor='tab:blue')

        #     # Create a second y-axis for Euclidean distance
        #     ax2 = ax1.twinx()
        #     line2, = ax2.plot(euclidean_dist, label=f"Euclidean Dist: {pair}", color='tab:orange', alpha=0.7)
        #     ax2.set_ylabel("Euclidean Distance", color='tab:orange')
        #     ax2.tick_params(axis='y', labelcolor='tab:orange')

        #     # Optional: Create a third y-axis for curvature
        #     ax3 = ax1.twinx()
        #     ax3.spines["right"].set_position(("outward", 60))  # Offset third axis
        #     line3, = ax3.plot(curvature, label=f"Curvature: {pair}", color='tab:green', alpha=0.7)
        #     ax3.set_ylabel("Curvature", color='tab:green')
        #     ax3.tick_params(axis='y', labelcolor='tab:green')

        #     # Title and x-label
        #     ax1.set_title(f"Pairwise Metrics for Trajectories {pair}")
        #     ax1.set_xlabel("Time step")
        #     ax1.grid(True)

        #     # Combine legends from all axes
        #     lines = [line1, line2, line3]
        #     labels = [line.get_label() for line in lines]
        #     ax1.legend(lines, labels, loc="upper right")

        #     plt.tight_layout()

        #     if Save_plots:
        #         plt.savefig(os.path.join(PLOTS_DIR, f"euclidean_cosine_curvature_in_projection_trajs_{i}_{j}.png"))
        #         plt.clf()
        #     else:
        #         plt.show()

        # # --- Sum over all pairwise comparisons ---
        # # n_traj = len(projected_trajectories)
        # n_traj = 10 # XX Otherwise it takes super long to run, we're just going to look at the first few
        # sum_cosine = np.zeros(minimum_trajectory_len)
        # sum_euclidean = np.zeros(minimum_trajectory_len)
        # sum_curvature = np.zeros(minimum_trajectory_len)

        # for i in range(n_traj): # GG XX this takes an eternity to run
        #     for j in range(i + 1, n_traj):
        #         t1, t2 = projected_trajectories[i], projected_trajectories[j]

        #         min_len = minimum_trajectory_len 
        #         cosine_sim = [
        #             F.cosine_similarity(t1[t].unsqueeze(0), t2[t].unsqueeze(0)).item()
        #             for t in range(min_len)
        #         ]

        #         euclidean_dist = np.array(torch.norm(t1[:min_len] - t2[:min_len], dim=1).cpu())

        #         curvature = []
        #         for t in range(1, min_len - 1):
        #             a, b, c = t1[t - 1], t1[t], t1[t + 1]
        #             ba = a - b
        #             bc = c - b
        #             cosine = torch.dot(ba, bc) / (torch.norm(ba) * torch.norm(bc) + 1e-8)
        #             cosine = torch.clamp(cosine, -1.0, 1.0)
        #             angle = torch.acos(cosine)
        #             curvature.append(angle.item())
        #         curvature = [0.0] + curvature + [0.0]

        #         sum_cosine[:min_len] += cosine_sim
        #         sum_euclidean[:min_len] += euclidean_dist
        #         sum_curvature[:min_len] += curvature

        # plt.figure(figsize=(12, 4))
        # ax1 = plt.gca()

        # # Cosine similarity on primary y-axis
        # line1, = ax1.plot(sum_cosine, label="Sum Cosine Similarity", color='tab:blue', alpha=0.7)
        # ax1.set_ylabel("Cosine Similarity", color='tab:blue')
        # ax1.tick_params(axis='y', labelcolor='tab:blue')

        # # Create a second y-axis for Euclidean distance
        # ax2 = ax1.twinx()
        # line2, = ax2.plot(sum_euclidean, label="Sum Euclidean Distance", color='tab:orange', alpha=0.7)
        # ax2.set_ylabel("Euclidean Distance", color='tab:orange')
        # ax2.tick_params(axis='y', labelcolor='tab:orange')

        # # Optional: Create a third y-axis for curvature
        # ax3 = ax1.twinx()
        # ax3.spines["right"].set_position(("outward", 60))  # Offset third axis
        # line3, = ax3.plot(sum_curvature, label="Sum Curvature", color='tab:green', alpha=0.7)
        # ax3.set_ylabel("Curvature", color='tab:green')
        # ax3.tick_params(axis='y', labelcolor='tab:green')

        # # Title and x-label
        # ax1.set_title(f"Metrics for Trajectories Sum")
        # ax1.set_xlabel("Time step")
        # ax1.grid(True)

        # # Combine legends from all axes
        # lines = [line1, line2, line3]
        # labels = [line.get_label() for line in lines]
        # ax1.legend(lines, labels, loc="upper right")

        # plt.tight_layout()
        # if Save_plots:
        #     plt.savefig(os.path.join(PLOTS_DIR, "avg_euclidean_cosine_curvature_in_projection.png"))
        #     plt.clf()
        # else:
        #     plt.show()




        # # --- looking at diplacement at each step ---
        # all_parallel = []
        # all_perpendicular = []
        # all_magnitude = []


        # sum_parallel = torch.zeros(minimum_trajectory_len, device=device)
        # sum_perpendicular = torch.zeros(minimum_trajectory_len, device=device)
        # sum_magnitude = torch.zeros(minimum_trajectory_len, device=device)

        # for traj in trajectories:

        #     magnitude, parallel, perpendicular = calculate_displacement_components(traj)

        #     sum_parallel[:minimum_trajectory_len-1] += parallel[:minimum_trajectory_len-1] # the -1s here are the worse coding solution ive done in my life
        #     sum_perpendicular[:minimum_trajectory_len-1] += perpendicular[:minimum_trajectory_len-1]
        #     sum_magnitude[:minimum_trajectory_len-1] += magnitude[:minimum_trajectory_len-1]



        # # Plot the summed magnitudes for all trajectories
        # plt.figure(figsize=(10, 6))
        # plt.plot(sum_parallel.cpu(), label="Sum Parallel Magnitude", alpha=0.5)
        # plt.plot(sum_perpendicular.cpu(), label="Sum Perpendicular Magnitude", alpha=0.5)
        # plt.plot(magnitude.cpu(), label="Sum Magnitude")
        # plt.title("Difference between consecutive hidden states Components Across All Trajectories")
        # plt.xlabel("Time step")
        # plt.ylabel("Magnitude")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()

        # if Save_plots:
        #     plt.savefig(os.path.join(PLOTS_DIR, "sum_displacement_analysis.png"))
        #     plt.clf()
        # else:
        #     plt.show()

        # ### Uninteresting
            # auto_euclidean_matrix = euclidean_dist_matrix_auto(trajectories[idx])
            # plt.imshow(auto_euclidean_matrix.cpu().numpy(), cmap='coolwarm', aspect='equal', vmin=0, vmax=torch.max(auto_euclidean_matrix).item())
            # plt.colorbar()
            # plt.title(f"Recurrence plot trajectory {idx}\nEuclidean Distance")
            # plt.xlabel("Generation Step")
            # plt.ylabel("Generation Step")
            # plt.tight_layout()
            # if Save_plots:
            #     plt.savefig(os.path.join(PLOTS_DIR, f"recurrence_plot_euclidean_traj_{idx}.png"))
            #     plt.clf()
            # else:
            #     plt.show()