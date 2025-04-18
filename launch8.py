import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import psutil  
import os
import json
import pickle
from torch.nn import functional as F
from scipy import stats


# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================
# Model parameters
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32 if DEVICE == "cuda" else torch.float32)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading model: {MODEL_NAME}")
# Move model to device and set eval mode (ONCE)
model.to(DEVICE)
model.eval()  # Disables dropout/batch norm if any

# =============================================================================
# SYSTEM METRICS
# =============================================================================
start_time = time.time()
cpu_start = psutil.cpu_percent()
ram_start = psutil.virtual_memory().used

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}\n\n")
else:
    print("device in NOT cuda\n\n")

# =============================================================================
# PARAMETERS
# =============================================================================

N_INITIAL_CONDITIONS = 64
SAVE_RESULTS = True
MAX_NEW_TOKENS = 2048
REPETITION_PENALTY = 1.1 

# semmi
RESULTS_DIR = "./launch_szerda/" # semmi
RADIUS_INITIAL_CONDITIONS = 0.01 # semmi
TEMPERATURE = 0
TOP_P = 1
TOP_K = 1
EMBEDDING_DIM = model.config.hidden_size


prompts = [
    "Provide a complete guide to mastering basic cooking skills for beginners. Cover essential techniques (chopping, sautéing, boiling, baking), must-have kitchen tools, pantry staples to keep on hand, and how to follow recipes effectively. Include five simple but versatile recipes that help build fundamental skills, with detailed instructions for each."
]
print("Prompts: ")
print(prompts)

radiuses = [0.01, 0.02, 0.04]
temps = [0, 0.6]
top_ps = [1, 0.95]
top_ks = [1, 50]



# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_directories():
    """Create directories for saving results if they don't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


# This is based on svd
# The singular values are the length of the ellipsoid axes in the embedding space
def calculate_hypervolume_and_axes(trajectories, n_axes=1):
    min_len = min(t.shape[0] for t in trajectories)
    trimmed = [t[:min_len] for t in trajectories]
    stacked = torch.stack(trimmed)  # [N, seq_len, D]
    N, seq_len, D = stacked.shape
    device = stacked.device

    X_centered = stacked - stacked.mean(dim=0, keepdim=True)
    X_centered = X_centered.permute(1, 0, 2)
    G = torch.bmm(X_centered, X_centered.transpose(1, 2))

    hypervolumes = torch.zeros(seq_len, device=device)
    axes_lengths = torch.zeros(seq_len, n_axes, device=device)
    denom = torch.sqrt(torch.tensor(max(N-1, 1), dtype=torch.float32, device=device))

    for t in range(seq_len):
        eigvals = torch.linalg.eigvalsh(G[t])
        svals = torch.flip(eigvals, dims=[0]).sqrt() # the singular values
        lengths = svals / denom if N > 1 else torch.zeros_like(svals)
        valid_lengths = lengths[:n_axes]
        axes_lengths[t, :len(valid_lengths)] = valid_lengths
        hypervolumes[t] = torch.prod(valid_lengths[valid_lengths > 0]) # GG XX should use all axes
    
    return hypervolumes, axes_lengths




# def compute_exponential_slopes_with_stats(axis_lengths, n_axis):
#     """
#     For each of the first `n_axis` components in `axis_lengths`, fit a line to the log-transformed values.
#     Returns a list of [slope, 95% CI half-width, R²] for each component.
#     """
#     axis_array = np.array(axis_lengths, dtype=np.float64)
#     axis_array = np.nan_to_num(axis_array, nan=1e-8, posinf=1e-8, neginf=1e-8)
#     axis_array[axis_array <= 0] = 1e-8  # avoid log(0) and negative values

#     logs = np.log(axis_array[:, :n_axis])
#     t = np.arange(logs.shape[0])
#     results = []

#     for i in range(n_axis):
#         y = logs[:, i]
#         slope, intercept, r_value, p_value, std_err = stats.linregress(t, y)
#         ci_half_width = 1.96 * std_err  # 95% CI half-width
#         r_squared = r_value ** 2
#         results.append([slope, ci_half_width, r_squared])

#     return results


def generate_perturbed_prompts(tokenizer, initial_prompt, n_conditions, radius): # GG also implement this for word modification, replacement instead of vector
    # Get both input_ids and attention_mask
    encoded = tokenizer(initial_prompt, return_tensors="pt")
    initial_tokens = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)
    
    with torch.no_grad():
        initial_embedding = model.get_input_embeddings()(initial_tokens)
    
    perturbations = torch.randn(
        n_conditions, 
        *initial_embedding.shape[1:], 
        device=DEVICE
    ) * radius
    
    perturbed_embeddings = initial_embedding + perturbations
    
    return initial_tokens, attention_mask, perturbed_embeddings  # Return attention mask too

def calculate_entropy(probabilities):
    mask = probabilities > 0
    return torch.where(mask, probabilities * torch.log2(probabilities), 0.0).sum(dim=-1).neg()




def debug_fractal_dimension(trajectory, max_dim=None):
    """Calculate fractal dimension of a trajectory using PyTorch operations"""
    if isinstance(trajectory, torch.Tensor):
        points = trajectory.reshape(-1, EMBEDDING_DIM)
    else:
        points = torch.cat([t.reshape(1, -1) for t in trajectory], dim=0)
    
    n_samples, n_features = points.shape
    max_possible_components = min(n_samples, n_features)
    n_components = min(max_dim or max_possible_components, max_possible_components)
    
    if n_components < n_features:
        points_centered = points - points.mean(dim=0)
        _, S, V = torch.linalg.svd(points_centered)
        reduced_points = points_centered @ V[:, :n_components]
    else:
        reduced_points = points
    
    mins = reduced_points.min(dim=0).values
    maxs = reduced_points.max(dim=0).values
    
    box_sizes = []
    box_counts = []
    
    for k in range(1, 10):
        size = 2 ** -k
        box_sizes.append(size)
        num_boxes = torch.ceil((maxs - mins) / size).to(torch.long)
        box_indices = torch.floor((reduced_points - mins) / size).to(torch.long)
        
        unique_boxes = torch.unique(box_indices, dim=0)
        box_counts.append(unique_boxes.shape[0])
    
    log_sizes = torch.log(torch.tensor(box_sizes))
    log_counts = torch.log(torch.tensor(box_counts))
    
    A = torch.vstack([log_sizes, torch.ones_like(log_sizes)]).T
    slope, _ = torch.linalg.lstsq(A, log_counts).solution
    
    return -slope.item()




# Compute Lyapunov coefficients from the distances
def get_lyapunov(distance_matrix, seq_len, n_coeffs, n_traj):
    time_steps = torch.arange(seq_len, device=DEVICE).float()
    X = torch.stack([time_steps, torch.ones_like(time_steps)], dim=1)
    coeffs = []
    
    for k in range(min(n_coeffs, n_traj-1)):
        y = torch.log(distance_matrix[0,k+1] + 1e-10)
        coeffs.append(torch.linalg.lstsq(X, y).solution[0].item())
    return coeffs
def calculate_combined_metrics(trajectories, n_coeffs=3):
    """
    Computes trajectory distances AND Lyapunov coefficients in one pass.
    Returns: (distances_dict, lyapunov_coeffs_euclidean, lyapunov_coeffs_cosine)
    """
    if len(trajectories) < 2:
        return {}, [], []
    
    # Ensure trajectories are stacked as [n_traj, seq_len, hidden_dim] tensor
    # traj_tensor = torch.stack(trajectories) if isinstance(trajectories, list) else trajectories
    if isinstance(trajectories, list):
        min_len = min(len(t) for t in trajectories)
        traj_tensor = torch.stack([t[:min_len] for t in trajectories])
    else:
        traj_tensor = trajectories
        min_len = traj_tensor.shape[1]
    n_traj, seq_len, hidden_dim = traj_tensor.shape
    
    # Pre-allocate output tensors
    cosine_dists = torch.zeros((n_traj, n_traj, seq_len), device=DEVICE)
    euclidean_dists = torch.zeros_like(cosine_dists)
    
    # Reference trajectory (first one)
    # ref_traj = traj_tensor[0]  # [seq_len, hidden_dim] XX GG UNUSED
    
    # Compute all pairwise distances
    for i in range(n_traj):
        for j in range(i+1, n_traj):
            # Vectorized distance calculations
            traj_i = traj_tensor[i]
            traj_j = traj_tensor[j]
            
            # Cosine distance: 1 - cos_sim
            traj_i_norm = F.normalize(traj_i, dim=-1)
            traj_j_norm = F.normalize(traj_j, dim=-1)
            cosine_dists[i,j] = 1 - (traj_i_norm * traj_j_norm).sum(dim=-1)
            
            # Euclidean distance
            euclidean_dists[i,j] = torch.norm(traj_i - traj_j, dim=-1)
    
    # Build distances dictionary
    distances = {}
    for i in range(n_traj):
        for j in range(i+1, n_traj):
            key = f"{i}-{j}"
            distances[key] = { # GGG is it necessary to make into cpu numpy?
                "cosine": cosine_dists[i,j].cpu().numpy(),
                "euclidean": euclidean_dists[i,j].cpu().numpy()
            }
    # GG since I only use get_lyapunov here maybe it's just easier to write the code here instead of making a function
    lyapunov_euclidean = get_lyapunov(euclidean_dists, seq_len, n_coeffs, n_traj) 
    lyapunov_cosine = get_lyapunov(cosine_dists, seq_len, n_coeffs, n_traj)
    
    return distances, lyapunov_euclidean, lyapunov_cosine



# =============================================================================
# MAIN FUNCTION
# =============================================================================

def analyze_llm_chaos(initial_prompt):
    """Main function to analyze chaotic behavior in LLM text generation."""
   
    print(f"Generating {N_INITIAL_CONDITIONS} perturbed initial conditions...")

    initial_tokens, attention_mask, perturbed_embeddings = generate_perturbed_prompts(
        tokenizer, initial_prompt, N_INITIAL_CONDITIONS, RADIUS_INITIAL_CONDITIONS
    )

    trajectories = []
    generated_texts = []
    entropy_values = []
    trajectory_hidden_lengths = []

    for i in range(N_INITIAL_CONDITIONS):
        print(f"\nProcessing initial condition {i+1}/{N_INITIAL_CONDITIONS}")
        current_embedding = perturbed_embeddings[i].unsqueeze(0)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs_embeds=current_embedding,
                attention_mask=attention_mask,  # Pass the attention mask
                num_return_sequences=1,
                max_length=MAX_NEW_TOKENS,
                do_sample=TEMPERATURE > 0,
                temperature=TEMPERATURE if TEMPERATURE > 0 else None,
                top_k=TOP_K if ((TOP_K > 0) and (TEMPERATURE > 0)) else None,
                top_p=TOP_P if ((TOP_P < 1.0) and (TEMPERATURE > 0)) else None,
                output_scores=True,
                return_dict_in_generate=True,
                output_hidden_states=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=REPETITION_PENALTY,
                eos_token_id=tokenizer.eos_token_id
            )
            
            hidden_states = outputs.hidden_states
            trajectory = torch.stack([h[-1][0, 0, :] for h in hidden_states])  # [seq_len, hidden_dim] GG more efficient ?
            trajectory_hidden_lengths.append((trajectory.norm(dim=1, keepdim=True)).tolist())  # [seq_len, 1] ### GG XX (trajectory.norm(dim=1, keepdim=True)).tolist() gives a list of a single float [1.2], so I should take the [0] or convert differently


            step_entropies = []
            for step_scores in outputs.scores: # GG make more efficient
                probs = torch.softmax(step_scores / max(TEMPERATURE, 1e-10), dim=-1)
                entropy = calculate_entropy(probs)
                step_entropies.append(entropy.item())
            
            trajectories.append(trajectory)
            generated_text = tokenizer.decode(
                outputs.sequences[0], 
                skip_special_tokens=True
            )
            generated_texts.append(generated_text)
            print("\nNumber of tokens: ", len(outputs.sequences[0]))
            if len(outputs.sequences[0]) >= MAX_NEW_TOKENS:
                print("------ Response finished early due to reaching max_new_tokens.")
            # print("-------- generated_text -----------")
            # print(generated_text) # XXXX
            
            entropy_values.append(step_entropies)
    
    print("\nAnalyzing trajectories...")
    
    distances = {}
    fractal_dimensions = []
    for i, trajectory in enumerate(trajectories):
        fractal_dim = debug_fractal_dimension(trajectory)
        fractal_dimensions.append(fractal_dim)
        print(f"Fractal dimension of trajectory {i}: {fractal_dim}")
    
    distances, lyapunov_coeffs_euclidean, lyapunov_coeffs_cosine = calculate_combined_metrics(trajectories)
    hypervolumes, axis_lengths = calculate_hypervolume_and_axes(trajectories, n_axes=4)


    if lyapunov_coeffs_euclidean:
        print(f"Lyapunov coefficients (Euclidean): {lyapunov_coeffs_euclidean}")
    if lyapunov_coeffs_cosine:
        print(f"Lyapunov coefficients (Cosine): {lyapunov_coeffs_cosine}")

    # alternative_lyap = compute_exponential_slopes_with_stats(axis_lengths, 4)
    # print("Alternative Lyapunov calculation: ")
    # print( " [value], [interval_half_width (+-) ], [r2] " )
    # print(alternative_lyap)
    
    if SAVE_RESULTS:
        results = {
            "initial_prompt": initial_prompt,
            "generated_texts": generated_texts,
            # "entropy_values": [list(map(float, e)) for e in entropy_values],
            "fractal_dimensions": fractal_dimensions,
            # "lyapunov_exponents": lyapunov_coeffs_euclidean + lyapunov_coeffs_cosine if lyapunov_coeffs_euclidean and lyapunov_coeffs_cosine else [], #XXXX
            "lyapunov_exponents_euclidean": lyapunov_coeffs_euclidean, # XXXX
            "lyapunov_exponents_cosine": lyapunov_coeffs_cosine,
            # "alternative_lyap": alternative_lyap,
            # "hypervolumes": hypervolumes.tolist(),
            # "axis_lengths": axis_lengths.tolist(),
            # "trajectory_hidden_lengths": trajectory_hidden_lengths,
            "config": {
                "model_name": MODEL_NAME,
                "temperature": TEMPERATURE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "n_initial_conditions": N_INITIAL_CONDITIONS,
                "radius_initial_conditions": RADIUS_INITIAL_CONDITIONS
            }
        }
        
        with open(f"{RESULTS_DIR}results_summary.json", "w") as f:
            json.dump(results, f, indent=2)
        
        with open(f"{RESULTS_DIR}trajectories.pkl", "wb") as f:
            pickle.dump(trajectories, f)

        with open(f"{RESULTS_DIR}hypervolume.pkl", "wb") as f:
            pickle.dump(hypervolumes, f)

        with open(f"{RESULTS_DIR}trajectory_hidden_lengths.pkl", "wb") as f:
            pickle.dump(trajectory_hidden_lengths, f)
        
        with open(f"{RESULTS_DIR}distances.pkl", "wb") as f:
            pickle.dump(distances, f)

        with open(f"{RESULTS_DIR}entropy_values.pkl", "wb") as f:
            pickle.dump(entropy_values, f)

        with open(f"{RESULTS_DIR}axis_lengths.pkl", "wb") as f:
            pickle.dump(axis_lengths, f)
  
    return {
        "fractal_dimensions": fractal_dimensions,
        "lyapunov_exponents": lyapunov_coeffs_euclidean + lyapunov_coeffs_cosine if lyapunov_coeffs_euclidean and lyapunov_coeffs_cosine else []
    }


# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    for rrr in radiuses:
        for ttt in range(len(temps)):
            TEMPERATURE = temps[ttt]
            TOP_P = top_ps[ttt]
            TOP_K = top_ks[ttt]
            RADIUS_INITIAL_CONDITIONS = rrr
            RESULTS_DIR = f"./launch_pentek_{TEMPERATURE}_{RADIUS_INITIAL_CONDITIONS}/"
            
            print(" ---------------------------------- ")
            print("radius: ", RADIUS_INITIAL_CONDITIONS)
            print("T = ", TEMPERATURE)

            setup_directories()   
            results = analyze_llm_chaos(prompts[0])
            
            print("\n=== RESULTS SUMMARY ===")

            if results['lyapunov_exponents']:
                print(f"Average Lyapunov exponent: {np.mean(results['lyapunov_exponents']):.4f}")
            
            print(f"Average fractal dimension: {np.mean(results['fractal_dimensions']):.4f}")

    end_time = time.time()
    execution_time = end_time - start_time
    print("\n\n=== PERFORMANCE METRICS ===\n")
    print(f"Total execution time: {execution_time:.2f} seconds\n")
    print(f"CPU usage: {psutil.cpu_percent()}%\n")
    print(f"RAM used: {(psutil.virtual_memory().used - ram_start)/1024/1024:.2f} MB\n")
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.max_memory_allocated()/1024/1024:.2f} MB\n")
    else:
        print("cuda not available at the end")