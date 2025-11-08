# --- Recurrence Quantification Analysis ---
def analyze_recurrence(recurrence_binary):
    recurrence_binary = recurrence_binary.astype(np.uint8)

    N = recurrence_binary.shape[0]
    assert recurrence_binary.shape[0] == recurrence_binary.shape[1], "Recurrence matrix must be square."

    # --- Recurrence Rate ---
    RR = np.sum(recurrence_binary) / (N * N)

    # --- Diagonal lines ---
    def get_diagonal_line_lengths(recurrence):
        lengths = []
        for offset in range(-N + 1, N):
            diag = np.diagonal(recurrence, offset=offset)
            if diag.size == 0:
                continue
            counts = count_lines(diag)
            lengths.extend(counts)
        return np.array(lengths)

    # --- Vertical lines ---
    def get_vertical_line_lengths(recurrence):
        lengths = []
        for col in recurrence.T:
            counts = count_lines(col)
            lengths.extend(counts)
        return np.array(lengths)

    # --- Helper: Count continuous 1's ---
    def count_lines(array):
        counts = []
        count = 0
        for v in array:
            if v == 1:
                count += 1
            elif count > 0:
                counts.append(count)
                count = 0
        if count > 0:
            counts.append(count)
        return counts

    diag_lengths = get_diagonal_line_lengths(recurrence_binary)
    vert_lengths = get_vertical_line_lengths(recurrence_binary)

    # Only consider lines of length >= 2
    diag_lengths = diag_lengths[diag_lengths >= 2]
    vert_lengths = vert_lengths[vert_lengths >= 2]

    # --- Determinism ---
    if np.sum(recurrence_binary) == 0:
        DET = 0.0
    else:
        DET = np.sum(diag_lengths) / np.sum(recurrence_binary)

    # --- Average Line Length (diagonals) ---
    if diag_lengths.size == 0:
        L = 0.0
    else:
        L = np.mean(diag_lengths)

    # --- Longest Diagonal Line ---
    if diag_lengths.size == 0:
        Lmax = 0
    else:
        Lmax = np.max(diag_lengths)

    # --- Entropy of Diagonal Lines ---
    if diag_lengths.size == 0:
        Entr_diag = 0.0
    else:
        hist, _ = np.histogram(diag_lengths, bins=np.arange(1, np.max(diag_lengths) + 2))
        p = hist / np.sum(hist)
        Entr_diag = entropy(p, base=np.e)

    # --- Laminarity (vertical lines) ---
    if np.sum(recurrence_binary) == 0:
        LAM = 0.0
    else:
        LAM = np.sum(vert_lengths) / np.sum(recurrence_binary)

    # --- Trapping Time (vertical lines) ---
    if vert_lengths.size == 0:
        TT = 0.0
    else:
        TT = np.mean(vert_lengths)

    # --- Longest Vertical Line ---
    if vert_lengths.size == 0:
        Vmax = 0
    else:
        Vmax = np.max(vert_lengths)

    # --- Entropy of Vertical Lines ---
    if vert_lengths.size == 0:
        Entr_vert = 0.0
    else:
        hist_v, _ = np.histogram(vert_lengths, bins=np.arange(1, np.max(vert_lengths) + 2))
        p_v = hist_v / np.sum(hist_v)
        Entr_vert = entropy(p_v, base=np.e)

    return np.array([RR, DET, Lmax, Entr_diag, LAM, Vmax, Entr_vert])
    # return np.array([RR, DET, L, Lmax, Entr_diag, LAM, TT, Vmax, Entr_vert])
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import entropy

# cosine_sim_last_last = torch.load("./results_pecs/interstellar_propulsion_review/cosine_sim_last_last.pt")




filenames = [
    "C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/code_LLM/results_pecs/interstellar_propulsion_review/cosine_sim_first_first.pt",
    "C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/code_LLM/results_pecs/interstellar_propulsion_review/cosine_sim_last_last.pt",
    "C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/code_LLM/results_pecs/quantum_consciousness_hallucination/cosine_sim_first_first.pt",
    "C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/code_LLM/results_pecs/quantum_consciousness_hallucination/cosine_sim_last_last.pt",
]
names = [
    "Prompt 1, first layer",
    "Prompt 1, last layer",
    "Prompt 2, first layer",
    "Prompt 2, last layer"
]



filenames = [


    "C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/git_repo/TDK_LLM/for_rqa_pentek/cos_similarity_traj_2.pt",
    "C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/git_repo/TDK_LLM/for_rqa_pentek/cos_similarity_traj_3.pt",
    "C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/git_repo/TDK_LLM/for_rqa_pentek/cos_similarity_traj_1.pt",
    "C:/Users/grego/OneDrive/Documents/BME_UNI_WORK/TDK_2025/git_repo/TDK_LLM/for_rqa_pentek/cos_similarity_traj_6.pt"
]
names = [
    "Prompt a",
    "Prompt b",
    "Prompt c",
    "Prompt d"
]




threshold = 0.3

metrics_names = [
    'Recurrence Rate',
    'Determinism',
    # 'Average Line Length',
    'Longest Diagonal Line',
    'Entropy of Diagonal Lines',
    'Laminarity',
    # 'Trapping Time',
    'Longest Vertical Line',
    'Entropy of Vertical Lines'
]

# Collect metrics for all files
all_metrics = []
for fname in filenames:
    cosine_sim = torch.load(fname, map_location=torch.device('cpu'))
    cosine_sim_np = cosine_sim.numpy()
    distances_np = 1 - cosine_sim_np
    recurrence_np = distances_np < threshold
    np.fill_diagonal(recurrence_np, 0)
    metrics = analyze_recurrence(recurrence_np)  # Call to the moved function
    all_metrics.append(metrics)

# Create DataFrame for LaTeX output
df_all = pd.DataFrame(all_metrics, columns=metrics_names, index=names)

print("\n==== Recurrence Quantification Analysis (RQA) Metrics for All Files ====\n")
print(df_all)

# Minimalist LaTeX table output
print("\nCopy-paste this into LaTeX:")
print("\\begin{tabular}{l" + "c"*len(metrics_names) + "}")
print("\\toprule")
print("Name & " + " & ".join(metrics_names) + " \\" + "\\")
print("\\midrule")
for idx, row in df_all.iterrows():
    vals = [f"{v:.4f}" for v in row]
    print(f"{idx} & " + " & ".join(vals) + " \\" + "\\")
print("\\bottomrule")
print("\\end{tabular}")

# Optionally, plot recurrence plot for the first file only
cosine_sim = torch.load(filenames[0], map_location=torch.device('cpu'))
cosine_sim_np = cosine_sim.numpy()
distances_np = 1 - cosine_sim_np
recurrence_np = distances_np < threshold
np.fill_diagonal(recurrence_np, 0)
plt.figure(figsize=(10, 8))
plt.imshow(recurrence_np, cmap='binary', origin='lower')
plt.title(f"Recurrence Plot ({names[0]})\n Threshold: {threshold}")
plt.xlabel("Time Index")
plt.ylabel("Time Index")
plt.colorbar(label="Recurrence")
plt.show()
plt.close()



