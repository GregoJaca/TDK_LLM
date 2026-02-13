import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize_scalar
from sklearn.metrics import r2_score
import os
import json

# Configuration
OUTPUT_DIR = "uj2/fit_lyapu/"
PAIRS_TO_PROCESS = [(5, 7), (46, 48)]
END_LOW = 50
END_HIGH = 110

os.makedirs(OUTPUT_DIR, exist_ok=True)

def fit_exponential_segment(filename, start_idx, end_idx, yname="Distance", plot_all=True, save_path=None, ax=None, scale="linear"):
    data = np.load(filename)
    x = np.arange(len(data))
    x_fit = x[start_idx:end_idx+1]
    y_fit = data[start_idx:end_idx+1]
    
    def exp_func(x, a, lambd):
        return a * np.exp(lambd * x)
        
    popt, _ = curve_fit(exp_func, x_fit, y_fit, bounds=([0,0],[np.inf,np.inf]), maxfev=1000)
    y_pred = exp_func(x_fit, *popt)
    r2 = r2_score(y_fit, y_pred)
    
    if save_path:
        plt.figure(figsize=(10, 6))
        
        # Styling
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('Index', fontsize=24)
        plt.ylabel(yname, fontsize=24)
        
        if scale == "log":
            plt.semilogy(x, data, 'k.', alpha=0.5, label='All data')
            plt.semilogy(x_fit, y_fit, 'ro', label='Fit region')
            plt.semilogy(x_fit, y_pred, 'b-', linewidth=2, label=f'Exp fit ($\lambda$={popt[1]:.3f}, $R^2$={r2:.3f})')
        else:
            plt.plot(x, data, 'k.', alpha=0.5, label='All data')
            plt.plot(x_fit, y_fit, 'ro', label='Fit region')
            plt.plot(x_fit, y_pred, 'b-', linewidth=2, label=f'Exp fit ($\lambda$={popt[1]:.3f}, $R^2$={r2:.3f})')

        plt.legend(fontsize=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    return popt[0], popt[1], r2

def process_selected_pairs():
    # Original path template strictly preserved
    fname_template = r"/home/grego/LLM/launch_sep/interstellar_propulsion_review_0_0.00035/results/sentence-transformers_all-mpnet-base-v2_traj{i}/20251102-134758-14f74f/plots/cos_timeseries_{x}_{y}.npy"
    
    for x, y in PAIRS_TO_PROCESS:
        # Construct filename
        fname = fname_template.replace('{x}', str(x)).replace('{y}', str(y))
        
        if not os.path.exists(fname):
             # Try to anticipate if {i} needs handling? 
             # For now, respecting user instruction to keep path exactly as was.
             # If it fails, user will see it.
             print(f"File not found: {fname}")
             # We can't proceed with this file if it doesn't exist
             # But maybe the user has a trick or this is running in an environment where it exists
             pass

        print(f"Processing pair {x}-{y} from {fname}")
        
        try:
             # Load data to find best fit range
             data = np.load(fname)
             
             def neg_r2(end_idx):
                end_idx = int(end_idx)
                if end_idx < END_LOW or end_idx > END_HIGH or end_idx >= len(data):
                    return 1e6
                try:
                    _, _, r2 = fit_exponential_segment(fname, 0, end_idx, plot_all=False)
                    return -r2
                except Exception:
                    return 1e6

             res = minimize_scalar(neg_r2, bounds=(END_LOW, min(END_HIGH, len(data)-1)), method='bounded')
             best_end = int(res.x)
             
             # Save Linear Plot
             save_path_lin = os.path.join(OUTPUT_DIR, f"fit_{x}_{y}_linear.pdf")
             fit_exponential_segment(fname, 0, best_end, plot_all=True, save_path=save_path_lin, scale="linear")
             
             # Save Log Plot
             save_path_log = os.path.join(OUTPUT_DIR, f"fit_{x}_{y}_log.pdf")
             fit_exponential_segment(fname, 0, best_end, plot_all=True, save_path=save_path_log, scale="log")
             
        except Exception as e:
            print(f"Error processing {x}-{y}: {e}")
    
if __name__ == "__main__":
    process_selected_pairs()
