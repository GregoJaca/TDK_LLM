import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize_scalar
from sklearn.metrics import r2_score
import os
import json

def fit_exponential_segment(filename, start_idx, end_idx, yname="Distance", plot_all=True, save_path=None):
    data = np.load(filename)
    x = np.arange(len(data))
    x_fit = x[start_idx:end_idx+1]
    y_fit = data[start_idx:end_idx+1]
    def exp_func(x, a, lambd):
        return a * np.exp(lambd * x)
    popt, _ = curve_fit(exp_func, x_fit, y_fit, bounds=([0,0],[np.inf,np.inf]), maxfev=400)
    y_pred = exp_func(x_fit, *popt)
    r2 = r2_score(y_fit, y_pred)
    plt.figure(figsize=(8,4))
    if plot_all:
        plt.plot(x, data, 'k.', alpha=0.5, label='All data')
        plt.plot(x_fit, y_fit, 'ro', label='Fit region')
    else:
        plt.plot(x_fit, y_fit, 'ro', label='Fit region')
    plt.plot(x_fit, y_pred, 'b-', label=f'Exp fit ($lambda$={popt[1]:.3f}, $R^2$={r2:.3f})')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel(yname)
    plt.title('Exponential Fit to Trajectory Divergence')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return popt[0], popt[1], r2

pairs_to_exclude = [(1,2), (1,3), (1,4)]

def batch_fit_and_save():
    END_LOW = 50
    END_HIGH = 110
    N = 100
    lambdas = []
    low_r2_pairs = []
    os.makedirs('exp_fit_embed', exist_ok=True)
    for x in range(N):
        for y in range(x+1,N):
            if (x,y) not in pairs_to_exclude:
                fname = r"/home/grego/LLM/launch_sep/interstellar_propulsion_review_0_0.00035/results/sentence-transformers_all-mpnet-base-v2_traj{i}/20251102-134758-14f74f/plots/cos_timeseries_{x}_{y}.npy"
                fname = fname.replace('{x}', str(x)).replace('{y}', str(y))

                data = np.load(fname)
                def neg_r2(end_idx):
                    end_idx = int(end_idx)
                    if end_idx < END_LOW or end_idx > END_HIGH:
                        return 1e6
                    try:
                        _, _, r2 = fit_exponential_segment(fname, 0, end_idx, plot_all=False)
                        return -r2
                    except Exception:
                        return 1e6

                res = minimize_scalar(neg_r2, bounds=(END_LOW, END_HIGH), method='bounded')
                best_end = int(res.x)
                try:
                    _, lam, best_r2 = fit_exponential_segment(fname, 0, best_end, plot_all=False)
                except Exception:
                    best_r2 = -np.inf
                    lam = None

                if lam is not None and best_r2 > 0.85:
                    lambdas.append({'x': x, 'y': y, 'lambda': lam, 'r2': best_r2})
                    save_path = f'exp_fit_embed/fit_{x}_{y}.png'
                    fit_exponential_segment(fname, 0, best_end, plot_all=True, save_path=save_path)
                else:
                    low_r2_pairs.append({'x': x, 'y': y, 'r2': best_r2})
        np.save('exp_fit_embed/lambda_list.npy', lambdas)
        with open('exp_fit_embed/low_r2_pairs.json', 'w') as f:
            json.dump(low_r2_pairs, f)


batch_fit_and_save()