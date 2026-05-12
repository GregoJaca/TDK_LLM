import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

class JacobianPlotter:
    """
    A modular and flexible class to plot exact Jacobian measurements.
    Designed so that researchers can easily inherit, modify, or extract individual plotting methods.
    """
    def __init__(self, json_path, output_dir=None):
        self.json_path = json_path
        
        # Determine output directory
        if output_dir is None:
            base_dir = os.path.dirname(json_path)
            prompt_id = os.path.basename(json_path).replace("mlp_jacobian_measurements_", "").replace(".json", "")
            self.output_dir = os.path.join(base_dir, f"plots_{prompt_id}")
        else:
            self.output_dir = output_dir
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load Data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        self.layers = sorted([int(k) for k in self.data["layers"].keys()])
        self.num_layers = len(self.layers)
        self.num_tokens = self.data["metadata"]["seq_len"]

    def _get_token_metric(self, metric_key, sub_key=None):
        """
        Extracts a token-level metric across all layers.
        Returns array of shape [num_layers, num_tokens]
        """
        result = []
        for l in self.layers:
            layer_data = self.data["layers"][str(l)]
            if sub_key:
                val = layer_data[metric_key][sub_key]
            else:
                val = layer_data[metric_key]
            result.append(val)
        return np.array(result)

    def _get_scalar_metric(self, category, metric_name):
        """
        Extracts a scalar metric (like a weight matrix norm) across all layers.
        Returns array of shape [num_layers]
        """
        result = []
        for l in self.layers:
            val = self.data["layers"][str(l)][category][metric_name]
            result.append(val)
        return np.array(result)

    def plot_spectral_norms(self, show=False):
        """
        Plots the local Jacobian Spectral Norm ||J_MLP||_2 across layers.
        Since we have values for each token, it plots the mean and shaded standard deviation.
        """
        norms = self._get_token_metric("spectral_norms") # [layers, tokens]
        means = norms.mean(axis=1)
        stds = norms.std(axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.layers, means, marker='o', label='Mean Spectral Norm', color='darkred')
        plt.fill_between(self.layers, means - stds, means + stds, color='darkred', alpha=0.2, label='±1 Std Dev')
        
        # Reference line for expansive mapping
        plt.axhline(y=1.0, color='black', linestyle='--', label='Neutral Boundary (||J||_2 = 1)')
        
        plt.title(r"Local Jacobian Spectral Norm $\|J_{MLP}\|_2$ across Layers", fontsize=14)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel(r"Spectral Norm $\|J\|_2$", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        out_path = os.path.join(self.output_dir, "spectral_norms.png")
        plt.savefig(out_path, dpi=300)
        print(f"Saved: {out_path}")
        if show: plt.show()
        plt.close()

    def plot_lambda_true(self, show=False):
        """
        Plots the actual mean squared singular value lambda_true = (1/d) * ||J||_F^2.
        """
        l_true = self._get_token_metric("lambda_true") # [layers, tokens]
        means = l_true.mean(axis=1)
        stds = l_true.std(axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.layers, means, marker='s', label=r'Mean $\bar{\lambda}_{true}$', color='navy')
        plt.fill_between(self.layers, means - stds, means + stds, color='navy', alpha=0.2)
        
        plt.axhline(y=1.0, color='black', linestyle='--', label='Neutral Boundary')
        
        plt.title(r"Mean Squared Singular Value $\bar{\lambda}_{true}$ across Layers", fontsize=14)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel(r"$\bar{\lambda}_{true} = \frac{1}{d} \|J\|_F^2$", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        out_path = os.path.join(self.output_dir, "lambda_true.png")
        plt.savefig(out_path, dpi=300)
        print(f"Saved: {out_path}")
        if show: plt.show()
        plt.close()

    def plot_weight_svds(self, show=False):
        """
        Plots the maximum singular values of the raw MLP weight matrices.
        """
        gate_svd = self._get_scalar_metric("weight_metrics", "W_gate_max_SVD")
        up_svd = self._get_scalar_metric("weight_metrics", "W_up_max_SVD")
        down_svd = self._get_scalar_metric("weight_metrics", "W_down_max_SVD")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.layers, gate_svd, marker='^', label=r'$W_{gate}$ Max SVD')
        plt.plot(self.layers, up_svd, marker='v', label=r'$W_{up}$ Max SVD')
        plt.plot(self.layers, down_svd, marker='d', label=r'$W_{down}$ Max SVD')
        
        plt.axhline(y=1.0, color='black', linestyle='--', label='Neutral Boundary')
        
        plt.title("Maximum Singular Values of SwiGLU Weight Matrices", fontsize=14)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Max Singular Value", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        out_path = os.path.join(self.output_dir, "weight_svds.png")
        plt.savefig(out_path, dpi=300)
        print(f"Saved: {out_path}")
        if show: plt.show()
        plt.close()

    def plot_scaled_frobenius(self, show=False):
        """
        Plots the dimensionally scaled Frobenius traces of the weight matrices.
        """
        gate_f2 = self._get_scalar_metric("weight_metrics", "W_gate_scaled_F2")
        up_f2 = self._get_scalar_metric("weight_metrics", "W_up_scaled_F2")
        down_f2 = self._get_scalar_metric("weight_metrics", "W_down_scaled_F2")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.layers, gate_f2, marker='^', label=r'$W_{gate}$ Scaled $\| \cdot \|_F^2$ (div by 1536)')
        plt.plot(self.layers, up_f2, marker='v', label=r'$W_{up}$ Scaled $\| \cdot \|_F^2$ (div by 1536)')
        plt.plot(self.layers, down_f2, marker='d', label=r'$W_{down}$ Scaled $\| \cdot \|_F^2$ (div by 8960)')
        
        plt.axhline(y=1.0, color='black', linestyle='--', label='Neutral Boundary')
        
        plt.title("Scaled Frobenius Traces of Weight Matrices", fontsize=14)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Scaled Squared Frobenius Norm", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        out_path = os.path.join(self.output_dir, "scaled_frobenius.png")
        plt.savefig(out_path, dpi=300)
        print(f"Saved: {out_path}")
        if show: plt.show()
        plt.close()

    def plot_activation_densities(self, show=False):
        """
        Plots the state-dependent activation terms: 
        Mean squared magnitudes of S(x) and D(x) arrays across the hidden dimensions.
        """
        S_x = self._get_token_metric("activation_density", "S_x_sq_mean") # [layers, tokens]
        D_x = self._get_token_metric("activation_density", "D_x_sq_mean") # [layers, tokens]
        
        s_means = S_x.mean(axis=1)
        d_means = D_x.mean(axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.layers, s_means, marker='o', label=r'Mean $S(x)^2$', color='teal')
        plt.plot(self.layers, d_means, marker='x', label=r'Mean $D(x)^2$', color='orange')
        
        plt.title("Activation Density / Magnitude Across Layers", fontsize=14)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Mean Squared Magnitude", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        out_path = os.path.join(self.output_dir, "activation_densities.png")
        plt.savefig(out_path, dpi=300)
        print(f"Saved: {out_path}")
        if show: plt.show()
        plt.close()
        
    def plot_all(self):
        """Generates all defined plots."""
        print(f"--- Generating plots for {os.path.basename(self.json_path)} ---")
        self.plot_spectral_norms()
        self.plot_lambda_true()
        self.plot_weight_svds()
        self.plot_scaled_frobenius()
        self.plot_activation_densities()
        print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Plot Jacobian measurement JSON results.")
    parser.add_argument("--json_path", type=str, required=True, 
                        help="Path to the generated mlp_jacobian_measurements_{prompt_id}.json file")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Directory to save the plots. Defaults to a new folder next to the JSON.")
    args = parser.parse_args()
    
    if not os.path.exists(args.json_path):
        print(f"Error: JSON file not found at {args.json_path}")
        return
        
    plotter = JacobianPlotter(args.json_path, args.output_dir)
    plotter.plot_all()

if __name__ == "__main__":
    main()
