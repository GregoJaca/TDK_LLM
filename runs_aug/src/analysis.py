import torch
from scipy import stats
import torch.nn.functional as F

def calculate_hypervolume_and_axes(trajectories, n_axes=1):
    N, seq_len, D = trajectories.shape
    device = trajectories.device

    X_centered = trajectories - trajectories.mean(dim=0, keepdim=True)
    X_centered = X_centered.permute(1, 0, 2)
    G = torch.bmm(X_centered, X_centered.transpose(1, 2))

    hypervolumes = torch.zeros(seq_len, device=device)
    axes_lengths = torch.zeros(seq_len, n_axes, device=device)
    denom = torch.sqrt(torch.tensor(max(N - 1, 1), dtype=torch.float32, device=device))

    for t in range(seq_len):
        eigvals = torch.linalg.eigvalsh(G[t])
        svals = torch.flip(eigvals, dims=[0]).sqrt()
        lengths = svals / denom if N > 1 else torch.zeros_like(svals)
        valid_lengths = lengths[:n_axes]
        axes_lengths[t, :len(valid_lengths)] = valid_lengths
        hypervolumes[t] = torch.prod(valid_lengths[valid_lengths > 0])
    
    return hypervolumes, axes_lengths


def _normalize_states(X):
    norms = X.norm(dim=1, keepdim=True)
    norms[norms == 0] = 1.0
    return X / norms


def _pca_svd_volume(X, n_axes=None, eps=1e-12):
    # X: (T, D) samples x dims
    if n_axes is None:
        n_axes = min(X.shape)
    Xc = X - X.mean(dim=0, keepdim=True)
    # compute SVD in float64 for numerical stability
    Xc64 = Xc.to(torch.float64)
    U, S, Vh = torch.linalg.svd(Xc64, full_matrices=False)
    svals = S[:n_axes].clamp(min=eps)
    denom = torch.sqrt(torch.tensor(max(X.shape[0] - 1, 1), dtype=torch.float64, device=Xc64.device))
    lengths = (svals / denom).to(torch.float64)
    log_volume = torch.sum(torch.log(lengths))
    volume = torch.exp(log_volume)
    return volume.to(torch.float64), lengths.to(torch.float64), log_volume.to(torch.float64)


def _logdet_cov_volume(X, n_axes=None, eps=1e-12):
    # X: (T, D)
    # compute covariance (D x D)
    Xc = (X - X.mean(dim=0, keepdim=True)).to(torch.float64)
    cov = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = torch.flip(eigvals, dims=[0])
    if n_axes is not None and n_axes < eigvals.shape[0]:
        eigvals = eigvals[:n_axes]
    eigvals = eigvals.clamp(min=eps)
    logdet = 0.5 * torch.sum(torch.log(eigvals))
    log_volume = logdet
    volume = torch.exp(log_volume)
    lengths = torch.sqrt(eigvals)
    return volume.to(torch.float64), lengths.to(torch.float64), log_volume.to(torch.float64)


def compute_layer_volumes_per_trajectory(layer_trajectories, method='pca_svd', normalize=False, n_axes=None):
    # layer_trajectories: list of tensors each (T, D) for a single trajectory for a layer
    device = layer_trajectories[0].device
    n_traj = len(layer_trajectories)
    volumes = torch.zeros(n_traj, dtype=torch.float64)
    log_volumes = torch.zeros(n_traj, dtype=torch.float64)
    axes = []
    for i, X in enumerate(layer_trajectories):
        X = X.cpu()
        if normalize:
            X = _normalize_states(X)
        if method == 'pca_svd':
            vol, lengths, logv = _pca_svd_volume(X, n_axes=n_axes)
        elif method == 'logdet_cov':
            vol, lengths, logv = _logdet_cov_volume(X, n_axes=n_axes)
        else:
            raise ValueError(f"Unknown layer volume method: {method}")
        volumes[i] = vol
        log_volumes[i] = logv
        axes.append(lengths.cpu())
    axes = torch.stack(axes, dim=0)
    return volumes, axes, log_volumes


def compute_all_layers_volumes(hidden_states_storages, method='pca_svd', normalize=False, n_axes=None):
    # hidden_states_storages: dict[layer_idx] = tensor (N_conditions, T, D)
    device = next(iter(hidden_states_storages.values())).device
    result = {}
    for layer_idx, tensor in hidden_states_storages.items():
        # tensor: (N, T, D) where N = n_initial_conditions
        n, T, D = tensor.shape
        per_trajectory_volumes = []
        per_trajectory_axes = []
        per_trajectory_logvol = []
        for i in range(n):
            X = tensor[i].cpu()
            if normalize:
                X = _normalize_states(X)
            if method == 'pca_svd':
                vol, lengths, logv = _pca_svd_volume(X, n_axes=n_axes)
            elif method == 'logdet_cov':
                vol, lengths, logv = _logdet_cov_volume(X, n_axes=n_axes)
            else:
                raise ValueError(f"Unknown layer volume method: {method}")
            per_trajectory_volumes.append(vol.cpu())
            per_trajectory_logvol.append(logv.cpu())
            per_trajectory_axes.append(lengths.cpu())
        per_trajectory_axes = torch.stack(per_trajectory_axes, dim=0)
        result[int(layer_idx)] = {
            'volumes': torch.stack(per_trajectory_volumes).cpu(),
            'axes': per_trajectory_axes,
            'log_volumes': torch.stack(per_trajectory_logvol).cpu(),
        }
    return result

