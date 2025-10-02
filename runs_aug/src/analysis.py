import torch
from scipy import stats
import torch.nn.functional as F
from src.config import Analysis as AnalysisConfig

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


def _choose_n_axes_by_explained_variance(X, min_explained=0.9, max_axes=None):
    # X: (T, D)
    Xc = X - X.mean(dim=0, keepdim=True)
    Xc64 = Xc.to(torch.float64)
    # compute singular values
    U, S, Vh = torch.linalg.svd(Xc64, full_matrices=False)
    eigvals = S**2
    total = eigvals.sum()
    if total == 0:
        return 1
    cum = torch.cumsum(eigvals, dim=0) / total
    r = int((cum >= min_explained).nonzero(as_tuple=True)[0][0].item()) + 1 if (cum >= min_explained).any() else eigvals.shape[0]
    if max_axes is not None:
        r = min(r, max_axes)
    return max(1, r)


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
        elif method == 'pair_dist':
            # Compute pairwise distance matrix (cosine distance by default)
            # We follow the same windowing behavior as compute_metric_matrices:
            # if sliding windows are used, represent each window by its mean vector.
            # Then compute normalized cosine similarity matrix S = An @ An.T and
            # distance matrix D = 1 - S, symmetric; sum all elements -> scalar volume.
            A = X.numpy()
            T = A.shape[0]
            # Fetch pairwise window params from AnalysisConfig if present
            win_use = getattr(AnalysisConfig, 'LAYER_VOLUME_PAIR_USE_WINDOW', True)
            win_size = getattr(AnalysisConfig, 'LAYER_VOLUME_PAIR_WINDOW_SIZE', None)
            disp = getattr(AnalysisConfig, 'LAYER_VOLUME_PAIR_DISPLACEMENT', 1)

            if win_size is not None and win_size > 0 and T >= win_size and win_use:
                starts = list(range(0, T - win_size + 1, disp))
                reps = [A[s: s + win_size].mean(axis=0) for s in starts]
            else:
                starts = list(range(0, T))
                reps = [A[s] for s in starts]

            if len(reps) == 0:
                vol = torch.tensor(0.0, dtype=torch.float64)
                lengths = torch.zeros(0, dtype=torch.float64)
                logv = torch.tensor(float('nan'), dtype=torch.float64)
            else:
                # Try to reuse the project's metric-matrix utilities when available
                traj_np = A
                try:
                    # Try to use data_analysis APIs: iterate enabled metrics and sum their matrices
                    from data_analysis import config as da_config
                    from data_analysis import compute_metric_matrices as da_cmm
                    cfg_metrics = getattr(da_config, 'CONFIG', {}).get('metrics', {}) or {}
                    sw_use = getattr(AnalysisConfig, 'LAYER_VOLUME_PAIR_USE_WINDOW', True)
                    sw_size = getattr(AnalysisConfig, 'LAYER_VOLUME_PAIR_WINDOW_SIZE', None)
                    sw_disp = getattr(AnalysisConfig, 'LAYER_VOLUME_PAIR_DISPLACEMENT', 1)

                    total_sum = 0.0
                    any_metric = False
                    for metric_name, metric_cfg in cfg_metrics.items():
                        if not isinstance(metric_cfg, dict):
                            continue
                        if not bool(metric_cfg.get('enabled', False)):
                            continue
                        any_metric = True
                        # Special-case matrix builders available in data_analysis
                        try:
                            if hasattr(da_cmm, '_compute_cross_cos_matrix') and metric_name == 'cos':
                                M = da_cmm._compute_cross_cos_matrix(traj_np, window_size=sw_size, displacement=sw_disp, use_window=sw_use)
                            elif hasattr(da_cmm, '_compute_matrix_for_trajectory'):
                                # this function mirrors compute_metric_matrices._compute_matrix_for_trajectory signature
                                # use the metric module loader behavior from that script
                                # We call it with the same arguments used there
                                M = da_cmm._compute_matrix_for_trajectory(
                                    traj=traj_np,
                                    metric_mod=__import__(f"src.metrics.{metric_name}", fromlist=['dummy']) if True else None,
                                    use_window=sw_use,
                                    window_size=sw_size,
                                    displacement=sw_disp,
                                    out_root=None,
                                    metric_name=metric_name,
                                    traj_idx=0,
                                )
                            else:
                                # no helper for this metric; skip
                                continue
                            import numpy as _np
                            s = float(_np.nansum(M))
                            total_sum += s
                        except Exception:
                            # skip metric on failure and continue with others
                            continue

                    if any_metric and total_sum is not None:
                        vol = torch.tensor(float(total_sum), dtype=torch.float64)
                        import numpy as _np
                        R = _np.vstack([_np.asarray(r, dtype=float).reshape(1, -1) for r in reps])
                        lengths = torch.tensor(_np.linalg.norm(R, axis=-1), dtype=torch.float64)
                        logv = torch.tensor(float('nan'), dtype=torch.float64)
                    else:
                        # Fallback: cosine-only local computation
                        import numpy as _np
                        R = _np.vstack([_np.asarray(r, dtype=float).reshape(1, -1) for r in reps])
                        norms = _np.linalg.norm(R, axis=-1, keepdims=True)
                        norms[norms == 0] = 1e-12
                        Rn = R / norms
                        S = Rn @ Rn.T
                        D = 1.0 - S
                        D = 0.5 * (D + D.T)
                        sum_all = float(_np.nansum(D))
                        vol = torch.tensor(sum_all, dtype=torch.float64)
                        lengths = torch.tensor(_np.linalg.norm(R, axis=-1), dtype=torch.float64)
                        logv = torch.tensor(float('nan'), dtype=torch.float64)
                except Exception:
                    # any import or runtime error -> fallback to local cosine-distance sum
                    import numpy as _np
                    R = _np.vstack([_np.asarray(r, dtype=float).reshape(1, -1) for r in reps])
                    norms = _np.linalg.norm(R, axis=-1, keepdims=True)
                    norms[norms == 0] = 1e-12
                    Rn = R / norms
                    S = Rn @ Rn.T
                    D = 1.0 - S
                    D = 0.5 * (D + D.T)
                    sum_all = float(_np.nansum(D))
                    vol = torch.tensor(sum_all, dtype=torch.float64)
                    lengths = torch.tensor(_np.linalg.norm(R, axis=-1), dtype=torch.float64)
                    logv = torch.tensor(float('nan'), dtype=torch.float64)
        else:
            raise ValueError(f"Unknown layer volume method: {method}")
        volumes[i] = vol
        log_volumes[i] = logv
        axes.append(lengths.cpu())
    axes = torch.stack(axes, dim=0)
    return volumes, axes, log_volumes


def compute_all_layers_volumes(hidden_states_storages, method='pca_svd', normalize=False, n_axes=None):
    # hidden_states_storages: dict[layer_idx] = tensor (N_conditions, T, D) OR list of (T, D) tensors
    result = {}
    for layer_idx, tensor_or_list in hidden_states_storages.items():
        # Normalize input to a tensor of shape (N, T, D)
        if isinstance(tensor_or_list, list):
            if len(tensor_or_list) == 0:
                raise ValueError(f"Empty trajectory list for layer {layer_idx}")
            tensor = torch.stack([t.cpu() for t in tensor_or_list], dim=0)
        elif isinstance(tensor_or_list, torch.Tensor):
            tensor = tensor_or_list.cpu()
        else:
            # Try to convert
            tensor = torch.as_tensor(tensor_or_list).cpu()

        # tensor: (N, T, D) where N = n_initial_conditions
        n, T, D = tensor.shape
        per_trajectory_volumes = []
        per_trajectory_axes = []
        per_trajectory_logvol = []
        for i in range(n):
            X = tensor[i]
            if normalize:
                X = _normalize_states(X)
            # choose number of axes if not provided
            chosen_n_axes = n_axes
            if chosen_n_axes is None:
                min_expl = getattr(AnalysisConfig, 'LAYER_VOLUME_EXPLAINED_VAR', 0.9)
                chosen_n_axes = _choose_n_axes_by_explained_variance(X, min_explained=min_expl, max_axes=None)
            if method == 'pca_svd':
                vol, lengths, logv = _pca_svd_volume(X, n_axes=chosen_n_axes)
            elif method == 'logdet_cov':
                vol, lengths, logv = _logdet_cov_volume(X, n_axes=chosen_n_axes)
            elif method == 'pair_dist':
                vol, lengths, logv = None, None, None
                # reuse the per-trajectory function to compute pair_dist
                try:
                    vols, axes_vals, logs = compute_layer_volumes_per_trajectory([X], method='pair_dist', normalize=False, n_axes=chosen_n_axes)
                    # compute_layer_volumes_per_trajectory returns volumes tensor of length 1
                    vol = vols[0]
                    lengths = axes_vals[0]
                    logv = logs[0]
                except Exception:
                    raise
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

