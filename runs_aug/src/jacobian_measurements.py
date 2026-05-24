import torch
from torch.func import functional_call, vmap, jacrev

def compute_mlp_jacobian_metrics(model, x_norm, layer_idx):
    """
    Computes exact Jacobian of the MLP function w.r.t its input (x_norm) for all tokens in the sequence.
    Also extracts scaled Frobenius norms of the MLP weight matrices.
    
    Args:
        model: The HuggingFace causal LM.
        x_norm: [seq_len, hidden_size] or [batch, seq_len, hidden_size] tensor.
        layer_idx: The integer index of the layer being analyzed.
        
    Returns:
        dict containing:
            - "spectral_norms": list of floats, the 2-norm of the Jacobian for each token.
            - "lambda_true": list of floats, the normalized Frobenius norm squared of the Jacobian.
            - "activation_density": dict with "S_x_sq_mean" and "D_x_sq_mean" for each token.
            - "weight_metrics": dict with scaled frobenius norms and max singular values.
    """
    if x_norm.dim() == 3:
        x_norm = x_norm.squeeze(0) # [seq_len, hidden_size]
        
    mlp = model.model.layers[layer_idx].mlp
    
    x_norm_32 = x_norm.to(torch.float32)
    original_dtype = next(mlp.parameters()).dtype
    mlp.to(torch.float32)
    
    params = dict(mlp.named_parameters())
    buffers = dict(mlp.named_buffers())
    
    def mlp_func(p, b, x):
        # We add unsqueeze(0) because linear layers expect at least 2D (or 1D works in torch > 1.11 usually)
        # But Qwen2MLP is safe with 2D [1, hidden_size]
        return functional_call(mlp, (p, b), (x.unsqueeze(0),)).squeeze(0)
    
    # Compute Jacobian w.r.t the 3rd argument (x)
    jac_fn = jacrev(mlp_func, argnums=2)
    # vmap over the batch dimension of x (which is dim 0), while params/buffers are broadcasted (None)
    batch_jac_fn = vmap(jac_fn, in_dims=(None, None, 0))
    
    with torch.no_grad():
        # Process the exact Jacobian in chunks to prevent OOM on long prompts.
        # jacrev computes vjp for 1536 basis vectors. vmap adds seq_len batching.
        # This requires enormous intermediate memory (chunk_size * 1536 * 8960 * 4 bytes).
        chunk_size = 16 
        seq_len = x_norm_32.size(0)
        
        spectral_norms_list = []
        lambda_true_list = []
        
        for i in range(0, seq_len, chunk_size):
            x_chunk = x_norm_32[i:i+chunk_size]
            # [chunk_size, hidden_size, hidden_size]
            jac_chunk = batch_jac_fn(params, buffers, x_chunk)
            
            # Calculate metrics immediately and discard the massive jacobian tensor
            sn_chunk = torch.linalg.matrix_norm(jac_chunk, ord=2)
            j_f2_chunk = torch.sum(jac_chunk ** 2, dim=(1, 2))
            lt_chunk = j_f2_chunk / jac_chunk.shape[-1]
            
            spectral_norms_list.append(sn_chunk)
            lambda_true_list.append(lt_chunk)
            
            del jac_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        spectral_norms = torch.cat(spectral_norms_list, dim=0)
        lambda_true = torch.cat(lambda_true_list, dim=0)
        
        # Calculate scaled Gramian metrics for weights
        if hasattr(mlp, 'gate_proj') and hasattr(mlp, 'up_proj'):
            W_gate = mlp.gate_proj.weight.data
            W_up = mlp.up_proj.weight.data
        elif hasattr(mlp, 'gate_up_proj'):
            W_gate_up = mlp.gate_up_proj.weight.data
            split_dim = W_gate_up.shape[0] // 2
            W_gate = W_gate_up[:split_dim, :]
            W_up = W_gate_up[split_dim:, :]
        else:
            # Fallback for models that might use different naming but still have SwiGLU
            # Attempt to find linear layers
            linears = [m for m in mlp.modules() if isinstance(m, torch.nn.Linear)]
            if len(linears) >= 3:
                W_gate = linears[0].weight.data
                W_up = linears[1].weight.data
            elif len(linears) == 2:
                W_gate_up = linears[0].weight.data
                split_dim = W_gate_up.shape[0] // 2
                W_gate = W_gate_up[:split_dim, :]
                W_up = W_gate_up[split_dim:, :]
            else:
                raise AttributeError(f"Could not automatically determine MLP weight matrices for {type(mlp)}")
                
        if hasattr(mlp, 'down_proj'):
            W_down = mlp.down_proj.weight.data
        else:
            linears = [m for m in mlp.modules() if isinstance(m, torch.nn.Linear)]
            if len(linears) >= 2:
                W_down = linears[-1].weight.data
            else:
                raise AttributeError(f"Could not automatically determine MLP down_proj for {type(mlp)}")
        
        gate_f2 = torch.sum(W_gate ** 2).item()
        up_f2 = torch.sum(W_up ** 2).item()
        down_f2 = torch.sum(W_down ** 2).item()
        
        # Scaled according to theoretical input dimension derivation
        scaled_gate = gate_f2 / W_gate.shape[1] # 1536
        scaled_up = up_f2 / W_up.shape[1]       # 1536
        scaled_down = down_f2 / W_down.shape[1] # 8960
        
        # Max singular values of the weights
        svd_gate = torch.linalg.svdvals(W_gate).max().item()
        svd_up = torch.linalg.svdvals(W_up).max().item()
        svd_down = torch.linalg.svdvals(W_down).max().item()
        
        # Activation Density
        h_gate = torch.nn.functional.linear(x_norm_32, W_gate)
        h_up = torch.nn.functional.linear(x_norm_32, W_up)
        
        sig_h_gate = torch.sigmoid(h_gate)
        S_x = h_gate * sig_h_gate
        silu_prime_h_gate = sig_h_gate * (1 + h_gate * (1 - sig_h_gate))
        D_x = h_up * silu_prime_h_gate
        
        S_x_sq_mean = (S_x ** 2).mean(dim=-1)
        D_x_sq_mean = (D_x ** 2).mean(dim=-1)
        
        
    # Restore original dtype
    mlp.to(original_dtype)
    
    return {
        "spectral_norms": spectral_norms.cpu().numpy().tolist(),
        "lambda_true": lambda_true.cpu().numpy().tolist(),
        "activation_density": {
            "S_x_sq_mean": S_x_sq_mean.cpu().numpy().tolist(),
            "D_x_sq_mean": D_x_sq_mean.cpu().numpy().tolist()
        },
        "weight_metrics": {
            "W_gate_scaled_F2": scaled_gate,
            "W_up_scaled_F2": scaled_up,
            "W_down_scaled_F2": scaled_down,
            "W_gate_max_SVD": svd_gate,
            "W_up_max_SVD": svd_up,
            "W_down_max_SVD": svd_down
        }
    }


def extract_attn_weights(self_attn, config):
    # Try to find q_proj, k_proj, v_proj
    q_proj = getattr(self_attn, "q_proj", None)
    k_proj = getattr(self_attn, "k_proj", None)
    v_proj = getattr(self_attn, "v_proj", None)
    o_proj = getattr(self_attn, "o_proj", None)
    
    num_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    d_model = getattr(config, "hidden_size", None)
    d_head = getattr(self_attn, "head_dim", None)
    if d_head is None and num_heads is not None and d_model is not None:
        d_head = d_model // num_heads
        
    if q_proj is not None and k_proj is not None and v_proj is not None:
        W_Q = q_proj.weight.data
        W_K = k_proj.weight.data
        W_V = v_proj.weight.data
    else:
        # Check for fused QKV projection
        qkv_proj = getattr(self_attn, "qkv_proj", None)
        if qkv_proj is None:
            # Fallback: look for any linear layers with 'qkv' in the name
            for name, module in self_attn.named_children():
                if "qkv" in name.lower() and isinstance(module, torch.nn.Linear):
                    qkv_proj = module
                    break
        if qkv_proj is not None:
            W_QKV = qkv_proj.weight.data
            q_dim = num_heads * d_head
            kv_dim = num_kv_heads * d_head
            W_Q = W_QKV[:q_dim, :]
            W_K = W_QKV[q_dim : q_dim + kv_dim, :]
            W_V = W_QKV[q_dim + kv_dim :, :]
        else:
            raise AttributeError("Could not find QKV projection layers in attention module")
            
    if o_proj is None:
        # Look for any linear layers with 'o' or 'out' in the name
        for name, module in self_attn.named_children():
            if ("o_proj" in name or "out_proj" in name or name.lower() == "o" or name.lower() == "out") and isinstance(module, torch.nn.Linear):
                o_proj = module
                break
    if o_proj is not None:
        W_O = o_proj.weight.data
    else:
        raise AttributeError("Could not find output projection layer in attention module")
        
    return W_Q, W_K, W_V, W_O, num_heads, num_kv_heads, d_head


def slice_tensor(v, n, M):
    if not isinstance(v, torch.Tensor):
        return v
    shape = v.shape
    if len(shape) == 2:
        return v[:, :n]
    elif len(shape) == 3:
        return v[:, :n]
    elif len(shape) == 4:
        if shape[3] == M:
            return v[:, :, :n, :n]
        else:
            return v[:, :, :n, :]
    return v


def slice_arg(v, n, M):
    if isinstance(v, torch.Tensor):
        return slice_tensor(v, n, M)
    elif isinstance(v, tuple):
        return tuple(slice_arg(x, n, M) for x in v)
    elif isinstance(v, list):
        return [slice_arg(x, n, M) for x in v]
    elif isinstance(v, dict):
        return {k: slice_arg(val, n, M) for k, val in v.items()}
    return v


def to_float32(v):
    if isinstance(v, torch.Tensor):
        if torch.is_floating_point(v):
            return v.to(torch.float32)
        return v
    elif isinstance(v, tuple):
        return tuple(to_float32(x) for x in v)
    elif isinstance(v, list):
        return [to_float32(x) for x in v]
    elif isinstance(v, dict):
        return {k: to_float32(val) for k, val in v.items()}
    return v


def to_device(v, device):
    if isinstance(v, torch.Tensor):
        return v.to(device)
    elif isinstance(v, tuple):
        return tuple(to_device(x, device) for x in v)
    elif isinstance(v, list):
        return [to_device(x, device) for x in v]
    elif isinstance(v, dict):
        return {k: to_device(val, device) for k, val in v.items()}
    return v


def compute_attn_jacobian_metrics(model, layer_idx, captured_args, captured_kwargs, N_list, K, device):
    """
    Computes exact global spectral norm of attention block's Jacobian (Algorithm A),
    static weight amplifiers (Algorithm B), dynamic attention entropy (Algorithm C),
    spectral gap of attention matrix (Algorithm D), and token-wise sensitivity (Algorithm E).
    """
    import numpy as np
    from torch.func import jvp, vjp
    
    attn = model.model.layers[layer_idx].self_attn
    config = model.config
    
    # Extract weights
    W_Q, W_K, W_V, W_O, num_heads, num_kv_heads, d_head = extract_attn_weights(attn, config)
    d_model = config.hidden_size
    
    # Cast weights to float32 for computation
    W_Q_32 = W_Q.to(torch.float32)
    W_K_32 = W_K.to(torch.float32)
    W_V_32 = W_V.to(torch.float32)
    W_O_32 = W_O.to(torch.float32)
    
    # Reshape Q, K, V, O to heads
    W_Q_heads = W_Q_32.view(num_heads, d_head, d_model)
    W_K_heads = W_K_32.view(num_kv_heads, d_head, d_model)
    W_V_heads = W_V_32.view(num_kv_heads, d_head, d_model)
    W_O_heads = W_O_32.T.reshape(num_heads, d_head, d_model)
    
    group_size = num_heads // num_kv_heads
    
    # Algorithm B1: Routing Amplifier
    routing_norms = []
    for h in range(num_heads):
        h_kv = h // group_size
        prod = torch.matmul(W_Q_heads[h], W_K_heads[h_kv].T) # [d_head, d_head]
        norm = torch.linalg.matrix_norm(prod, ord=2).item()
        routing_norms.append(norm)
    mean_routing_norm = float(np.mean(routing_norms))
    
    # Algorithm B2: Mixing Amplifier
    mixing_norms = []
    for h in range(num_heads):
        h_kv = h // group_size
        prod = torch.matmul(W_V_heads[h_kv], W_O_heads[h].T) # [d_head, d_head]
        norm = torch.linalg.matrix_norm(prod, ord=2).item()
        mixing_norms.append(norm)
    mean_mixing_norm = float(np.mean(mixing_norms))
    
    # Setup for functional evaluation
    # Switch layer to float32
    original_dtype = next(attn.parameters()).dtype
    attn.to(torch.float32)
    
    # Locate the hidden states tensor in the captured arguments
    hs_loc = None
    if "hidden_states" in captured_kwargs:
        hs_loc = ("kwargs", "hidden_states")
        x_norm_full_raw = captured_kwargs["hidden_states"]
    else:
        for idx, arg in enumerate(captured_args):
            if isinstance(arg, torch.Tensor) and arg.ndim == 3:
                hs_loc = ("args", idx)
                x_norm_full_raw = arg
                break
        else:
            # Fallback to search kwargs for a 3D tensor
            for k, v in captured_kwargs.items():
                if isinstance(v, torch.Tensor) and v.ndim == 3:
                    hs_loc = ("kwargs", k)
                    x_norm_full_raw = v
                    break
                    
    if hs_loc is None:
        raise ValueError(f"Could not locate hidden_states tensor in captured arguments. args types: {[type(a) for a in captured_args]}, kwargs keys: {list(captured_kwargs.keys())}")
        
    M = x_norm_full_raw.shape[1]
    x_norm_full = x_norm_full_raw.squeeze(0).to(torch.float32) # [M, d_model]
    
    if torch.isnan(x_norm_full).any():
        print(f"[Layer {layer_idx}] WARNING: Captured input hidden states contain NaN! The forward pass has likely overflowed.", flush=True)
        
    # Calculate static routing matrix W_QK and its dominant singular vector (Algorithm F / Weight Alignment)
    # W_K_heads is [num_kv_heads, d_head, d_model]
    W_K_expanded = torch.cat([W_K_heads[h // group_size] for h in range(num_heads)], dim=0) # [num_heads * d_head, d_model]
    W_QK = torch.matmul(W_Q_32.T, W_K_expanded) # [d_model, d_model]
    _, _, V_T = torch.linalg.svd(W_QK, full_matrices=False)
    u_route = V_T[0, :] # [d_model]
    
    results_per_N = {}
    
    for n in N_list:
        if n > M:
            continue
            
        x_norm_n = x_norm_full[:n, :].to(device)
        
        # Define functional wrapper attn_func(x) for Jacobian VJP/JVP
        def attn_func(x):
            # x: [n, d_model]
            x_3d = x.unsqueeze(0)
            
            # Slice and prepare args
            sliced_args = []
            for idx, arg in enumerate(captured_args):
                if hs_loc[0] == "args" and hs_loc[1] == idx:
                    sliced_args.append(x_3d)
                else:
                    sliced_args.append(to_device(to_float32(slice_arg(arg, n, M)), device))
                    
            # Slice and prepare kwargs
            sliced_kwargs = {}
            for k, v in captured_kwargs.items():
                if hs_loc[0] == "kwargs" and hs_loc[1] == k:
                    sliced_kwargs[k] = x_3d
                else:
                    sliced_kwargs[k] = to_device(to_float32(slice_arg(v, n, M)), device)
            # Force cache to be disabled during Jacobian computations
            sliced_kwargs["use_cache"] = False
            sliced_kwargs["past_key_value"] = None
                    
            # Call self_attn
            out_tuple = attn(*sliced_args, **sliced_kwargs)
            return out_tuple[0].squeeze(0)
            
        # Algorithm A: Exact Global Spectral Norm via Power Iteration
        v = torch.randn(n, d_model, device=device, dtype=torch.float32)
        v = v / torch.norm(v)
        
        for _ in range(K):
            # Compute pushforward
            _, u = jvp(attn_func, (x_norm_n,), (v,))
            u = u.detach()
            # Compute pullback setup
            _, vjp_func = vjp(attn_func, x_norm_n)
            # Compute pullback vector
            w = vjp_func(u)[0]
            w = w.detach()
            # Rayleigh quotient
            sigma_sq = torch.sum(w * v) / torch.sum(v * v)
            # Normalize
            v = w / torch.norm(w)
            
        attn_spectral_norm = float(torch.sqrt(torch.clamp(sigma_sq, min=0.0)).item())
        
        # Algorithm E: Token-Wise Sensitivity
        token_sensitivity = torch.norm(v, dim=-1) # [n]
        token_sensitivity_sum = token_sensitivity.sum()
        if token_sensitivity_sum > 0:
            token_sensitivity = token_sensitivity / token_sensitivity_sum
        token_sensitivity_profile = token_sensitivity.cpu().numpy().tolist()
        
        # Singular Vector-Weight Alignment Index (\alpha_\ell)
        v_peak = v[0, :] # Token 0 is empirically the peak sensitivity index
        alignment = torch.abs(torch.dot(v_peak.float(), u_route.float())) / (torch.norm(v_peak.float()) * torch.norm(u_route.float()))
        weight_alignment_index = float(alignment.item())
        
        x_norm_mean = torch.norm(x_norm_n.float(), dim=-1).mean().item()
        
        # Algorithm C & D: Extract Attention weights and compute Entropy / Spectral Gap
        with torch.no_grad():
            x_3d = x_norm_n.unsqueeze(0)
            
            sliced_args = []
            for idx, arg in enumerate(captured_args):
                if hs_loc[0] == "args" and hs_loc[1] == idx:
                    sliced_args.append(x_3d)
                else:
                    sliced_args.append(to_device(to_float32(slice_arg(arg, n, M)), device))
                    
            sliced_kwargs = {}
            for k, v in captured_kwargs.items():
                if hs_loc[0] == "kwargs" and hs_loc[1] == k:
                    sliced_kwargs[k] = x_3d
                else:
                    sliced_kwargs[k] = to_device(to_float32(slice_arg(v, n, M)), device)
            sliced_kwargs["output_attentions"] = True
            # Force cache to be disabled during Jacobian computations
            sliced_kwargs["use_cache"] = False
            sliced_kwargs["past_key_value"] = None
            
            out_tuple = attn(*sliced_args, **sliced_kwargs)
            attn_weights = out_tuple[1]
            
        if attn_weights is not None:
            # Squeeze batch dimension -> [num_heads, n, n]
            A = attn_weights.squeeze(0).to(torch.float32)
            
            # Algorithm C: Shannon Entropy
            epsilon = 1e-12
            A_clamped = torch.clamp(A, min=0.0)
            entropy_matrix = - A_clamped * torch.log2(A_clamped + epsilon)
            # Mask lower triangular part (causal mask)
            mask = torch.tril(torch.ones(n, n, device=device))
            entropy_matrix = entropy_matrix * mask
            row_entropy = entropy_matrix.sum(dim=-1) # [num_heads, n]
            mean_attention_entropy = float(row_entropy.mean().item())
            min_attn_entropy = float(row_entropy.min().item())
            max_attn_entropy = float(row_entropy.max().item())
            mean_max_weight = float(A.max(dim=-1)[0].mean().item())
            
            # Algorithm D: Spectral Gap
            svd_vals = torch.linalg.svdvals(A) # [num_heads, n]
            if n >= 2:
                sigma_2 = svd_vals[:, 1] # [num_heads]
                spectral_gap = 1.0 - sigma_2
                mean_spectral_gap = float(spectral_gap.mean().item())
            else:
                mean_spectral_gap = 1.0 # default for n < 2
        else:
            mean_attention_entropy = 0.0
            min_attn_entropy = 0.0
            max_attn_entropy = 0.0
            mean_max_weight = 0.0
            mean_spectral_gap = 1.0
            
        # Calculate exact theoretical mean max entropy for causal attention
        # Row i (from 1 to n) has maximum entropy log2(i)
        exact_h_max = torch.mean(torch.log2(torch.arange(1, n + 1, dtype=torch.float32, device=device)))
        entropy_ratio = mean_attention_entropy / exact_h_max.item() if exact_h_max.item() > 0.0 else 0.0
            
        results_per_N[n] = {
            "attn_spectral_norm": attn_spectral_norm,
            "mean_attn_entropy": mean_attention_entropy,
            "min_attn_entropy": min_attn_entropy,
            "max_attn_entropy": max_attn_entropy,
            "entropy_ratio": entropy_ratio,
            "mean_max_weight": mean_max_weight,
            "x_norm_mean": x_norm_mean,
            "mean_spectral_gap": mean_spectral_gap,
            "token_sensitivity_profile": token_sensitivity_profile,
            "weight_alignment_index": weight_alignment_index
        }
        
        # Free GPU memory
        del x_norm_n, v, w, u, vjp_func
        if attn_weights is not None:
            del attn_weights, A
        torch.cuda.empty_cache()
        
    # Restore original dtype
    attn.to(original_dtype)
    
    return {
        "routing_weight_norm": mean_routing_norm,
        "mixing_weight_norm": mean_mixing_norm,
        "seq_lengths": results_per_N
    }

