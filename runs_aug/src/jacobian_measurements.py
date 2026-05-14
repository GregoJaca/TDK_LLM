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
        W_gate = mlp.gate_proj.weight.data
        W_up = mlp.up_proj.weight.data
        W_down = mlp.down_proj.weight.data
        
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
