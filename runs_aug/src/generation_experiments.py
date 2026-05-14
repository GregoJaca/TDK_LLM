import os
import torch
from src.utils import ensure_dir

def generate_and_save_hidden_states(
    model, 
    attention_mask, 
    perturbed_embeddings, 
    selected_layers, 
    results_dir, 
    run_name,
    max_new_tokens=0,
    batch_size=64
):
    """
    Evaluates the model on the perturbed embeddings and saves the hidden states.
    If max_new_tokens == 0: Performs a single forward pass over the prompt (prefill).
    If max_new_tokens > 0: Autoregressively generates tokens and concatenates the states.
    """
    ensure_dir(results_dir)
    n_conditions = perturbed_embeddings.shape[0]
    
    layer_storages = {layer_idx: [] for layer_idx in selected_layers}
    
    model.eval()
    for i in range(0, n_conditions, batch_size):
        current_embedding = perturbed_embeddings[i:i+batch_size]
        current_batch_size = current_embedding.shape[0]
        
        # Expand attention mask to match current batch size
        current_attention_mask = attention_mask.expand(current_batch_size, -1)
        
        with torch.no_grad():
            if max_new_tokens == 0:
                outputs = model(
                    inputs_embeds=current_embedding,
                    attention_mask=current_attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                for layer_idx in selected_layers:
                    states = outputs.hidden_states[layer_idx].cpu() # [batch, seq_len, hidden_size]
                    for b in range(current_batch_size):
                        layer_storages[layer_idx].append(states[b])
            else:
                outputs = model.generate(
                    inputs_embeds=current_embedding,
                    attention_mask=current_attention_mask,
                    max_new_tokens=max_new_tokens,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    do_sample=False # Deterministic generation for consistent perturbation analysis
                )
                
                for layer_idx in selected_layers:
                    # Prompt hidden states: outputs.hidden_states[0][layer_idx] -> [batch, seq_len, hidden_size]
                    prompt_states = outputs.hidden_states[0][layer_idx].cpu()
                    
                    # Generated hidden states: outputs.hidden_states[step][layer_idx] -> [batch, 1, hidden_size]
                    gen_states = []
                    for step_idx in range(1, len(outputs.hidden_states)):
                        step_state = outputs.hidden_states[step_idx][layer_idx].cpu()
                        gen_states.append(step_state)
                        
                    for b in range(current_batch_size):
                        p_state = prompt_states[b]
                        g_states = [s[b] for s in gen_states]
                        full_trajectory = torch.cat([p_state] + g_states, dim=0)
                        layer_storages[layer_idx].append(full_trajectory)
                    
    # Save the tensors per layer
    # Output shape for each file: [n_conditions, seq_len (+ max_new_tokens), hidden_size]
    for layer_idx in selected_layers:
        stacked_states = torch.stack(layer_storages[layer_idx], dim=0) 
        filepath = os.path.join(results_dir, f"{run_name}_layer_{layer_idx}.pt")
        torch.save(stacked_states, filepath)
