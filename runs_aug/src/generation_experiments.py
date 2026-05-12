import os
import torch
from src.utils import ensure_dir

def generate_and_save_hidden_states(
    model, 
    attention_mask, 
    perturbed_embeddings, 
    selected_layers, 
    results_dir, 
    run_name
):
    """
    Evaluates the model on the perturbed embeddings and saves the hidden states.
    Performs a single forward pass over the prompt (no autoregressive generation)
    to observe how the perturbation propagates to the hidden states across layers.
    """
    ensure_dir(results_dir)
    n_conditions = perturbed_embeddings.shape[0]
    
    layer_storages = {layer_idx: [] for layer_idx in selected_layers}
    
    model.eval()
    for i in range(n_conditions):
        current_embedding = perturbed_embeddings[i].unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(
                inputs_embeds=current_embedding,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            for layer_idx in selected_layers:
                # outputs.hidden_states is a tuple of (num_layers + 1) tensors
                # each tensor is of shape [batch_size, seq_len, hidden_size]
                # Extract the sequence hidden states for the current condition
                state = outputs.hidden_states[layer_idx][0].cpu() # [seq_len, hidden_size]
                layer_storages[layer_idx].append(state)
                
    # Save the tensors per layer
    # Output shape for each file: [n_conditions, seq_len, hidden_size]
    for layer_idx in selected_layers:
        stacked_states = torch.stack(layer_storages[layer_idx], dim=0) 
        filepath = os.path.join(results_dir, f"{run_name}_layer_{layer_idx}.pt")
        torch.save(stacked_states, filepath)
