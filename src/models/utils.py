"""Useful functions for getting distance information and embedding information."""

import functools
import random
import torch
from torch.nn.functional import softmax
from transformers import GPTNeoXForCausalLM, AutoTokenizer


import matplotlib.pyplot as plt
import seaborn as sns

## TODO: have to write a unit test for the weights swap --- 
## subtract the modified matrices from the intact matrix 
## (check that the correct indices are different)

def generate_revisions():
    ## TODO: Ensure this is correct
    # Fixed initial steps
    revisions = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
    # Add every 1,000 steps afterward
    revisions.extend(range(2000, 144000, 1000))  # Adjust range as needed
    # Format each step as "stepX"
    return [f"step{step}" for step in revisions]


def generate_revisions_test():
    # Fixed initial steps
    revisions = [143000]
    # revisions = [143000]
    return [f"step{step}" for step in revisions]

def generate_revisions_post512():

    revisions = [512, 1000]
    # Add every 1,000 steps afterward
    revisions.extend(range(2000, 144000, 1000))  # Adjust range as needed
    # Format each step as "stepX"
    return [f"step{step}" for step in revisions]



def find_sublist_index(mylist, sublist):
    """Find the first occurence of sublist in list.
    Return the start and end indices of sublist in list"""

    for i in range(len(mylist)):
        if mylist[i] == sublist[0] and mylist[i:i+len(sublist)] == sublist:
            return i, i+len(sublist)
    return None

@functools.lru_cache(maxsize=None)  # This will cache results, handy later...



def run_model(model, tokenizer, sentence, device):
    """Run model, return hidden states and attention"""
    # Tokenize sentence
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    # Run model
    with torch.no_grad():
        output = model(**inputs, output_attentions=True)
        hidden_states = output.hidden_states
        attentions = output.attentions

    return {'hidden_states': hidden_states,
            'attentions': attentions,
            'tokens': inputs}


### ... grab the embeddings for your target tokens
def get_embedding(hidden_states, inputs, tokenizer, target, layer, device):
    """Extract embedding for TARGET from set of hidden states and token ids."""
    
    # Tokenize target
    target_enc = tokenizer.encode(target, return_tensors="pt",
                                  add_special_tokens=False).to(device)
    
    # Get indices of target in input tokens
    target_inds = find_sublist_index(
        inputs["input_ids"][0].tolist(),
        target_enc[0].tolist()
    )

    # Get layer
    selected_layer = hidden_states[layer][0]

    #grab just the embeddings for your target word's token(s)
    token_embeddings = selected_layer[target_inds[0]:target_inds[1]]

    #if a word is represented by >1 tokens, take mean
    #across the multiple tokens' embeddings
    embedding = torch.mean(token_embeddings, dim=0)
    
    return embedding



def calculate_attention_entropy(attention_distribution):
    """
    Calculate entropy over an attention distribution.
    
    Args:
        attention_distribution (torch.Tensor): Attention weights for a single token 
                                                (1D tensor of size seq_len).
    
    Returns:
        float: Entropy value.
    """
    # Normalize the attention distribution using softmax
    attention_probs = softmax(attention_distribution, dim=-1)
    
    # Avoid log(0) by masking zeros
    attention_probs = attention_probs + 1e-12  # Small epsilon to prevent NaN
    
    # Compute entropy: -sum(p * log(p))
    entropy = -torch.sum(attention_probs * torch.log(attention_probs)).item()
    
    return entropy

def get_attention_and_entropy_for_head(
    attentions, inputs, tokenizer, target, disambiguating, layer, head, device
):
    """
    Get entropy over attention from a target token to all tokens,
    and the attention from a target token to a specific disambiguating token
    for a specified head in a given layer.
    
    Args:
        model: Pretrained Transformer model.
        tokenizer: Corresponding tokenizer.
        sentence (str): Input sentence.
        target (str): Target word.
        disambiguating (str): Disambiguating word.
        layer (int): Layer index for attention extraction.
        head (int): Head index for attention extraction.
        device (str): Device to run computations on (e.g., 'cpu', 'cuda').
    
    Returns:
        dict: Contains entropy of attention distribution, attention to disambiguating token, 
              and attention distribution.
    """
    # Tokenize target and disambiguating words
    target_enc = tokenizer.encode(target, return_tensors="pt", add_special_tokens=False).to(device)
    disambiguating_enc = tokenizer.encode(disambiguating, return_tensors="pt", add_special_tokens=False).to(device)
    
    # Find indices of target and disambiguating words in input tokens
    target_inds = find_sublist_index(
        inputs["input_ids"][0].tolist(),
        target_enc[0].tolist()
    )
    disambiguating_inds = find_sublist_index(
        inputs["input_ids"][0].tolist(),
        disambiguating_enc[0].tolist()
    )
    
    if target_inds is None:
        raise ValueError(f"Target word '{target}' not found in the tokenized input.")
    if disambiguating_inds is None:
        raise ValueError(f"Disambiguating word '{disambiguating}' not found in the tokenized input.")
    
    # Extract attention from the specified layer
    attention_layer = attentions[layer][0]  # Shape: (num_heads, seq_len, seq_len)
    
    # Select the specified head
    attention_head = attention_layer[head]  # Shape: (seq_len, seq_len)
    
    # Get attention distribution for the target token(s)
    target_attention = attention_head[target_inds[0]:target_inds[1]]  # Shape: (target_len, seq_len)
    
    # Average over multiple tokens if target spans multiple subwords
    attention_distribution = torch.mean(target_attention, dim=0)  # Shape: (seq_len)
    
    # Calculate entropy over the attention distribution
    attention_probs = softmax(attention_distribution, dim=-1)
    entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-12)).item()
    
    # Calculate attention to the disambiguating token(s)
    disambiguating_attention = attention_distribution[
        disambiguating_inds[0]:disambiguating_inds[1]
    ]
    attention_to_disambiguating = torch.mean(disambiguating_attention).item()
    
    return {
        "entropy": entropy,
        "attention_to_disambiguating": attention_to_disambiguating,
        "attention_distribution": attention_distribution
    }

### ... grab the number of trainable parameters in the model

def count_parameters(model):
    """credit: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model"""
    
    total_params = 0
    for name, parameter in model.named_parameters():
        
        # if the param is not trainable, skip it
        if not parameter.requires_grad:
            continue
        
        # otherwise, count it towards your number of params
        params = parameter.numel()
        total_params += params
    # print(f"Total Trainable Params: {total_params}")
    
    return total_params

def visualize_matrix(matrix, title, figname,show, color_scale_tuple):
    plt.figure(figsize=(6, 6))
    if color_scale_tuple == None: 
        sns.heatmap(matrix, cmap="coolwarm")
    else: 
        sns.heatmap(matrix, cmap="coolwarm", vmin = color_scale_tuple[0], vmax = color_scale_tuple[1])
    plt.title(title)
    if not figname == None: 
        plt.savefig(figname)
    if show == 1: 
        plt.show()



def select_random_heads(num_layers, num_heads_per_layer, num_samples=1, seed=None, target_layer=None, blocked_heads=None):
    """
    Randomly select (layer, head) pairs, with options to restrict to a specific layer and avoid specific heads.

    Args:
        num_layers (int): Total number of layers.
        num_heads_per_layer (int): Number of attention heads per layer.
        num_samples (int): Number of (layer, head) pairs to sample.
        seed (int, optional): Seed for reproducibility.
        target_layer (int, optional): If set, only sample from this layer.
        blocked_heads (list, optional): List of head indices to exclude from sampling.

    Returns:
        dict: {'layers': [...], 'heads': [...]}
    """
    if seed is not None:
        random.seed(seed)
    
    blocked_heads = set(blocked_heads or [])

    if target_layer is not None:
        all_combinations = [
            (target_layer, head)
            for head in range(num_heads_per_layer)
            if head not in blocked_heads
        ]
    else:
        all_combinations = [
            (layer, head)
            for layer in range(num_layers)
            for head in range(num_heads_per_layer)
            if head not in blocked_heads
        ]

    if num_samples > len(all_combinations):
        raise ValueError("Not enough valid (layer, head) combinations to sample from.")

    selected = random.sample(all_combinations, num_samples)
    layer_indices = [layer for layer, _ in selected]
    head_indices = [head for _, head in selected]

    LAYERS_HEADS_IDX = {"layers": layer_indices,
                        "heads": head_indices}

    return LAYERS_HEADS_IDX



# ----
# CAUSAL MANIPULATION
# 1. zero out a given matrix (QK for target head)

# 2. assign randomized values to a given matrix (QK for target head)

# 3. rescale a given matrix (QK for target head) by some scalar value (negative to positive)
# ----

def modify_attention_weights(mpath, model, checkpoint, modification, layer_idx, head_idx, device):
    """
    Modify the Q and K weight matrices for a specific layer and attention head.
    
    Args:
        modification: string, indicating what kind of ablation/operation to perform on 
                      a given head's Q & K matrices
        qkv_weight: the attention matrix for a given layer from a Pythia model, e.g. 
                    model.config.layers[layer_idx].attention.query_key_value.weight
                    will contain all of the layer's heads' q, k, and v matrices concatenated
        hidden_size: model embedding dimension, from the model config, # columns of qkv_weight
        num_heads: number of attention heads per model layer, from the model config
        head_size: attention head dimensionality, hidden_size//num_heads
        head_idx: head to ablate in this layer, specified by investigator
        device:   machine location to allocate model to, e.g. 
                  torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    """

    # Pythia14M combines Q, K, V into one projection
    # Extract dimensions
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_size = hidden_size // num_heads

    # Specify the starting indices for Q and K for the specific head
    q_start_idx = head_idx * head_size * 3  # *3 because QKV are concatenated
    k_start_idx = q_start_idx + head_size

    if modification == "ablate_copy_step1": 

        step1_model = GPTNeoXForCausalLM.from_pretrained(
                        mpath,
                        revision="step1",
                        return_dict_in_generate = True,
                        output_hidden_states = True).to(device)

        qkv_step1 = step1_model.gpt_neox.layers[layer_idx].attention.query_key_value.weight
        q_modification = qkv_step1.data[q_start_idx:q_start_idx+head_size, :]
        k_modification = qkv_step1.data[k_start_idx:k_start_idx+head_size, :]

        q_modification.to(device)
        k_modification.to(device)

    elif modification == "rescue_copy_step143000": 

        step143000_model = GPTNeoXForCausalLM.from_pretrained(
                        mpath,
                        revision="step143000",
                        return_dict_in_generate = True,
                        output_hidden_states = True).to(device)

        qkv_step143000 = step143000_model.gpt_neox.layers[layer_idx].attention.query_key_value.weight
        q_modification = qkv_step143000.data[q_start_idx:q_start_idx+head_size, :]
        k_modification = qkv_step143000.data[k_start_idx:k_start_idx+head_size, :]

        q_modification.to(device)
        k_modification.to(device)

    elif modification == "ablate_zero":

        q_modification = torch.zeros(head_size, hidden_size).to(device)
        k_modification = torch.zeros(head_size, hidden_size).to(device)
    
    
    # Modify the model with the new weights
    model.gpt_neox.layers[layer_idx].attention.query_key_value.weight.data[q_start_idx:q_start_idx+head_size, :] = q_modification
    model.gpt_neox.layers[layer_idx].attention.query_key_value.weight.data[k_start_idx:k_start_idx+head_size, :] = k_modification
        

    # # Get the attention module for the specified layer of the model to be modified
    # attention_module = model.gpt_neox.layers[layer_idx].attention
    
    # # Get the current Q and K weight matrices
    # # For Pythia, these are typically in the query_key_value projection
    # qkv_weight = attention_module.query_key_value.weight

    # qkv_weight.data[q_start_idx:q_start_idx+head_size, :] = q_modification
    # qkv_weight.data[k_start_idx:k_start_idx+head_size, :] = k_modification

    
    return model, q_start_idx, k_start_idx, head_size, hidden_size



