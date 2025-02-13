"""Useful functions for getting distance information and embedding information."""

import functools
import torch
from torch.nn.functional import softmax



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


def get_attention_and_entropy_for_head_modified(
    attentions, inputs, tokenizer, target, disambiguating, layer, head, device
):
    """
    Get entropy over attention from a target token to all tokens,
    attention from a target token to a specific disambiguating token,
    number of tokens in the target and disambiguating words,
    and attention to the token just before the target.
    
    Args:
        attentions: Attention weights from the model.
        inputs: Tokenized input data.
        tokenizer: Corresponding tokenizer.
        target (str): Target word.
        disambiguating (str): Disambiguating word.
        layer (int): Layer index for attention extraction.
        head (int): Head index for attention extraction.
        device (str): Device to run computations on (e.g., 'cpu', 'cuda').
    
    Returns:
        dict: Contains entropy of attention distribution, attention to disambiguating token,
              attention to the token just before the target, number of tokens in target,
              and number of tokens in disambiguating word.
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
    
    # Get number of tokens in target and disambiguating words
    num_target_tokens = len(target_enc[0])
    num_disambiguating_tokens = len(disambiguating_enc[0])
    
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
    
    # Get attention to the token just before the target
    target_prev_index = max(0, target_inds[0] - 1)  # Ensure index is not negative
    attention_to_previous = attention_distribution[target_prev_index].item()
    
    return {
        "entropy": entropy,
        "attention_to_disambiguating": attention_to_disambiguating,
        "attention_to_previous": attention_to_previous,
        "num_target_tokens": num_target_tokens,
        "num_disambiguating_tokens": num_disambiguating_tokens,
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
    print(f"Total Trainable Params: {total_params}")
    
    return total_params