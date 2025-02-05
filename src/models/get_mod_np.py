"""

Compute logprob ratios for original and reversed modifier-noun phrases.

Specifically, we calculate the ratio of the log-probability of an original modifier NP construction ("desperate act") 
vs. the reversed version ("act desperate"), each presented after an *end of sequence token* to each Pythia model.

See V2 for an alternative version that presents the original and reversed versions in entire sentences (e.g., "It was a desperate act") 
and analyzes the relative log-probability of the modifier-NP phrases respectively.
"""

import numpy as np
import os
import pandas as pd
import transformers
import torch


from scipy.spatial.distance import cosine
from tqdm import tqdm
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import statsmodels.formula.api as smf


import utils



### Models to test
MODELS = [
          # 'EleutherAI/pythia-14m',
         'EleutherAI/pythia-70m',
         # 'EleutherAI/pythia-160m',
          # 'EleutherAI/pythia-410m',
          # 'EleutherAI/pythia-1b',
          # 'EleutherAI/pythia-1.4b',
          # 'EleutherAI/pythia-2.8b',
          # 'EleutherAI/pythia-6.9b',
          # 'EleutherAI/pythia-12b',
          ]

STIMULI = "data/raw/rawc/rawc_mod_np.csv"




def get_logprob(model, tokenizer, text, device):
    """Compute logprob of entire sequence in TEXT, after adding start token."""

    ## Add start token
    input_with_start = tokenizer.bos_token + " " + text

    ## Tokenize
    input_ids = tokenizer(input_with_start, return_tensors="pt").input_ids.to(device)
    # Get model output logits
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits
    
    # Calculate token probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Shift input IDs to align tokens with their log probabilities
    shifted_input_ids = input_ids[:, 1:]  # Remove first token (start-of-sequence token)
    shifted_log_probs = log_probs[:, :-1, :]  # Remove last log_probs to match sequence length
    
    # Gather log probabilities for the actual sequence tokens
    sequence_log_probs = shifted_log_probs.gather(2, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
    
    # Sum log probabilities for the entire sequence
    total_log_prob = sequence_log_probs.sum().item()
    
    return total_log_prob


### Handle logic for a dataset/model
def main(df, mpath, revisions):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("number of checkpoints:", len(revisions))



    for checkpoint in tqdm(revisions):
        print(checkpoint)

        ### Set up save path, filename, etc.
        savepath = "data/processed/rawc/pythia/mod_np_cxn/"
        if not os.path.exists(savepath): 
            os.mkdir(savepath)
        if "/" in mpath:
            filename = "rawc-mod_np_cxn_model-" + mpath.split("/")[1] + "-" + checkpoint +  ".csv"
        else:
            filename = "rawc-mod_np_cxn_model-" + mpath +  "-" + checkpoint + ".csv"

        print("Checking if we've already run this analysis...")
        if os.path.exists(os.path.join(savepath,filename)):
            print("Already run this model for this checkpoint.")
            continue

        ### if it doesn't exist, run it.
        model = GPTNeoXForCausalLM.from_pretrained(
            mpath,
            revision=checkpoint,
            output_hidden_states = True
        )
        model.to(device) # allocate model to desired device

        tokenizer = AutoTokenizer.from_pretrained(mpath, revision=checkpoint)


        n_layers = model.config.num_hidden_layers
        print("number of layers:", n_layers)
        n_heads = model.config.num_attention_heads
        print("number of heads:", n_heads)
    
        n_params = utils.count_parameters(model)
    
        results = []

        ### TODO: Why tqdm not working here?
        for (ix, row) in tqdm(df.iterrows(), total=df.shape[0]):

            ### Get targets
            original_mod = row['mod_np']
            reversed_mod = row['reversed_mod_np']

            ### Get logprobs
            original_logprob = get_logprob(model, tokenizer, original_mod, device)
            reversed_logprob = get_logprob(model, tokenizer, reversed_mod, device)
   
            ### Add to results dictionary
            results.append({
                'sentence': row['sentence'],
                'word': row['word'],
                'string': row['string'],
                'disambiguating_word': row['disambiguating_word'],
                'non_adjective': row['non_adjective'],
                'original_logprob': original_logprob,
                'reversed_logprob': reversed_logprob,
                'ratio': original_logprob - reversed_logprob
            })


        df_results = pd.DataFrame(results)
        df_results['n_params'] = np.repeat(n_params,df_results.shape[0])
        df_results['mpath'] = mpath
        df_results['revision'] = checkpoint
        df_results['step'] = int(checkpoint.replace("step", ""))
        
    
        df_results.to_csv(os.path.join(savepath,filename), index=False)


if __name__ == "__main__":

    ## Read stimuli
    df = pd.read_csv(STIMULI)

    ### Get revisions
    revisions = utils.generate_revisions()

    ## Run main
    main(df, MODELS[0], revisions)