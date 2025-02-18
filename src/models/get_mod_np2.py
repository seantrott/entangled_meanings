"""

Compute logprob ratios for original and reversed modifier-noun phrases.

Specifically, calculates log-prob for modifier NP in context ("desperate act" in "It was a desperate act")
for ocmparison to reversed ("act desperate" in "It was an act desperate"). 

See V1 for an alternative approach. 
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
         # 'EleutherAI/pythia-70m',
         # 'EleutherAI/pythia-160m',
           'EleutherAI/pythia-410m',
          # 'EleutherAI/pythia-1b',
          # 'EleutherAI/pythia-1.4b',
          # 'EleutherAI/pythia-2.8b',
          # 'EleutherAI/pythia-6.9b',
          # 'EleutherAI/pythia-12b',
          ]

STIMULI = "data/raw/rawc/rawc_mod_np.csv"




def next_seq_prob(model, tokenizer, seen, unseen):
    """Get p(unseen | seen)

    Parameters
    ----------
    model : transformers.PreTrainedModel
        Model to use for predicting tokens
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for Model
    seen : str
        Input sequence
    unseen: str
        The sequence for which to calculate a probability
    """
    # Get ids for tokens
    input_ids = tokenizer.encode(seen, return_tensors="pt")
    unseen_ids = tokenizer.encode(unseen)

    # Loop through unseen tokens & store log probs
    log_probs = []
    for unseen_id in unseen_ids:
        
        # Run model on input
        with torch.no_grad():
            logits = model(input_ids).logits

        # Get next token prediction logits
        next_token_logits = logits[0, -1]
        next_token_probs = torch.softmax(next_token_logits, 0) # Normalize

        # Get probability for relevant token in unseen string & store
        prob = next_token_probs[unseen_id]
        log_probs.append(torch.log(prob))

        # Add input tokens incrementally to input
        input_ids = torch.cat((input_ids, torch.tensor([[unseen_id]])), 1)

    # Add log probs together to get total log probability of sequence
    total_log_prob = sum(log_probs)
    
    return total_log_prob.item()



### Handle logic for a dataset/model
def main(df, mpath, revisions):

    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("number of checkpoints:", len(revisions))



    for checkpoint in tqdm(revisions):
        print(checkpoint)

        ### Set up save path, filename, etc.
        savepath = "data/processed/rawc/pythia/mod_np_cxn2/"
        if not os.path.exists(savepath): 
            os.mkdir(savepath)
        if "/" in mpath:
            filename = "rawc-mod_np_cxn2_model-" + mpath.split("/")[1] + "-" + checkpoint +  ".csv"
        else:
            filename = "rawc-mod_np_cxn2_model-" + mpath +  "-" + checkpoint + ".csv"

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
        # model.to(device) # allocate model to desired device

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
            original_mod = " " + row['mod_np']
            reversed_mod = " " + row['reversed_mod_np']

            ### Get targets
            original_sentence = row['sentence'].replace(original_mod, "").replace(".", "")
            reversed_sentence = row['reversed_and_modified_sentence'].replace(reversed_mod, "").replace(".", "")

            ### Get logprobs
            original_logprob = next_seq_prob(model, tokenizer, original_sentence, original_mod)
            reversed_logprob = next_seq_prob(model, tokenizer, reversed_sentence, reversed_mod)
   
            ### Add to results dictionary
            results.append({
                'sentence': row['sentence'],
                'word': row['word'],
                'string': row['string'],
                'reversed_sentence': row['reversed_and_modified_sentence'],
                'mod_np': original_mod,
                'reversed_mod': reversed_mod,
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
    revisions = utils.generate_revisions_test()

    ## Run main
    main(df, MODELS[0], revisions)