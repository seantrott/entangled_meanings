"""Run a modified Pythia14-M instance over the RAW-C dataset, over pre-training checkpoints."""


### TODO: Cache already-run models/checkpoints somewhere so we don't have to run again.

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


### Models to modify and test
MODELS = [
         'EleutherAI/pythia-14m',
          ]

STIMULI = "data/raw/rawc/rawc_stimuli.csv"

MODIFICATIONS = ["zeroed"]


def main(df, mpath, revisions, modification, layer_idx, head_idx):
    """
    Modify the Q and K weight matrices for a specific layer and attention head.
    
    Args:
        model: The Pythia model
        revisions: List of model checkpoints
        layer_idx: Index of the layer to modify
        head_idx: Index of the attention head to modify
        q_modification: Tensor with modification to apply to the Q weights
        k_modification: Tensor with modification to apply to the K weights
    
    Returns:
        The modified model
    """

    # Specify the device you'll allocate the model to
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("number of checkpoints:", len(revisions))

    # Start modifying from the checkpoint just before step1000, where the inflection point 
    # in performance comes in
    for checkpoint in tqdm(revisions):

        # Set up save path, filename, etc.
        savepath = "../../data/processed/rawc/pythia-QKmod-zero/distances/"
        if not os.path.exists(savepath): 
            os.mkdir(savepath)
        if "/" in mpath:
            filename = "rawc-distances_model-" + mpath.split("/")[1] + "-" + checkpoint +  ".csv"
        else:
            filename = "rawc-distances_model-" + mpath +  "-" + checkpoint + ".csv"

        print("Checking if we've already run this analysis...")
        if os.path.exists(os.path.join(savepath,filename)):
            print("Already run this model for this checkpoint.")
            continue


        # Make a deep copy of the model to avoid modifying the original
        modified_model = GPTNeoXForCausalLM.from_pretrained(
            mpath,
            revision=checkpoint,
            return_dict_in_generate = True,
            output_hidden_states = True
        )
        modified_model.to(device) # allocate model to desired device

        tokenizer = AutoTokenizer.from_pretrained(mpath, revision=checkpoint)
    
        # Get the attention module for the specified layer
        attention_module = modified_model.gpt_neox.layers[layer_idx].attention
        
        # Get the current Q and K weight matrices
        # For Pythia, these are typically in the query_key_value projection
        qkv_weight = attention_module.query_key_value.weight
        
        # Take a look
        # utils.visualize_matrix(qkv_weight.detach(),"pre-mod")

        # Pythia14M combines Q, K, V into one projection
        # Extract dimensions
        hidden_size = modified_model.config.hidden_size
        num_heads = modified_model.config.num_attention_heads
        head_size = hidden_size // num_heads
        
        #### -----------------------------
        #### ----- APPLY MODIFICATION ----
        #### -----------------------------
        
        utils.modify_attention_weights(modification, qkv_weight, hidden_size, num_heads, head_size, head_idx, device)
        # utils.visualize_matrix(qkv_weight.detach(),"post-mod")

        print(f"Modified Q and K weights for layer {layer_idx}, head {head_idx}")

        n_layers = model.config.num_hidden_layers
        print("number of layers:", n_layers)
        
        results = []

        for (ix, row) in tqdm(df.iterrows(), total=df.shape[0]):

            ### Get word
            target = " {w}".format(w = row['string'])

            ### Run model for each sentence
            s1_outputs = utils.run_model(modified_model, tokenizer, row['sentence1'], device)
            s2_outputs = utils.run_model(modified_model, tokenizer, row['sentence2'], device)

            ### Now, for each layer...
            for layer in range(n_layers+1): # `range` is non-inclusive for the last value of interval
    
                ### Get embeddings for word
                s1 = utils.get_embedding(s1_outputs['hidden_states'], s1_outputs['tokens'], tokenizer, target, layer, device)
                s2 = utils.get_embedding(s2_outputs['hidden_states'], s2_outputs['tokens'], tokenizer, target, layer, device)
    
                ### Now calculate cosine distance 
                #.  note, tensors need to be copied to cpu to make this run;
                #.  still faster to do this copy than to just have everything
                #.  running on the cpu
                if device.type == "mps":  
                    model_cosine = cosine(s1.cpu(), s2.cpu())
    
                else: 
                    model_cosine = cosine(s1, s2)
    
    
                if row['same'] == True:
                    same_sense = "Same Sense"
                else:
                    same_sense = "Different Sense"
    
    
                ### Figure out how many tokens you're
                ### comparing across sentences
                n_tokens_s1 = len(tokenizer.encode(row['sentence1']))
                n_tokens_s2 = len(tokenizer.encode(row['sentence2']))
    
                ### Add to results dictionary
                results.append({
                    'sentence1': row['sentence1'],
                    'sentence2': row['sentence2'],
                    'word': row['word'],
                    'string': row['string'],
                    'Same_sense': same_sense,
                    'Distance': model_cosine,
                    'Layer': layer,
                    'mean_relatedness': row['mean_relatedness'],
                    'S1_ntokens': n_tokens_s1,
                    'S2_ntokens': n_tokens_s2
                })
    
        df_results = pd.DataFrame(results)
        df_results['token_diffs'] = np.abs(df_results['S1_ntokens'].values-df_results['S2_ntokens'].values)
        df_results['n_params'] = np.repeat(n_params,df_results.shape[0])
        df_results['mpath'] = mpath
        df_results['revision'] = checkpoint
        df_results['step'] = int(checkpoint.replace("step", ""))
        df_results['model_modification'] = np.repeat(modification,df_results.shape[0])
        
        
        df_results.to_csv(os.path.join(savepath,filename), index=False)



if __name__ == "__main__":


    ## Read stimuli
    df = pd.read_csv(STIMULI)
    df_just_n = df[df['Class']=='N']

    ### Get revisions
    revisions = utils.generate_revisions_post1000()

    modification = MODIFICATIONS[0]

    ## Run main
    main(df_just_n, MODELS[0], revisions, modification, layer_idx, head_idx)

