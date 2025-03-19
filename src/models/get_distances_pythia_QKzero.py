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
MODELS = ['EleutherAI/pythia-14m']

STIMULI = "../../data/raw/rawc/rawc_stimuli.csv"

MODIFICATIONS = ["ablate_zero",
                 "ablate_copy_step1",
                 "rescue_copy_step143000"]


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
        savepath = f"../../data/processed/rawc/pythia-QKmod/distances_{modification}_layer{layer_idx}head{head_idx}/"
        if not os.path.exists(savepath): 
            os.makedirs(savepath)
        if "/" in mpath:
            filename = "rawc-distances_model-" + mpath.split("/")[1] + "-" + checkpoint +  ".csv"
        else:
            filename = "rawc-distances_model-" + mpath +  "-" + checkpoint + ".csv"

        # Set up figure savepath
        figsavepath = f"../../data/processed/rawc/pythia-QKmod/figures/{modification}"
        if not os.path.exists(figsavepath): 
            os.makedirs(figsavepath)

        # Skip this checkpoint's analysis if you've already run it before
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
        
        premod_qkv_weight = modified_model.gpt_neox.layers[layer_idx].attention.query_key_value.weight
        figpremod = figsavepath + f"/QKVmatrix_{checkpoint}_l{layer_idx}h{head_idx}.pdf"
        if not os.path.exists(figpremod): 
            utils.visualize_matrix(premod_qkv_weight.data.detach(), "qkv_pre-", figpremod, 0, None)

        #### -----------------------------
        #### ----- APPLY MODIFICATION ----
        #### -----------------------------

        modified_model, tokenizer, qkv_weight, q_start_idx, k_start_idx = utils.modify_attention_weights(mpath, checkpoint, modification, layer_idx, head_idx, device)
        figpostmod = figsavepath + f"/QKVmatrix_{checkpoint}_l{layer_idx}h{head_idx}_mod_{modification}.pdf"
        color_scale_max = premod_qkv_weight.data.max().item()
        utils.visualize_matrix(qkv_weight.detach(),"qkv post-" + modification, figpostmod, 0, color_scale_max)

        n_layers = modified_model.config.num_hidden_layers
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

                #for record-keeping
                num_params = utils.count_parameters(modified_model)
    
        df_results = pd.DataFrame(results)
        df_results['token_diffs'] = np.abs(df_results['S1_ntokens'].values-df_results['S2_ntokens'].values)
        df_results['n_params'] = np.repeat(num_params,df_results.shape[0])
        df_results['mpath'] = mpath
        df_results['revision'] = checkpoint
        df_results['step'] = int(checkpoint.replace("step", ""))
        df_results['model_modification'] = np.repeat(modification,df_results.shape[0])
        df_results['layer_modified'] = np.repeat(layer_idx,df_results.shape[0])
        df_results['head_modified'] = np.repeat(head_idx,df_results.shape[0])
        
        
        df_results.to_csv(os.path.join(savepath,filename), index=False)



if __name__ == "__main__":


    ## Read stimuli
    df = pd.read_csv(STIMULI)
    df_just_n = df[df['Class']=='N']

    ### Get revisions
    revisions = utils.generate_revisions_post1000()

    ## Specify layer/head to modify
    layer_idx = 2
    head_idx = 1

    modification = MODIFICATIONS[1]
    
    ## Run main
    main(df_just_n, MODELS[0], revisions, modification, layer_idx, head_idx)

