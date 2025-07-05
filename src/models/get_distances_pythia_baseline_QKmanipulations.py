# get distances for pythia-14M, for manipulations of randomly selected heads' QK matrices

"""Run a modified Pythia14-M instance over the RAW-C dataset, over pre-training checkpoints."""


import numpy as np
import os
import pandas as pd
import transformers
import torch


from scipy.spatial.distance import cosine
from tqdm import tqdm
from transformers import GPTNeoXForCausalLM, AutoTokenizer


import utils


### Models to modify and test
MODELS = ['EleutherAI/pythia-410m']

STIMULI = "data/raw/rawc/rawc_stimuli.csv"

MODIFICATIONS = ["ablate_zero",
                 "ablate_copy_step1"]


### Set up config (for 14m, randomly selecting non-target heads from target layer)
num_layers = 6 #Pythia-14M
num_heads_per_layer = 4 #Pythia-14M
target_layer = 2 #layer index to from which to sample random heads
num_interventions = 3 #three rounds of separate interventions
num_samples_list = [2,1,1] # (for 14m)
blocked_heads = [0,1]

LAYERS_HEADS_IDX = {}
for intervention in range(num_interventions):
    num_samples = num_samples_list[intervention]
    LAYERS_HEADS_IDX[intervention] = utils.select_random_heads(num_layers, num_heads_per_layer, num_samples=num_samples, seed=None, target_layer=target_layer, blocked_heads=blocked_heads)

LAYERS_HEADS_IDX = {
    "layers": [v["layers"] for v in LAYERS_HEADS_IDX.values()],
    "heads":  [sorted(v["heads"]) for v in LAYERS_HEADS_IDX.values()]
}



### For 410m, we manually the heads at the bottom of the disamb_index distribution
LAYERS_HEADS_IDX = {"layers": [[23],[23],[7], [19], [8], [22]],
                    "heads": [[2],[5],[8], [1], [7],[12]]}


def main(df, mpath, revisions, modification, layer_indices, head_indices):
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
        savepath = f"data/processed/rawc/pythia/pythia-QKmod-random-heads/distances_{modification}_layer{layer_indices}head{head_indices}/"
        if not os.path.exists(savepath): 
            os.makedirs(savepath)
        if "/" in mpath:
            filename = "rawc-distances_model-" + mpath.split("/")[1] + "-" + checkpoint +  ".csv"
        else:
            filename = "rawc-distances_model-" + mpath +  "-" + checkpoint + ".csv"

        # Set up figure savepath
        figsavepath = f"data/processed/rawc/pythia/pythia-QKmod-random-heads/figures/{modification}"
        if not os.path.exists(figsavepath): 
            os.makedirs(figsavepath)

        # Skip this checkpoint's analysis if you've already run it before
        print("Checking if we've already run this analysis...")
        if os.path.exists(os.path.join(savepath,filename)):
            print("Already run this model for this checkpoint.")
            continue

        print(savepath)

        # Load the model & tokenizer (this will be the intact model)
        model = GPTNeoXForCausalLM.from_pretrained(
            mpath,
            revision=checkpoint,
            return_dict_in_generate = True,
            output_hidden_states = True
        )
        model.to(device) # allocate model to desired device

        tokenizer = AutoTokenizer.from_pretrained(mpath, revision=checkpoint)

        n_params = utils.count_parameters(model)
       

        #### -----------------------------
        #### ----- APPLY MODIFICATION ----
        #### -----------------------------
        ## TODO: need to check that the flow of nested for loops is working
        for layer_idx, head_idx in zip(layer_indices,head_indices):

            premod_qkv_weight = model.gpt_neox.layers[layer_idx].attention.query_key_value.weight.data.cpu().numpy()
            # figpremod = figsavepath + f"/QKVmatrix_{checkpoint}_l{layer_idx}h{head_idx}.pdf"
            # if not os.path.exists(figpremod): 
                #utils.visualize_matrix(premod_qkv_weight, "qkv_pre-", figpremod, 0, None)
        

            # Note: the model object gets modified even when you just call it "model", so after each iteration 
            # of the `utils.modify_attention_weights' call, the next-iteration's `model' will be the modified version
            # *not* the intact version---I just change the name of the output for clarity in the block of code that 
            # runs the stimuli to get the target word representations
            modified_model, q_start_idx, k_start_idx, head_size, hidden_size = utils.modify_attention_weights(mpath, model, checkpoint, modification, layer_idx, head_idx, device)
            
            qkv_weight = modified_model.gpt_neox.layers[layer_idx].attention.query_key_value.weight.data.cpu().numpy()
            # figpostmod = figsavepath + f"/QKVmatrix_{checkpoint}_l{layer_idx}h{head_idx}_mod_{modification}.pdf"
            # color_scale_max = premod_qkv_weight.max().item()
            # color_scale_min = premod_qkv_weight.min().item()
            # color_scale_tuple = (color_scale_min,color_scale_max)
            # utils.visualize_matrix(qkv_weight,"qkv post-" + modification, figpostmod, 0, color_scale_tuple)

            # Check that you've modified only the intended rows/columns

            # Figure out the set of indices that you had intended to modify
            rows_intended = torch.tensor(range(q_start_idx,k_start_idx+head_size))
            
            # Check if only the target block is modified
            block_premod = premod_qkv_weight[rows_intended, :]
            block_postmod = qkv_weight[rows_intended, :]    

            # Compare outside the target block using a mask to avoid grabbing the within-block indices
            mask = np.ones(premod_qkv_weight.shape[0], dtype=bool)
            mask[rows_intended] = False

            outside_premod = premod_qkv_weight[mask, :]
            outside_postmod = qkv_weight[mask, :]

            if np.array_equal(block_premod,block_postmod) and modification != "ablate_copy_step1": 
                raise ValueError("The target matrix blocks are identical! They should be different.")

            if not np.array_equal(outside_premod, outside_postmod):
                raise ValueError("Unintended modifications detected outside the target block!")


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
    
        df_results = pd.DataFrame(results)
        df_results['token_diffs'] = np.abs(df_results['S1_ntokens'].values-df_results['S2_ntokens'].values)
        df_results['n_params'] = n_params # np.repeat(num_params,df_results.shape[0])
        df_results['mpath'] = mpath
        df_results['revision'] = checkpoint
        df_results['step'] = int(checkpoint.replace("step", ""))
        df_results['model_modification'] = np.repeat(modification,df_results.shape[0])
        df_results['layers_modified'] = np.repeat(str(layer_indices),df_results.shape[0])
        df_results['heads_modified'] = np.repeat(str(head_indices),df_results.shape[0])
        
        
        df_results.to_csv(os.path.join(savepath,filename), index=False)



if __name__ == "__main__":


    ## Read stimuli
    df = pd.read_csv(STIMULI)
    df_just_n = df[df['Class']=='N']

    ### Get revisions
    ### Now doing pre512
    revisions = utils.generate_revisions_test()


    print(LAYERS_HEADS_IDX)

    ## Specify layer/head to modify
    for layer_indices, head_indices in zip(LAYERS_HEADS_IDX["layers"],LAYERS_HEADS_IDX["heads"]):
        print(f"Running with layers: {layer_indices}, and heads: {head_indices}")

        for modification in MODIFICATIONS:
            ## Run main
            main(df_just_n, MODELS[0], revisions, modification, layer_indices, head_indices)

