"""Run the Pythia suite over the RAW-C dataset over pre-training checkpoints."""


### TODO: Cache already-run models/checkpoints somewhere so we don't have to run again.

import numpy as np
import os
import pandas as pd
import transformers
import torch


from scipy.spatial.distance import cosine
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


import utils



### Models to test
MODELS = [
         # 'EleutherAI/pythia-14m',
         'EleutherAI/pythia-70m',
         'EleutherAI/pythia-160m',
         'EleutherAI/pythia-410m',
          'EleutherAI/pythia-1b',
          #  'EleutherAI/pythia-1.4b',
          # 'EleutherAI/pythia-2.8b',
          # 'EleutherAI/pythia-6.9b',
          # 'EleutherAI/pythia-12b',
          ]

STIMULI = "data/raw/rawc/rawc_stimuli.csv"



def run_model(model, tokenizer, sentence):
    """Run model, return hidden states and attention"""
    # Tokenize sentence
    inputs = tokenizer(sentence, return_tensors="pt")

    # Run model
    with torch.no_grad():
        output = model(**inputs, output_attentions=True, output_hidden_states=True)
        hidden_states = output.hidden_states
        attentions = output.attentions

    return {'hidden_states': hidden_states,
            'attentions': attentions,
            'tokens': inputs}


def get_embedding(hidden_states, inputs, tokenizer, target, layer):
    """Extract embedding for TARGET from set of hidden states and token ids."""
    
    # Tokenize target
    target_enc = tokenizer.encode(target, return_tensors="pt",
                                  add_special_tokens=False)
    
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

def find_sublist_index(mylist, sublist):
    """Find the first occurence of sublist in list.
    Return the start and end indices of sublist in list"""

    for i in range(len(mylist)):
        if mylist[i] == sublist[0] and mylist[i:i+len(sublist)] == sublist:
            return i, i+len(sublist)
    return None


### Handle logic for a dataset/model
def main(df, mpath, revisions):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("number of checkpoints:", len(revisions))

    for checkpoint in tqdm(revisions):

        ### Set up save path, filename, etc.
        savepath = "data/processed/rawc/pythia/distances/"
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

        model = AutoModelForCausalLM.from_pretrained(
            mpath,
            revision=checkpoint,
            # output_hidden_states = True,
            device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(mpath, revision=checkpoint)


        n_layers = model.config.num_hidden_layers
        print("number of layers:", n_layers)
    
        n_params = utils.count_parameters(model)
    
        results = []

        ### TODO: Why tqdm not working here?
        for (ix, row) in tqdm(df.iterrows(), total=df.shape[0]):

            ### Get word
            target = " {w}".format(w = row['string'])

            ### Run model for each sentence
            s1_outputs = run_model(model, tokenizer, row['sentence1'])
            s2_outputs = run_model(model, tokenizer, row['sentence2'])

            ### Now, for each layer...
            for layer in range(n_layers+1): # `range` is non-inclusive for the last value of interval
    
                ### Get embeddings for word
                s1 = get_embedding(s1_outputs['hidden_states'], s1_outputs['tokens'], tokenizer, target, layer)
                s2 = get_embedding(s2_outputs['hidden_states'], s2_outputs['tokens'], tokenizer, target, layer)
    
                ### Now calculate cosine distance 
                model_cosine = cosine(s1.cpu(), s2.cpu())
    
    
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
        
        
        ### Hurray! Save your cosine distance results to load into R
        #.  for analysis
    
        savepath = "data/processed/rawc/pythia/distances/"
        if not os.path.exists(savepath): 
            os.mkdir(savepath)
    
        if "/" in mpath:
            filename = "rawc-distances_model-" + mpath.split("/")[1] + "-" + checkpoint +  ".csv"
        else:
            filename = "rawc-distances_model-" + mpath +  "-" + checkpoint + ".csv"
    
        df_results.to_csv(os.path.join(savepath,filename), index=False)


if __name__ == "__main__":

    ## Read stimuli
    df = pd.read_csv(STIMULI)
    df_just_n = df[df['Class']=='N']

    ### Get revisions
    revisions = utils.generate_revisions_test()

    ## Run main
    for mpath in MODELS:
        print("Running:", mpath)
        main(df_just_n, mpath, revisions)

