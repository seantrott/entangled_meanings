"""Run the Pythia suite over the RAW-C dataset over pre-training checkpoints."""


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
         'EleutherAI/pythia-14m',
         # 'EleutherAI/pythia-70m',
         # 'EleutherAI/pythia-160m',
         # 'EleutherAI/pythia-410m',
          # 'EleutherAI/pythia-1b',
          #  'EleutherAI/pythia-1.4b',
          # 'EleutherAI/pythia-2.8b',
          # 'EleutherAI/pythia-6.9b',
          # 'EleutherAI/pythia-12b',
          ]

STIMULI = "data/raw/rawc/rawc_stimuli.csv"



### Get distances

### Handle logic for a dataset/model
def main(df, mpath, revisions):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("number of checkpoints:", len(revisions))

    for checkpoint in tqdm(revisions):

        for seed in range(1, 10):

            seed_name = "seed" + str(seed)

            ### Set up save path, filename, etc.
            savepath = "data/processed/rawc/pythia/distances_rs/"
            if not os.path.exists(savepath): 
                os.mkdir(savepath)
            if "/" in mpath:
                filename = "rawc-distances-rs_model-" + mpath.split("/")[1] + "-" + checkpoint + "-" + seed_name + ".csv"
            else:
                filename = "rawc-distances-rs_model-" + mpath +  "-" + checkpoint + "-" + seed_name + ".csv"

            print("Checking if we've already run this analysis...")
            print(filename)
            if os.path.exists(os.path.join(savepath,filename)):
                print("Already run this model for this checkpoint.")
                continue

            model_name = mpath + "-" + seed_name
            print(model_name)

            model = GPTNeoXForCausalLM.from_pretrained(
                model_name,
                revision=checkpoint,
                output_hidden_states = True
            )
            model.to(device) # allocate model to desired device

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
                s1_outputs = utils.run_model(model, tokenizer, row['sentence1'], device)
                s2_outputs = utils.run_model(model, tokenizer, row['sentence2'], device)

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
            df_results['seed_name'] = seed_name
            df_results['seed'] = seed
            df_results['step'] = int(checkpoint.replace("step", ""))
            
            
            ### Hurray! Save your cosine distance results to load into R
            #.  for analysis
        
        
            df_results.to_csv(os.path.join(savepath,filename), index=False)


if __name__ == "__main__":

    ## Read stimuli
    df = pd.read_csv(STIMULI)
    df_just_n = df[df['Class']=='N']

    ### Get revisions
    revisions = utils.generate_revisions()

    ## Run main
    main(df_just_n, MODELS[0], revisions)

