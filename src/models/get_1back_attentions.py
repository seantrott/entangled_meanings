"""Get 1-back attention head information for Pythia suite over pre-training."""


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
          # 'EleutherAI/pythia-1.4b',
          # 'EleutherAI/pythia-2.8b',
          # 'EleutherAI/pythia-6.9b',
          # 'EleutherAI/pythia-12b',
          ]

STIMULI = "data/raw/rawc/individual_sentences.csv"



### Handle logic for a dataset/model
def main(df, mpath, revisions):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("number of checkpoints:", len(revisions))



    for checkpoint in tqdm(revisions):
        print(checkpoint)

        ### Set up save path, filename, etc.
        savepath = "data/processed/rawc/pythia/attention_1back/"
        if not os.path.exists(savepath): 
            os.mkdir(savepath)
        if "/" in mpath:
            filename = "rawc-attentions-1back_model-" + mpath.split("/")[1] + "-" + checkpoint +  ".csv"
        else:
            filename = "rawc-attentions-1back_model-" + mpath +  "-" + checkpoint + ".csv"

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

            ### Get word
            target = " {w}".format(w = row['string'])
            disambiguating_word = " {w}".format(w = row['disambiguating_word']) # row['string']
            sentence = row['sentence']

            ### Run model for each sentence
            model_outputs = utils.run_model(model, tokenizer, sentence, device)

            ### Now, for each layer...
            for layer in range(n_layers): 

                for head in range(n_heads): 
    
                    ### Get attention weights for the given head
                    attn_weights = model_outputs['attentions'][layer][0, head]  # Shape: (seq_len, seq_len)

                    ### Extract attention to previous token (diagonal just below main diagonal)
                    seq_len = attn_weights.shape[0]
                    if seq_len > 1:  # Ensure sequence is long enough
                        prev_token_attention = torch.diagonal(attn_weights, offset=-1).mean().item()
                    else:
                        prev_token_attention = None  # Skip if not applicable

                    ### Add to results dictionary
                    results.append({
                        'sentence': row['sentence'],
                        'Head': head,
                        'Layer': layer,
                        'Attention': prev_token_attention  # Storing 1-back attention score
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
    df_just_n = df[df['Class']=='N']

    ### Get revisions
    revisions = utils.generate_revisions()

    ## Run main
    main(df_just_n, MODELS[0], revisions)
