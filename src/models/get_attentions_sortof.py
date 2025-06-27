"""Get attention head information for Pythia suite over pre-training,
for the stimuli that separate the adjective cue from the target ambiguous noun."""


import numpy as np
import os
import pandas as pd
import transformers
import torch


from scipy.spatial.distance import cosine
from tqdm import tqdm
from transformers import GPTNeoXForCausalLM, AutoTokenizer


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

STIMULI = "data/raw/rawc/individual_sentences_separate_adjcue.csv"


def run_model(model, tokenizer, sentence):
    """Run model, return hidden states and attention"""
    # Tokenize sentence
    inputs = tokenizer(sentence, return_tensors="pt")

    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}


    # Run model
    with torch.no_grad():
        output = model(**inputs, output_attentions=True, output_hidden_states=True)
        hidden_states = output.hidden_states
        attentions = output.attentions

    return {'hidden_states': hidden_states,
            'attentions': attentions,
            'tokens': inputs}


### Handle logic for a dataset/model
def main(df, mpath, revisions):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("number of checkpoints:", len(revisions))



    for checkpoint in tqdm(revisions):
        print(checkpoint)

        ### Set up save path, filename, etc.
        savepath = "data/processed/rawc/pythia/attention_sortof_check/"
        if not os.path.exists(savepath): 
            os.mkdir(savepath)
        if "/" in mpath:
            filename = "rawc-attentions_model-sortof-check-" + mpath.split("/")[1] + "-" + checkpoint +  ".csv"
        else:
            filename = "rawc-attentions_model-sortof-check-" + mpath +  "-" + checkpoint + ".csv"

        print("Checking if we've already run this analysis...")
        if os.path.exists(os.path.join(savepath,filename)):
            print("Already run this model for this checkpoint.")
            continue

        ### if it doesn't exist, run it.
        model = GPTNeoXForCausalLM.from_pretrained(
            mpath,
            revision=checkpoint,
            output_hidden_states = True,
            device_map="auto"
        )

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
            target = " {w}".format(w = row['string_ambiguous_word'])
            disambiguating_word = " {w}".format(w = row['disambiguating_word']) # row['string']
            sentence = row['sentence_new']

            ### Inserted string
            inserted_string = row['inserted_string']
            if "-" in inserted_string: ## remove "skin-type", etc.
                continue
            inserted_string_final_token = " {w}".format(w = inserted_string.split()[-1])

            ### Run model for each sentence
            model_outputs = utils.run_model(model, tokenizer, sentence)



            ### Now, for each layer...
            for layer in range(n_layers): 

                for head in range(n_heads): 
    
                    ### Get heads
                    ### TODO: Store attention to each token index maybe, and track which is disambiguating word, which is target, etc.?

                    attention_info = utils.get_attention_and_entropy_for_head(model_outputs['attentions'], model_outputs['tokens'], tokenizer, 
                                                                             target, disambiguating_word, layer, head, device)



                    ### Attention to inserted string
                    attention_info_inserted_string = utils.get_attention_and_entropy_for_head(model_outputs['attentions'], model_outputs['tokens'], tokenizer, 
                                                                             target, inserted_string_final_token, layer, head, device)


                    ### Get 1-back still
                    attn_weights = model_outputs['attentions'][layer][0, head]  # Shape: (seq_len, seq_len)
                    seq_len = attn_weights.shape[0]
                    prev_token_attention = torch.diagonal(attn_weights, offset=-1).mean().item()
        
        
                    ### Add to results dictionary
                    results.append({
                        'sentence': row['sentence_new'],
                        'word': row['word'],
                        'string': row['string_ambiguous_word'],
                        'disambiguating_word': disambiguating_word,
                        'Attention': attention_info['attention_to_disambiguating'],
                        'inserted_string': inserted_string,
                        'inserted_string_final_token': inserted_string_final_token,
                        'Attention_inserted_string': attention_info_inserted_string['attention_to_disambiguating'],
                        'Attention_1back': prev_token_attention,
                        'Entropy': attention_info['entropy'],
                        'Head': head,
                        'Layer': layer
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
    # df_just_n = df[df['Class']=='N']

    ### Get revisions
    revisions = utils.generate_revisions()

    ## Run main
    main(df, MODELS[0], revisions)