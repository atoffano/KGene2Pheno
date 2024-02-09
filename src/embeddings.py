
from tqdm import tqdm
import numpy as np

import pandas as pd
import torch
import os
from torch import cuda

from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import DataLoader
from torchkge.data_structures import KnowledgeGraph
from torchkge.models import *

from src.utils import timer_func

def generate_emb(emb_model, batch, sampler, device):
    """
    Generate embeddings for a batch of samples using the trained embedding model.

    Parameters
    ----------
    emb_model : torchkge.models.xxx
        The trained embedding model.
    batch : torch.tensor
        A tensor containing the batch data (h_idx, t_idx, r_idx).
    sampler : torchkge.sampling.NegativeSampler
        The negative sampler used to generate negative samples.
    device : torch.device
        The device to perform the computation on.

    Returns
    -------
    torch.tensor
        A tensor containing the entity indices (h_idx, t_idx, n_h_idx, n_t_idx), 
        true triple head and tail node embeddings (pos_x), corrupted triple head and tail node embeddings (neg_x), 
        true triple relations (relation), and corrupted triple relations (neg_relation).
    """

    # Generate positive samples
    h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]

    # Generate negative samples by corrupting the tail
    n_h_idx, n_t_idx = sampler.corrupt_batch(h_idx, t_idx, r_idx) 

    # Get entity embeddings for the batch
    if type(emb_model) == ComplExModel:
        # Real embeddings
        h = emb_model.re_ent_emb(h_idx.to(device))
        t = emb_model.re_ent_emb(t_idx.to(device))
        n_t = emb_model.re_ent_emb(n_t_idx.to(device))
        n_h = emb_model.re_ent_emb(n_h_idx.to(device))

        # Imaginary embeddings
        im_h = emb_model.im_ent_emb(h_idx.to(device))
        im_t = emb_model.im_ent_emb(t_idx.to(device))
        im_n_t = emb_model.im_ent_emb(n_t_idx.to(device))
        im_n_h = emb_model.im_ent_emb(n_h_idx.to(device))
    else:
        h = emb_model.ent_emb(h_idx.to(device))
        t = emb_model.ent_emb(t_idx.to(device))
        n_t = emb_model.ent_emb(n_t_idx.to(device))
        n_h = emb_model.ent_emb(n_h_idx.to(device))

    # Create a ground truth for samples
    neg_relation = torch.tensor([-1]*len(r_idx)).to(device)
    relation = torch.tensor([0]*len(r_idx)).to(device)
    for i, r_type in enumerate(r_idx):
        relation[i] = r_type.item()

    # Concat head and tail embeddings
    if type(emb_model) == ComplExModel: # Take into account imaginary embeddings
        pos_x = torch.cat((h, im_h, t, im_t), dim=1)
        neg_x = torch.cat((n_h, im_n_h, n_t, im_n_t), dim=1)
    else:
        pos_x = torch.cat((h, t), dim=1)
        neg_x = torch.cat((n_h, n_t), dim=1)

    ent_idxs = (h_idx, t_idx, n_h_idx, n_t_idx) # Store the entity indices
    return ent_idxs, pos_x, neg_x, relation, neg_relation

def get_emb(emb_model, idx):
    # Returns the corresponding embeddings for the given indices
    with torch.no_grad():
        if type(emb_model) == ComplExModel:
            # Real embeddings
            emb = emb_model.re_ent_emb(idx)

            # Imaginary embeddings
            im_emb = emb_model.im_ent_emb(idx)
            emb = torch.cat((emb, im_emb), dim=1)

        else:
            emb = emb_model.ent_emb(idx)
    return emb

@timer_func
def generate(emb_model, dataset, config, timestart, device):
    """
    Generate a dataset containing embeddings of both head and tail node, to be used by the classifier using the specified trained embedding model.
    This creates a pd.DataFrame with the following layout: [embedding_head, embedding_tail, head_str, relation_str, tail_str]

    Parameters
    ----------
    emb_model : torchkge.models.xxx
        The trained embedding model.
    dataset : torchkge.data_structures.KnowledgeGraph
        The dataset to generate embeddings for.
    config : bool
        CLI arguments.
    device : torch.device
        The device to perform the computation on.

    Returns
    -------
    pd.DataFrame
    """

    emb_model.to(device)
    dataloader = DataLoader(dataset, batch_size=512, use_cuda='None')
    sampler = BernoulliNegativeSampler(dataset)

    with torch.no_grad():
        all_pred, all_truth, all_h_idx, all_t_idx = torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)
        for batch in tqdm(dataloader, desc='Generating embeddings for dataset'):
            ent_idxs, pos_x , neg_x, relation, neg_relation = generate_emb(emb_model, batch, sampler, device)

            # Store the batch data in their respective tensors
            pred = torch.cat((pos_x, neg_x), dim=0)
            truth = torch.cat((relation, neg_relation), dim=0)
            all_pred = torch.cat((all_pred, pred), dim=0)
            all_truth = torch.cat((all_truth, truth), dim=0)
            all_h_idx = torch.cat((all_h_idx, torch.cat((ent_idxs[0].to(device), ent_idxs[2].to(device)), dim=0)), dim=0) # Store head entity indices
            all_t_idx = torch.cat((all_t_idx, torch.cat((ent_idxs[1].to(device), ent_idxs[3].to(device)), dim=0)), dim=0) # Store tail entity indices

    ent2ix = {v: k for k, v in dataset.ent2ix.items()} # Mapping of entity indices to entity names
    all_t_idx = all_t_idx.cpu().numpy()
    all_t_idx_dict = np.vectorize(ent2ix.get)(all_t_idx) # Match head entity indices to entity names
    all_h_idx = all_h_idx.cpu().numpy()
    all_h_idx_dict = np.vectorize(ent2ix.get)(all_h_idx) # Match tail entity indices to entity names

    rel2ix = {v: k for k, v in dataset.rel2ix.items()} # Mapping of relation indices to entity names
    rel2ix[-1] = 'no_link_known'
    all_truth = all_truth.cpu().numpy()
    all_truth_dict = np.vectorize(rel2ix.get)(all_truth) # Match relation indices to relation names

    # make all embeddings as a df with 50 features (default) per node (total 100 for head [0:49] and tail [50:99]) and add all_truth as a qualitative column
    df = pd.DataFrame(all_pred.cpu().numpy())
    df['head'] = all_h_idx_dict
    df['relation'] = all_truth_dict
    df['tail'] = all_t_idx_dict

    #shuffle df rows randomly
    df = df.sample(frac=1).reset_index(drop=True)

    if config['save_embeddings'] == True:
        if not os.path.exists('data/embeddings'):
            os.makedirs('data/embeddings')
        df.to_csv(f'data/embeddings/{timestart}_{config["method"]}', index=False)

    return df
    
if __name__ == '__main__':
    # # Loads a model and triple data, before exporting the pair of head and tail node embeddings as csv
    import os
    os.chdir('KGene2pheno')
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    # Model and dataset parameters
    datapath = '' # Path to the dataset in .n3 format
    emb_model = ComplExModel(emb_dim=50, n_entities=99699, n_relations=9) # Arguments vary depending on the model. Refer to torchkge documentation for specifics
    model_path = '' # Path to the .pt model
    save_path = '' # Path to save the embeddings

    # Dataset loading
    print("Loading train dataset..")
    df = pd.read_csv(datapath, skiprows=[0], usecols=[1, 2, 3], header=None, names=['from', 'to', 'rel'])
    kg = KnowledgeGraph(df)
    
    # Model loading
    print("Loading model..")
    emb_model.load_state_dict(torch.load(model_path))

    # Move everything to CUDA if available
    use_cuda = cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
        cuda.empty_cache()
        emb_model.to(device)
    else:
        device = torch.device('cpu')

    generate(emb_model, kg, save_path, device=device)
