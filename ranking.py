
from tqdm import tqdm
from datetime import datetime as dt
import numpy as np

import pandas as pd
import torch

from torch import cuda

from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import DataLoader
from torchkge.data_structures import KnowledgeGraph
from torchkge.models import *
from torchkge.evaluation import *

def generate_emb(h_idx, t_idx, r_idx, n_h_idx, n_t_idx):
    # Get entity embeddings for the batch
    h = emb_model.ent_emb(h_idx)
    t = emb_model.ent_emb(t_idx)
    n_t = emb_model.ent_emb(n_t_idx)
    n_h = emb_model.ent_emb(n_h_idx)


    # Create a ground truth for samples
    neg_ground_truth = torch.tensor([-1]*len(r_idx)).to(device)
    ground_truth = torch.tensor([0]*len(r_idx)).to(device)
    for i, r_type in enumerate(r_idx):
        ground_truth[i] = r_type.item()

    # Concat head and tail embeddings
    pos_x = torch.cat((h, t), dim=1)
    neg_x = torch.cat((n_h, n_t), dim=1)

    return pos_x, neg_x, ground_truth, neg_ground_truth

def generate_samples(emb_model, batch, sampler):
    # Generate positive samples
    h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]

    # Generate negative samples by corrupting the tail
    n_h_idx, n_t_idx = sampler.corrupt_batch(h_idx, t_idx, r_idx) 

    return h_idx, t_idx, r_idx, n_h_idx, n_t_idx
    
def generate_scores(evaluator, h_idx, t_idx, r_idx, n_h_idx, n_t_idx):
    h_idx, t_idx, r_idx, n_h_idx, n_t_idx = h_idx.to(device), t_idx.to(device), r_idx.to(device), n_h_idx.to(device), n_t_idx.to(device)
    h_emb, t_emb, _, candidates = evaluator.model.inference_prepare_candidates(h_idx, t_idx, r_idx, entities=False)
    n_h_emb, n_t_emb, _, candidates = evaluator.model.inference_prepare_candidates(n_h_idx, n_t_idx, r_idx, entities=False)

    # Compute scores for all relation types on each positive and negative sample
    scores_pos = evaluator.model.inference_scoring_function(h_emb, t_emb, candidates)
    scores_neg = evaluator.model.inference_scoring_function(n_h_emb, n_t_emb, candidates)

    return scores_pos, scores_neg

def generate_data(emb_model, kg_train, kg_val, kg_test):
    timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    rel2ix = {v: k for k, v in kg_train.rel2ix.items()} # Mapping of entity indices to entity names
    rel2ix[-1] = 'no_link_known' # Handles corrupted triples

    emb_model.to(device)

    for dataset, name in zip([kg_train, kg_val, kg_test], ['train', 'val', 'test']):
        dataloader = DataLoader(dataset, batch_size=1024, use_cuda='None')
        sampler = BernoulliNegativeSampler(dataset)
        evaluator = RelationPredictionEvaluator(emb_model, dataset)
        with torch.no_grad():
            all_pred, all_truth, all_scores = torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)
            for batch in tqdm(dataloader, desc=f'Generating embeddings for ' + name + ' set'):

                h_idx, t_idx, r_idx, n_h_idx, n_t_idx = generate_samples(emb_model, batch, sampler)
                h_idx, t_idx, r_idx, n_h_idx, n_t_idx = h_idx.to(device), t_idx.to(device), r_idx.to(device), n_h_idx.to(device), n_t_idx.to(device)

                pos_x , neg_x, ground_truth, neg_ground_truth = generate_emb(h_idx, t_idx, r_idx, n_h_idx, n_t_idx)
                scores_pos, scores_neg = generate_scores(evaluator, h_idx, t_idx, r_idx, n_h_idx, n_t_idx)

                # Get max score index for each sample
                _, scores_pos_idx = torch.max(scores_pos, dim=1)
                _, scores_neg_idx = torch.max(scores_neg, dim=1)


                # append predictions and ground truths to respective tensors
                pred = torch.cat((pos_x, neg_x), dim=0)
                truth = torch.cat((ground_truth, neg_ground_truth), dim=0)
                scores = torch.cat([scores_pos_idx, scores_neg_idx], dim=0)
                all_pred = torch.cat((all_pred, pred), dim=0)
                all_truth = torch.cat((all_truth, truth), dim=0)
                all_scores = torch.cat((all_scores, scores), dim=0)
        
        # Each row consists of:
        # 2 node embeddings, concatenated (50 features per node : head [0:49] and tail [50:99])
        # 'ground_truth' column with the relation type of the sample
        # 'score_prediction' column with the relation type with the highest score predicted by the for the sample (Hit@1)
        # 'true_triple' column with 1 if the triple is true and 0 if it is corrupted
        
        df = pd.DataFrame(all_pred.cpu().numpy())
        
        all_truth = all_truth.cpu().numpy()
        all_truth_dict = np.vectorize(rel2ix.get)(all_truth)
        df['ground_truth'] = all_truth_dict
        
        rel2ix_dataset = {v: k for k, v in dataset.rel2ix.items()} # Mapping of entity indices to entity names
        all_scores = all_scores.cpu().numpy()
        all_scores_dict = np.vectorize(rel2ix_dataset.get)(all_scores)
        df['score_prediction'] = all_scores_dict

        df['true_triple'] = df['ground_truth'].apply(lambda x: 1 if x != 'no_link_known' else 0)
        
        # Shuffle df rows randomly
        df = df.sample(frac=1).reset_index(drop=True)
        df.to_csv(f"/home/antoine/gene_pheno_pred/{timestamp}_{name}.csv", index=False)

if __name__ == '__main__':
    # # Loads a model and the relevant test data, and run on a test set
    # inference_from_checkpoint('/home/antoine/gene_pheno_pred/models/TorchKGE/TransH_2023-03-13 17:08:16.530738.pt', '/home/antoine/gene_pheno_pred/emb_models/TorchKGE/TransH_2023-03-13 17:08:16.530738_kg_val.csv')
    import os
    os.chdir('/home/antoine/gene_pheno_pred')
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    os.environ["WANDB_API_KEY"]="4e5748d6c6f3917c78cdc38a516a1bac776faf58"

    # Dataset loading
    print("Loading train dataset..")
    df = pd.read_csv('/home/antoine/gene_pheno_pred/models/TorchKGE/ConvKB_2023-04-07 18:57:08.619362_kg_train.csv', skiprows=[0], usecols=[1, 2, 3], header=None, names=['from', 'to', 'rel'])
    kg_train = KnowledgeGraph(df)

    print("Loading val dataset..")
    df = pd.read_csv('/home/antoine/gene_pheno_pred/models/TorchKGE/ConvKB_2023-04-07 18:57:08.619362_kg_val.csv', skiprows=[0], usecols=[1, 2, 3], header=None, names=['from', 'to', 'rel'])
    kg_val = KnowledgeGraph(df)

    print("Loading test dataset..")
    df = pd.read_csv('/home/antoine/gene_pheno_pred/models/TorchKGE/ConvKB_2023-04-07 18:57:08.619362_kg_test.csv', skiprows=[0], usecols=[1, 2, 3], header=None, names=['from', 'to', 'rel'])
    kg_test = KnowledgeGraph(df)
    
    # Model loading
    print("Loading model..")
    emb_model = ConvKBModel(50, 10, 675845,10)
    emb_model.load_state_dict(torch.load('/home/antoine/gene_pheno_pred/models/TorchKGE/ConvKB_2023-04-07 18:57:08.619362.pt'))

    # Move everything to CUDA if available
    use_cuda = cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
        cuda.empty_cache()
        emb_model.to(device)
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    generate_data(emb_model, kg_train, kg_val, kg_test)
