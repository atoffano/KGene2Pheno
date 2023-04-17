
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

def generate_emb(emb_model, batch, sampler):

    # Generate positive samples
    h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]

    # Generate negative samples by corrupting the tail
    n_h, n_t = sampler.corrupt_batch(h_idx, t_idx, r_idx) 

    # Get entity embeddings for the batch
    h = emb_model.ent_emb(h_idx.to(device))
    t = emb_model.ent_emb(t_idx.to(device))
    n_t = emb_model.ent_emb(n_t.to(device))
    n_h = emb_model.ent_emb(n_h.to(device))

    # Create a ground truth for samples
    neg_ground_truth = torch.tensor([11]*len(r_idx)).to(device)
    ground_truth = torch.tensor([0]*len(r_idx)).to(device)
    for i, r_type in enumerate(r_idx):
        ground_truth[i] = r_type.item()

    # Concat head and tail embeddings
    pos_x = torch.cat((h, t), dim=1)
    neg_x = torch.cat((n_h, n_t), dim=1)

    return pos_x, neg_x, ground_truth, neg_ground_truth

def generate_data(emb_model, kg_train, kg_val, kg_test):
    timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")

    for dataset, name in zip([kg_train, kg_val, kg_test], ['train', 'val', 'test']):
        dataloader = DataLoader(dataset, batch_size=512, use_cuda='None')
        sampler = BernoulliNegativeSampler(dataset)
        emb_model.to(device)
        with torch.no_grad():
            all_pred, all_truth = torch.tensor([]).to(device), torch.tensor([]).to(device)
            for batch in tqdm(dataloader, desc='Generating embeddings for ' + name):
                pos_x , neg_x, ground_truth, neg_ground_truth = generate_emb(emb_model, batch, sampler)

                # append predictions and ground truths to respective tensors
                pred = torch.cat((pos_x, neg_x), dim=0)
                truth = torch.cat((ground_truth, neg_ground_truth), dim=0)
                all_pred = torch.cat((all_pred, pred), dim=0)
                all_truth = torch.cat((all_truth, truth), dim=0)

        rel2ix = {v: k for k, v in dataset.rel2ix.items()} # Mapping of entity indices to entity names
        rel2ix[11] = 'no_link_known'
        all_truth = all_truth.cpu().numpy()
        all_truth_dict = np.vectorize(rel2ix.get)(all_truth)
        
        # make all embeddings as a df with 50 features per node (total 100 for head [0:49] and tail [50:99]) and add all_truth as a qualitative column
        df = pd.DataFrame(all_pred.cpu().numpy())
        df['ground_truth'] = all_truth_dict

        #shuffle df rows randomly
        df = df.sample(frac=1).reset_index(drop=True)
        df.to_csv(f"/home/antoine/gene_pheno_pred/{timestamp}_{name}.csv", index=False)


def get_scores(emb_model, verbose=True):
    b_size = 512
    for dataset, name in zip([kg_train, kg_val, kg_test], ['train', 'val', 'test']):
        with torch.no_grad():
            evaluator = RelationPredictionEvaluator(emb_model, dataset)
            sampler = BernoulliNegativeSampler(dataset)
            dataloader = DataLoader(evaluator.kg, batch_size=b_size)

            # use_cuda = next(evaluator.model.parameters()).is_cuda

            # if use_cuda:
            #     dataloader = DataLoader(evaluator.kg, batch_size=b_size, use_cuda='batch')
            #     evaluator.rank_true_rels = evaluator.rank_true_rels.cuda()
            #     evaluator.filt_rank_true_rels = evaluator.filt_rank_true_rels.cuda()
            # else:
            #     dataloader = DataLoader(evaluator.kg, batch_size=b_size)
            all_scores = torch.tensor([]).to(device)
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                                    unit='batch', disable=(not verbose),
                                    desc='Relation prediction evaluation'):

                # Generate positive samples and negative samples by corrupting the tail
                h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]
                n_h_idx, n_t_idx = sampler.corrupt_batch(h_idx, t_idx, r_idx)

                # Load to GPU if necessary
                h_idx, t_idx, r_idx, n_h_idx, n_t_idx = h_idx.to(device), t_idx.to(device), r_idx.to(device), n_h_idx.to(device), n_t_idx.to(device)

                h_emb, t_emb, r_emb, candidates = evaluator.model.inference_prepare_candidates(h_idx, t_idx, r_idx, entities=False)
                n_h_emb, n_t_emb, r_emb, candidates = evaluator.model.inference_prepare_candidates(n_h_idx, n_t_idx, r_idx, entities=False)

                scores_pos = evaluator.model.inference_scoring_function(h_emb, t_emb, candidates)
                scores_neg = evaluator.model.inference_scoring_function(n_h_emb, n_t_emb, candidates)

                # Append a column of ones to the scores of positive samples, and a column of zeros to the scores of negative samples.
                # This will serve as a ground truth for the samples.
                scores_pos = torch.cat([scores_pos, torch.ones(scores_pos.shape[0], 1, dtype=torch.int).to(device)], dim=1)
                scores_neg = torch.cat([scores_neg, torch.zeros(scores_pos.shape[0], 1, dtype=torch.int).to(device)], dim=1)
                
                scores = torch.cat((scores_pos, scores_neg), dim=0)
                all_scores = torch.cat((all_scores, scores), dim=0)

            df = pd.DataFrame(all_scores.cpu().numpy())

            #shuffle df rows randomly
            df = df.sample(frac=1).reset_index(drop=True)
            col = list(dataset.rel2ix.keys())
            col.append('ground_truth')
            df.to_csv(f"/home/antoine/gene_pheno_pred/ConvKB_scores_{name}.csv", header=col, index=False)

if __name__ == '__main__':
    # # Loads a model and the relevant test data, and run on a test set
    # inference_from_checkpoint('/home/antoine/gene_pheno_pred/models/TorchKGE/TransH_2023-03-13 17:08:16.530738.pt', '/home/antoine/gene_pheno_pred/emb_models/TorchKGE/TransH_2023-03-13 17:08:16.530738_kg_val.csv')
    import os
    os.chdir('/home/antoine/gene_pheno_pred')
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
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

    from torch import empty, zeros, cat
    from tqdm.autonotebook import tqdm

    from torchkge.data_structures import SmallKG
    from torchkge.exceptions import NotYetEvaluatedError
    from torchkge.sampling import PositionalNegativeSampler
    from torchkge.utils import DataLoader, get_rank, filter_scores
    from torchkge.evaluation import *

    get_scores(emb_model)
