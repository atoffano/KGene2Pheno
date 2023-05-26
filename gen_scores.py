import numpy as np
from tqdm import tqdm
import pandas as pd
import torch

from torch import cuda

from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import DataLoader
from torchkge.data_structures import KnowledgeGraph
from torchkge.models import *

from torchkge.evaluation import *

def get_scores(emb_model, kg_train, kg_val, kg_test, verbose=True):
    b_size = 4096
    all_keys_rel2ix = kg_train.rel2ix.keys()
    for dataset, name in zip([kg_train, kg_val, kg_test], ['train', 'val', 'test']):
        with torch.no_grad():

            evaluator = RelationPredictionEvaluator(emb_model, dataset)
            sampler = BernoulliNegativeSampler(dataset)
            dataloader = DataLoader(evaluator.kg, batch_size=b_size)

            all_scores, all_true_triple, all_rel_type = torch.tensor([], device=device), torch.tensor([], device=device), torch.tensor([], device=device)
            for _, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                                    unit='batch', disable=(not verbose),
                                    desc='Calculating scores for ' + name + ' set..'):

                # Generate positive samples and negative samples by corrupting the tail
                h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]
                n_h_idx, n_t_idx = sampler.corrupt_batch(h_idx, t_idx, r_idx)

                # Load to GPU if necessary
                h_idx, t_idx, r_idx, n_h_idx, n_t_idx = h_idx.to(device), t_idx.to(device), r_idx.to(device), n_h_idx.to(device), n_t_idx.to(device)

                # Get the ground truth for relation types
                rel_type = torch.tensor([r_type.item() for r_type in r_idx], device=device)
                neg_rel_type = torch.tensor([-1]*len(r_idx), device=device)
                rel_type = torch.cat((rel_type, neg_rel_type), dim=0)

                h_emb, t_emb, _, candidates = evaluator.model.inference_prepare_candidates(h_idx, t_idx, r_idx, entities=False)
                n_h_emb, n_t_emb, _, candidates = evaluator.model.inference_prepare_candidates(n_h_idx, n_t_idx, r_idx, entities=False)

            
                # Compute scores for all relation types on each positive and negative sample
                scores_pos = evaluator.model.inference_scoring_function(h_emb, t_emb, candidates)
                scores_neg = evaluator.model.inference_scoring_function(n_h_emb, n_t_emb, candidates)

                # Append a column of ones to the scores of positive samples, and a column of zeros to the scores of negative samples.
                # This will serve as a ground truth for the samples.
                scores = torch.cat([scores_pos, scores_neg], dim=0)

                true_triple = torch.ones(scores_pos.shape[0], 1, dtype=torch.int, device=device)
                neg_triple = torch.zeros(scores_neg.shape[0], 1, dtype=torch.int, device=device)
                true_triple = torch.cat((true_triple, neg_triple), dim=0)
                # scores_pos = torch.cat([scores_pos, torch.ones(scores_pos.shape[0], 1, dtype=torch.int).to(device)], dim=1)
                # scores_neg = torch.cat([scores_neg, torch.zeros(scores_neg.shape[0], 1, dtype=torch.int).to(device)], dim=1)
                
                # Append a col matching the ground truth for the relation type
                # scores_pos = torch.cat((scores_pos, rel_type.unsqueeze(-1)), dim=-1)
                # scores_neg = torch.cat((scores_neg, neg_rel_type.unsqueeze(-1)), dim=-1)

                # scores = torch.cat((scores_pos, scores_neg), dim=0)
                all_rel_type = torch.cat((all_rel_type, rel_type), dim=0)
                all_scores = torch.cat((all_scores, scores), dim=0)
                all_true_triple = torch.cat((all_true_triple, true_triple), dim=0)

            rel2ix = {v: k for k, v in dataset.rel2ix.items()} # Mapping of entity indices to entity names
            rel2ix[-1] = 'no_link_known'            
            all_rel_type = np.vectorize(rel2ix.get)(all_rel_type.cpu().numpy())

            #create df from tensors
            df = pd.DataFrame(all_scores.cpu().numpy())
            df['rel_type'] = [rel2ix.get(x, x) for x in all_rel_type]
            df['true_triple'] = all_true_triple.cpu().numpy()
            df = df.sample(frac=1).reset_index(drop=True) # shuffle df rows randomly

            col = list(all_keys_rel2ix)
            col.append('rel_type')
            col.append('true_triple')
            df.to_csv(f"/home/antoine/gene_pheno_pred/ConvKB_scores_{name}.csv", header=col, index=False)

if __name__ == '__main__':
    # # Loads a model and the relevant test data, and run on a test set
    # inference_from_checkpoint('/home/antoine/gene_pheno_pred/models/TorchKGE/TransH_2023-03-13 17:08:16.530738.pt', '/home/antoine/gene_pheno_pred/emb_models/TorchKGE/TransH_2023-03-13 17:08:16.530738_kg_val.csv')
    import os
    os.chdir('/home/antoine/gene_pheno_pred')
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    os.environ["WANDB_API_KEY"]="4e5748d6c6f3917c78cdc38a516a1bac776faf58"

    # # Dataset loading
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

    get_scores(emb_model, kg_train, kg_val, kg_test)
