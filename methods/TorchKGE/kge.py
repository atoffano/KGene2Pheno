import pandas as pd
from tqdm import tqdm
from datetime import datetime as dt
import wandb

import torch

# from ignite.engine import Engine, Events
# from ignite.handlers import EarlyStopping
# from ignite.metrics import RunningAverage
from torch.optim import Adam
from torch import cuda

from torchkge.evaluation import *
from torchkge.models import *
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, LogisticLoss, BinaryCrossEntropyLoss, DataLoader
from torchkge.data_structures import KnowledgeGraph

def train(method, dataset, config, timestart):

    # Dataset loading and splitting
    df = pd.read_csv(dataset, sep=' ', header=None, names=['from', 'rel', 'to'])
    kg = KnowledgeGraph(df)

    print(f'{dt.now()} Number of triples: {kg.n_facts}')
    print(f'{dt.now()} Number of distinct entities: {kg.n_ent}')
    print(f'{dt.now()} Number of relations: {kg.n_rel}\n')

    kg_train, kg_val, kg_test = kg.split_kg(share=config['split_ratio'], validation=True)

    # Define the emb_model, criterion, optimizer, sampler and dataloader
    match method:
        case "TransE":
            emb_model = TransEModel(config['ent_emb_dim'], kg_train.n_ent, kg_train.n_rel, dissimilarity_type=config['dissimilarity_type'])
        case "TransH":
            emb_model = TransHModel(config['ent_emb_dim'], kg_train.n_ent, kg_train.n_rel)
        case "TransR":
            emb_model = TransRModel(config['ent_emb_dim'], config['rel_emb_dim'], kg_train.n_ent, kg_train.n_rel)
        case "TransD":
            emb_model = TransDModel(config['ent_emb_dim'], config['rel_emb_dim'], kg_train.n_ent, kg_train.n_rel)
        case "TorusE":
            emb_model = TorusEModel(config['ent_emb_dim'], kg_train.n_ent, kg_train.n_rel, dissimilarity_type=config['dissimilarity_type']) #dissim type one of  ‘torus_L1’, ‘torus_L2’, ‘torus_eL2’.
        case "RESCAL":
            emb_model = RESCALModel(config['ent_emb_dim'], kg_train.n_ent, kg_train.n_rel)
        case "DistMult":
            emb_model = DistMultModel(config['ent_emb_dim'], kg_train.n_ent, kg_train.n_rel)
        case "HolE":
            emb_model = HolEModel(config['ent_emb_dim'], kg_train.n_ent, kg_train.n_rel)
        case "ComplEx":
            emb_model = ComplExModel(config['ent_emb_dim'], kg_train.n_ent, kg_train.n_rel)
        case "ANALOGY":
            emb_model = AnalogyModel(config['ent_emb_dim'], kg_train.n_ent, kg_train.n_rel, config['scalar_share'])
        case "ConvKB":
            if config['init_transe']:
                    print('ok')
                    init_model = TransEModel(emb_dim=50, n_entities=675845, n_relations=10, dissimilarity_type='L1')
                    init_model.load_state_dict(torch.load('/home/antoine/gene_pheno_pred/models/TorchKGE/TransE_2023-03-22 14:54:57.152481.pt'))
                    ent_emb, rel_emb = init_model.get_embeddings()
                    emb_model = ConvKBModel(config['ent_emb_dim'], config['n_filters'], kg_train.n_ent, kg_train.n_rel)
                    print(emb_model.ent_emb.weight.data[0])
                    emb_model.ent_emb.weight.data = ent_emb
                    print(emb_model.rel_emb.weight.data[0])
                    emb_model.rel_emb.weight.data = rel_emb

        case _:
            raise ValueError(f"Method {method} not supported.")
    wandb.watch(emb_model, log="all")

    match config['loss_fn']:
        case "margin":
            criterion = MarginLoss(margin=config['margin'])
        case "bce":
            criterion = LogisticLoss()
        case 'logistic':
            criterion = BinaryCrossEntropyLoss()
        case _:
            raise ValueError(f"Loss function {config['loss_fn']} not supported.")
 
    dataloader = DataLoader(kg_train, batch_size=config['batch_size'], use_cuda='None')
    optimizer = Adam(emb_model.parameters(), lr=config['lr'], weight_decay=1e-5)
    sampler = BernoulliNegativeSampler(kg_train)

    print(f'{dt.now()} Size of training set:\n\t{len(kg_train)} triples\n\t{kg_train.n_ent} entitites\n\t{kg_train.n_rel} relations')
    print(f'{dt.now()} Size of validation set\n\t{len(kg_train)} triples\n\t{kg_val.n_ent} entitites\n\t{kg_val.n_rel} relations')
    print(f'{dt.now()} Size of test set\n\t{len(kg_test)} triples\n\t{kg_test.n_ent} entitites\n\t{kg_test.n_rel} relations')
    
    # Log parameters
    print(f'\n{dt.now()} - PARAMETERS')
    for i in config.items():
        print(f'\t {i[0]} : {i[1]}' )

    # Move everything to CUDA if available
    use_cuda = cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
        cuda.empty_cache()
        emb_model.to(device)
        criterion.to(device)
    else:
        device = torch.device('cpu')

    # Train the emb_model
    iterator = tqdm(range(config['n_epochs']), unit='epoch')
    for epoch in iterator:
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            h, t, r = batch[0], batch[1], batch[2]
            n_h, n_t = sampler.corrupt_batch(h, t, r) # generate negative samples

            optimizer.zero_grad()

            # forward + backward + optimize
            h, t, r, n_h, n_t = h.to(device), t.to(device), r.to(device), n_h.to(device), n_t.to(device)
            pos, neg = emb_model(h, t, r, n_h, n_t)
            loss = criterion(pos, neg)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'{dt.now()} - Epoch {epoch + 1} | mean loss: { running_loss / len(dataloader)}')
        wandb.log({'loss': running_loss / len(dataloader)})

        if config['normalize_parameters']:
            emb_model.normalize_parameters()
    print(f'{dt.now()} - Finished Training !')

    if config['save_model']:
        torch.save(emb_model.state_dict(), f'models/TorchKGE/{method}_{timestart}.pt')
    if config['save_data']:
        kg_train.get_df().to_csv(f'models/TorchKGE/{method}_{timestart}_kg_train.csv')
        kg_test.get_df().to_csv(f'models/TorchKGE/{method}_{timestart}_kg_test.csv')
        kg_val.get_df().to_csv(f'models/TorchKGE/{method}_{timestart}_kg_val.csv')

    evaluate_emb_model(emb_model, kg_val)
    return emb_model, kg_train, kg_val, kg_test

def evaluate_emb_model(emb_model, kg_eval):
    print(f'{dt.now()} - Evaluating..')
    evaluator = RelationPredictionEvaluator(emb_model, kg_eval)
    evaluator.evaluate(b_size=4, verbose=True)

    # Log results to logfile
    print(f'{dt.now()} - Results:')
    evaluator.print_results(k=[i for i in range(1, 11)])

    # Log results to wandb
    get_results(evaluator)


def inference_from_checkpoint(emb_model_path, test_path):
    use_cuda = cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
        cuda.empty_cache()
    else:
        device = torch.device('cpu')
    emb_model = TransHModel(50,675845,10)
    emb_model.load_state_dict(torch.load(emb_model_path))
    emb_model.to(device)
    kg_test = KnowledgeGraph(pd.read_csv(test_path))
    evaluate_emb_model(emb_model, kg_test)

def get_results(evaluator):
    for k in range(1, 11):
        wandb.log({f'Hit@{k}': evaluator.hit_at_k(k)[0]})
    wandb.log({'Mean Rank': evaluator.mean_rank()[0]})
    wandb.log({'MRR': int(evaluator.mrr()[0])})

if __name__ == '__main__':
    # # Loads a model and the relevant test data, and run on a test set
    # inference_from_checkpoint('/home/antoine/gene_pheno_pred/models/TorchKGE/TransH_2023-03-13 17:08:16.530738.pt', '/home/antoine/gene_pheno_pred/emb_models/TorchKGE/TransH_2023-03-13 17:08:16.530738_kg_val.csv')
    import os
    os.chdir('/home/antoine/gene_pheno_pred')
    # train('TransE', "celedebug.txt", dt.now())