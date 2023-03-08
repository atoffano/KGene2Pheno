import pandas as pd
from tqdm import tqdm
from datetime import datetime as dt

import torch
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from ignite.metrics import RunningAverage
from torch.optim import Adam
from torch import cuda

from torchkge.evaluation import LinkPredictionEvaluator
from torchkge.models import *
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, LogisticLoss, BinaryCrossEntropyLoss, DataLoader
from torchkge.data_structures import KnowledgeGraph

import methods.TorchKGE.config as config

def train(method, dataset):

    eval_epoch = config.eval_epoch  # do link prediction evaluation each 20 epochs
    n_epochs = config.n_epochs
    patience = config.patience
    batch_size = config.batch_size
    lr = config.lr
    margin = config.margin
    loss_fn = config.loss_fn

    ent_emb_dim = config.ent_emb_dim
    rel_emb_dim = config.rel_emb_dim
    scalar_share = config.scalar_share
    n_filters = config.n_filters

    df = pd.read_csv(dataset, sep=' ', header=None, names=['from', 'rel', 'to'])
    kg = KnowledgeGraph(df)
    kg_train, kg_val, kg_test = kg.split_kg(share=0.8, validation=True)

    # Define the model, criterion, optimizer, sampler and dataloader
    match method:
        case "TransE":
            model = TransEModel(ent_emb_dim, kg_train.n_ent, kg_train.n_rel, dissimilarity_type='L2')
        case "TransH":
            model = TransHModel(ent_emb_dim, kg_train.n_ent, kg_train.n_rel)
        case "TransR":
            model = TransRModel(ent_emb_dim, rel_emb_dim, kg_train.n_ent, kg_train.n_rel)
        case "TransD":
            model = TransDModel(ent_emb_dim, rel_emb_dim, kg_train.n_ent, kg_train.n_rel)
        case "TorusE":
            model = TorusEModel(ent_emb_dim, kg_train.n_ent, kg_train.n_rel, dissimilarity_type='torus_L2') #dissim type one of  ‘torus_L1’, ‘torus_L2’, ‘torus_eL2’.
        case "RESCAL":
            model = RESCALModel(ent_emb_dim, kg_train.n_ent, kg_train.n_rel)
        case "DistMult":
            model = DistMultModel(ent_emb_dim, kg_train.n_ent, kg_train.n_rel)
        case "HolE":
            model = HolEModel(ent_emb_dim, kg_train.n_ent, kg_train.n_rel)
        case "ComplEx":
            model = ComplExModel(ent_emb_dim, kg_train.n_ent, kg_train.n_rel)
        case "ANALOGY":
            model = AnalogyModel(ent_emb_dim, kg_train.n_ent, kg_train.n_rel, scalar_share)
        case "ConvKB":
            model = ConvKBModel(ent_emb_dim, n_filters, kg_train.n_ent, kg_train.n_rel)
        case _:
            raise ValueError(f"Method {method} not supported.")

    match loss_fn:
        case "margin":
            criterion = MarginLoss(margin)
        case "bce":
            criterion = LogisticLoss()
        case 'logistic':
            criterion = BinaryCrossEntropyLoss()
        case _:
            raise ValueError(f"Loss function {loss_fn} not supported.")
 
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    sampler = BernoulliNegativeSampler(kg_train)
    dataloader = DataLoader(kg_train, batch_size=batch_size, use_cuda='None')
    evaluator = LinkPredictionEvaluator(model, kg_val)
    
    print(dt.now() + ' Size of training set: {} triples'.format(len(kg_train)))
    print(dt.now() + ' Size of validation set: {} triples'.format(len(kg_val)))
    print(dt.now() + ' Size of test set: {} triples'.format(len(kg_test)) + '\n')
    
    # Move everything to CUDA if available
    use_cuda = cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
        cuda.empty_cache()
        model.to(device)
        criterion.to(device)
    else:
        device = torch.device('cpu')

    # Train the model
    iterator = tqdm(range(n_epochs), unit='epoch')
    for epoch in iterator:
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            h, t, r = batch[0], batch[1], batch[2]
            n_h, n_t = sampler.corrupt_batch(h, t, r) # generate negative samples

            optimizer.zero_grad()

            # forward + backward + optimize
            h, t, r, n_h, n_t = h.to(device), t.to(device), r.to(device), n_h.to(device), n_t.to(device)
            pos, neg = model(h, t, r, n_h, n_t)
            loss = criterion(pos, neg)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print('\n' + dt.now() + ' Epoch {} | mean loss: {:.5f}'.format(epoch + 1, running_loss / len(dataloader)))
        model.normalize_parameters()

    print(dt.now() + ' - Evaluating..')
    evaluator.evaluate(b_size=64, verbose=True)
    evaluator.print_results()
