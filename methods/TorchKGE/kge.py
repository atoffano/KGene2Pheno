import pandas as pd
from tqdm import tqdm
from datetime import datetime as dt
import yaml
import numpy as np

import wandb

import torch
import torch.nn as nn

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

def train(method, dataset, timestart, args):

    # Config
    if args.default_config:
        with open(f'methods/TorchKGE/config/{method}.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = vars(args)

    wandb.config = {
    "architecture": "config['method']",
    "learning_rate": config['lr'],
    "epochs": config['n_epochs'],
    "batch_size": config['batch_size'],
    "embedding_dim": config['ent_emb_dim'],
    "loss": config['loss_fn'],
    "dataset": config['dataset'],
    "split_ratio": config['split_ratio'],
    "margin": config['margin'],
    "dissimilarity_type": config['dissimilarity_type'],
    "rel_emb_dim": config['rel_emb_dim'],
    "n_filters": config['n_filters'],
    "scalar_share": config['scalar_share'],
    }
    wandb.login()
    wandb.init(project="cigap", config=config)

    # Dataset loading
    df = pd.read_csv(dataset, sep=' ', header=None, names=['from', 'rel', 'to'])
    kg = KnowledgeGraph(df)

    print(f'{dt.now()} Number of triples: {kg.n_facts}')
    print(f'{dt.now()} Number of distinct entities: {kg.n_ent}')
    print(f'{dt.now()} Number of relations: {kg.n_rel}\n')

    kg_train, kg_val, kg_test = kg.split_kg(share=config['split_ratio'], validation=True)

    # Define the model, criterion, optimizer, sampler and dataloader
    match method:
        case "TransE":
            model = TransEModel(config['ent_emb_dim'], kg_train.n_ent, kg_train.n_rel, dissimilarity_type=config['dissimilarity_type'])
        case "TransH":
            model = TransHModel(config['ent_emb_dim'], kg_train.n_ent, kg_train.n_rel)
        case "TransR":
            model = TransRModel(config['ent_emb_dim'], config['rel_emb_dim'], kg_train.n_ent, kg_train.n_rel)
        case "TransD":
            model = TransDModel(config['ent_emb_dim'], config['rel_emb_dim'], kg_train.n_ent, kg_train.n_rel)
        case "TorusE":
            model = TorusEModel(config['ent_emb_dim'], kg_train.n_ent, kg_train.n_rel, dissimilarity_type=config['dissimilarity_type']) #dissim type one of  ‘torus_L1’, ‘torus_L2’, ‘torus_eL2’.
        case "RESCAL":
            model = RESCALModel(config['ent_emb_dim'], kg_train.n_ent, kg_train.n_rel)
        case "DistMult":
            model = DistMultModel(config['ent_emb_dim'], kg_train.n_ent, kg_train.n_rel)
        case "HolE":
            model = HolEModel(config['ent_emb_dim'], kg_train.n_ent, kg_train.n_rel)
        case "ComplEx":
            model = ComplExModel(config['ent_emb_dim'], kg_train.n_ent, kg_train.n_rel)
        case "ANALOGY":
            model = AnalogyModel(config['ent_emb_dim'], kg_train.n_ent, kg_train.n_rel, config['scalar_share'])
        case "ConvKB":
            model = ConvKBModel(config['ent_emb_dim'], config['n_filters'], kg_train.n_ent, kg_train.n_rel)
        case _:
            raise ValueError(f"Method {method} not supported.")
    wandb.watch(model, log="all")

    match config['loss_fn']:
        case "margin":
            criterion = MarginLoss(margin=config['margin'])
        case "bce":
            criterion = LogisticLoss()
        case 'logistic':
            criterion = BinaryCrossEntropyLoss()
        case _:
            raise ValueError(f"Loss function {config['loss_fn']} not supported.")
 
    optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)

    sampler = BernoulliNegativeSampler(kg_train)
    dataloader = DataLoader(kg_train, batch_size=config['batch_size'], use_cuda='None')

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
        model.to(device)
        criterion.to(device)
    else:
        device = torch.device('cpu')

    # Train the model
    iterator = tqdm(range(config['n_epochs']), unit='epoch')
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
        print(f'{dt.now()} - Epoch {epoch + 1} | mean loss: { running_loss / len(dataloader)}')
        wandb.log({'loss': running_loss / len(dataloader)})

        if config['normalize_parameters']:
            model.normalize_parameters()
    print(f'{dt.now()} - Finished Training !')

    if config['save_model']:
        torch.save(model.state_dict(), f'models/TorchKGE/{method}_{timestart}.pt')
    if config['save_data']:
        kg_train.get_df().to_csv(f'models/TorchKGE/{method}_{timestart}_kg_train.csv')
        kg_test.get_df().to_csv(f'models/TorchKGE/{method}_{timestart}_kg_test.csv')
        kg_val.get_df().to_csv(f'models/TorchKGE/{method}_{timestart}_kg_val.csv')

    evaluate_model(model, kg_val)


def train_classifier(model, dataloader):
        # Move everything to CUDA if available
    use_cuda = cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
        cuda.empty_cache()
        model.to(device)
    else:
        device = torch.device('cpu')

    model.to(device)
    learning_rate=0.1
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    n_epochs=1
    input_dim=100     # how many Variables are in the dataset
    hidden_dim = 50 # hidden layers
    output_dim=11    # number of classes

    dataloader = DataLoader(kg_train, batch_size=512, use_cuda='None')
    classifier=MultiClassifier(input_dim,hidden_dim,output_dim).to(device)
    criterion=nn.CrossEntropyLoss()
    sampler = BernoulliNegativeSampler(kg_train)

    # Train the model
    iterator = tqdm(range(n_epochs), unit='epoch')
    for epoch in iterator:
        running_pos_loss, running_neg_loss = 0.0, 0.0
        for batch in tqdm(dataloader):

            # Generate positive samples
            h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]
            # Generate negative samples by corrupting the tail
            n_h, n_t = sampler.corrupt_batch(h_idx, t_idx, r_idx) 
            
            # Get entity embeddings for the batch
            h = model.ent_emb(h_idx.to(device))
            t = model.ent_emb(t_idx.to(device))
            n_t = model.ent_emb(n_t.to(device))
            n_h = model.ent_emb(n_h.to(device))

            # Create a ground truth one-hot tensor for each sample like [x, x, x, x, ..., x, 0.0]
            ground_truth = np.zeros((len(h_idx), 11)) # Couille là
            for i, r_type in enumerate(r_idx):
                label_vec = np.zeros(11)
                label_vec[r_type] = 1.0
                ground_truth[i] = label_vec
            ground_truth = torch.tensor(ground_truth, dtype=torch.float32).to(device)
            
            # Create a ground truth one-hot tensor for each corrupted sample (negative samples) like [0.0, 0.0, 0.0, 0.0 ... 0.0, 1.0]
            neg_ground_truth = np.zeros((len(h_idx), 11))
            for i in range(len(neg_ground_truth)):
                vec = np.zeros(11)
                vec[10] = 1.0 # Last index denotes the absence of a link in the graph
                neg_ground_truth[i] = vec
            neg_ground_truth = torch.tensor(neg_ground_truth, dtype=torch.float32).to(device)

            optimizer.zero_grad()

            # forward + backward + optimize
            # ..on positive samples
            pos_x = classifier(h, t)
            pos_loss = criterion(pos_x, ground_truth)
            running_pos_loss += pos_loss.item()

            # ..on negative samples
            neg_x = classifier(n_h, n_t)
            neg_loss = criterion(neg_x, neg_ground_truth)
            running_neg_loss += neg_loss.item()

            # Backpropagate
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()
        print(f'{dt.now()} - Epoch {epoch + 1} | mean loss: {running_pos_loss / len(dataloader)}\nmean positive loss: {running_pos_loss / len(dataloader)}\nmean negative loss: { running_neg_loss / len(dataloader)}')
        # wandb.log({'classif_loss': running_loss / len(dataloader)})
    test_classifier(model, classifier)

def test_classifier(model, classifier):
    # Move everything to CUDA if available
    use_cuda = cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
        cuda.empty_cache()
        model.to(device)
    else:
        device = torch.device('cpu')

    # Dataset loading
    df = pd.read_csv('/home/antoine/gene_pheno_pred/models/TorchKGE/ConvKB_2023-03-16 17:02:16.120236_kg_val.csv', skiprows=[0], usecols=[1, 2, 3], header=None, names=['from', 'to', 'rel'])
    kg_val = KnowledgeGraph(df)
    dataloader = DataLoader(kg_val, batch_size=512, use_cuda='None')

    model.to(device)
    criterion=nn.CrossEntropyLoss()
    print(f'{dt.now()} - Running inference of classifier on validation set..')

    with torch.no_grad():
        running_loss = 0.0
        for _, batch in enumerate(dataloader):
            h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]
            h = model.ent_emb(h_idx.to(device))
            t = model.ent_emb(t_idx.to(device))
            # n_h, n_t = sampler.corrupt_batch(h, t, r) # generate negative samples
            
            # Create a ground truth tensor for the batch
            ground_truth = np.zeros((len(h_idx), 11))
            for i, rel_index in enumerate(r_idx):
                label_vec = np.zeros(11)
                label_vec[rel_index] = 1.0 # Relation class
                ground_truth[i] = label_vec
            ground_truth = torch.tensor(ground_truth, dtype=torch.float32).to(device)

            # forward + backward + optimize
            x = classifier(h, t)
            loss = criterion(x, ground_truth)
            running_loss += loss.item()
        print(f'{dt.now()} - mean loss: { running_loss / len(dataloader)}')


def evaluate_model(model, kg_eval):
    print(f'{dt.now()} - Evaluating..')
    evaluator = RelationPredictionEvaluator(model, kg_eval)
    evaluator.evaluate(b_size=4, verbose=True)

    # Log results to logfile
    print(f'{dt.now()} - Results:')
    evaluator.print_results(k=[i for i in range(1, 11)])

    # Log results to wandb
    get_results(evaluator)


def inference_from_checkpoint(model_path, test_path):
    use_cuda = cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
        cuda.empty_cache()
    else:
        device = torch.device('cpu')
    model = TransHModel(50,675845,10)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    kg_test = KnowledgeGraph(pd.read_csv(test_path))
    evaluate_model(model, kg_test)

def get_results(evaluator):
    for k in range(1, 11):
        wandb.log({f'Hit@{k}': evaluator.hit_at_k(k)[0]})
    wandb.log({'Mean Rank': evaluator.mean_rank()[0]})
    wandb.log({'MRR': int(evaluator.mrr()[0])})


class MultiClassifier(nn.Module):
    def __init__(self,emb_dim, hidden_size, nb_classes):
        super(MultiClassifier, self).__init__()
        self.linear1=nn.Linear(emb_dim, hidden_size)
        self.linear2=nn.Linear(hidden_size, nb_classes)

        
    def forward(self, h, t):
        x = torch.cat((h, t), 1)
        # x = torch.add(h, t)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

if __name__ == '__main__':
    # # Loads a model and the relevant test data, and run on a test set
    # inference_from_checkpoint('/home/antoine/gene_pheno_pred/models/TorchKGE/TransH_2023-03-13 17:08:16.530738.pt', '/home/antoine/gene_pheno_pred/models/TorchKGE/TransH_2023-03-13 17:08:16.530738_kg_val.csv')
    import os
    os.chdir('/home/antoine/gene_pheno_pred')
    # train('TransE', "celedebug.txt", dt.now())


    # Dataset loading
    df = pd.read_csv('/home/antoine/gene_pheno_pred/models/TorchKGE/ConvKB_2023-03-16 17:02:16.120236_kg_train.csv', skiprows=[0], usecols=[1, 2, 3], header=None, names=['from', 'to', 'rel'])
    kg_train = KnowledgeGraph(df)
    # Model loading
    model = ConvKBModel(50, 500, 675845,10)
    model.load_state_dict(torch.load('/home/antoine/gene_pheno_pred/models/TorchKGE/ConvKB_2023-03-15 13:03:19.327774.pt'))
    train_classifier(model, kg_train)