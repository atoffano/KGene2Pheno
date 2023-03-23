
from tqdm import tqdm
from datetime import datetime as dt
import yaml
import numpy as np

import wandb

import pandas as pd
import torch
import torch.nn as nn

from torch.optim import Adam
from torch import cuda

from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import DataLoader
from torchkge.data_structures import KnowledgeGraph
from torchkge.models import *

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


def classif_forward(emb_model, classifier, batch, sampler, criterion):
    # Generate positive samples
    h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]
    # Generate negative samples by corrupting the tail
    n_h, n_t = sampler.corrupt_batch(h_idx, t_idx, r_idx) 
    
    # Get entity embeddings for the batch
    h = emb_model.ent_emb(h_idx.to(device))
    t = emb_model.ent_emb(t_idx.to(device))
    n_t = emb_model.ent_emb(n_t.to(device))
    n_h = emb_model.ent_emb(n_h.to(device))

    # Create a ground truth one-hot tensor for each sample like [x, x, x, x, ..., x, 0.0]
    ground_truth = np.zeros((len(h_idx), 11))
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

    # Positive samples
    pos_x = classifier(h, t)
    pos_loss = criterion(pos_x, ground_truth)

    # Negative samples
    neg_x = classifier(n_h, n_t)
    neg_loss = criterion(neg_x, neg_ground_truth)
    return  pos_loss, neg_loss
    
def train_classifier(emb_model, kg_train, kg_val):
    dataloader = DataLoader(kg_train, batch_size=512, use_cuda='None')

    emb_model.to(device)
    learning_rate=0.1
    optimizer = Adam(emb_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    n_epochs=10
    input_dim=100     # how many Variables are in the dataset
    hidden_dim = 50 # hidden layers
    output_dim=11    # number of classes

    dataloader = DataLoader(kg_train, batch_size=512, use_cuda='None')

    classifier=MultiClassifier(input_dim,hidden_dim,output_dim).to(device)
    criterion=nn.CrossEntropyLoss()
    sampler = BernoulliNegativeSampler(kg_train)

    # Train the classifier
    iterator = tqdm(range(n_epochs), unit='epoch')
    for epoch in iterator:
        running_pos_loss, running_neg_loss = 0.0, 0.0
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            pos_loss, neg_loss = classif_forward(emb_model, classifier, batch, sampler, criterion)
            running_pos_loss += pos_loss.item()
            running_neg_loss += neg_loss.item()
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()

        print(f'{dt.now()} - Epoch {epoch + 1} | mean loss: {((running_pos_loss + running_neg_loss) / 2) / len(dataloader)}\nmean positive loss: {running_pos_loss / len(dataloader)}\nmean negative loss: { running_neg_loss / len(dataloader)}')
        wandb.log({'Classifier pos_loss': running_pos_loss / len(dataloader), 'Classifier neg_loss': running_neg_loss / len(dataloader), 'Classifier Loss': loss / len(dataloader)})
    
    # Test the classifier
    test_classifier(emb_model, classifier, kg_val)


def test_classifier(emb_model, classifier, kg_val):
    print(f'{dt.now()} - Running inference of classifier on validation set..')
    dataloader = DataLoader(kg_val, batch_size=512, use_cuda='None')
    criterion=nn.CrossEntropyLoss()
    sampler = BernoulliNegativeSampler(kg_val)
    running_pos_loss, running_neg_loss = 0.0, 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            pos_loss, neg_loss = classif_forward(emb_model, classifier, batch, sampler, criterion)
            running_pos_loss += pos_loss.item()
            running_neg_loss += neg_loss.item()
            loss = pos_loss + neg_loss
    print(f'{dt.now()} - mean loss: {running_pos_loss / len(dataloader)}\nmean positive loss: {running_pos_loss / len(dataloader)}\nmean negative loss: { running_neg_loss / len(dataloader)}')
    wandb.log({'Classifier pos_loss': running_pos_loss / len(dataloader), 'Classifier neg_loss': running_neg_loss / len(dataloader), 'Classifier Loss': loss / len(dataloader)})

if __name__ == '__main__':
    # # Loads a model and the relevant test data, and run on a test set
    # inference_from_checkpoint('/home/antoine/gene_pheno_pred/models/TorchKGE/TransH_2023-03-13 17:08:16.530738.pt', '/home/antoine/gene_pheno_pred/emb_models/TorchKGE/TransH_2023-03-13 17:08:16.530738_kg_val.csv')
    import os
    os.chdir('/home/antoine/gene_pheno_pred')
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    os.environ["WANDB_API_KEY"]="4e5748d6c6f3917c78cdc38a516a1bac776faf58"

    # train('TransE', "celedebug.txt", dt.now())


    # Dataset loading
    df = pd.read_csv('/home/antoine/gene_pheno_pred/models/TorchKGE/TransE_2023-03-22 14:54:57.152481_kg_train.csv', skiprows=[0], usecols=[1, 2, 3], header=None, names=['from', 'to', 'rel'])
    kg_train = KnowledgeGraph(df)
    df = pd.read_csv('/home/antoine/gene_pheno_pred/models/TorchKGE/TransE_2023-03-22 14:54:57.152481_kg_val.csv', skiprows=[0], usecols=[1, 2, 3], header=None, names=['from', 'to', 'rel'])
    kg_val = KnowledgeGraph(df)

    # Model loading
    emb_model = TransEModel(50, 675845,10)
    emb_model.load_state_dict(torch.load('/home/antoine/gene_pheno_pred/models/TorchKGE/TransE_2023-03-22 14:54:57.152481.pt'))

    # Move everything to CUDA if available
    use_cuda = cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
        cuda.empty_cache()
        emb_model.to(device)
    else:
        device = torch.device('cpu')

    # Config
    wandb.config = {
    "learning_rate": 0.1,
    "epochs": 10,
    "batch_size": 512,
    "loss": 'CrossEntropyLoss',
    "input_dim" : 100,
    "hidden_dim" : 50,
    "output_dim" : 11
    }
    wandb.login()
    wandb.init(project="cigap", config=wandb.config)

    train_classifier(emb_model, kg_train, kg_val)
