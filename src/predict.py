
from tqdm import tqdm
import numpy as np

import pandas as pd
import torch

from torch import cuda

from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import DataLoader
from torchkge.data_structures import KnowledgeGraph
from torchkge.models import *

from utils import timer_func

print("Loading val dataset..")
df = pd.read_csv('/home/antoine/gene_pheno_pred/models/ComplEx_2023-05-27 20:08:45.227169_kg_train.csv', skiprows=[0], usecols=[1, 2, 3], header=None, names=['from', 'to', 'rel'])
kg_val = KnowledgeGraph(df)


# Model loading
print("Loading model..")
emb_model = TransEModel(emb_dim=50, n_entities=99699, n_relations=9)
emb_model.load_state_dict(torch.load('/home/antoine/gene_pheno_pred/models/non-reified_ComplEx_2023-05-05 17:04:52.671826.pt'))

# Move everything to CUDA if available
use_cuda = cuda.is_available()
if use_cuda:
    device = torch.device('cuda')
    cuda.empty_cache()
    emb_model.to(device)
else:
    device = torch.device('cpu')

