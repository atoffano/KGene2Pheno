
from tqdm import tqdm
import numpy as np

import pandas as pd
import torch

from torch import cuda

from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import DataLoader
from torchkge.data_structures import KnowledgeGraph
from torchkge.models import *
from torchkge.inference import *

from utils import timer_func

# Model loading
print("Loading model..")
emb_model = TransEModel(emb_dim=50, n_entities=125, n_relations=3)
emb_model.load_state_dict(torch.load('/home/antoine/KGene2Pheno/models/TransE_2023-06-21 14:52:34.233254.pt'))
torch.cuda.set_device(0)

# Move everything to CUDA if available
# use_cuda = cuda.is_available()
# if use_cuda:
#     device = torch.device('cuda')
#     cuda.empty_cache()
#     emb_model.to(device)
# else:
#     device = torch.device('cpu')
# print(device)
df = pd.read_csv('/home/antoine/KGene2Pheno/data/raw/toy-example.txt', sep=' ', header=None, names=['from', 'rel', 'to'])
kg = KnowledgeGraph(df) # Create a knowledge graph from the dataframe

known_entities = [kg.ent2ix['https://wormbase.org/species/c_elegans/gene/WBGene00017644'], kg.ent2ix['https://wormbase.org/wbdata/pheno-443']]
known_relations = [kg.rel2ix['http://www.w3.org/1999/02/22-rdf-syntax-ns#type'], kg.rel2ix['http://www.semanticweb.org/needed-terms#001']]
known_entities = torch.tensor(known_entities)
known_relations = torch.tensor(known_relations)
# known_entities = known_entities.to(device)
# known_relations = known_relations.to(device)
ent_inf = EntityInference(emb_model, known_entities, known_relations, top_k=10, missing='tails')

# ent_inf.evaluate(b_size=50, verbose=True)

use_cuda = next(ent_inf.model.parameters()).is_cuda

b_size=264
verbose=True
if use_cuda:
    dataloader = DataLoader_(ent_inf.known_entities, ent_inf.known_relations, batch_size=b_size, use_cuda='batch')
    ent_inf.predictions = ent_inf.predictions.cuda()
else:
    dataloader = DataLoader_(ent_inf.known_entities, ent_inf.known_relations, batch_size=b_size)

for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                        unit='batch', disable=(not verbose),
                        desc='Inference'):
    known_ents, known_rels = batch[0], batch[1]
    if ent_inf.missing == 'heads':
        _, t_emb, rel_emb, candidates = ent_inf.model.inference_prepare_candidates(tensor([]).long(), known_ents,
                                                                                known_rels,
                                                                                entities=True)
        scores = ent_inf.model.inference_scoring_function(candidates, t_emb, rel_emb)
    else:
        h_emb, _, rel_emb, candidates = ent_inf.model.inference_prepare_candidates(known_ents, tensor([]).long(),
                                                                                known_rels,
                                                                                entities=True)
        scores = ent_inf.model.inference_scoring_function(h_emb, candidates, rel_emb)

    if ent_inf.dictionary is not None:
        scores = filter_scores(scores, ent_inf.dictionary, known_ents, known_rels, None)

    scores, indices = scores.sort(descending=True)
    print(scores)

    ent_inf.predictions[i * b_size: (i+1)*b_size] = indices[:, :ent_inf.top_k]
    ent_inf.scores[i*b_size: (i+1)*b_size] = scores[:, :ent_inf.top_k]

    print(ent_inf.scores)
    print(ent_inf.predictions)


if use_cuda:
    ent_inf.predictions = ent_inf.predictions.cpu()
    ent_inf.scores = ent_inf.scores.cpu()

# Transform all indices back to entities
ix2ent = {v: k for k, v in kg.ent2ix.items()}
predictions = {}
for i in range(len(ent_inf.predictions)):
    key_ix = known_entities[i].item()
    key_ix_str = ix2ent[key_ix]
    predictions[key_ix_str] = []
    for p in range(len(ent_inf.predictions[i])):
        ix = ent_inf.predictions[i][p].item()
        predictions[key_ix_str].append(ix2ent[ix])

print(predictions)



# predictions = [ent_inf.predictions[i] for i in ix2ent[ent_inf.predictions[i]]]