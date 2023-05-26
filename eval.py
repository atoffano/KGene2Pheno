
def evaluate_emb_model(emb_model, kg_eval):
    """
    Evaluate the trained embedding model on a knowledge graph.

    Parameters
    ----------
    emb_model : torchkge.models.xxx
        The embedding model to be evaluated.
    kg_eval : torchkge.data_structures.KnowledgeGraph
        The knowledge graph used for evaluation.

    Returns
    -------
    None
    """
    criterion = BinaryCrossEntropyLoss()
    sampler = BernoulliNegativeSampler(kg_eval)
    dataloader = DataLoader(kg_eval, batch_size=3072, use_cuda='None')
    # test the embedding model
    running_loss = 0.0
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            h, t, r = batch[0], batch[1], batch[2]
            n_h, n_t = sampler.corrupt_batch(h, t, r) # generate negative samples

            # forward + backward + optimize
            h, t, r, n_h, n_t = h.to(device), t.to(device), r.to(device), n_h.to(device), n_t.to(device)
            pos, neg = emb_model(h, t, r, n_h, n_t)
            loss = criterion(pos, neg)
            running_loss += loss.item()
    print('Average loss on test set: {}'.format(running_loss / len(dataloader)))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
if __name__ == '__main__':
    import pandas as pd
    from tqdm import tqdm
    from datetime import datetime as dt
    import wandb
    import time
    import torch
    import numpy as np
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
    from sklearn.metrics import confusion_matrix

        # # Loads a model and the relevant test data, and run on a test set
    # inference_from_checkpoint('/home/antoine/gene_pheno_pred/models/TorchKGE/TransH_2023-03-13 17:08:16.530738.pt', '/home/antoine/gene_pheno_pred/emb_models/TorchKGE/TransH_2023-03-13 17:08:16.530738_kg_val.csv')
    import os
    os.chdir('/home/antoine/gene_pheno_pred')
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    print("Loading val dataset..")
    df = pd.read_csv('/home/antoine/gene_pheno_pred/models/TorchKGE/ComplEx_2023-05-04 20:36:44.446498_kg_test.csv', skiprows=[0], usecols=[1, 2, 3], header=None, names=['from', 'to', 'rel'])
    kg_eval = KnowledgeGraph(df)

    print("Loading test dataset..")
    df = pd.read_csv('/home/antoine/gene_pheno_pred/models/TorchKGE/ComplEx_2023-05-04 20:36:44.446498_kg_train.csv', skiprows=[0], usecols=[1, 2, 3], header=None, names=['from', 'to', 'rel'])
    kg_eval2 = KnowledgeGraph(df)


    # Model loading
    print("Loading model..")
    emb_model = ComplExModel(50, 675845, 10)
    emb_model.load_state_dict(torch.load('/home/antoine/gene_pheno_pred/models/TorchKGE/ComplEx_2023-05-04 20:36:44.446498.pt'))

    # Move everything to CUDA if available
    use_cuda = cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
        cuda.empty_cache()
        emb_model.to(device)
    else:
        device = torch.device('cpu')

    # Evaluate the model on the knowledge graph
    evaluate_emb_model(emb_model, kg_eval)
    evaluate_emb_model(emb_model, kg_eval2)