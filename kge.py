from rdfpandas.graph import to_dataframe
import pandas as pd

import torch
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from ignite.metrics import RunningAverage
from torch.optim import Adam

from torchkge.evaluation import LinkPredictionEvaluator
from torchkge.models import TransEModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from torchkge.data_structures import KnowledgeGraph

def process_batch(engine, batch, sampler, optimizer, model, criterion):
    h, t, r = batch[0], batch[1], batch[2]
    n_h, n_t = sampler.corrupt_batch(h, t, r)

    optimizer.zero_grad()

    pos, neg = model(h, t, r, n_h, n_t)
    loss = criterion(pos, neg)
    loss.backward()
    optimizer.step()

    return loss.item()


def linkprediction_evaluation(engine, model, kg_val, eval_epoch):
    model.normalize_parameters()

    loss = engine.state.output

    # validation MRR measure
    if engine.state.epoch % eval_epoch == 0:
        evaluator = LinkPredictionEvaluator(model, kg_val)
        evaluator.evaluate(b_size=256, verbose=False)
        val_mrr = evaluator.mrr()[1]
    else:
        val_mrr = 0

    print('Epoch {} | Train loss: {}, Validation MRR: {}'.format(
        engine.state.epoch, loss, val_mrr))

    try:
        if engine.state.best_mrr < val_mrr:
            engine.state.best_mrr = val_mrr
        return val_mrr

    except AttributeError as e:
        if engine.state.epoch == 1:
            engine.state.best_mrr = val_mrr
            return val_mrr
        else:
            raise e

def train(method, dataset, data_format):
    device = torch.device('cuda')

    eval_epoch = 20  # do link prediction evaluation each 20 epochs
    max_epochs = 1000
    patience = 40
    batch_size = 32768
    emb_dim = 100
    lr = 0.0004
    margin = 0.5

    df = pd.read_csv(dataset, sep=' ', header=None)
    kg = KnowledgeGraph(df)
    kg_train, kg_val, kg_test = kg.split_kg(share=0.8, validation=True)

    # Define the model, optimizer and criterion
    model = TransEModel(emb_dim, kg_train.n_ent, kg_train.n_rel,
                        dissimilarity_type='L2')
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = MarginLoss(margin)
    sampler = BernoulliNegativeSampler(kg_train, kg_val=kg_val, kg_test=kg_test)

    # Define the engine
    trainer = Engine(process_batch(sampler=sampler, optimizer=optimizer, model=model, criterion=criterion))

    # Define the moving average
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'margin')

    # Add early stopping
    handler = EarlyStopping(patience=patience,
                            score_function=linkprediction_evaluation,
                            trainer=trainer,
                            model=model,
                            kg_val=kg_val,
                            eval_epoch=eval_epoch)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

    # Training
    train_iterator = DataLoader(kg_train, batch_size, use_cuda='all')
    trainer.run(train_iterator,
                epoch_length=len(train_iterator),
                max_epochs=max_epochs)

    print('Best score {:.3f} at epoch {}'.format(handler.best_score,
                                                trainer.state.epoch - handler.patience))
