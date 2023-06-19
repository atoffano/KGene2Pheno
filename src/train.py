import pandas as pd
from tqdm import tqdm
from datetime import datetime as dt
import os
# import wandb

# from ignite.engine import Engine, Events
# from ignite.handlers import EarlyStopping
# from ignite.metrics import RunningAverage
import torch
from torch.optim import Adam
from torch import cuda

from torchkge.evaluation import *
from torchkge.models import *
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, LogisticLoss, BinaryCrossEntropyLoss, DataLoader
from torchkge.data_structures import KnowledgeGraph
from sklearn.metrics import confusion_matrix

from src.utils import timer_func

@timer_func
def train(method, dataset, config, timestart, logger, device):
    """
    Train the embedding model using the specified method and dataset.

    Parameters
    ----------
    method : str
        The embedding method to use.
    dataset : str
        The file location of the dataset.
    config : dict
        CLI arguments.
    timestart : datetime.datetime
        The starting time of the training. Used for logging.

    Returns
    -------
    emb_model : torchkge.models.xxx
        The trained embedding model.
    kg_train : torchkge.data_structures.KnowledgeGraph
        The training knowledge graph.
    kg_test : torchkge.data_structures.KnowledgeGraph
        The test knowledge graph.
    """

    # Dataset loading and splitting
    df = pd.read_csv(dataset, sep=' ', header=None, names=['from', 'rel', 'to'])
    kg = KnowledgeGraph(df) # Create a knowledge graph from the dataframe

    logger.info('Splitting knowledge graph..')
    kg_train, kg_test = split(kg, split_ratio=config['split_ratio'], validation=False) 

    if kg.n_ent != kg_train.n_ent:
        raise ValueError('Some entities are not present in the training set. \n \
                         All entities should be seen during training, else the model will not be able to generate an embedding for unseen entities.')
    # logger.info number of entities and relations in each set:
    logger.info(f'\n Train set')
    logger.info(f'Number of entities: {kg_train.n_ent}')
    logger.info(f'Number of relations: {kg_train.n_rel}')
    logger.info(f'Number of triples: {kg_train.n_facts}')

    logger.info(f'\n Test set')
    logger.info(f'Number of entities: {kg_test.n_ent}')
    logger.info(f'Number of relations: {kg_test.n_rel}')
    logger.info(f'Number of triples: {kg_test.n_facts}')
    
    if kg_train.rel2ix != kg_test.rel2ix:
        logger.info('/!\ WARNING ! /!\ \nNumber of relations are not the same in all sets. \n \
              This is usually due to a relation not being present in the validation or test set. \n \
              It usually boils down to one directed relation type leading to an unconnected node. A classical example is the "label" relation.')

    # Initialize the embedding model
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
                # Decide whether to train a TransE from scratch or load a pretrained one
                if config['init_transe'] == []:
                    # Train a TransE model to initialize the embeddings.
                    # Config will be the same as the one used for ConvKB unless a path is specified as arg.
                    init_model, _, _, _ = train('TransE', dataset, config, timestart)
                    logger.info('TransE model trained.')

                else:
                    # Load a pretrained TransE model
                    try:
                        path, emb_dim, dissimilarity_type = config['init_transe']
                        emb_dim = int(emb_dim)
                    except:
                        raise ValueError(f"init_transe should have the following args: path, emb_dim, dissimilarity_type or a boolean.")
                    init_model = TransEModel(emb_dim=emb_dim, n_entities=kg_train.n_ent, n_relations=kg_train.n_rel, dissimilarity_type=dissimilarity_type)
                    init_model.load_state_dict(torch.load(path))
                    logger.info(f'TransE model loaded from {path}')

                # Initialize the ConvKB model's weights with the TransE embeddings
                ent_emb, rel_emb = init_model.get_embeddings()
                emb_model = ConvKBModel(config['ent_emb_dim'], config['n_filters'], kg_train.n_ent, kg_train.n_rel)
                emb_model.ent_emb.weight.data = ent_emb
                emb_model.rel_emb.weight.data = rel_emb
                logger.info('ConvKB model initialized with TransE embeddings.')

        case _:
            raise ValueError(f"Method {method} not supported.")
        
    # wandb.watch(emb_model, log="all")

    # Define the loss function, the dataloaders, the optimizer and the negative samplers
    # Add your own custom losses as another case
    match config['loss_fn']:
        case "margin":
            criterion = MarginLoss(margin=config['margin'])
        case "logistic":
            criterion = LogisticLoss()
        case "bce":
            criterion = BinaryCrossEntropyLoss()
        case _:
            raise ValueError(f"Loss function {config['loss_fn']} not supported.")
 
    dataloader = DataLoader(kg_train, batch_size=config['batch_size'], use_cuda='None')
    optimizer = Adam(emb_model.parameters(), lr=config['lr'], weight_decay=1e-5)
    sampler = BernoulliNegativeSampler(kg_train)

    test_dataloader = DataLoader(kg_test, batch_size=config['batch_size'], use_cuda='None')
    test_sampler = BernoulliNegativeSampler(kg_test)

    # Move to gpu if available
    emb_model.to(device)
    criterion.to(device)

    # Log parameters
    logger.info(f'\n{dt.now()} - PARAMETERS')
    for i in config.items():
        logger.info(f'\t {i[0]} : {i[1]}' )

    # Training loop
    iterator = tqdm(range(config['n_epochs']), unit='epoch')
    for epoch in iterator:
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            h, t, r = batch[0], batch[1], batch[2] # Generate batches of positive triples
            n_h, n_t = sampler.corrupt_batch(h, t, r) # Generate negative samples

            optimizer.zero_grad()

            # forward + backward + optimize
            h, t, r, n_h, n_t = h.to(device), t.to(device), r.to(device), n_h.to(device), n_t.to(device)
            pos, neg = emb_model(h, t, r, n_h, n_t)
            loss = criterion(pos, neg)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        validation_loss = val_loss(test_dataloader, test_sampler, emb_model, criterion, device) # Compute validation loss
        logger.info(f'{dt.now()} - Epoch {epoch + 1} | mean loss: { running_loss / len(dataloader)}, val loss: {validation_loss}')
        # wandb.log({'loss': running_loss / len(dataloader)})
        # wandb.log({'val_loss': validation_loss})

        if config['normalize_parameters']: # Normalize embeddings after each epoch
            emb_model.normalize_parameters()

    logger.info(f'{dt.now()} - Finished Training !')

    # Save the model and/or the data
    if os.path.exists('models') == False:
        os.mkdir('models')

    if config['save_model']:
        torch.save(emb_model.state_dict(), f'models/{method}_{timestart}.pt')
    if config['save_data']:
        kg_train.get_df().to_csv(f'models/{method}_{timestart}_kg_train.csv')
        kg_test.get_df().to_csv(f'models/{method}_{timestart}_kg_test.csv')

    # Evaluate the model on a task to get performance (Hit@k, MRR)
    evaluate_emb_model(emb_model, kg_test, config["eval_task"], device, logger=logger)
    return emb_model, kg_train, kg_test

@timer_func
def split(kg, split_ratio=0.8, validation=False):
    return kg.split_kg(share=split_ratio, validation=validation)

def val_loss(dataloader, sampler, emb_model, criterion, device='cpu'):
    """
    Compute the validation loss of the embedding model.

    Parameters
    ----------
    dataloader : torchkge.utils.DataLoader
        The dataloader for the validation set.
    sampler : torchkge.sampling.NegativeSampler
        The negative sampler.
    emb_model : torchkge.models.xxx
        The embedding model.
    criterion : torchkge.utils.LossFunction
        The loss function.

    Returns
    -------
    float
        The validation loss, averaged over all batches of the validation set.
    """
    with torch.no_grad():
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            h, t, r = batch[0], batch[1], batch[2]
            n_h, n_t = sampler.corrupt_batch(h, t, r) # generate negative samples

            # forward
            h, t, r, n_h, n_t = h.to(device), t.to(device), r.to(device), n_h.to(device), n_t.to(device)
            pos, neg = emb_model(h, t, r, n_h, n_t)
            loss = criterion(pos, neg)
            running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate_emb_model(emb_model, kg_eval, task, device, logger):
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
        
    logger.info(f'{dt.now()} - Evaluating..')
    b_size = 264 # Lower batch size if OOM error during evaluation

    match task:
        case 'link-prediction':
            evaluator = LinkPredictionEvaluator(emb_model, kg_eval)
            evaluator.evaluate(b_size=b_size, verbose=True)

        case 'triple-classification':
            evaluator = TripleClassificationEvaluator(emb_model, kg_eval)
            evaluator.evaluate(b_size=b_size, verbose=True)

        case 'relation-prediction':
            evaluator = RelationPredictionEvaluator(emb_model, kg_eval)

            use_cuda = next(evaluator.model.parameters()).is_cuda

            if use_cuda:
                dataloader = DataLoader(evaluator.kg, batch_size=b_size, use_cuda='batch')
                evaluator.rank_true_rels = evaluator.rank_true_rels.cuda()
                evaluator.filt_rank_true_rels = evaluator.filt_rank_true_rels.cuda()
            else:
                dataloader = DataLoader(evaluator.kg, batch_size=b_size)

            all_scores, all_true_ranks = torch.tensor([]).to(device), torch.tensor([]).to(device)
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                                    unit='batch', disable=(not True),
                                    desc='Relation prediction evaluation'):
                h_idx, t_idx, r_idx = batch[0], batch[1], batch[2]
                h_emb, t_emb, r_emb, candidates = evaluator.model.inference_prepare_candidates(h_idx, t_idx, r_idx, entities=False)

                scores = evaluator.model.inference_scoring_function(h_emb, t_emb, candidates)
                filt_scores = filter_scores(scores, evaluator.kg.dict_of_rels, h_idx, t_idx, r_idx)

                if not evaluator.directed:
                    scores_bis = evaluator.model.inference_scoring_function(t_emb, h_emb, candidates)
                    filt_scores_bis = filter_scores(scores_bis, evaluator.kg.dict_of_rels, h_idx, t_idx, r_idx)

                    scores = cat((scores, scores_bis), dim=1)
                    filt_scores = cat((filt_scores, filt_scores_bis), dim=1)

                true_data = scores.gather(1, r_idx.long().view(-1, 1))
                
                ranks = (scores >= true_data).sum(dim=1).detach()
                evaluator.rank_true_rels[i * b_size: (i + 1) * b_size] = ranks
                evaluator.filt_rank_true_rels[i * b_size: (i + 1) * b_size] = (filt_scores >= true_data).sum(dim=1).detach()

                ranks = (scores >= true_data).sum(dim=1).detach()
                max_scores = torch.argmax(scores, dim=1)
                all_scores = torch.cat((all_scores, max_scores), dim=0)
                all_true_ranks = torch.cat((all_true_ranks, ranks), dim=0)

            evaluator.evaluated = True

            if use_cuda:
                evaluator.rank_true_rels = evaluator.rank_true_rels.cpu()
                evaluator.filt_rank_true_rels = evaluator.filt_rank_true_rels.cpu()


            # Compute the confusion matrix for the relation prediction task
            logger.info(kg_eval.rel2ix)
            # Convert tensors to numpy arrays
            all_scores_np = all_scores.cpu().numpy()
            all_true_ranks_np = all_true_ranks.cpu().numpy()
            cm = confusion_matrix(all_true_ranks_np, all_scores_np)

            # logger.info the confusion matrix
            logger.info('\n' + cm)

    # Log results to logfile
    logger.info(f'{dt.now()} - EMBEDDING MODEL EVALUATION RESULTS:')
    logger.info(f'Task : {task}')
    for k in [1, 3, 5, 10]:
        logger.info(f'Hit@{k} : {evaluator.hit_at_k(k)[0]}')
    logger.info(f'Mean Rank : {evaluator.mean_rank()[0]}')
    logger.info(f'MRR : {evaluator.mrr()[0]}')

    # Log results to wandb
    # for k in range(1, 11):
    #     wandb.log({f'Hit@{k}': evaluator.hit_at_k(k)[0]})
    # wandb.log({'Mean Rank': evaluator.mean_rank()[0]})
    # wandb.log({'MRR': evaluator.mrr()[0]})

if __name__ == '__main__':
    pass