import argparse
from utils import *
import os
import sys
import pandas as pd
import yaml
import wandb
import torch
from torch import cuda

import classifier
import train
import embeddings

def main():
    '''Parse the command line arguments'''
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument('--keywords', nargs='*', default=None, help='Multiple keywords')
    parser.add_argument('--method', required=True, help='Name of the method')
    parser.add_argument('--dataset', required=True, help='Name of the dataset')

    parser.add_argument('--query', default=None, help='A SPARQL query')
    parser.add_argument('--data_format', default=None, help='Format of the dataset')
    parser.add_argument('--ouput', default='./', help='Directory to store the data')

    # Add torchkge arguments
    parser.add_argument('--default_config', action='store_true', help='Use the default config file for the given method.')

    parser.add_argument('--n_epochs', required=False, default=20, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', required=False, default=128, type=int, help='Batch size')
    parser.add_argument('--lr', required=False, default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', required=False, default=0.0001, type=float, help='Weight decay')
    parser.add_argument('--loss_fn', required=False, default="margin", type=str, help='loss function. ne of "margin", "bce", "logistic".')
    parser.add_argument('--ent_emb_dim', required=False, default=50, type=int, help='Size of entity embeddings')

    parser.add_argument('--split_ratio', required=False, default=0.8, type=float, help='train/test ratio')
    parser.add_argument('--dissimilarity_type', required=False, default='L1', type=str, help='Either "L1" or "L2", representing the type of dissimilarity measure to use')
    parser.add_argument('--margin', required=False, default=1, type=float, help='margin value.')

    # TorusE
    parser.add_argument('--rel_emb_dim', required=False, default=50, type=int, help='Size of entity embeddings')

    # ConvKB
    parser.add_argument('--n_filters', required=False, default=10, type=int, help='Number of filters (ConvKB)')
    parser.add_argument('--init_transe', nargs='*', required=False, default=True, help='Whether to init convKB with transe embeddings')

    # ANALOGY
    parser.add_argument('--scalar_share', required=False, default=0.5, type=float, help='Share of the diagonal elements of the relation-specific matrices to be scalars. By default it is set to 0.5 according to the original paper..')
    
    parser.add_argument('--normalize_parameters', action='store_true', help='whether to normalize entity embeddings')

    parser.add_argument('--classifier', nargs='*', required=False, default=False, help='train a classifier on the embeddings')
    parser.add_argument('--get_embeddings', action='store_true', help='Compute embeddings for the given method')

    parser.add_argument('--save_model', action='store_true', help='whether to save the model weights')
    parser.add_argument('--save_data', action='store_true', help='whether to save the data splits')

    
    args = parser.parse_args()

    # Change directory to the current file path
    current_file_path = os.path.realpath(__file__)
    current_dir = os.path.dirname(current_file_path)
    os.chdir(current_dir)

    if os.path.exists("query_result.txt") == True:
        os.remove("query_result.txt")
        
    timestart = dt.now()
    print(f"Start time: {timestart}")

    # Gather data, either from local file or SPAQL endpoint
    if args.query: # Query the SPARQL endpoint
        dataset = load_by_query(args.query)

    elif args.dataset and args.method != "PhenoGeneRanker":
        match args.dataset:
            case "celegans":
                dataset = load_celegans(args.keywords, sep=' ')
            case "local_celegans": # Celegans dataset from local file (to avoid sparql interaction)
                dataset = "local_celegans.txt"
            case "toy-example": # Debug dataset
                dataset = "toy-example.txt"
            case _:
                raise Exception("Dataset not supported.")
    
    # Legacy code for incomplete PhenoGeneRanker implementation
    # elif args.dataset and args.method == "PhenoGeneRanker":
    #     dataset = load_pgr(args.keywords, sep='\t')
    else:
        raise Exception("No dataset or query provided.")

    # Load config and init wandb tracking
    if args.default_config: # TODO : add default config for each method
        with open(f'methods/TorchKGE/config/{args.method}.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = vars(args)

    wandb.config = {
    "architecture": config['method'],
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
    wandb.init(project="cigap-base", config=config)

    # CUDA parameters
    use_cuda = cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
        cuda.empty_cache()
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # Train embedding model with the selected method
    if config['method'] and config['method'] in ["TransE", "TransH", "TransR", "TransD", "TorusE", "RESCAL", "DistMult", "HolE", "ComplEx", "ANALOGY", "ConvKB"]:
        emb_model, kg_train, kg_test, timestart = train.train(config['method'], dataset, config, timestart, device)
    else:
        raise Exception("Method not supported. Check spelling ?")

    if dataset not in ['toy-example.txt', 'local_celegans.txt']:
        os.remove(dataset) # Don't keep the dataset file if it was downloaded from the SPARQL endpoint

    if config['classifier']:
        data_path = f'{current_dir}/models/{config["method"]}_{timestart}_kg_test.csv'
        embeddings.generate(emb_model, kg_test, data_path, device)
        classifier.binary_classifier(config['classifier'], data_path, timestart, save=config['save_model'])

    if config['get_embeddings']:
        pass

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    os.environ["WANDB_API_KEY"]="4e5748d6c6f3917c78cdc38a516a1bac776faf58"
    main()
