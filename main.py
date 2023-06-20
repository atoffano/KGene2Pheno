import argparse
import os
import sys
import pandas as pd
# import wandb
import logging
import torch
from torch import cuda

from src.utils import *
import src.classifier
import src.train
import src.embeddings

def main():
    '''Parse the command line arguments'''
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument('--keywords', nargs='*', default=None, help='Multiple keywords')
    parser.add_argument('--method', required=True, help='Name of the method')
    parser.add_argument('--dataset', required=False, help='Name of the dataset')
    parser.add_argument('--query', default=None, help='A SPARQL query')

    parser.add_argument('--normalize_parameters', action='store_true', help='whether to normalize entity embeddings')

    parser.add_argument('--train_classifier', nargs='*', help='train a classifier on the embeddings')

    parser.add_argument('--save_model', action='store_true', help='whether to save the model weights')
    parser.add_argument('--save_data', action='store_true', help='whether to save the data split')
    parser.add_argument('--save_embeddings', action='store_true', help='whether to save the embeddings as csv')

    # Add torchkge arguments

    parser.add_argument('--n_epochs', required=False, default=20, type=int, help='Number of epochs')
    parser.add_argument('--batch_size', required=False, default=128, type=int, help='Batch size')
    parser.add_argument('--lr', required=False, default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', required=False, default=0.0001, type=float, help='Weight decay')
    parser.add_argument('--loss_fn', required=False, default="margin", type=str, help='loss function. ne of "margin", "bce", "logistic".')
    parser.add_argument('--ent_emb_dim', required=False, default=50, type=int, help='Size of entity embeddings')
    parser.add_argument('--eval_task', required=False, default="relation-prediction", type=str, help='Task on which to evaluate the embedding model. One of "link-prediction", "relation-prediction".')
    parser.add_argument('--split_ratio', required=False, default=0.8, type=float, help='train/test ratio')
    parser.add_argument('--dissimilarity_type', required=False, default='L1', type=str, help='Either "L1" or "L2", representing the type of dissimilarity measure to use')
    parser.add_argument('--margin', required=False, default=1, type=float, help='margin value.')

    # TorusE
    parser.add_argument('--rel_emb_dim', required=False, default=50, type=int, help='Size of relation embeddings')

    # ConvKB
    parser.add_argument('--n_filters', required=False, default=10, type=int, help='Number of ConvKB filters')
    parser.add_argument('--init_transe', nargs='*', required=False, default=False, help='Whether to initialize ConvKB with transe embeddings. Takes the following nargs: [path to .pt TransE model] [TransE entity embedding size] [TransE dissimilarity_type]')
    
    args = parser.parse_args()
    config = vars(args)


    # Change directory to the current file path
    current_file_path = os.path.realpath(__file__)
    current_dir = os.path.dirname(current_file_path)
    os.chdir(current_dir)

    if os.path.exists("query_result.txt") == True: # Remove query artifacts from previous runs
        os.remove("query_result.txt")

    # Start time  
    timestart = dt.now() 


    # Create a logger
    if os.path.exists("logs") == False:
        os.mkdir("logs")
    logging.basicConfig(filename=f'logs/{timestart}_{config["method"]}_{config["dataset"]}.log',
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s')
    logger = logging.getLogger()

    # Create a handler for console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the console handler
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    logger.info(f"Start time: {timestart}")


    # Set device
    use_cuda = cuda.is_available()
    if use_cuda:
        device = torch.device('cuda')
        cuda.empty_cache()
    else:
        device = torch.device('cpu')
    logger.info(f"Device: {device}\n")


    # Gather data, either from local file or SPARQL endpoint
    if args.query: # Query the SPARQL endpoint
        dataset = load_by_query(args.query)

    elif args.dataset and args.method != "PhenoGeneRanker":
        match args.dataset:
            case "celegans":
                dataset = load_celegans(args.keywords, sep=' ')
            case "local": # Celegans dataset from local file (to avoid sparql interaction)
                dataset = "data/raw/local.txt"
            case "toy-example": # Debug dataset
                dataset = "data/raw/toy-example.txt"
            case _:
                raise Exception("Dataset not supported.")
    
    # Legacy code for incomplete PhenoGeneRanker implementation
    # elif args.dataset and args.method == "PhenoGeneRanker":
    #     dataset = load_pgr(args.keywords, sep='\t')
    else:
        raise Exception("No dataset or query provided.")


    # wandb.config = {
    # "architecture": config['method'],
    # "learning_rate": config['lr'],
    # "epochs": config['n_epochs'],
    # "batch_size": config['batch_size'],
    # "embedding_dim": config['ent_emb_dim'],
    # "loss": config['loss_fn'],
    # "dataset": config['dataset'],
    # "split_ratio": config['split_ratio'],
    # "margin": config['margin'],
    # "dissimilarity_type": config['dissimilarity_type'],
    # "rel_emb_dim": config['rel_emb_dim'],
    # "n_filters": config['n_filters'],
    # "scalar_share": config['scalar_share'],
    # }
    # wandb.login()
    # wandb.init(project="kgene", config=config)

    # Train embedding model with the selected method
    if config['method'] and config['method'] in ["TransE", "TransH", "TransR", "TransD", "TorusE", "RESCAL", "DistMult", "HolE", "ComplEx", "ANALOGY", "ConvKB"]:
        emb_model, kg_train, kg_test= src.train.train(config['method'], dataset, config, timestart, logger, device)
        logger.info("Training of Embedding Model done !\n")
    else:
        raise Exception("Method not supported. Check spelling ?")

    if dataset not in ['data/raw/local.txt', 'data/raw/toy-example.txt']:
        os.remove(dataset) # Do not keep the dataset file if it was downloaded from the SPARQL endpoint

    # Train classifier
    if config['train_classifier']:
        logger.info("Converting test set to embeddings...")
        data = src.embeddings.generate(emb_model, kg_test, config, timestart, device)
        logger.info("Test set converted. It will be used to train the classifier\n")
        logger.info("Training classifier...")
        src.classifier.train_classifier(config['train_classifier'], data, timestart, logger=logger, device=device, save=config['save_model'])
        logger.info("Classifier trained !\n")

    if config['save_embeddings']:
        for kg, name in zip([kg_train, kg_test], ["train", "test"]):
            kg = src.embeddings.generate(emb_model, kg, config, timestart, device=device)
            kg.to_csv(f"data/embeddings/{config['method']}_{config['dataset']}_{name}_embeddings.csv", index=False)
    
    # if config['get_scores']:
    #     get_scores(emb_model, dataset, config)
        

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    #os.environ["WANDB_API_KEY"]=""
    main()
