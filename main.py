import argparse
from utils import *
import os
import sys

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
    parser.add_argument('--n_filters', required=False, default=500, type=int, help='Number of filters (ConvKB)')

    # ANALOGY
    parser.add_argument('--scalar_share', required=False, default=0.5, type=float, help='Share of the diagonal elements of the relation-specific matrices to be scalars. By default it is set to 0.5 according to the original paper..')

    parser.add_argument('--normalize_parameters', action='store_true', help='whether to normalize entity embeddings')
    parser.add_argument('--save_model', action='store_true', help='whether to save the model weights')
    parser.add_argument('--save_data', action='store_true', help='whether to save the data splits')

    
    args = parser.parse_args()
    
    # Change directory
    current_file_path = os.path.realpath(__file__)
    current_dir = os.path.dirname(current_file_path)
    os.chdir(current_dir)

    if os.path.exists("query_result.txt") == True:
        os.remove("query_result.txt")
        
    # Create log file
    timestart = dt.now()
    log_file = open(f"log/{str(timestart).replace(' ', '_')}_{args.method}.log", "w")
    sys.stdout = log_file
    sys.stderr = log_file

    if args.query:
        dataset = load_by_query(args.query)

    elif args.dataset and args.method != "PhenoGeneRanker":
        match args.dataset:
            case "celegans":
                dataset = load_celegans(args.keywords, sep=' ')
            case "celedebug":
                dataset = "celedebug.txt"
                #dataset = load_biokg(args.keywords)
            case "toy-example":
                dataset = "toy-example.txt"
            case _:
                raise Exception("Dataset not supported.")
            
    elif args.dataset and args.method == "PhenoGeneRanker":
        dataset = load_pgr(args.keywords, sep='\t')
   
    else:
        raise Exception("No dataset or query provided.")

    train_model(args.method, dataset, timestart, args)
    if dataset != 'toy-example.txt':
        os.remove(dataset) 

    # Close log file
    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    
def train_model(method, dataset, timestart, args):
    if method in ["TransE", "TransH", "TransR", "TransD", "TorusE", "RESCAL", "DistMult", "HolE", "ComplEx", "ANALOGY", "ConvKB"]:
        import methods.TorchKGE.kge as kge
        emb_model = kge.train(method, dataset, timestart, args)
    elif method == "MultiVERSE":
        pass
    elif method == "PhenoGeneRanker":
        pass
    elif method == "DeepPheno":
        pass
    elif method == "HybridGNN":
        pass
    elif method == "AnyBURL":
        import subprocess

        # Check data file validity
        with open(dataset, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if not str.isalpha(line[0]):
                    raise Exception("AnyBURL expects that each identifier that appears in the data file starts with a alphabetic character and consists of at least 2 letters.")
        
        # Add training path to config file
        with open('methods/AnyBURL/config-learn.properties', 'a') as f:
            lines = f.readlines()
            f.write(f'PATH_TRAINING = {dataset}')
        
            subprocess.run(["java", "-Xmx10G", "-cp", "methods/AnyBURL/AnyBURL-23-1.jar", "de.unima.ki.anyburl.Learn", "config-learn.properties"], capture_output=True)

        # Remove the changes made to the config file
        with open('methods/AnyBURL/config-learn.properties', 'w') as f:
            f.write(lines)
    elif method == "GraphLP":
        pass
    elif method == "Relphormer":
        gen_vocab(dataset)
        match_id_names(dataset)
        split_dataset(dataset)


if __name__ == "__main__":
    main()
