import argparse
from utils import *
import os

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

    #AnyBURL specific arguments
    parser.add_argument('--snapshots_at', nargs='*', default=100, help='Rule snapshots at which timestamps (s). Usually decent predictions at 1000, may need to be higher for larger datasets.')
    parser.add_argument('--mem', default=4, help='Memory to use (GB)')
    parser.add_argument('--threshold_correct_predictions', default=5, help='Threshold for correct predictions')
    parser.add_argument('--batch_time', default=2000, help='time for batch')
    parser.add_argument('--worker_threads', default=20, help='how many workers to use')
    parser.add_argument('--policy', default=2, help='Possible values are 1 (greedy policy) and 2 (weighted policy). Experiments do not show a significant difference even tough the weighted variant is probably more robust in the sense that its less negative affected by the specifics of rather specific datasets.')
    parser.add_argument('--reward', default=5, help='Possible values are 1 (correct predictions), 3 (correct predictions weighted by confidence with laplace smoothing), 5 (correct predictions weighted by confidence with laplace smoothing divided by (rule length-1)^2).')
    parser.add_argument('--epsilon', default=0.1, help='0.1 is the default setting, which allocates a core with a probability of 0.1 randomly. You can change this to a value of 0.0 to 1.0 (= random policy).')
    # Length of the rules
    parser.add_argument('--max_length_cyclic', default=3, help='called binary rules or cyclic rules')
    parser.add_argument('--zero_rules_active', default=False, help='rules that try to capture very simple relation specific frequencies')
    parser.add_argument('--max_length_acyclic', default=1, help='these rules are called AC1 and AC2 rule or U_c or U_d rules (unary)')
    parser.add_argument('--max_length_grounded_cyclic', default=1, help='U_d or AC1 rules that are derived from a closed path')


    args = parser.parse_args()

    # Change directory
    current_file_path = os.path.realpath(__file__)
    current_dir = os.path.dirname(current_file_path)
    os.chdir(current_dir)

    if args.query:
        load_by_query(args.query)
    elif args.dataset:
        if args.dataset == "celegans":
            dataset = load_celegans(args.keywords)
        elif args.dataset == "ogb_biokg":
            pass
            #data_biokg = load_biokg(args.keywords)
    else:
        raise Exception("Please provide either a dataset or a query")

    train_model(args.method, args.dataset, args.data_format)

def train_model(method, dataset, data_format):
    if method in ["TransE", "TransH", "TransR", "TransD", "TorusE", "RESCAL", "DistMult", "HolE", "ComplEx", "ANALOGY", "ConvKB"]:
        import kge
        kge.train(method, dataset, data_format)
    elif method == "MultiVERSE":
        pass
    elif method == "PhenoGeneRanker":
        pass
    elif method == "DeepPheno":
        pass
    elif method == "MHGCN":
        pass
    elif method == "HybridGNN":
        pass
    elif method == "AnyBURL":
        pass
    elif method == "GraphLP":
        pass


if __name__ == "__main__":
    main()