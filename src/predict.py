
from tqdm import tqdm
import argparse
from termcolor import colored

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


def evaluate(ent_inf, b_size, filter_known_facts=True, verbose=True):
    """Performs evaluation on the given entity inference model.

    Args:
        ent_inf (object): The entity inference model.
        b_size (int): Batch size for data loading.
        filter_known_facts (bool, optional): Whether to filter known facts from the scores. Defaults to True.
        verbose (bool, optional): Whether to display progress information. Defaults to True.

    Returns:
        None

    Raises:
        None

    Notes:
        - This a modified copy of torchkge's EntityInference's evaluate func.
        - The `ent_inf` object should have the following attributes:
            - known_entities (list): List of known entities.
            - known_relations (list): List of known relations.
            - predictions (tensor): Tensor to store the predicted indices.
            - scores (tensor): Tensor to store the scores.
            - missing (str): Indicates missing heads or tails in the model.
            - model (object): The underlying inference model.
            - top_k (int): Number of top predictions to consider.
            - dictionary (object): Dictionary object for filtering known facts.

        - The `dataloader` object is initialized based on `known_entities`, `known_relations`, and `b_size`.

        - The inference is performed batch-wise using the `dataloader`.

        - The scoring function is applied based on the `missing` attribute.

        - If `filter_known_facts` is True, the scores are filtered using the `dictionary`, `known_ents`, `known_rels`, and None.

        - The top-k predictions and scores are stored in `ent_inf.predictions` and `ent_inf.scores`, respectively.
    """
    # if use_cuda:
    #     dataloader = DataLoader_(ent_inf.known_entities, ent_inf.known_relations, batch_size=b_size, use_cuda='batch')
    #     ent_inf.predictions = ent_inf.predictions.cuda()
    # else:
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

        if filter_known_facts:
            scores = filter_scores(scores, ent_inf.dictionary, known_ents, known_rels, None)

        scores, indices = scores.sort(descending=True)

        ent_inf.predictions[i * b_size: (i+1)*b_size] = indices[:, :ent_inf.top_k]
        ent_inf.scores[i*b_size: (i+1)*b_size] = scores[:, :ent_inf.top_k]


    # if use_cuda:
    #     ent_inf.predictions = ent_inf.predictions.cpu()
    #     ent_inf.scores = ent_inf.scores.cpu()

def format_predictions(ent_inf, kg):
    """Formats the predictions from the entity inference model.

    Args:
        ent_inf (object): The entity inference model.
        kg (object): The knowledge graph.

    Returns:
        dict: A dictionary containing the formatted predictions.

    Raises:
        None
    """
    ix2ent = {v: k for k, v in kg.ent2ix.items()} # Transform all indices back to entities
    predictions, score = {}, {}
    for i, known_entity in enumerate(ent_inf.known_entities):
        key_ix_str = ix2ent[known_entity.item()]
        predictions[key_ix_str] = [(ix2ent[ix.item()], score.item()) for ix, score in zip(ent_inf.predictions[i], ent_inf.scores[i])] # Match entity and its score
    return predictions

def load_model(argsmodel, kg):
    """Loads a pre-trained model from the specified path.

    Args:
        argsmodel (tuple): nargs containing: model type, the path to the .pt model file,
         the dimension of embeddings, and optionnaly the dissimilarity type / scalar share / nb_filters.


    Returns:
        object: The loaded embedding model.
    """
    argsmodel[2] = int(argsmodel[2]) # Convert dim number to int
    try:
        match argsmodel[0]:
            case "TransE":
                emb_model = TransEModel(argsmodel[2], kg.n_ent, kg.n_rel, dissimilarity_type=argsmodel[3])
            case "TransH":
                emb_model = TransHModel(argsmodel[2], kg.n_ent, kg.n_rel)
            case "TransR":
                emb_model = TransRModel(argsmodel[2], argsmodel[2], kg.n_ent, kg.n_rel)
            case "TransD":
                emb_model = TransDModel(argsmodel[2], argsmodel[2], kg.n_ent, kg.n_rel)
            case "TorusE":
                emb_model = TorusEModel(args.model[2], kg.n_ent, kg.n_rel, dissimilarity_type=argsmodel[3]) #dissim type one of  ‘torus_L1’, ‘torus_L2’, ‘torus_eL2’.
            case "RESCAL":
                emb_model = RESCALModel(argsmodel[2], kg.n_ent, kg.n_rel)
            case "DistMult":
                emb_model = DistMultModel(argsmodel[2], kg.n_ent, kg.n_rel)
            case "HolE":
                emb_model = HolEModel(argsmodel[2], kg.n_ent, kg.n_rel)
            case "ComplEx":
                emb_model = ComplExModel(argsmodel[2], kg.n_ent, kg.n_rel)
            case "ANALOGY":
                emb_model = AnalogyModel(argsmodel[2], kg.n_ent, kg.n_rel, scalar_share=argsmodel[3])
            case "ConvKB":
                ConvKBModel(argsmodel[2], kg.n_ent, kg.n_rel, nb_filters=argsmodel[3])
        emb_model.load_state_dict(torch.load(argsmodel[1]))
    except IndexError:
        raise IndexError("Index out of range. You may be missing one argument in --model.")
 
    return emb_model

def load_graph(graph_path):
    """Loads a knowledge graph from the specified .csv file.

    Args:
        graph_path (str): The path to the graph file.

    Returns:
        object: The loaded knowledge graph.
    """
    df = pd.read_csv(graph_path, sep=',', header=0, names=['from', 'rel', 'to'])
    kg = KnowledgeGraph(df)
    return kg

def parse_arguments():
    parser = argparse.ArgumentParser(description='Knowledge Graph Embedding Predictions')
    parser.add_argument('--model', type=str, nargs='+', help='[Model type] [Model path] [embedding dim] [Additional param : One of dissmimilary func (L1/L2) (TorusE/TransE), nb_filter (ConvKB), scalar share (ANALOGY)]', required=True)
    parser.add_argument('--filter_known_facts', action='store_true', help='Removes known facts from the predictions')
    parser.add_argument('--topk', type=int, default=10, help='Number of predictions to return (optional, default=10)')
    parser.add_argument('--graph', type=str, required=True, help='Path of the model\'s training data file as .csv(required)')
    parser.add_argument('--file', type=str, help='CSV file containing predictions in the format: head,relation,? or ?,relation,tail')
    parser.add_argument('--triple', type=str, nargs='+', help='URI of triple like [head] [relation] [?] or [head] [relation] [?] (optional)')
    parser.add_argument('--b_size', type=int, default=264, help='Batch size (optional, default=264)')
    parser.add_argument('--output', type=str, help='Path of the prediction output file')
    return parser.parse_args()

def main():
    """Main function for executing the entity inference process.

    Args:
        None

    Returns:
        None

    Raises:
        None

    Notes:
        - The function parses command line arguments using the `parse_arguments` function.

        - The embedding model is loaded from the specified model file.

        - The knowledge graph is loaded using the `load_graph` function.

        - The `known_entities` and `known_relations` lists are populated based on the input arguments (either `triple` or `file` for multiple queries).

        - The inference is performed with filtering on known facts using `ent_inf_filt`, and the predictions are formatted using `format_predictions`.

        - If `filter_known_facts` flag is False, the inference is performed without filtering known facts using `ent_inf`, and the predictions are formatted.

        - The dictionaries of filtered and unfiltered predictions are merged, and the results are printed to the console.

        - If an output file is specified, the predictions are saved to the file.

    """
    args = parse_arguments()

    # Load the knowledge graph if provided
    print("Loading graph..")
    if args.graph:
        kg = load_graph(args.graph)
    else:
        raise Exception("No knowledge graph provided")

    # Load the embedding model
    print("Loading model..")
    emb_model = load_model(args.model, kg)


    # Convert head, relation, tail to known_entities and known_relations
    known_entities = []
    known_relations = []

    if args.triple:
        h, r, t = args.triple
        missing = 'heads' if h == '?' else 'tails'
        if missing == 'tails':
            known_entities.append(kg.ent2ix[h])
        else:
            known_entities.append(kg.ent2ix[t])
        known_relations.append(kg.rel2ix[r])

    elif args.file:
        # read file, split by comma, convert to indices
        with open(args.file, 'r') as f:
            for line in f:
                h, r, t = line.strip().split(',')
                missing = 'heads' if h == '?' else 'tails'
                if missing == 'tails':
                    known_entities.append(kg.ent2ix[h])
                else:
                    known_entities.append(kg.ent2ix[t])
                known_relations.append(kg.rel2ix[r])
    

    known_entities = torch.tensor(known_entities, dtype=torch.long)
    known_relations = torch.tensor(known_relations, dtype=torch.long)

    # Inference with filter on known facts
    ent_inf_filt = EntityInference(emb_model, known_entities, known_relations, top_k=args.topk, missing='tails', dictionary=kg.dict_of_tails)
    evaluate(ent_inf_filt, args.b_size, filter_known_facts=True)
    filt_pred = format_predictions(ent_inf_filt, kg)
    
    if not args.filter_known_facts:
        # Inference without filtering known facts
        ent_inf = EntityInference(emb_model, known_entities, known_relations, top_k=args.topk, missing='tails', dictionary=kg.dict_of_tails)
        evaluate(ent_inf, args.b_size, filter_known_facts=False)
        unfilt_pred = format_predictions(ent_inf, kg)

        # Merge the dictionaries
        merged_data = {}
        for key in unfilt_pred:
            merged_data[key] = sorted(list(set(unfilt_pred[key] + filt_pred.get(key, []))), key=lambda x: x[1], reverse=True)


        # Print predictions to console
        for key in merged_data:
            values = merged_data[key]
            colored_values = []

            for value in values:
                if value not in filt_pred[key]:
                    colored_values.append(colored(str(value), 'green'))
                else:
                    colored_values.append(colored(str(value), 'light_yellow'))

            print(f"{key}: {', '.join(colored_values)}")
    else:
        # Print predictions to console
        for key in filt_pred:
            values = filt_pred[key]
            print(f"{key}: {', '.join([str(value) for value in values])}")

    
    ix2rel = {v: k for k, v in kg.rel2ix.items()}
    # Output predictions to file if provided
    if args.output:
        if args.filter_known_facts:
            with open(args.output, 'w') as f:
                for entity, relation in zip(filt_pred, known_relations):
                    values = filt_pred[entity]
                    for value in values:
                        if missing == 'tails':
                            f.write(f"{entity}\t{ix2rel[relation.item()]}\t{value[0]}\t{value[1]},PRED\n")
                        else:
                            f.write(f"{value[0]}\t{ix2rel[relation.item()]}\t{entity}\t{value[1]},PRED\n")
        else:
            with open(args.output, 'w') as f:
                for entity, relation in zip(merged_data, known_relations):
                    values = merged_data[entity]
                    for value in values:
                        if value not in filt_pred[entity]:
                            if missing == 'tails':
                                f.write(f"{entity}\t{ix2rel[relation.item()]}\t{value[0]}\t{value[1]}, KNOWN\n")
                            else:
                                f.write(f"{value[0]}\t{ix2rel[relation.item()]}\t{entity}\t{value[1]}, KNOWN\n")
                        else:
                            if missing == 'tails':
                                f.write(f"{entity}\t{ix2rel[relation.item()]}\t{value[0]}\t{value[1]}, PRED\n")
                            else:
                                f.write(f"{value[0]}\t{ix2rel[relation.item()]}\t{entity}\t{value[1]}, PRED\n")
                         
        print(f"Predictions saved to {args.output}")


if __name__ == '__main__':
    main()