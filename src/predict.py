
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
from embeddings import get_emb
from classifier import load_classifier, predict

def evaluate(ent_inf, b_size, filter_known_facts, verbose=True):
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

    with torch.no_grad():
        dataloader = DataLoader_(ent_inf.known_entities, ent_inf.known_relations, batch_size=b_size)
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                                unit='batch', disable=(not verbose),
                                desc='Inference'):
            known_ents, known_rels = batch[0], batch[1]
            query_emb, _, rel_emb, candidates = ent_inf.model.inference_prepare_candidates(known_ents, known_ents,
                                                                        known_rels,
                                                                        entities=True)
            if ent_inf.missing == 'heads': # Take into account whether we're predicting a head or a tail
                scores = ent_inf.model.inference_scoring_function(candidates, query_emb, rel_emb)
            else:
                scores = ent_inf.model.inference_scoring_function(query_emb, candidates, rel_emb)

            if filter_known_facts: # Remove already known facts
                scores = filter_scores(scores, ent_inf.dictionary, known_ents, known_rels, None)

            scores, indices = scores.sort(descending=True)
            # Isolate topk predictions

            ent_inf.predictions[i * b_size: (i+1)*b_size] = indices[:, :ent_inf.top_k]
            ent_inf.scores[i*b_size: (i+1)*b_size] = scores[:, :ent_inf.top_k]

def format_predictions(args, ent_inf, kg):
    """Formats the predictions from the entity inference model by converting indices to entities and matching them with both their scores and corresponding embeddings.

    Args:
        ent_inf (object): The entity inference model.
        kg (object): The knowledge graph.

    Returns:
        dict: A dictionary containing the formatted predictions in the form of {input_entity: [(predicted_entity, score, embedding_of_predicted_entity), ...]}.
    """
    if args.classifier:
       classifier = load_classifier(args.classifier)

    ix2ent = {v: k for k, v in kg.ent2ix.items()} # Transform all indices back to entities
    predictions, score = {}, {}

    known_idx = ent_inf.known_entities.numpy()
    known_idx_dict = np.vectorize(ix2ent.get)(known_idx) # Match head entity indices to entity names
    candidate_idx = ent_inf.predictions.numpy()
    candidate_idx_dict = np.vectorize(ix2ent.get)(candidate_idx) # Match tail entity indices to entity names
    print(ent_inf.known_entities.shape)
    print(ent_inf.predictions.shape)
    known_emb = get_emb(ent_inf.model, ent_inf.known_entities)
    candidate_emb = get_emb(ent_inf.model, ent_inf.predictions)
    print(candidate_emb.shape)
    print(known_emb.shape)
    exit()
    for i, known_entity in enumerate(ent_inf.known_entities):
        key_ix_str = ix2ent[known_entity.item()]
        predictions[key_ix_str] = []

        known_emb = get_emb(ent_inf.model, ent_inf.known_entities[i])
        candidate_emb = get_emb(ent_inf.model, ent_inf.predictions[i])
        print(candidate_emb.shape)
        print(known_emb.shape)
        if args.classifier:
            # concat tensors
            features = torch.cat((known_emb, candidate_emb), dim=0)
            # convert to df, with each index of the tensor being a feature
            features = pd.DataFrame(features.numpy()).T
            classifier_predictions = predict(classifier, features)



            classifier_predictions['annotation'] = known_idx_dict
            classifier_predictions['target'] = candidate_idx_dict
            # Remove columns containing embeddings
            filter_predictions = classifier_predictions.iloc[:, -5:]
            print(filter_predictions)

        exit()
        for ix, score in zip(ent_inf.predictions[i], ent_inf.scores[i]): # Match entity and its score
            predictions[key_ix_str].append((ix2ent[ix.item()], score.item())) # Get the embedding of the predicted entity
            if args.classifier:
                # get embedding of the predicted entity and the known entity embedding
                known_emb = get_emb(ent_inf.model, known_entity.item())
                candidate_emb = get_emb(ent_inf.model, ix.item())
                # concat tensors
                features = torch.cat((known_emb, candidate_emb), dim=0)
                # convert to df, with each index of the tensor being a feature
                features = pd.DataFrame(features.numpy()).T
                classifier_predictions = predict(classifier, features)
                classifier_predictions['annotation'] = ix2ent[ix.item()]
                classifier_predictions['target'] = ix2ent[known_entity.item()]
                # Remove columns containing embeddings
                filter_predictions = classifier_predictions.iloc[:, -5:]
                # Transform cols of the df into a list
                filter_predictions = filter_predictions.values.tolist()
                predictions[key_ix_str].append(filter_predictions) # Get the embedding of the predicted entity

    return predictions

def load_embedding_model(argsmodel, kg):
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
                emb_model = TorusEModel(argsmodel[2], kg.n_ent, kg.n_rel, dissimilarity_type=argsmodel[3]) #dissim type one of  ‘torus_L1’, ‘torus_L2’, ‘torus_eL2’.
            case "RESCAL":
                emb_model = RESCALModel(argsmodel[2], kg.n_ent, kg.n_rel)
            case "DistMult":
                emb_model = DistMultModel(argsmodel[2], kg.n_ent, kg.n_rel)
            case "HolE":
                emb_model = HolEModel(argsmodel[2], kg.n_ent, kg.n_rel)
            case "ComplEx":
                emb_model = ComplExModel(argsmodel[2], kg.n_ent, kg.n_rel)
            case "ANALOGY":
                emb_model = AnalogyModel(argsmodel[2], kg.n_ent, kg.n_rel, scalar_share=int(argsmodel[3]))
            case "ConvKB":
                emb_model = ConvKBModel(argsmodel[2], int(argsmodel[3]), kg.n_ent, kg.n_rel)
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
    df = pd.read_csv(graph_path, sep=' ', header=None, names=['from', 'rel', 'to'])
    kg = KnowledgeGraph(df)
    return kg

def parse_arguments():
    parser = argparse.ArgumentParser(description='Knowledge Graph Embedding Predictions')
    parser.add_argument('--model', type=str, nargs='+', help='[Model type] [Model path] [embedding dim] [Additional param : One of dissmimilary func (L1/L2) (TorusE/TransE), nb_filter (ConvKB), scalar share (ANALOGY)]', required=True)
    parser.add_argument('--filter_known_facts', action='store_true', help='Removes known facts from the predictions')
    parser.add_argument('--topk', type=int, default=10, help='Number of predictions to return (optional, default=10)')
    parser.add_argument('--graph', type=str, required=True, help='Path of the model\'s training data file as .csv(required)')
    parser.add_argument('--file', type=str, help='CSV file containing queries in the format: [head,relation,?] or [?,relation,tail]')
    parser.add_argument('--triple', type=str, nargs='+', help='URI of triple like [head] [relation] [?] or [?] [relation] [tail] (optional)')
    parser.add_argument('--b_size', type=int, default=264, help='Batch size (optional, default=264)')
    parser.add_argument('--classifier', type=str, help='Path of the classifier .pkl file. Adding this option will add predictions of a binary classifier on the existence of each link.')
    parser.add_argument('--output', type=str, help='Path of the prediction output file')
    return parser.parse_args()

def main():
    """Main function for executing the entity inference process.
        - The function parses command line arguments using the `parse_arguments` function.
        - The embedding model is loaded from the specified model file.
        - The knowledge graph is loaded using the `load_graph` function.
        - The `known_entities` and `known_relations` lists are populated based on the input arguments (either `triple` or `file` for multiple queries).
        - The inference is performed with filtering on known facts using `ent_inf_filt`, and the predictions are formatted using `format_predictions`.
        - If `filter_known_facts` flag is False, the inference is performed without filtering known facts using `ent_inf`, and the predictions are formatted.
        - The dictionaries of filtered and unfiltered predictions are merged, and the results are printed to the console.
        - If an output file is specified, the predictions are saved to the file.
    Prediction ranking follows a descending order of confidence: the higher the score the more confidence there is. Scores can not be directly compared between different models.
    """
    args = parse_arguments()

    if args.classifier:
        args.classifier = args.classifier.replace('.pkl', '') # Remove .pkl extension if present

    # Load the knowledge graph if provided
    print("Loading graph..")
    if args.graph:
        kg = load_graph(args.graph)
    else:
        raise Exception("No knowledge graph provided")

    # Load the embedding model
    print("Loading model..")
    emb_model = load_embedding_model(args.model, kg)


    # Convert head, relation, tail to known_entities and known_relations
    known_entities = []
    known_relations = []

    if args.triple:
        # convert triple to indices
        h, r, t = args.triple
        missing = 'heads' if h == '?' else 'tails'
        if missing == 'tails':
            known_entities.append(kg.ent2ix[h])
        else:
            known_entities.append(kg.ent2ix[t])
        known_relations.append(kg.rel2ix[r])

    elif args.file:
        # read file, split by comma, convert triples to indices
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

    # Prediction with filtering on known facts
    ent_inf_filt = EntityInference(emb_model, known_entities, known_relations, top_k=args.topk, missing=missing, dictionary=kg.dict_of_tails if missing == 'tails' else kg.dict_of_heads)
    evaluate(ent_inf_filt, args.b_size, filter_known_facts=True)
    filt_pred = format_predictions(args, ent_inf_filt, kg)
    
    if not args.filter_known_facts:
        # Prediction without filtering on known facts
        ent_inf = EntityInference(emb_model, known_entities, known_relations, top_k=args.topk, missing=missing, dictionary=kg.dict_of_tails if missing == 'tails' else kg.dict_of_heads)
        evaluate(ent_inf, args.b_size, filter_known_facts=False)
        unfilt_pred = format_predictions(args, ent_inf, kg)

        # Merge the filtered and unfiltered predictions to find known facts
        merged_data = {}
        for key in tqdm(unfilt_pred, desc='Identifying known facts'):
            merged_data[key] = sorted(list(set(unfilt_pred[key] + filt_pred.get(key, []))), key=lambda x: x[1], reverse=True)


        # Print predictions to console. Green values are known facts.
        for key in merged_data:
            values = merged_data[key]
            colored_values = [
                colored(str(value[:2]), 'green') if value[0] not in {x[0] for x in filt_pred[key]} else ''
                for value in values
            ]
            print(f"{key}: {', '.join(colored_values)}")
    else:
        for key, values in filt_pred.items():
            print(f"{key}: {', '.join(map(str, values))}")


    # Output predictions to the output file
    if args.output:
        ix2rel = {v: k for k, v in kg.rel2ix.items()} # Mapping of relation indices to entity names
        with open(args.output, 'w') as f:
            f.write(f"# Predicted using {args.model[0]} model - {args.model[1]}\n")
            f.write("head,relation,tail,score,source\n")
            if args.filter_known_facts:
                for entity, relation in zip(filt_pred, known_relations):
                    values = filt_pred[entity]
                    for value in values:
                        f.write(f"{entity}\t{ix2rel[relation.item()]}\t{value[0]}\t{value[1]},PRED\n" if missing == 'tails' else f"{value[0]}\t{ix2rel[relation.item()]}\t{entity}\t{value[1]},PRED\n")
            else:
                for entity, relation in zip(merged_data, known_relations):
                    values = merged_data[entity]
                    for value in values:
                        candidates = {x[0] for x in filt_pred[entity]}
                        known_fact = "KNOWN" if value[0] not in candidates else "PRED"
                        f.write(f"{entity}\t{ix2rel[relation.item()]}\t{value[0]}\t{value[1]},{known_fact}\n" if missing == 'tails' else f"{value[0]}\t{ix2rel[relation.item()]}\t{entity}\t{value[1]},{known_fact}\n")

    print(f"Predictions saved to {args.output}")



if __name__ == '__main__':
    main()