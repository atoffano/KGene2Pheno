
from tqdm import tqdm
import argparse
from termcolor import colored

import numpy as np
import pandas as pd
import torch
from torch import cuda
from torch.utils.data import Dataset

from torchkge.sampling import BernoulliNegativeSampler
# from torchkge.utils import DataLoader as DataLoader
from torchkge.data_structures import KnowledgeGraph
from torchkge.models import *
from torchkge.inference import *
 
from utils import *

from classifier import *

class KGDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        head, tail = self.data[index]
        return head, tail

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
    argsmodel[2] = int(argsmodel[2]) # Convert dim size to int
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
    df = pd.read_csv(graph_path, sep=' ', header=None, names=['from', 'rel', 'to'])
    kg = KnowledgeGraph(df)
    return kg

def parse_arguments():
    parser = argparse.ArgumentParser(description='Knowledge Graph Embedding Predictions')
    parser.add_argument('--model', type=str, nargs='+', help='[Model type] [Model path] [embedding dim] [Additional param : One of dissmimilary func (L1/L2) (TorusE/TransE), nb_filter (ConvKB), scalar share (ANALOGY)]', required=True)
    parser.add_argument('--filter_known_facts', action='store_true', help='Removes known facts from the predictions')
    parser.add_argument('--gene', type=str, help='Gene URI')
    parser.add_argument('--graph', type=str, required=True, help='Path of the model\'s training data file as .csv(required)')
    parser.add_argument('--file', type=str, help='CSV file containing queries in the format: [head,relation,?] or [?,relation,tail]')
    parser.add_argument('--triple', type=str, nargs='+', help='URI of triple like [head] [relation] [?] or [head] [relation] [?] (optional)')
    parser.add_argument('--b_size', type=int, default=264, help='Batch size (optional, default=264)')
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
    """
    args = parse_arguments()


    # Load the knowledge graph if provided
    print("Loading graph..")
    if args.graph:
        kg = load_graph(args.graph)
    else:
        raise Exception("No knowledge graph provided")

    df = kg.get_df()


    df_annot = df[df['rel'] == 'http://semanticscience.org/resource/SIO_001279']
    # Make a dict matching df_annot['from'] to df_annot['to']
    dict_annotations = dict(zip(df_annot['from'], df_annot['to']))

    df_gene = df[df['rel'] == 'http://www.semanticweb.org/needed-terms#001'] 

    # Find rows in df_gene where df_gene['to'] matches the target gene
    matching_rows = df_gene[df_gene['to'] == args.gene]

    # Find rows in matching_rows that share their annotation with df_annot
    matching_rows = matching_rows[matching_rows['from'].isin(df_annot['from'])]

    # Add col df_annot['to'] to result as 'phenotype' if df_annots['from'] matches result['from']
    matching_rows = matching_rows.merge(df_annot[['from', 'to']], on='from', how='left')

    known_annotations = matching_rows['from']
    known_annotations.drop_duplicates(inplace=True)
    known_annotations = known_annotations.rename('annotation')

    # Remove column `rel`
    df_annot.drop(['rel', 'to'], axis=1, inplace=True)

    # Rename col 'from' to 'annotation', add col 'gene'
    df_annot = df_annot.rename(columns={'from': 'annotation'})
    df_annot['gene'] = args.gene

    # Remove duplicates
    df_annot = df_annot.drop_duplicates()

    df_annot = df_annot.applymap(lambda x: kg.ent2ix[x])

    # Load the embedding model
    print("Loading model..")
    emb_model = load_model(args.model, kg)

    dataset = KGDataset(df_annot)
    batch_size = 264  # Specify your desired batch size
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    with torch.no_grad():
        h_idxs, t_idxs, h_emb, t_emb = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
        for h_idx, t_idx in dataloader:
            
            # Add batch to tensors
            h_idxs = torch.cat((h_idxs, h_idx), dim=0)
            t_idxs = torch.cat((t_idxs, t_idx), dim=0)

            # Get entity embeddings for the batch
            if type(emb_model) == ComplExModel:
                # Real embeddings
                h = emb_model.re_ent_emb(h_idx)
                t = emb_model.re_ent_emb(t_idx)

                # Imaginary embeddings
                im_h = emb_model.im_ent_emb(h_idx)
                im_t = emb_model.im_ent_emb(t_idx)

                # Concatenate real and imaginary embeddings
                h = torch.cat((h, im_h), dim=1)
                t = torch.cat((t, im_t), dim=1)

            else:
                h = emb_model.ent_emb(h_idx)
                t = emb_model.ent_emb(t_idx)
            
            # h, t to respective tensors
            h_emb = torch.cat((h_emb, h), dim=0)
            t_emb = torch.cat((t_emb, t), dim=0)

    ix2ent = {v: k for k, v in kg.ent2ix.items()} # Mapping of entity indices to entity names
    h_idx = h_idxs.numpy()
    h_idx_dict = np.vectorize(ix2ent.get)(h_idx) # Match head entity indices to entity names
    t_idx = t_idxs.numpy()
    t_idx_dict = np.vectorize(ix2ent.get)(t_idx) # Match tail entity indices to entity names

    # make all embeddings as a df with 50 features (default) per node (total 100 for head [0:49] and tail [50:99])
    df_emb = torch.cat((h_emb, t_emb), dim=1)
    df_emb = pd.DataFrame(df_emb.numpy())
    print(df_emb.shape)
    df_emb['annotation'] = h_idx_dict
    df_emb['gene'] = t_idx_dict

    predictions = classifier_inference('/home/antoine/KGene2Pheno/binary_classif/rf/rf_model_2023-06-26 13:00:36.058257', df_emb, output_path='/home/antoine/KGene2Pheno/classif_predictions.csv')
    
    # Keep only the last 5 cols
    filter_predictions = predictions.iloc[:, -5:]

    # Add col 'known' with value True if the filter_prediction['tail'] of the row is in known_phenotypes, else False
    filter_predictions['known'] = filter_predictions['annotation'].isin(known_annotations)
    
    # Add a col 'matching_phenotype' corresponding to the value of dict_annotations[filter_predictions['annotation']]
    filter_predictions['matching_phenotype'] = filter_predictions['annotation'].map(dict_annotations)


    # Remove all rows where 'prediction_label' = 0. COMMENT OUT TO KEEP NEGATIVE PREDICTIONS
    filter_predictions = filter_predictions[filter_predictions['prediction_label'] == 1]
    # Order rows by 'prediction_score1' in descending order
    filter_predictions = filter_predictions.sort_values(by='prediction_score_1', ascending=False)

    filter_predictions.to_csv('/home/antoine/KGene2Pheno/classif_predictions_filt.csv', index=False)

if __name__ == '__main__':
    main()