
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from torchkge.models import *
from torchkge.inference import *
 
from utils import *
from predict import load_embedding_model, load_graph
from classifier import *

class KGDataset(Dataset):
    """
    A custom dataset class for handling graph data stored in a pandas DataFrame.

    Parameters:
        dataframe (pandas.DataFrame): The input DataFrame containing the data.

    Attributes:
        data (numpy.ndarray): A 2D array representing the data from the DataFrame.

    """
    def __init__(self, dataframe):
        self.data = dataframe.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        head, tail = self.data[index]
        return head, tail

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

def annotation_matching(args, kg):
    """Perform annotation matching to target (a phenotype or a gene URI) on a knowledge graph.

    Args:
        args: An object containing command line arguments.
        kg: A KnowledgeGraph object.

    Returns:
        df_annot (pandas.DataFrame): A DataFrame containing matches between annotation and the target. It has the following columns:
            - annotation: The index corresponding to an annotation.
            - target: The index corresponding to the target.
        dict_annotations (dict): A dictionary containing all annotation nodes (keys) and their corresponding phenotype or gene (depending on target) URI (values).
        known_annotations (list): A list of all annotation nodes connected to the queried gene (args.gene) in the KnowledgeGraph kg.   
    """
    df = kg.get_df()

    target = args.gene if args.gene else args.phenotype # The target's URI.
    
    # Make a DataFrame containing all triples where the relation is 'has_phenotype' or 'has_gene', depending on the target type.
    if args.gene:
        df_annot = df[df['rel'] == 'http://semanticscience.org/resource/SIO_001279']
        df_target = df[df['rel'] == 'http://www.semanticweb.org/needed-terms#001']
    else:
        df_annot = df[df['rel'] == 'http://www.semanticweb.org/needed-terms#001']
        df_target = df[df['rel'] == 'http://semanticscience.org/resource/SIO_001279']

    # Make a dict containing all annotation nodes and their corresponding endpoint (phenotype or gene) URI.
    dict_annotations = dict(zip(df_annot['from'], df_annot['to']))
    
    # Bypass annotations to find matching phenotypes and genes 
    matching_rows = df_target[df_target['to'] == target]
    matching_rows = matching_rows[matching_rows['from'].isin(df_annot['from'])]
    matching_rows = matching_rows.merge(df_annot[['from', 'to']], on='from', how='left')

    # Store all endpoints connected to annotations connected to the target (ex: all phenotypes connected to the annotation connected to the input target gene)
    known_endpoints = matching_rows['to_y'] 
    known_endpoints.drop_duplicates(inplace=True)
    known_endpoints = known_endpoints.rename('tail')

    # Store all annotations connected to the target
    known_annotations = matching_rows['from']
    known_annotations.drop_duplicates(inplace=True)
    known_annotations = known_annotations.rename('annotation')

    # Make a DataFrame containing all annotations connected to the target and their corresponding phenotype or gene
    df_annot.drop(['rel', 'to'], axis=1, inplace=True)
    df_annot = df_annot.rename(columns={'from': 'annotation'})
    df_annot['target'] = target
    df_annot = df_annot.drop_duplicates()
    df_annot = df_annot.applymap(lambda x: kg.ent2ix[x]) # Convert all URIs to indices

    return df_annot, dict_annotations, known_annotations, known_endpoints

def parse_arguments():
    parser = argparse.ArgumentParser(description='Knowledge Graph Embedding Predictions')
    parser.add_argument('--model', type=str, nargs='+', required=True, help='[Model type] [Model path] [embedding dim] [Additional param : One of dissmimilary func (L1/L2) (TorusE/TransE), nb_filter (ConvKB), scalar share (ANALOGY)]', required=True)
    parser.add_argument('--filter_known_facts', action='store_true', help='Removes known facts from the predictions')
    parser.add_argument('--gene', type=str, help='Target gene URI')
    parser.add_argument('--phenotype', type=str, help='Target phenotype URI')
    parser.add_argument('--classifier', type=str, help='Path of the classifier model .pkl file')

    parser.add_argument('--graph', type=str, required=True, help='Path of the model\'s training data file as .csv(required)')
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

    df_annot, dict_annotations, known_annotations, known_endpoints = annotation_matching(args, kg)

    # Load the embedding model
    print("Loading model..")
    emb_model = load_embedding_model(args.model, kg)

    # Convert to embeddings
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

    # Put all embeddings in a df with 50 features (by default, x2 for ComplEx) per node (total 100 for head [0:49] and tail [50:99]) 
    df_emb = torch.cat((h_emb, t_emb), dim=1)
    df_emb = pd.DataFrame(df_emb.numpy())

    args.classifier = args.classifier.replace('.pkl', '') # Remove .pkl extension if present
    classifier = load_classifier(args.classifier)
    
    predictions = predict(classifier, df_emb)

    predictions['annotation'] = h_idx_dict
    predictions['target'] = t_idx_dict
    # Remove columns containing embedding
    filter_predictions = predictions.iloc[:, -5:]

    # Add col 'known' with value True if the filter_prediction['annotation'] of the row is in known_phenotypes, else False
    filter_predictions['known'] = filter_predictions['annotation'].isin(known_annotations)

    # Add a col 'matching_phenotype_or_gene' corresponding to the endpoint of the predicted annotation.
    # If the target is a phenotype, the endpoint is the gene, and vice versa
    filter_predictions['matching_phenotype_or_gene'] = filter_predictions['annotation'].map(dict_annotations)

    # If the matching phenotype or gene (the endpoint) of an annotation is known to be linked to the target (ie. the input gene or phenotype URI), add a col 'known_by_inference' with value True, else False.
    # This is useful because some annotations can be linked to an endpoint linked to the target, but the annotation itself may not linked to the target.
    # This results in the interaction being classified as not known, even though the interaction between the target and the endpoint is known to happen through another equivalent annotation.
    # Example: AnnotationA is linked to GeneA and an endpoint PhenotypeA in the graph. A prediction is made for AnnotationA and a target gene GeneT.
    # The prediction is classified as not known, because AnnotationA is not linked to GeneT. However, PhenotypeA is linked to GeneT through another annotation, Annotation2, so the interaction between GeneT and PhenotypeA is known to happen.
    # This is why we add the col 'known_by_inference' to keep track of these cases.
    filter_predictions['known_by_inference'] = filter_predictions['matching_phenotype_or_gene'].isin(known_endpoints)

    # Remove all rows where 'prediction_label' = 0. COMMENT OUT TO KEEP NEGATIVE PREDICTIONS
    filter_predictions = filter_predictions[filter_predictions['prediction_label'] == 1]

    # Order rows by confidence score.
    filter_predictions = filter_predictions.sort_values(by='prediction_score_1', ascending=False)

    filter_predictions.to_csv(args.output, index=False)

if __name__ == '__main__':
    main()