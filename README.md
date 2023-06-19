# Gene-Phenotype relation prediction
This repository stores the code required to derive insight from the wormbase database, by predicting links between genes and phenotypes in *C.Elegans* through various knowledge graph completion models.

## Prerequisites

Make sure you have the following dependencies installed:
- python 3.10 or higher
- pandas
- SPARQLWrapper
- torch
- torchkge
- sklearn
- pycaret

You can install the dependencies using the following command line or with the provided requirements.txt file:

```bash
pip install torchkge pycaret torch SPARQLWrapper pandas numpy
```
```bash
pip install -r requirements.txt
```

Also, due to a bug in the TorchKGE package, relation prediction evaluation of the TorusE model does not work, as a parameter that is not defined in TorusE is called. If you plan to use this model along with the mentionned evaluation method, an easy fix for this is to replace in torchkge/models/translation.py the line 765:

    candidates = candidates.expand(b_size, self.n_rel, self.rel_emb_dim)
by

	candidates = candidates.expand(b_size, self.n_rel, self.emb_dim)


## Usage

To use the code, follow these steps:

Clone the repository or download the code files to your local machine.

Run the script using the following command:

    python main.py [arguments]

Replace main.py with the name of the script file containing the code.

Specify the required command-line arguments to customize the behavior of the script. The available arguments are:

    --keywords: Specify multiple keywords to generate the dataset(optional). 
    Currently available: disease molecular_entity phenotype not-phenotype interaction disease_plus_ortho expression_value expression_pattern lifestage-ontology disease-ontology phenotype-ontology go-ontology go-annotation
    --method: Name of the method (required). One of TransE, TorusE, ComplEx, ConvKB.
    --dataset: Used to specify local datasets (optional). See 'Using a local dataset' for more information.
    --query: A SPARQL query (optional). Used to retrieve data from a query instead of using keywords.
    --normalize_parameters: Whether to normalize entity embeddings (optional). Defaults to False.
    --train_classifier: Train a classifier on the generated embeddings (optional). Specify the names of the classifiers to use as n arguments. See the PyCaret documentation for all available classifiers.
    --save_model: Whether to save the model weights (optional). Defaults to False.
    --save_data: Whether to save the data split (optional). Defaults to False.
    --save_embeddings: Whether to save the embeddings as csv (optional). Defaults to False.
    --n_epochs: Number of epochs (optional). Defaults to 20.
    --batch_size: Batch size (optional). Defaults to 128.
    --lr: Learning rate (optional). Defaults to 0.0001.
    --weight_decay: Weight decay (optional). Defaults to 0.0001.
    --loss_fn: Loss function. One of margin, bce, logistic (optional). Defaults to margin.
    --ent_emb_dim: Size of entity embeddings (optional). Defaults to 50.
    --split_ratio: Train/test ratio (optional). Defaults to 0.8.
    --dissimilarity_type: Either L1 or L2, representing the type of dissimilarity measure to use (optional). Defaults to L1. When using torus, replace L1 and L2 with torus_L1 and torus_L2, respectively.
    --margin: Margin value (optional). Defaults to 1.
    --rel_emb_dim: Size of entity embeddings (optional). Defaults to 50.
    --n_filters: Number of filters (ConvKB) (optional). Defaults to 10.
    --init_transe: Whether to initialize ConvKB with transe embeddings (optional). Defaults to True.

Note: Arguments marked as (required) are mandatory and must be provided.

The script will start executing the main function main(). It performs the following steps:

    Retrieves or generates the dataset based on the specified dataset or query.
    Trains an embedding model using the selected method.
    Trains a binary classifier using the generated embeddings.

All logs are saved in the logs folder. Models are saved in the models folder. Embeddings are saved in  data/embeddings.

## Keywords
Specific SPARQL queries are available as a baseline. They can be added or removed in a modular way by specifying them as nargs for the '--keyword' argument.

    molecular_entity: All molecular entities, including genes, ncRNA, pseudogenes, etc.
    phenotype: All phenotypes.
    not-phenotype: All negatively associated phenotypes.
    interaction: All interactions between molecular entities.
    disease: All diseases.
    disease_plus_ortho: All diseases including diseases inferred by orthology. Do not duplicate with 'disease'.
    expression_value: All bulk RNAseq expression values. Not recommended unprocessed as this is a very large amount of data.
    expression_pattern: All developmental expression patterns.
    lifestage-ontology: All lifestage ontology terms.
    disease-ontology: All disease ontology terms.
    phenotype-ontology: All phenotype ontology terms.
    go-ontology: All Gene Ontology terms.
    go-annotation: All Gene Ontology annotations.


## Examples

Here are a few examples of how to use the code:

### Querying a SPARQL endpoint:

    python main.py --query "SPARQL query" --method "method_name"


Training an embedding model with default configuration:

    python main.py --method "TransE" --keywords molecular_entity phenotype not-phenotype disease_plus_ortho interaction expression_pattern lifestage-ontology disease-ontology phenotype-ontology

Note: Replace "main.py" with the actual name of the script file, and replace "SPARQL query," "method_name," and "dataset_name" with your own values.

### Using a local dataset:

	python main.py --dataset "local" --method "method_name"

This will use the dataset stored in the data folder. The dataset must be in the form of a space separated file with the following columns:

    head: The head of the triple.
    relation: The relation of the triple.
    tail: The tail of the triple.

No index or header is required.


## Additional Information

For more information on the code and its functionality, refer to the comments within the code file for now. A full documentation will be provided in the future.
