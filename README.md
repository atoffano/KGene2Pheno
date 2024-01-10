# Gene-Phenotype relation prediction
This repository stores the code required to derive insight from the wormbase database, by predicting links between genes and phenotypes in *C.Elegans* through various knowledge graph completion models.

## Prerequisites

Make sure you have the following dependencies installed:
- python 3.10
- pandas
- SPARQLWrapper
- torch
- torchkge
- sklearn
- pycaret

You can install the dependencies using the following command line:

```bash
pip install torchkge pycaret torch SPARQLWrapper pandas numpy
```


Also, due to a bug in the TorchKGE package, relation prediction evaluation of the TorusE model does not work, as a parameter that is not defined in TorusE is called. If you plan to use this model along with the mentionned evaluation method, an easy fix for this is to replace in torchkge/models/translation.py the line 765:

    candidates = candidates.expand(b_size, self.n_rel, self.rel_emb_dim)
by

	candidates = candidates.expand(b_size, self.n_rel, self.emb_dim)

To use the code, follow these steps:

Clone the repository or download the code files to your local machine.

Test that everything works fine using:
    python main.py --method TransE --dataset toy-example --n_epochs 20 --batch_size 3072 --lr 0.0001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 10 --init_transe
    
Run the script using the following command:

    python main.py [arguments]

Replace main.py with the name of whichever script you want to run.

## Training a model

Models, whether embedding models or binary classifiers, can be trained using the main.py script.
Specify the required command-line arguments to customize the behavior of the script. 

Arguments:

    --keywords: Specify multiple keywords to generate the dataset(optional). 
    Currently available: disease molecular-entity phenotype not-phenotype interaction disease_plus_ortho expression_value expression_pattern lifestage-ontology disease-ontology phenotype-ontology go-ontology go-annotation
    --method: Name of the method (required). One of TransE, TorusE, ComplEx, ConvKB.
    --dataset: Used to specify local datasets (optional). See 'Using a local dataset' for more information.
    --query: A SPARQL query (optional). Used to retrieve data from a query instead of using keywords.
    --normalize_parameters: Whether to normalize entity embeddings (optional). Defaults to False.
    --train_classifier: Train a classifier on the generated embeddings (optional). Specify the names of the classifiers to use as n arguments. See the PyCaret documentation for all available classifiers.
    --save_model: Whether to save the model weights (optional). Defaults to False.
    --save_embeddings: Whether to save the embeddings as csv (optional). Defaults to False.
    --n_epochs: Number of epochs (optional). Defaults to 20.
    --batch_size: Batch size (optional). Defaults to 128.
    --lr: Learning rate (optional). Defaults to 0.0001.
    --weight_decay: Weight decay (optional). Defaults to 0.0001.
    --loss_fn: Loss function. One of margin, bce, logistic (optional). Defaults to margin.
    --ent_emb_dim: Size of entity embeddings (optional). Defaults to 50.
    --split_ratio: Train/test ratio (optional). Defaults to 0.8.
    --dissimilarity_type: Either L1 or L2, representing the type of dissimilarity measure to use (optional). Defaults to L1. When using torus, replace L1 and L2 with torus_L1 and torus_L2, respectively.
    --margin: Margin value (optional). Defaults to 1. Only used when loss_fn is margin.
    --rel_emb_dim: Size of entity embeddings (optional). Defaults to 50.
    --n_filters: Number of filters (ConvKB) (optional). Defaults to 10.
    --init_transe: Whether to initialize ConvKB with transe embeddings (optional, recommended). Takes the following nargs: [path to .pt TransE model] [TransE entity embedding size] [TransE dissimilarity_type].

Note: Arguments marked as (required) are mandatory and must be provided.
### Keywords
Specific SPARQL queries are available as a baseline. They can be added or removed in a modular way by specifying them as nargs for the '--keyword' argument.

    molecular-entity: All molecular entities, including genes, ncRNA, pseudogenes, etc.
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
### Example
    python main.py --keywords molecular-entity phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern lifestage-ontology --method ConvKB --dataset celegans --n_epochs 20 --batch_size 3072 --lr 0.0001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 10 --save_model --save_embeddings --init_transe True --train_classifier rf lr

The script will start executing the main function main(). It performs the following steps:

    Retrieves or generates the dataset based on the specified dataset or query.
    Trains an embedding model using the selected method.
    Trains a binary classifier using the generated embeddings.

All logs are saved in the logs folder. Models are saved in the models folder. Embeddings are saved in  data/embeddings.

data/raw/toy-example.txt can be used to test the training script. Simply replace the dataset argument with 'toy-example'. Do not use the --keyword argument. Under the hood, this works as a local dataset. Refer the the 'Using a local dataset' section for more information.


## Inference
Two scripts will perform link prediction between two nodes.
- predict.py will perform link prediction between two nodes using the underlyiong scoring function of the embedding model, and then optionally adds the predictions of a binary classifier on the existence of each link.
- predict_classif.py will perform link prediction specifically between gene and phenotype nodes using the predictions of a binary classifier on the existence of each link. It takes into account whether predicted links are known by inference on annotations (See code).
### predict.py
    python predict.py --model TransE '/home/gene_pheno_pred/models/TransE_2023-05-04 17:19:26.570766.pt' 50 L1 --graph '/home/KGene2Pheno/models/TransE_2023-05-04 17:19:26.570766_kg_train.csv' --file t.txt --output transe.txt --topk 1000 --classifier '/home/KGene2Pheno/binary_classif/rf/rf_model_2023-05-04 17:23:21.83576.pkl'

Arguments:

    --model: This argument is of type string (str). It expects multiple values to be passed in. The values vary depending on the model type. The first is always the name of the model, the second the path to the model and the third the embedding size. The third is optionnal, and is either the dssimilarity for TorusE and TransE, the number of filters for ConvKB or the scalar share for ANALOGY. 
    --filter_known_facts: This is a flag argument that doesn't require a value. When present, it removes the known facts from the predictions.
    --topk: This argument specifies the number of predictions to return.
    --graph: This argument expects the path to the model's training data file in CSV format (Required).
    --file: This argument expects the path to a CSV file containing queries. The queries can be in two formats: [head,relation,?] or [?,relation,tail]. Useful to chain multiple queries.
    --triple: This argument expects three values to be passed in, representing a triple. The triple can be in two formats: [head] [relation] [?] or [?] [relation] [tail]. This argument is optional.
    --b_size: This argument specifies the batch size.
    --classifier: Path to a classifier .pkl file. If this argument is provided, predictions of a binary classifier on the existence of each link will be added.
    --output: Path to save the prediction output file.

### predict_classif.py
    python predict_classif.py --model ComplEx '/home/KGene2Pheno/models/ComplEx_2023-06-26 13:00:36.058257.pt' 50 --graph '/home/KGene2Pheno/models/ComplEx_2023-06-26 12:53:57.459441_kg_train.csv' --phenotype https://wormbase.org/species/all/phenotype/WBPhenotype:0001588 --classifier '/home/KGene2Pheno/binary_classif/rf/rf_model_2023-06-26 13:00:36.058257.pkl' --output classif.txt

Arguments:

    --model: This argument is of type string (str). It expects multiple values to be passed in. The values vary depending on the model type. The first is always the name of the model, the second the path to the model and the third the embedding size. The third is optionnal, and is either the dssimilarity for TorusE and TransE, the number of filters for ConvKB or the scalar share for ANALOGY. 
    --filter_known_facts: This is a flag argument that doesn't require a value. When present, it removes the known facts from the predictions.
    --gene: Either the URI of the gene to predict links for or a ? if you're trying to predict the gene.
    --phenotype: Either the URI of the phenotype to predict links for or a ? if you're trying to predict the phenotype.
    --graph: This argument expects the path to the model's training data file in CSV format (Required).
    --b_size: This argument specifies the batch size.
    --classifier: Path to a classifier .pkl file. If this argument is provided, predictions of a binary classifier on the existence of each link will be added.
    --output: Path to save the prediction output file.
## Querying a SPARQL endpoint:

    python main.py --query "SPARQL query" 

You can directly interact with the SPARQL endpoint without using the keyword argument. The query must be enclosed in double quotes. Do not use the keyword argument if you are using the query argument.
--dataset must be celegans.
## Using a local dataset:

	python main.py --dataset ['path to file]

The dataset must be in the form of a space separated file with the following columns:

    head: The head of the triple.
    relation: The relation of the triple.
    tail: The tail of the triple.

No index or header is required.

## Additional Information

The pipeline used to train the embeddings models is based on [TorchKGE](https://torchkge.readthedocs.io/en/latest/). The pipeline used to train the binary classifiers is based on [PyCaret](https://pycaret.gitbook.io/docs/).

All logs are available in the logs folder.
All models are available in the models folder. Classification models are available in the binary_classif folder.
All embeddings are available in the data/embeddings folder.
All SPARQL queries are available in the queries folder.

For questions about the overall methodology, please refer to the MSc thesis resulting from this project which goes into detail on the methods used and the results obtained.
Results and figures presented in the thesis are available in the figures folder.

Wandb tracking has been used throughout the project. While tracking has been disabled, it can easily be added by uncommenting the relevant lines in the code.

For more information on the code and its functionality, refer to the comments within the code files.
