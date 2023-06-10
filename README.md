# Gene-Phenotype relation prediction
This repository stores the code required to derive insight from the wormbase database, by predicting links between genes and phenotypes in *C.Elegans* through various knowledge graph completion models.

## Prerequisites

Before using this code, make sure you have the following dependencies installed:
- python 3.10 or higher
- pandas
- pyyaml
- SPARQLWrapper
- torch
- torchkge
- sklearn
- pycaret

You can install the dependencies using the provided requirements.txt file:

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

Open a terminal or command prompt and navigate to the directory where the code files are located.

Run the script using the following command:

    python <script_name>.py [arguments]

Replace <script_name>.py with the name of the script file containing the code.

Specify the required command-line arguments to customize the behavior of the script. The available arguments are:

    --keywords: Specify multiple keywords (optional).
    --method: Name of the method (required).
    --dataset: Name of the dataset (required).
    --query: A SPARQL query (optional).
    --data_format: Format of the dataset (optional).
    --output: Directory to store the data (optional).

Note: Arguments marked as (required) are mandatory and must be provided.

The script will start executing the main function main(). It performs the following steps:

    Parses the command-line arguments provided.
    Changes the current working directory to the directory containing the script.
    Removes a file named "query_result.txt" if it exists.
    Retrieves or generates the dataset based on the specified dataset or query.
    Loads the configuration and initializes Weights & Biases (wandb) tracking.
    Trains an embedding model using the selected method.
    Removes the dataset file if it was downloaded from a SPARQL endpoint.
    Performs additional actions based on the configuration.

The script will print the start time and continue executing until completion.

Once the script finishes, you can analyze the results or perform any desired actions based on the configuration and the executed method.

## Examples

Here are a few examples of how to use the code:

### Querying a SPARQL endpoint:

    python script.py --query "SPARQL query" --method "method_name" --dataset "dataset_name"



### Using a local dataset:

python script.py --dataset "local_celegans" --method "method_name"

Training an embedding model with default configuration:

    python script.py --method "TransE" --dataset "celegans" --default_config

Note: Replace "script.py" with the actual name of the script file, and replace "SPARQL query," "method_name," and "dataset_name" with your own values.
Additional Information

For more information on the code and its functionality, refer to the comments within the code file and consult the documentation provided, if available.