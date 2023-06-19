from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
import warnings
import os
from datetime import datetime as dt
import pandas as pd
from time import time
import glob

def timer_func(func):
    # This function shows the execution time of the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

@timer_func
def load_celegans(keywords, sep):
    print(f'{dt.now()} - Querying celegans SPARQL endpoint with the following queries : {keywords}.')
    queries = queries_from_features(keywords)
    query_db(queries, sep)
    return 'query_result.txt'

def load_by_query(query):
    query = add_prefixes(query)
    query_db([query])

def query_db(queries, sep):
    """Queries the database with a SPARQL query that returns a graph (ie uses a CONSTRUCT clause)."""
    # Set up the SPARQL endpoint
    sparql = SPARQLWrapper("http://cedre-14a.med.univ-rennes1.fr:3030/WS286_2023march08_RDF/sparql")
    warnings.filterwarnings("ignore")
    for query in tqdm(queries, desc="Querying SPARQL endpoint..."):

        # Set the query
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)

        # Execute the query and parse the response
        results = sparql.query()
        results = results.convert()
        # Store the constructed subgraph
        try:
            with open('query_result.txt', 'a') as f:
                for s, p, o in results:
                    if any(x.toPython() == '' for x in (s, p, o)): # Checks if the triple is complete
                        continue
                    f.write(f'{s}{sep}{p}{sep}{o}\n')       
        except:
            raise Exception("Check that the query output is a triple like ?s ?p ?o. Reminder: You must use a CONSTRUCT query")
    print(f'{dt.now()} - Query executed !')

def queries_from_features(keywords):
    file_paths = glob.glob("sparql_queries/*")
    file_names = [os.path.basename(file_path).replace('.txt', '') for file_path in file_paths]
    prefixes = open("sparql_queries/PREFIXES.txt", "r").read()

    ret = []
    for keyword in keywords:
        if keyword not in file_names:
            raise Exception("Unknown keyword: " + keyword)
        query = open("sparql_queries/" + keyword + ".txt", "r").read()
        ret.append(f"{prefixes}\n{query}")
    return ret


# Legacy code to load data in the format required by PhenoGeneRanker. Implementation incomplete.
# def load_pgr(keywords, sep):
#     feature_type = {
#         "pgr-gene-pheno" : "bipartite",
#         "pgr-gene-gene" : "bipartite",
#         "pgr-gene-disease" : "bipartite",
#         "pgr-pheno-pheno" : "bipartite",
#         "pgr-disease-disease" : "bipartite",
#     }

#     for keyword in keywords:
#         # Retrieve data files
#         load_celegans([keyword], sep)
#         with open('query_result.txt', 'r') as f:
#             lines = f.readlines()
#         with open(f'{keyword}.txt', 'a') as f:
#             f.write(f'from{sep}to{sep}weight\n')
#             for line in lines:
#                 parsed = line.split(sep)
#                 parsed = parsed[0] + sep + parsed[2].strip('\n') + sep + '1.000\n'
#                 f.write(parsed)
#         os.remove('query_result.txt')

#     # Create input file
#     with open('input.txt', 'a') as f:
#         f.write(f'type{sep}file_name\n')
#         for keyword in keywords:
#             f.write(f'{feature_type[keyword]}{sep}{keyword}.txt\n')
#     return 'input.txt'

if __name__ == "__main__":
    # Change directory to the current file path
    current_file_path = os.path.realpath(__file__)
    current_dir = os.path.dirname(current_file_path)
    os.chdir(current_dir)

    if os.path.exists("query_result.txt") == True:
        os.remove("query_result.txt")

    keywords = ['gene', 'phenotype', 'interaction', 'disease_plus_ortho',
     'disease-ontology', 'phenotype-ontology', 'expression_pattern',
      'lifestage-ontology'] # , 'go-ontology', 'go-annotation'
    load_celegans(keywords, sep=' ')
    
    print("All queries executed, results saved in query_result.txt")