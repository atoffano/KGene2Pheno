from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
import warnings
import os
from datetime import datetime as dt
import pandas as pd


def load_celegans(keywords, sep):
    print(f'{dt.now()} - Querying celegans dataset withthe following features : {keywords}.')
    queries = queries_from_features(keywords)
    query_db(queries, sep)
    return 'query_result.txt'

def load_pgr(keywords, sep):
    feature_type = {
        "pgr-gene-pheno" : "bipartite",
        "pgr-gene-gene" : "gene",
        "pgr-gene-disease" : "bipartite",
        "pgr-pheno-pheno" : "phenotype",
        "pgr-disease-disease" : "phenotype",
    }

    for keyword in keywords:
        # Retrieve data files
        load_celegans([keyword], sep)
        with open('query_result.txt', 'r') as f:
            lines = f.readlines()
        with open(f'{keyword}.txt', 'a') as f:
            f.write(f'from{sep}to{sep}weight\n')
            for line in lines:
                parsed = line.split(sep)
                parsed = parsed[0] + sep + parsed[2].strip('\n') + sep + '1.000\n'
                f.write(parsed)
        os.remove('query_result.txt')

    # Create input file
    with open('input.txt', 'a') as f:
        f.write(f'type{sep}file_name\n')
        for keyword in keywords:
            f.write(f'{feature_type[keyword]}{sep}{keyword}.txt\n')
    return 'input.txt'

def split_dataset(dataset, split_ratio=0.8, validation=True):
    """Takes in a .tsv file of triples and splits it into a train, dev and test set."""
    import random

    # set the paths to the output train and test TSV files
    train_file = "train.txt"
    test_file = "test.txt"

    # open the input and output files
    with open(dataset, "r") as in_file, open(train_file, "w") as out_train_file, open(test_file, "w") as out_test_file:
        # read the lines from the input file
        lines = in_file.readlines()

        # shuffle the lines randomly
        random.shuffle(lines)

        # calculate the number of lines for the train and test files
        num_train_lines = int(len(lines) * split_ratio)
        num_test_lines = len(lines) - num_train_lines

        # write the lines to the train and test files
        out_train_file.writelines(lines[:num_train_lines])
        out_test_file.writelines(lines[num_train_lines:])
    
    if validation:
        # set the paths to the output train and test TSV files
        dev_file = "dev.txt"

        # open the input and output files
        with open(test_file, "r") as in_file:
            # read the lines from the input file
            lines = in_file.readlines()
        os.remove(test_file)
        
        with open(test_file, "w") as out_test_file, open(dev_file, "w") as out_dev_file:
            # calculate the number of lines for the train and test files
            num_dev_lines = len(lines) // 2
            num_test_lines = len(lines) - num_dev_lines

            # write the lines to the train and test files
            out_test_file.writelines(lines[:num_test_lines])
            out_dev_file.writelines(lines[num_dev_lines:])

def load_dataset(validation=True):
    df_train = pd.read_csv('train.txt', sep=' ', header=None, names=['from', 'rel', 'to'])
    df_test = pd.read_csv('test.txt', sep=' ', header=None, names=['from', 'rel', 'to'])

    if validation:
        df_val = pd.read_csv('dev.txt', sep=' ', header=None, names=['from', 'rel', 'to'])
    else:
        df_val = None

    return df_train, df_val, df_test

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
    features = {       
        "gene" : 
            """
            CONSTRUCT {
                ?geneid rdf:type ?type .
                ?geneid rdf:label ?lab .
            }
            WHERE {
                ?geneid rdf:type ?type .
                ?geneid rdfs:label ?lab .
                FILTER (
                    ?type = sio:000985 || #protein coding gene
                    ?type = sio:010035 || # gene
                    ?type = sio:000988 || # pseudogene
                    ?type = sio:001230 || # tRNA gene
                    ?type = sio:000790 || # non coding RNA gene (includes ncRNA, miRNA, linc RNA, piRNA, antisense lncRNA)
                    ?type = sio:001182 || # rRNA gene
                    ?type = sio:001227 || # scRNA gene
                    ?type = sio:001228 || # snRNA gene
                    ?type = sio:001229    # snoRNA gene
                )
            }
            """,

        "phenotype" :
           """
            CONSTRUCT {
                ?wbpheno nt:001 ?geneid .
                ?wbpheno ?rel ?pheno .
            }
            WHERE {
                ?wbpheno nt:001 ?geneid .
                ?wbpheno ?rel ?pheno .
                FILTER(?rel = sio:000281|| ?rel = sio:001279)
                ?wbpheno sio:000772 ?eco .
  				FILTER (REGEX(STR(?pheno), "^" + STR(wbpheno:))) # Exclude go: annotations
            }
            """,

        "interaction" :
           """
            CONSTRUCT {
                ?wbinter nt:001 ?geneid1 .
                ?wbinter nt:001 ?geneid2 .
                ?wbinter  sio:000628 ?interaction_type
            }
            WHERE {
                ?wbinter nt:001 ?geneid1 .
                ?wbinter nt:001 ?geneid2 .
                ?wbinter rdf:type ?rel .
 				?wbinter  sio:000628 ?interaction_type .
                FILTER (?geneid1 != ?geneid2)
            }
            """,

        "disease" :
           """
            CONSTRUCT {    
               ?wbdisease nt:001 ?geneid .
               ?wbdisease nt:009 ?doid .
            }
            WHERE {
              ?wbdisease nt:001 ?geneid .
              ?wbdisease nt:009 ?doid . # refers to disease associated with celegans gene
              FILTER NOT EXISTS{ ?wbdisease sio:000772 <http://purl.obolibrary.org/obo/ECO_0000201>. }  # without ortholog
            }
            """,

          "disease_plus_ortho" :
           """
            CONSTRUCT {    
               ?wbdisease nt:001 ?geneid .
               ?wbdisease nt:009 ?doid .
            }
            WHERE {
              ?wbdisease nt:001 ?geneid .
              ?wbdisease nt:009 ?doid . # refers to disease associated with celegans gene
            }
            """,

        "expression_value" :
           """
            CONSTRUCT {
                ?wbexpr_val nt:001 ?geneid .
                ?wbexpr_val sio:000300 ?expr_value .
                ?wbexpr_val nt:002 ?lifestage .                
            }
            WHERE {
                ?wbexpr_val nt:001 ?geneid .
                ?wbexpr_val sio:000300 ?expr_value .
                ?wbexpr_val nt:002 ?lifestage .
            }
            """,

        "expression_pattern" :
           """
            CONSTRUCT {                
                ?wbexpr_pat nt:001 ?geneid .
                ?wbexpr_pat nt:004 ?expr_id .
                ?wbexpr_pat nt:002 ?wblifestage .
                
            }
            WHERE {
                ?wbexpr_pat nt:001 ?geneid .
                ?wbexpr_pat nt:004 ?expr_id .
                ?wbexpr_pat nt:002 ?wblifestage .
                
            }
            """,

        "disease-ontology" :
            """
            CONSTRUCT {
                ?disease rdfs:subClassOf ?disease2 .
            }
            WHERE {
            ?wbdata nt:009 ?disease . # refers to disease associated with celegans gene
            ?disease rdfs:subClassOf+ ?disease2 .
            }
            """,
            
        "phenotype-ontology" :
            """
            CONSTRUCT {
                ?pheno rdfs:subClassOf ?pheno2 .
            }
            WHERE {
                ?wbdata sio:001279 ?pheno .
                ?pheno rdfs:subClassOf+ ?pheno2 .
            }
            """,

        "toy-example" :
            """
            CONSTRUCT {
                ?wbdata nt:001 ?gene .
                ?gene rdf:type ?type .
                ?wbdata sio:001279 ?allpheno . 
            }
            WHERE {
                ?wbdata nt:001 ?gene .
                ?gene rdf:type ?type .
                ?wbdata sio:001279 ?allpheno.
  				?allpheno rdfs:subClassOf* ?pheno
            FILTER ( ?pheno = <https://wormbase.org/species/all/phenotype/WBPhenotype:0000601>)
            FILTER (
                ?type = sio:000985 || #protein coding gene
                ?type = sio:010035 || # gene
                ?type = sio:000988 || # pseudogene
                ?type = sio:001230 || # tRNA gene
                ?type = sio:000790 || # non coding RNA gene (includes ncRNA, miRNA, linc RNA, piRNA, antisense lncRNA)
                ?type = sio:001182 || # rRNA gene
                ?type = sio:001227 || # scRNA gene
                ?type = sio:001228 || # snRNA gene
                ?type = sio:001229    # snoRNA gene
            )            
            }
            """
    }
    
    ret = []
    for keyword in keywords:
        if keyword not in features.keys():
            raise Exception("Unknown keyword: " + keyword)
        ret.append(add_prefixes(features[keyword]))
    return ret

def add_prefixes(query):
    prefixes = """
        PREFIX bfo: <http://purl.obolibrary.org/obo/BFO_>
        PREFIX dcterms: <http://purl.org/dc/terms/#>
        PREFIX doid: <https://disease-ontology.org/?id=DOID:>
        PREFIX eco: <http://purl.obolibrary.org/obo/ECO_>
        PREFIX pmid: <https://pubmed.ncbi.nlm.nih.gov/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX ro: <http://purl.obolibrary.org/obo/RO_>
        PREFIX sio: <http://semanticscience.org/resource/SIO_>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX ChEBI: <https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:>
        PREFIX EC: <https://enzyme.expasy.org/EC/>
        PREFIX InterPro: <https://www.ebi.ac.uk/interpro/search/text/>
        PREFIX go: <http://amigo.geneontology.org/amigo/term/GO:>
        PREFIX goref: <https://www.pombase.org/reference/GO_REF:> 
        PREFIX omim: <https://www.omim.org/entry/>
        PREFIX taxon: <http://purl.uniprot.org/taxonomy/>
        PREFIX UniPathway: <https://www.unipathway.org>
        PREFIX UniProtKB: <https://www.uniprot.org/uniprot/>
        PREFIX UniProtKB-KW: <https://www.uniprot.org/keywords/>
        PREFIX UniProtKB-SubCell: <https://www.uniprot.org/locations/>
        PREFIX UniRule: <https://www.uniprot.org/unirule/>
        PREFIX wbbt: <https://wormbase.org/species/all/anatomy_term/WBbt:>
        PREFIX wbdata: <https://wormbase.org/wbdata/>
        PREFIX wbexp: <https://wormbase.org/species/all/expr_pattern/Expr>
        PREFIX wbgene: <https://wormbase.org/species/c_elegans/gene/WBGene>
        PREFIX wbgtype: <https://wormbase.org/species/c_elegans/genotype/WBGenotype>
        PREFIX wbinter: <https://wormbase.org/wbinter/>
        PREFIX wbls: <https://wormbase.org/search/life_stage/>
        PREFIX wbperson: <https://wormbase.org/resources/person/WBPerson>
        PREFIX wbpheno: <https://wormbase.org/species/all/phenotype/WBPhenotype:>
        PREFIX wbref: <https://wormbase.org/resources/paper/WBPaper>
        PREFIX wbrnai: <https://wormbase.org/species/c_elegans/rnai/WBRNAi>
        PREFIX wbstrain: <https://wormbase.org/species/c_elegans/strain/WBStrain>
        PREFIX wbtransg: <https://wormbase.org/species/c_elegans/transgene/WBTransgene>
        PREFIX wbvar: <https://wormbase.org/species/c_elegans/variation/WBVar>
        PREFIX wbversion: <https://wormbase.org/about/wormbase_release_>
        PREFIX CGD: <http://www.candidagenome.org/cgi-bin/locus.pl?dbid=>
        PREFIX dictyBase: <http://dictybase.org/gene/>
        PREFIX FB: <https://flybase.org/reports/>
        PREFIX FLYBASE: <https://flybase.org/reports/>
        PREFIX HGNC: <https://www.genenames.org/data/gene-symbol-report/#!/hgnc_id/HGNC:>
        PREFIX MGI: <http://www.informatics.jax.org/marker/>
        PREFIX PANTHER: <http://pantherdb.org/>
        PREFIX PomBase: <https://www.pombase.org/gene/>
        PREFIX RGD: <https://rgd.mcw.edu/rgdweb/report/gene/main.html?id=>
        PREFIX RHEA: <https://www.rhea-db.org/rhea/>
        PREFIX SGD: <https://www.yeastgenome.org/locus/>
        PREFIX TAIR: <https://www.arabidopsis.org/servlets/TairObject?accession=locus:>
        PREFIX ZFIN: <https://www.zfin.org/>
        PREFIX nt: <http://www.semanticweb.org/needed-terms#>"""

    return f"{prefixes}\n{query}"