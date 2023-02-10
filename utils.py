from SPARQLWrapper import SPARQLWrapper, JSON

def load_celegans(keywords):
    queries = queries_from_features(keywords)
    query_db(queries)

def load_by_query(query):
    query = add_prefixes(query)
    query_db([query])
    

def query_db(queries):
    """Queries the database with a SPARQL query that returns a graph (ie uses a CONSTRUCT clause)."""
    # Set up the SPARQL endpoint
    sparql = SPARQLWrapper("http://cedre-14a.med.univ-rennes1.fr:3030/WS285_27sep2022_rdf/sparql")
    
    for query in queries:

        # Set the query
        print(query)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)

        # Execute the query and parse the response
        results = sparql.query()
        results = results.convert()
        # Store the constructed subgraph
        try:
            with open('query_result.txt', 'a') as f:
                for s, p, o in results:
                    f.write(f'{s} {p} {o}\n')
        except:
            raise Exception("Check that the query output is a triple like ?s ?p ?o")

def queries_from_features(keywords):
    features = {       
        "gene-gene" : 
            """
            CONSTRUCT {
                ?wbinter nt:001 ?gene1 .
                ?wbinter nt:001 ?gene2 .
                ?wbinter rdfs:type ?rel .
            }
            WHERE {
                ?wbinter nt:001 ?gene1 .
                ?wbinter nt:001 ?gene2 .
                ?wbinter rdfs:type ?rel .
                FILTER (?rel = "Physical")
                FILTER (?gene1 != ?gene2)
            }
            LIMIT 10 # REMOVE ONCE TESTING DONE
            """,

        "gene-diseases" : 
            """
            CONSTRUCT {
                ?wbdata nt:009 ?disease .
                ?wbdata ro:0002331 ?omim .
                ?wbdata sio:000558 ?human_ortholog .
            }
            WHERE {
                ?wbdata nt:001 ?gene .
                ?gene rdf:type ?type .
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
            ?gene rdfs:label ?glab .
            
            # Get all diseases associated with those genes.
            ?wbdata nt:009 ?disease . # refers to disease associated with celegans gene
            ?wbdata ro:0002331 ?omim . # refers to corresponding human equivalent of a celegans disease
            ?wbdata sio:000558 ?human_ortholog . # human ortholog associated with current gene
            FILTER (?omim != 	<https://www.omim.org/entry/>)
            ?disease rdfs:label ?lab .
            }
            """,
        
        "gene-phenotypes" :
            """
            CONSTRUCT {
                ?wbdata ?rel1 ?pheno .
                ?wbdata ?rel ?gene .} 
                WHERE {
                ?wbdata nt:001 ?gene .
                ?gene rdf:type ?type .
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
            ?gene rdfs:label ?glab .
            # Get all phenotypes associated with those genes.
            ?wbdata sio:001279 ?pheno .
            ?pheno rdfs:label ?lab .
            FILTER (?rel1 = sio:001279) .
            FILTER (?rel = nt:001) .
            } limit 100
            """,
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