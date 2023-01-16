import csv
from SPARQLWrapper import SPARQLWrapper, JSON

# Set up the SPARQL endpoint
sparql = SPARQLWrapper("http://cedre-14a.med.univ-rennes1.fr:3030/WS285_27sep2022_rdf/sparql")
from SPARQLWrapper import SPARQLWrapper, JSON

# Define the SPARQL query to select all triples in the graph
sparql.setQuery("""
    SELECT ?s ?p ?o
WHERE {
       ?s ?p ?o .
} LIMIT 10
""")

# Set the return format to JSON
sparql.setReturnFormat(JSON)

# Execute the query and parse the response
results = sparql.query().convert()

# Write the results to a CSV file
with open("triples.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["subject", "predicate", "object"])
    for result in results["results"]["bindings"]:
        writer.writerow([result["s"]["value"], result["p"]["value"], result["o"]["value"]])