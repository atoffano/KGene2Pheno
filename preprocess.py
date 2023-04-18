from utils import split_dataset, load_dataset

rel2text = {'http://semanticscience.org/resource/SIO_000281': 'not associated with phenotype',
             'http://semanticscience.org/resource/SIO_000628': 'refers to',
               'http://semanticscience.org/resource/SIO_001279': 'associated with phenotype',
                 'http://www.semanticweb.org/needed-terms#001': 'refers to gene',
                   'http://www.semanticweb.org/needed-terms#002': 'refers to life stage',
                     'http://www.semanticweb.org/needed-terms#004': 'refers to expression pattern',
                       'http://www.semanticweb.org/needed-terms#009': 'refers to disease',
                         'http://www.w3.org/1999/02/22-rdf-syntax-ns#label': 'has for label',
                           'http://www.w3.org/1999/02/22-rdf-syntax-ns#type': 'is of type',
                             'http://www.w3.org/2000/01/rdf-schema#subClassOf': 'is a subclass of'}
entities = set()

with open('/home/antoine/gene_pheno_pred/local_celegans.txt', 'r') as dataset:
          for line in dataset.readlines():
              # replace second element in tsv by matching value in dict:
              h, r, t = line.strip().split(' ')
              entities.add(h)
              entities.add(t)

with open('/home/antoine/gene_pheno_pred/methods/Relphormer/dataset/celegans/entities.txt', 'a') as entity_txt:
    with open('/home/antoine/gene_pheno_pred/methods/Relphormer/dataset/celegans/entity2text.txt', 'a') as entity2text:
        for entity in entities:
            entity2text.write(f'{entity}\t{entity}\n')
            entity_txt.write(f'{entity}\n')

with open('/home/antoine/gene_pheno_pred/methods/Relphormer/dataset/celegans/relations.txt', 'a') as relations_txt:
  with open('/home/antoine/gene_pheno_pred/methods/Relphormer/dataset/celegans/relation2text.txt', 'a') as entity2text:
      for rel, rel_text in rel2text.items():
          entity2text.write(f'{rel}\t{rel_text}\n')
          relations_txt.write(f'{rel}\n')

dataset = '/home/antoine/gene_pheno_pred/local_celegans.txt'
split_dataset(dataset, split_ratio=0.8, dev_set=True)
kg_train, kg_val, kg_test = load_dataset(dataset)

kg_train.to_csv(f'/home/antoine/gene_pheno_pred/methods/Relphormer/dataset/celegans/train.tsv', sep='\t', header=False, index=False)
kg_test.to_csv(f'/home/antoine/gene_pheno_pred/methods/Relphormer/dataset/celegans/test.tsv', sep='\t', header=False, index=False)
kg_val.to_csv(f'/home/antoine/gene_pheno_pred/methods/Relphormer/dataset/celegans/dev.tsv', sep='\t', header=False, index=False)
