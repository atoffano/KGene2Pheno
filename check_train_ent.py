
import torchkge
import pandas as pd

df = pd.read_csv('/home/antoine/gene_pheno_pred/local_celegans_go.txt', sep=' ', header=None, names=['from', 'rel', 'to'])
kg = torchkge.KnowledgeGraph(df)
kg_train, kg_val, kg_test = kg.split_kg(share=0.8, validation=True)

print(kg_train.rel2ix)
print(kg_val.rel2ix)

# pdtrain = kg_train.get_df()
# pdval = kg_val.get_df()
# pdtest = kg_test.get_df()
# pdkg = kg.get_df()

# print(pdtrain.applymap(lambda x: x == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#label').sum().sum())
# print(pdval.applymap(lambda x: x == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#label').sum().sum())
# print(pdtest.applymap(lambda x: x == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#label').sum().sum())
# print(pdkg.applymap(lambda x: x == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#label').sum().sum())


# print(pdtrain.applymap(lambda x: x == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type').sum().sum())
# print(pdval.applymap(lambda x: x == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type').sum().sum())
# print(pdtest.applymap(lambda x: x == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type').sum().sum())
# print(pdkg.applymap(lambda x: x == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type').sum().sum())

