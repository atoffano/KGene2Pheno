# required for all methods
eval_epoch = 20  # do link prediction evaluation each 20 epochs
n_epochs = 20
patience = 40
batch_size = 64
lr = 0.0004
loss_fn = 'margin'  # one of 'margin', 'bce', 'logistic'.
margin = 0.5 # if loss_fn == 'margin'

ent_emb_dim = 128 # Dimension of the entities embedding
n_entities = None # Number of entities in the current data set
n_relations = None # Number of relations in the current data set

# TransR and TransD
rel_emb_dim = 128 # Dimension of the relations embedding

# TransE and TorusE
dissimilarity_type = 'L2' # Either 'L1' or 'L2', representing the type of dissimilarity measure to use

# ANALOGY
scalar_share = 0.5 # (float) – Share of the diagonal elements of the relation-specific matrices to be scalars. By default it is set to 0.5 according to the original paper.

# ConvKB
n_filters = 3 # (int) – Number of filters used for convolution.

