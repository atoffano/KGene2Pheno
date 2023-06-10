
#!/bin/bash

# export WANDB_MODE=offline
export WANDB_API_KEY=4e5748d6c6f3917c78cdc38a516a1bac776faf58
JOBNAME='kgene'

for ((i=0; i<5; i++))
do
    python main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern lifestage-ontology --method ConvKB --dataset local_celegans --n_epochs 20 --batch_size 6138 --lr 0.0001 --normalize_parameters --loss_fn bce --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 10 --save_model --save_data --init_transe '/home/antoine/gene_pheno_pred/models/TransE_2023-05-04 17:19:26.570766.pt' 50 'L1' 
    # python main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern lifestage-ontology --method TorusE --dataset local_celegans --n_epochs 20 --batch_size 2048 --lr 0.0001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type torus_L1 --margin 1 --n_filters 10 --save_model --save_data --init_transe True
    # python main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern lifestage-ontology --method ComplEx --dataset local_celegans --n_epochs 20 --batch_size 3072 --lr 0.0001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 10 --save_model --save_data --init_transe True
    # python main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern lifestage-ontology --method TransE --dataset local_celegans --n_epochs 50 --batch_size 256 --lr 0.0001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 10 --save_model --save_data --init_transe True
done
