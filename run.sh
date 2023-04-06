#!/bin/bash

# No Q system
export WANDB_API_KEY=4e5748d6c6f3917c78cdc38a516a1bac776faf58
# methods=("TransE")
# for method in "${methods[@]}"; do
#     for epoch in 50; do
#         for batch_size in 3072 4092; do
#             python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method $method --dataset local_celegans --n_epochs $epoch --batch_size $batch_size --lr 0.0001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 500
#             wait
#         done
#     done
# done



python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method TransE --dataset local_celegans --n_epochs 50 --batch_size 256 --lr 0.0001 --normalize_parameters --loss_fn margin --ent_emb_dim 100 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 500 --save_model --save_data
wait
python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method TransE --dataset local_celegans --n_epochs 50 --batch_size 256 --lr 0.0001 --normalize_parameters --loss_fn margin --ent_emb_dim 200 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 500 --save_model --save_data
wait
python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method TransE --dataset local_celegans --n_epochs 50 --batch_size 256 --lr 0.0001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L2 --margin 1 --n_filters 500 --save_model --save_data
wait

python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method ComplEx --dataset local_celegans --n_epochs 20 --batch_size 3072 --lr 0.0001 --normalize_parameters --loss_fn margin --ent_emb_dim 100 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 500 --save_model --save_data
wait
python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method ComplEx --dataset local_celegans --n_epochs 20 --batch_size 3072 --lr 0.0001 --normalize_parameters --loss_fn margin --ent_emb_dim 200 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 500 --save_model --save_data
wait
python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method ComplEx --dataset local_celegans --n_epochs 20 --batch_size 3072 --lr 0.0001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L2 --margin 1 --n_filters 500 --save_model --save_data
wait

python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method TorusE --dataset local_celegans --n_epochs 20 --batch_size 2048 --lr 0.001 --normalize_parameters --loss_fn margin --ent_emb_dim 100 --split_ratio 0.8 --dissimilarity_type torus_L1 --margin 1 --n_filters 500 --save_model --save_data
wait
python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method TorusE --dataset local_celegans --n_epochs 20 --batch_size 2048 --lr 0.001 --normalize_parameters --loss_fn margin --ent_emb_dim 200 --split_ratio 0.8 --dissimilarity_type torus_L1 --margin 1 --n_filters 500 --save_model --save_data
wait
python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method TorusE --dataset local_celegans --n_epochs 20 --batch_size 2048 --lr 0.001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type torus_L2 --margin 1 --n_filters 500 --save_model --save_data
wait


python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method RESCAL --dataset local_celegans --n_epochs 100 --batch_size 64 --lr 0.001 --normalize_parameters --loss_fn margin --ent_emb_dim 100 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 500 --save_model --save_data
wait
python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method RESCAL --dataset local_celegans --n_epochs 100 --batch_size 64 --lr 0.001 --normalize_parameters --loss_fn margin --ent_emb_dim 200 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 500 --save_model --save_data
wait
python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method RESCAL --dataset local_celegans --n_epochs 100 --batch_size 64 --lr 0.001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L2 --margin 1 --n_filters 500 --save_model --save_data
wait
