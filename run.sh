#!/bin/bash

# No Q system
export WANDB_API_KEY=4e5748d6c6f3917c78cdc38a516a1bac776faf58
export CUDA_VISIBLE_DEVICES=1
# methods=("TorusE" "RESCAL" "ComplEx" "ConvKB")
# for method in "${methods[@]}"; do
#     for epoch in 20 50 100; do
#         for batch_size in 256 128 64; do
#             python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method $method --dataset local_celegans --n_epochs $epoch --batch_size $batch_size --lr 0.0001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 500
#             wait
#         done
#     done
# done

# python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method ComplEx --dataset local_celegans --n_epochs 20 --batch_size 256 --lr 0.0001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 500 --save_model --save_data
# wait
# python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method ComplEx --dataset local_celegans --n_epochs 20 --batch_size 512 --lr 0.0001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 500 --save_model --save_data
# wait
# python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method ComplEx --dataset local_celegans --n_epochs 20 --batch_size 1024 --lr 0.0001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 500 --save_model --save_data
# wait
# python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method ComplEx --dataset local_celegans --n_epochs 20 --batch_size 256 --lr 0.001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 500 --save_model --save_data
# wait
# python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method ComplEx --dataset local_celegans --n_epochs 20 --batch_size 256 --lr 0.001 --normalize_parameters --loss_fn margin --ent_emb_dim 100 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 500 --save_model --save_data
# wait
# python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method ComplEx --dataset local_celegans --n_epochs 20 --batch_size 256 --lr 0.001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L1 --margin 0.5 --n_filters 500 --save_model --save_data
# wait
# python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method ComplEx --dataset local_celegans --n_epochs 20 --batch_size 256 --lr 0.001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L2 --margin 1 --n_filters 500 --save_model --save_data
# wait
# python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method ComplEx --dataset local_celegans --n_epochs 20 --batch_size 256 --lr 0.0001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.9 --dissimilarity_type L1 --margin 1 --n_filters 500 --save_model --save_data
# wait
# python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method ConvKB --dataset local_celegans --n_epochs 20 --batch_size 256 --lr 0.001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 50 --save_model --save_data
# wait
# python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method ConvKB --dataset local_celegans --n_epochs 20 --batch_size 256 --lr 0.01 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 50 --save_model --save_data
# wait
# python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method ConvKB --dataset local_celegans --n_epochs 20 --batch_size 512 --lr 0.001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 50 --save_model --save_data
# wait

ipython -c /home/antoine/gene_pheno_pred/auto_classifier.ipynb
wait

# # Q system, bug on gpu access
# methods=("TransE" "TorusE" "RESCAL" "ComplEx" "ConvKB")
# for method in "${methods[@]}"; do
#     for epoch in 50 100 200; do
#         for batch_size in 256 128 64; do
#             command="bash -i -c 'conda activate aled && python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method $method --dataset celegans --n_epochs $epoch --batch_size $batch_size --lr 0.0001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 500'"
#             echo $command | qsub -V -l nodes=cedre-22a:gpus=100 -q cedre-22a-gpu1
#         done
#     done
# done


# methods=("TransE")
# for method in "${methods[@]}"; do
#     command="bash -i -c 'conda activate aled && python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method $method --dataset celedebug --n_epochs 1 --batch_size 256 --lr 0.0001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 500 --scalar_share 0.5'"
#     echo $command | qsub -V -l nodes=cedre-22a:gpus=100 -q cedre-22a-gpu1
# done