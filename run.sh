#!/bin/bash

methods=("TransE" "TorusE" "RESCAL" "ComplEx" "ConvKB")

for method in "${methods[@]}"; do
    for epoch in 50 100 200; do
        for batch_size in 64 128 256; do
            command="bash -i -c 'conda activate aled && python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method $method --dataset celegans --n_epochs $epoch --batch_size $batch_size --lr 0.0001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 500'"
            echo $command | qsub -V -l nodes=cedre-22a:gpus=100 -q cedre-22a-gpu1
        done
    done
done


# methods=("TransE")
# for method in "${methods[@]}"; do
#     command="bash -i -c 'conda activate aled && python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method $method --dataset celedebug --n_epochs 1 --batch_size 256 --lr 0.0001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L1 --margin 1 --n_filters 500 --scalar_share 0.5'"
#     echo $command | qsub -V -l nodes=cedre-22a:gpus=100 -q cedre-22a-gpu1
# done
