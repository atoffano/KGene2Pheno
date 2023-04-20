#!/bin/bash

export WANDB_API_KEY=4e5748d6c6f3917c78cdc38a516a1bac776faf58
JOBNAME='TransE'

methods=("TransE")
for method in "${methods[@]}"; do
    for dist in 'L1'; do
        for loss in "margin"; do
            for epoch in 10 20 50; do
                for batch_size in 256 512 1024 2048; do
                    for n_filters in 50; do
                        srun -A sre@v100 --job-name ${JOBNAME} --error ${JOBNAME}.err --output ${JOBNAME}.out --constraint v100-16g --gres gpu:1 --cpus-per-task 10 --qos qos_gpu-dev --hint nomultithread python -u /gpfswork/rech/sre/uki62ne/KGene2Pheno --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method $method --dataset local_celegans --n_epochs $epoch --batch_size $batch_size --lr 0.001 --normalize_parameters --loss_fn $loss --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type $dist --margin 1 --n_filters $n_filters --save_model --save_data --init_transe True
                        wait
                    done
                done
            done
        done
    done
done