#!/bin/bash

JOBNAME='ConvKB'
#SBATCH --job-name=${JOBNAME} # nom du job
#SBATCH --output=${JOBNAME}%j.out # fichier de sortie (%j = job ID)
#SBATCH --error=${JOBNAME}%j.err # fichier d’erreur (%j = job ID)
#SBATCH --constraint=v100-16g # demander des GPU a 16 Go de RAM
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=1 # reserver n taches (ou processus)
#SBATCH --gres=gpu:1 # reserver n GPU
#SBATCH --cpus-per-task=10 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=01:59:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --qos=qos_gpu-dev # QoS
#SBATCH --hint=nomultithread # desactiver l’hyperthreading

export WANDB_API_KEY=4e5748d6c6f3917c78cdc38a516a1bac776faf58

methods=("ConvKB")
for method in "${methods[@]}"; do
    for dist in 'L1' 'L2'; do
        for loss in "logistic" "margin"; do
            for epoch in 5 10 20; do
                for batch_size in 10230 8184 6138 4092; do
                    for n_filters in 5 10 25 50; do
                            python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method $method --dataset local_celegans --n_epochs $epoch --batch_size $batch_size --lr 0.001 --normalize_parameters --loss_fn $loss --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type $dist --margin 1 --n_filters $n_filters --save_model --save_data --init_transe True
                            wait
                        done
                    done
                done
            done
        done
    done
done


# methods=("ConvKB")
# for method in "${methods[@]}"; do
#     for dist in 'L1' 'L2'; do
#         for loss in "logistic" "margin"; do
#             for epoch in 5 10 20; do
#                 for batch_size in 10230 8184 6138 4092; do
#                     for n_filters in 5 10 25 50; do
#                             set -x # activer l’echo des commandes
#                             srun python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method $method --dataset local_celegans --n_epochs $epoch --batch_size $batch_size --lr 0.001 --normalize_parameters --loss_fn $loss --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type $dist --margin 1 --n_filters $n_filters --save_model --save_data --init_transe True
#                             wait
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done



# python /home/antoine/gene_pheno_pred/main.py --keywords gene phenotype interaction disease_plus_ortho disease-ontology phenotype-ontology expression_pattern --method RESCAL --dataset local_celegans --n_epochs 100 --batch_size 64 --lr 0.001 --normalize_parameters --loss_fn margin --ent_emb_dim 50 --split_ratio 0.8 --dissimilarity_type L2 --margin 1 --n_filters 500 --save_model --save_data
# wait
