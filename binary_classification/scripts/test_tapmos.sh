#!/usr/bin/env bash

device=3
METHOD=TAPMOS
adjust_type=bright
gmm_random_state=0

for input_seed in 0 1 2 3 4 663 1458 1708 1955 7130; do
    for clusters in 2; do
        loader_index=1
        for adjust_scale in $(seq 0.0 0.1 1.9) $(seq 2.0 0.5 7.5); do
            CUDA_VISIBLE_DEVICES=${device} python test_tapmos.py \
            --name NAS_DETECTION_${adjust_type}_test_${METHOD}/seed_${input_seed}/cluster${clusters}/ \
            --logdir results/logs_ood_scores \
            --ood_feats_path results/penultimate_ftrs/seed_${input_seed}/ftrs_${adjust_type}_${loader_index}.npy \
            --id_feats_path results/penultimate_ftrs/seed_${input_seed}/ftrs_${adjust_type}_0.npy \
            --train_feats_path results/penultimate_ftrs/seed_${input_seed}/ftrs_${adjust_type}_train.npy \
            --num_of_clusters ${clusters} \
            --result_path results \
            --clustering_model_path results/clustering_models/seed_${input_seed} \
            --gmm_random_state ${gmm_random_state} \
            --adjust_scale ${adjust_scale} \
            --adjust_type ${adjust_type} \
            --seed ${input_seed}
            ((loader_index=loader_index+1))
            
        done
    done
done