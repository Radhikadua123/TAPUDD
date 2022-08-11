#!/usr/bin/env bash

device=3
METHOD=GMM_FULL_Mahala
num_of_clusters_=32
adjust_type=bright
gmm_random_state=0

for input_seed in 0 1 2 3 4 663 1458 1708 1955 7130; do
    for clusters in 1 2 3 4 5 6 7 8 9 10 16 32; do
        loader_index=1
        for adjust_scale in $(seq 0.0 0.1 1.9) $(seq 2.0 0.5 7.5); do
            CUDA_VISIBLE_DEVICES=${device} python test_tap_mahalanobis.py \
            --name NAS_DETECTION_${adjust_type}_test_${METHOD}/seed_${input_seed}/gmm_random_state_${gmm_random_state}/cluster${clusters}/ \
            --logdir /home/radhika/assign/binary_classification/results_ce/logs/ood_scores \
            --ood_feats_path /home/radhika/assign/binary_classification/results_ce/seed_${input_seed}/penultimate_ftrs/ftrs_${adjust_type}_${loader_index}.npy \
            --id_feats_path /home/radhika/assign/binary_classification/results_ce/seed_${input_seed}/penultimate_ftrs/ftrs_${adjust_type}_0.npy \
            --train_feats_path /home/radhika/assign/binary_classification/results_ce/seed_${input_seed}/penultimate_ftrs/ftrs_${adjust_type}_train.npy \
            --num_of_clusters ${clusters} \
            --result_path /home/radhika/assign/binary_classification/results_ce \
            --clustering_model_path /home/radhika/assign/binary_classification/results_ce/clustering_models/seed_${input_seed}/gmm_random_state_${gmm_random_state} \
            --gmm_random_state ${gmm_random_state} \
            --adjust_scale ${adjust_scale} \
            --adjust_type ${adjust_type} \
            --seed ${input_seed}
            ((loader_index=loader_index+1))
            
        done
    done
done
