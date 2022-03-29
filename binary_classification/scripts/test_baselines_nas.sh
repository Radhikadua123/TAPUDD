#!/usr/bin/env bash

device=3
adjust_type=bright
data_path="/home/data_storage/radhika/data/datasets/boneage_data_kaggle/"
path="./results_ce"

for METHOD in Mahalanobis ; do
    for inp_seed in 0 1 2 3 4 663 1458 1708 1955 7130; do
        loader_index=1
        for adjust_scale in $(seq 0.0 0.1 1.9) $(seq 2.0 0.5 7.5); do
            CUDA_VISIBLE_DEVICES=${device} python test_baselines_NAS.py \
            --name NAS_DETECTION_${adjust_type}_test_${METHOD}/seed_${inp_seed}/ \
            --batch 256 \
            --logdir /home/radhika/assign/binary_classification/results_ce/logs/ood_scores \
            --score ${METHOD} \
            --adjust_type ${adjust_type} \
            --adjust_scale ${adjust_scale} \
            --loader_count ${loader_index} \
            --data_path ${data_path} \
            --seed ${inp_seed} \
            --mahalanobis_param_path /home/radhika/assign/binary_classification/results_ce/tune_mahalanobis/seed_${inp_seed} \
            --result_path ${path} 
            ((loader_index=loader_index+1))
        done
    done
done