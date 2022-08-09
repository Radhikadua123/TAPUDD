#!/usr/bin/env bash

device=2
adjust_type=bright
method=$1
data_path=$2
path="./results"

for METHOD in ${method} ; do
    for inp_seed in 0 1 2 3 4 663 1458 1708 1955 7130; do
        loader_index=1
        for adjust_scale in $(seq 0.0 0.1 1.9) $(seq 2.0 0.5 7.5); do
            CUDA_VISIBLE_DEVICES=${device} python test_baselines_NAS.py \
            --name NAS_DETECTION_${adjust_type}_test_${METHOD}/seed_${inp_seed}/ \
            --batch 256 \
            --logdir results/logs_ood_scores \
            --score ${METHOD} \
            --adjust_type ${adjust_type} \
            --adjust_scale ${adjust_scale} \
            --loader_count ${loader_index} \
            --data_path ${data_path} \
            --seed ${inp_seed} \
            --mahalanobis_param_path results/tune_mahalanobis/seed_${inp_seed} \
            --result_path ${path} 
            ((loader_index=loader_index+1))
        done
    done
done
