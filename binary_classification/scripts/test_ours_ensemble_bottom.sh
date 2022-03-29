#!/usr/bin/env bash

device=2
METHOD=GMM_FULL_Mahala_Ensemble_Bottom
adjust_type=bright
gmm_random_state=0
top=False

for input_seed in 0 1 2 3 4 663 1458 1708 1955 7130 ; do
    loader_index=1
    for adjust_scale in $(seq 0.0 0.1 1.9) $(seq 2.0 0.5 7.5); do
        CUDA_VISIBLE_DEVICES=${device} python test_ours_ensemble.py \
        --name NAS_DETECTION_${adjust_type}_test_${METHOD}/seed_${input_seed}/gmm_random_state_${gmm_random_state}/ \
        --logdir /home/radhika/assign/binary_classification/results_ce/logs/ood_scores \
        --gmm_random_state ${gmm_random_state} \
        --adjust_scale ${adjust_scale} \
        --adjust_type ${adjust_type} \
        --seed ${input_seed} \
        --top ${top}
        ((loader_index=loader_index+1))
    done
done