#!/usr/bin/env bash
device=2
data_path=$1
path="./results"

for inp_seed in 0 1 2 3 4 663 1458 1708 1955 7130; do
    CUDA_VISIBLE_DEVICES=${device} python tune_mahalanobis_hyperparameter.py \
    --name seed_${inp_seed} \
    --logdir results/tune_mahalanobis \
    --batch 8 \
    --data_path ${data_path} \
    --seed ${inp_seed} \
    --result_path ${path} 
done
