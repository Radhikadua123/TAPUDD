#!/usr/bin/env bash
device=3
data_path="/home/data_storage/radhika/data/datasets/boneage_data_kaggle/"
path="./results_ce"

for inp_seed in 663 1458 1708 1955 7130; do
    CUDA_VISIBLE_DEVICES=${device} python tune_mahalanobis_hyperparameter.py \
    --name seed_${inp_seed} \
    --logdir /home/radhika/assign/binary_classification/results_ce/tune_mahalanobis \
    --batch 8 \
    --data_path ${data_path} \
    --seed ${inp_seed} \
    --result_path ${path} 
done
