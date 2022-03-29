#!/usr/bin/env bash

device=0
data_path="/home/data_storage/radhika/data/datasets/boneage_data_kaggle/"

path="./results_ce"
for inp_seed in 4 1955; do
    CUDA_VISIBLE_DEVICES=${device} python get_features.py --seed ${inp_seed} --flag_adjust --variation 'bright'  --result_path $path --data_path $data_path
done