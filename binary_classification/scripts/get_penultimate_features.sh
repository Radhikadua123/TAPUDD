#!/usr/bin/env bash

device=2
data_path=$1

path="./results"
for inp_seed in 0 1 2 3 4 663 1458 1708 1955 7130; do
    CUDA_VISIBLE_DEVICES=${device} python get_features.py --seed ${inp_seed} --flag_adjust --variation 'bright'  --result_path $path --data_path $data_path
done