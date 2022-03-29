#!/usr/bin/env bash

device=3
data_path="/home/data_storage/radhika/data/datasets/boneage_data_kaggle/"

path="./results_ce"
for inp_seed in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=${device} python main.py --run_mode 'train' --seed ${inp_seed} --result_path ${path} --data_path ${data_path} --epoch 100
    CUDA_VISIBLE_DEVICES=${device} python main.py --run_mode 'test' --seed ${inp_seed} --result_path ${path} --data_path ${data_path}
    CUDA_VISIBLE_DEVICES=${device} python main.py --run_mode 'test' --seed ${inp_seed}  --result_path ${path} --variation="bright" --data_path ${data_path}
done