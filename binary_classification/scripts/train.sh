#!/usr/bin/env bash

device=2
data_path=$1

path="./results"
for inp_seed in 0 1 2 3 4 663 1458 1708 1955 7130 ; do
    CUDA_VISIBLE_DEVICES=${device} python main.py --run_mode 'train' --seed ${inp_seed} --result_path ${path} --data_path ${data_path} --epoch 100
    CUDA_VISIBLE_DEVICES=${device} python main.py --run_mode 'test' --seed ${inp_seed} --result_path ${path} --data_path ${data_path} --epoch 100
    CUDA_VISIBLE_DEVICES=${device} python main.py --run_mode 'ood_test' --seed ${inp_seed}  --result_path ${path} --variation="bright" --data_path ${data_path}
done