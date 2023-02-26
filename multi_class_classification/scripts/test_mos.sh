#!/usr/bin/env bash

MODEL=BiT-S-R101x1
dataset_path=$1
ood_dataset_path=$2

for OUT_DATA in Places SUN iNaturalist dtd ; do
    CUDA_VISIBLE_DEVICES=7 python test_mos.py \
    --name test_${MODEL}_mos_${OUT_DATA} \
    --in_datadir ${dataset_path}/val \
    --out_datadir ${ood_dataset_path}/${OUT_DATA} \
    --model ${MODEL} \
    --model_path checkpoints/finetune/finetune_group_softmax_${MODEL}/bit.pth.tar \
    --logdir checkpoints/test_log \
    --group_config group_config/taxonomy_level0.npy
done 