#!/usr/bin/env bash

device=7
MODEL=BiT-S-R101x1
METHOD=$1
dataset_path=$2
ood_dataset_path=$3

for OUT_DATA in Places SUN iNaturalist dtd ; do
    CUDA_VISIBLE_DEVICES=${device} python test_baselines.py \
    --name test_${MODEL}_${METHOD}_${OUT_DATA} \
    --in_datadir ${dataset_path}/val \
    --out_datadir ${ood_dataset_path}/${OUT_DATA} \
    --model ${MODEL} \
    --model_path checkpoints/finetune/finetune_flat_softmax_${MODEL}/bit.pth.tar \
    --batch 16 \
    --logdir checkpoints/test_log \
    --score ${METHOD} \
    --mahalanobis_param_path checkpoints/tune_mahalanobis/tune_mahalanobis_${MODEL}

done