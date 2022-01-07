#!/usr/bin/env bash

METHOD=$1
OUT_DATA=$2
MODEL=BiT-S-R101x1
adjust_type=bright
for METHOD in ODIN ; do
    for adjust_scale in $(seq 0.0 0.1 1.9) $(seq 2.0 0.5 7.5); do
        CUDA_VISIBLE_DEVICES=3 python test_baselines_NAS.py \
        --name NAS_DETECTION_${adjust_type}_test_${MODEL}_${METHOD}/${adjust_scale} \
        --in_datadir /home/data_storage/imagenet/v12/val \
        --out_datadir /home/data_storage/radhika/data/ood_data/${OUT_DATA} \
        --model ${MODEL} \
        --model_path /home/radhika/mos1/checkpoints/finetune/finetune_flat_softmax_${MODEL}/bit.pth.tar \
        --batch 16 \
        --logdir /home/radhika/mos1/checkpoints/test_log \
        --score ${METHOD} \
        --mahalanobis_param_path /home/radhika/mos1/checkpoints/tune_mahalanobis/tune_mahalanobis_${MODEL} \
        --adjust_type ${adjust_type} \
        --adjust_scale ${adjust_scale}
    done
done