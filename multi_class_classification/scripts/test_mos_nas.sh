#!/usr/bin/env bash

OUT_DATA=$1
MODEL=BiT-S-R101x1

adjust_type=bright
for adjust_scale in $(seq 0 0.1 1.9) $(seq 2.0 0.5 7.5); do
    CUDA_VISIBLE_DEVICES=3 python test_mos_NAS.py \
    --name NAS_DETECTION_${adjust_type}_test_${MODEL}_mos/${adjust_scale} \
    --in_datadir /home/data_storage/imagenet/v12/val \
    --out_datadir /home/data_storage/radhika/data/ood_data/${OUT_DATA} \
    --model ${MODEL} \
    --model_path /home/radhika/mos1/checkpoints/finetune/finetune_group_softmax_${MODEL}/bit.pth.tar \
    --logdir /home/radhika/mos1/checkpoints/test_log \
    --group_config group_config/taxonomy_level0.npy \
    --adjust_type ${adjust_type} \
    --adjust_scale ${adjust_scale}
done
