#!/usr/bin/env bash

MODEL=BiT-S-R101x1

for OUT_DATA in Places SUN iNaturalist dtd ; do
    CUDA_VISIBLE_DEVICES=2 python test_mos.py \
    --name test_${MODEL}_mos_${OUT_DATA} \
    --in_datadir /home/data_storage/imagenet/v12/val \
    --out_datadir /home/data_storage/radhika/data/ood_data/${OUT_DATA} \
    --model ${MODEL} \
    --model_path /home/radhika/mos1/checkpoints/finetune/finetune_group_softmax_${MODEL}/bit.pth.tar \
    --logdir /home/radhika/mos1/checkpoints/test_log \
    --group_config group_config/taxonomy_level0.npy
done 