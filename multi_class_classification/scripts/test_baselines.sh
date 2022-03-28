#!/usr/bin/env bash

device=3
MODEL=BiT-S-R101x1
for METHOD in KL_Div; do
    for OUT_DATA in Places SUN iNaturalist dtd ; do
        CUDA_VISIBLE_DEVICES=${device} python test_baselines.py \
        --name test_${MODEL}_${METHOD}_${OUT_DATA} \
        --in_datadir /home/data_storage/imagenet/v12/val \
        --out_datadir /home/data_storage/radhika/data/ood_data/${OUT_DATA} \
        --model ${MODEL} \
        --model_path /home/radhika/mos1/checkpoints/finetune/finetune_flat_softmax_${MODEL}/bit.pth.tar \
        --batch 16 \
        --logdir /home/radhika/mos1/checkpoints/test_log \
        --score ${METHOD} \
        --mahalanobis_param_path /home/radhika/mos1/checkpoints/tune_mahalanobis/tune_mahalanobis_${MODEL}
    done
done