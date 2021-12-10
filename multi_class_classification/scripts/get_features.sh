#!/usr/bin/env bash
OUT_DATA=$1
MODEL=BiT-S-R101x1

CUDA_VISIBLE_DEVICES=2,3 python get_penultimate_features.py \
--name get_features_${MODEL} \
--model ${MODEL} \
--model_path /home/radhika/mos1/checkpoints/finetune/finetune_flat_softmax_${MODEL}/bit.pth.tar \
--logdir /home/radhika/mos1/checkpoints/ \
--datadir /home/data_storage/imagenet/v12 \
--train_list data_lists/imagenet2012_train_list.txt \
--val_list data_lists/imagenet2012_val_list.txt \
--batch 32 \
--in_datadir /home/data_storage/imagenet/v12/val \
--out_datadir /home/data_storage/radhika/data/ood_data/${OUT_DATA} \