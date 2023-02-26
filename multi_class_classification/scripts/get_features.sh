#!/usr/bin/env bash
OUT_DATA=$1
dataset_path=$2
ood_dataset_path=$3
MODEL=BiT-S-R101x1

CUDA_VISIBLE_DEVICES=7 python get_penultimate_features.py \
--name get_features_${MODEL} \
--model ${MODEL} \
--model_path checkpoints/finetune/finetune_flat_softmax_${MODEL}/bit.pth.tar \
--logdir checkpoints/ \
--datadir ${dataset_path} \
--train_list data_lists/imagenet2012_train_list.txt \
--val_list data_lists/imagenet2012_val_list.txt \
--batch 32 \
--in_datadir ${dataset_path}/val \
--out_datadir ${ood_dataset_path}/${OUT_DATA} \
--out_data ${OUT_DATA} \