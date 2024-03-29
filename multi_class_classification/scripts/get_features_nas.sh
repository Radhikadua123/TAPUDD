#!/usr/bin/env bash
OUT_DATA=$1
MODEL=BiT-S-R101x1

adjust_type=gaussian_noise
CUDA_VISIBLE_DEVICES=3 python get_penultimate_features_NAS.py \
--name get_features_${MODEL} \
--model ${MODEL} \
--model_path /home/radhika/mos1/checkpoints/finetune/finetune_flat_softmax_${MODEL}/bit.pth.tar \
--logdir /home/radhika/mos1/checkpoints/ \
--datadir /home/data_storage/imagenet/v12 \
--train_list data_lists/imagenet2012_train_list.txt \
--val_list data_lists/imagenet2012_val_list.txt \
--batch 512 \
--in_datadir /home/data_storage/imagenet/v12/val \
--out_datadir /home/data_storage/ood_datasets/data/ood_data/${OUT_DATA} \
--adjust_type ${adjust_type}