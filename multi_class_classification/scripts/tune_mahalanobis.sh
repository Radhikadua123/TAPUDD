#!/usr/bin/env bash

MODEL=BiT-S-R101x1

CUDA_VISIBLE_DEVICES=2,3 python tune_mahalanobis_hyperparameter.py \
--name tune_mahalanobis_${MODEL} \
--model ${MODEL} \
--model_path /home/radhika/mos1/checkpoints/finetune/finetune_flat_softmax_${MODEL}/bit.pth.tar \
--logdir /home/radhika/mos1/checkpoints/tune_mahalanobis \
--datadir /home/data_storage/imagenet/v12 \
--train_list data_lists/imagenet2012_train_list.txt \
--val_list data_lists/imagenet2012_val_list.txt \
--batch 8 \
