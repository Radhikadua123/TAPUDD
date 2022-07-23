#!/usr/bin/env bash

dataset_path=$1
MODEL=BiT-S-R101x1

CUDA_VISIBLE_DEVICES=1 python finetune.py \
--name finetune_group_softmax_${MODEL} \
--model ${MODEL} \
--bit_pretrained_dir bit_pretrained_models \
--logdir checkpoints/finetune \
--dataset imagenet2012 \
--eval_every 200 \
--datadir ${dataset_path} \
--train_list data_lists/imagenet2012_group_softmax_train_list.txt \
--val_list data_lists/imagenet2012_group_softmax_val_list.txt \
--num_block_open 0 \
--finetune_type group_softmax \
--group_config  group_config/taxonomy_level0.npy
