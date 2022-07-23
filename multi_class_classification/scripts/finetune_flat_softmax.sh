#!/usr/bin/env bash

dataset_path=$1
MODEL=BiT-S-R101x1

CUDA_VISIBLE_DEVICES=0 python finetune.py \
--name finetune_flat_softmax_${MODEL} \
--model ${MODEL} \
--bit_pretrained_dir bit_pretrained_models \
--logdir checkpoints/finetune \
--dataset imagenet2012 \
--eval_every 200 \
--datadir ${dataset_path} \
--train_list data_lists/imagenet2012_train_list.txt \
--val_list data_lists/imagenet2012_val_list.txt \
--num_block_open 0 \
--finetune_type flat_softmax \
