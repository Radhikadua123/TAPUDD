#!/usr/bin/env bash

MODEL=BiT-S-R101x1

CUDA_VISIBLE_DEVICES=2,3 python finetune.py \
--name finetune_group_softmax_${MODEL} \
--model ${MODEL} \
--bit_pretrained_dir bit_pretrained_models \
--logdir /home/radhika/mos1/checkpoints/finetune \
--dataset imagenet2012 \
--eval_every 10 \
--datadir /home/data_storage/imagenet/v12 \
--train_list data_lists/imagenet2012_group_softmax_train_list.txt \
--val_list data_lists/imagenet2012_group_softmax_val_list.txt \
--num_block_open 0 \
--finetune_type group_softmax \
--group_config  group_config/taxonomy_level0.npy
