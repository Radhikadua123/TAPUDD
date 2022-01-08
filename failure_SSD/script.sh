#!/usr/bin/env bash

for inten in $(seq 0.0 0.1 1.9) $(seq 2.0 0.5 7.5); do 
    CUDA_VISIBLE_DEVICES=3 python -u eval_ssd.py \
    --arch resnet50 \
    --training-mode SimCLR \
    --dataset cifar10 \
    --exp-name ssd_cifar10_bf_${inten} \
    --normalize \
    --intensity ${inten} \
    --ckpt pretrained_model_cifar10/last.pth
done