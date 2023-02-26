#!/usr/bin/env bash

dataset_path=$1
ood_dataset_path=$2
METHOD=GMM_FULL_Mahala
MODEL=BiT-S-R101x1
num_of_clusters_=32
num_samples_for_fitting_model=1281167

for ((num_samples=1281167; num_samples<=num_samples_for_fitting_model; num_samples += 200000)); do
    for clusters in 1 2 3 4; do
        for OUT_DATA in iNaturalist dtd Places SUN; do
            CUDA_VISIBLE_DEVICES=7 python test_ours.py \
            --name test_${MODEL}_${METHOD}_${OUT_DATA}/num_samples${num_samples}/cluster${clusters} \
            --in_datadir ${dataset_path}/val \
            --out_datadir ${ood_dataset_path}/${OUT_DATA} \
            --datadir ${dataset_path} \
            --train_list data_lists/imagenet2012_train_list.txt \
            --val_list data_lists/imagenet2012_val_list.txt \
            --model ${MODEL} \
            --model_path checkpoints/finetune/finetune_flat_softmax_${MODEL}/bit.pth.tar \
            --batch 512 \
            --logdir checkpoints/test_log \
            --ood_feats_path checkpoints/features/ood_data/${OUT_DATA}/feats.npy \
            --num_of_clusters ${clusters} \
            --num_of_samples ${num_samples}
        done
    done
done