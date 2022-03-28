#!/usr/bin/env bash

METHOD=GMM_FULL_Mahala
MODEL=BiT-S-R101x1
num_of_clusters_=32
num_samples_for_fitting_model=1281167

for ((num_samples=1281167; num_samples<=num_samples_for_fitting_model; num_samples += 200000)); do
    for clusters in 1 2 3 4 5 6 7 8 9 10 16 32; do
        for OUT_DATA in iNaturalist dtd Places SUN; do
            CUDA_VISIBLE_DEVICES=2,3 python test_ours.py \
            --name test_${MODEL}_${METHOD}_${OUT_DATA}/num_samples${num_samples}/cluster${clusters} \
            --in_datadir /home/data_storage/imagenet/v12/val \
            --out_datadir /home/data_storage/ood_datasets/data/ood_data/${OUT_DATA} \
            --datadir /home/data_storage/imagenet/v12 \
            --train_list data_lists/imagenet2012_train_list.txt \
            --val_list data_lists/imagenet2012_val_list.txt \
            --model ${MODEL} \
            --model_path /home/radhika/mos1/checkpoints/finetune/finetune_flat_softmax_${MODEL}/bit.pth.tar \
            --batch 512 \
            --logdir /home/radhika/mos1/checkpoints/test_log \
            --ood_feats_path /home/radhika/assign/MOS/features/ood_data/${OUT_DATA}/feats.npy \
            --num_of_clusters ${clusters} \
            --num_of_samples ${num_samples}
        done
    done
done