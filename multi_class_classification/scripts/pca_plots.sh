#!/usr/bin/env bash

METHOD=GMM_FULL_Mahala
OUT_DATA_SET=$1
MODEL=BiT-S-R101x1
num_of_clusters_=10
num_samples_for_fitting_model=1281167

for OUT_DATA in iNaturalist; do
    for density_type in gaussian_kde; do
        CUDA_VISIBLE_DEVICES=2,3 python pca_visualization.py \
        --name test_${MODEL}_${METHOD}_${OUT_DATA} \
        --in_datadir /home/data_storage/imagenet/v12/val \
        --out_datadir /home/data_storage/radhika/data/ood_data/${OUT_DATA} \
        --datadir /home/data_storage/imagenet/v12 \
        --train_list data_lists/imagenet2012_train_list.txt \
        --val_list data_lists/imagenet2012_val_list.txt \
        --model ${MODEL} \
        --model_path /home/radhika/mos1/checkpoints/finetune/finetune_flat_softmax_${MODEL}/bit.pth.tar \
        --batch 512 \
        --logdir /home/radhika/mos1/checkpoints/test_log \
        --ood_feats_path /home/radhika/assign/MOS/features/ood_data/${OUT_DATA}/feats.npy \
        --dataset_name ${OUT_DATA} \
        --density True \
        --density_type ${density_type} \
        --separate False
    done
done