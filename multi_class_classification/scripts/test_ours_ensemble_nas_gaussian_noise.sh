#!/usr/bin/env bash

METHOD=GMM_FULL_Mahala_Ensemble
OUT_DATA_SET=$1
MODEL=BiT-S-R101x1
num_of_clusters_=32
num_samples_for_fitting_model=1281167
adjust_type=gaussian_noise

name_base_models=NAS_DETECTION_${adjust_type}_test_${MODEL}_GMM_FULL_Mahala

type=average
top=False
best=False
average=True
extremes=False
scaling=False

for ((num_samples=1281167; num_samples<=num_samples_for_fitting_model; num_samples += 200000)); do
    for OUT_DATA in $(seq 1 1 21); do
        CUDA_VISIBLE_DEVICES=2,3 python test_ours_ensemble_nas.py \
        --name NAS_DETECTION_${adjust_type}_test_${MODEL}_${METHOD}/${type}/${OUT_DATA}/num_samples${num_samples}/ \
        --logdir /home/radhika/mos1/checkpoints/test_log \
        --ood_scores_path /home/radhika/mos1/checkpoints/test_log/${name_base_models}/${OUT_DATA}/num_samples${num_samples} \
        --id_scores_path /home/radhika/mos1/checkpoints/test_log/${name_base_models}/1/num_samples${num_samples} \
        --top ${top} \
        --best ${best} \
        --average ${average} \
        --extremes ${extremes} \
        --scaling ${scaling}
    done
done


type=average_scaling
top=False
best=False
average=True
extremes=False
scaling=True

for ((num_samples=1281167; num_samples<=num_samples_for_fitting_model; num_samples += 200000)); do
    for OUT_DATA in $(seq 1 1 21); do
        CUDA_VISIBLE_DEVICES=2,3 python test_ours_ensemble_nas.py \
        --name NAS_DETECTION_${adjust_type}_test_${MODEL}_${METHOD}/${type}/${OUT_DATA}/num_samples${num_samples}/ \
        --logdir /home/radhika/mos1/checkpoints/test_log \
        --ood_scores_path /home/radhika/mos1/checkpoints/test_log/${name_base_models}/${OUT_DATA}/num_samples${num_samples} \
        --id_scores_path /home/radhika/mos1/checkpoints/test_log/${name_base_models}/1/num_samples${num_samples} \
        --top ${top} \
        --best ${best} \
        --average ${average} \
        --extremes ${extremes} \
        --scaling ${scaling}
    done
done


type=trimmed_average
top=False
best=False
average=False
extremes=False
scaling=False

for ((num_samples=1281167; num_samples<=num_samples_for_fitting_model; num_samples += 200000)); do
    for OUT_DATA in $(seq 1 1 21); do
        CUDA_VISIBLE_DEVICES=2,3 python test_ours_ensemble_nas.py \
        --name NAS_DETECTION_${adjust_type}_test_${MODEL}_${METHOD}/${type}/${OUT_DATA}/num_samples${num_samples}/ \
        --logdir /home/radhika/mos1/checkpoints/test_log \
        --ood_scores_path /home/radhika/mos1/checkpoints/test_log/${name_base_models}/${OUT_DATA}/num_samples${num_samples} \
        --id_scores_path /home/radhika/mos1/checkpoints/test_log/${name_base_models}/1/num_samples${num_samples} \
        --top ${top} \
        --best ${best} \
        --average ${average} \
        --extremes ${extremes} \
        --scaling ${scaling}
    done
done

type=trimmed_average_scaling
top=False
best=False
average=False
extremes=False
scaling=True

for ((num_samples=1281167; num_samples<=num_samples_for_fitting_model; num_samples += 200000)); do
    for OUT_DATA in $(seq 1 1 21); do
        CUDA_VISIBLE_DEVICES=2,3 python test_ours_ensemble_nas.py \
        --name NAS_DETECTION_${adjust_type}_test_${MODEL}_${METHOD}/${type}/${OUT_DATA}/num_samples${num_samples}/ \
        --logdir /home/radhika/mos1/checkpoints/test_log \
        --ood_scores_path /home/radhika/mos1/checkpoints/test_log/${name_base_models}/${OUT_DATA}/num_samples${num_samples} \
        --id_scores_path /home/radhika/mos1/checkpoints/test_log/${name_base_models}/1/num_samples${num_samples} \
        --top ${top} \
        --best ${best} \
        --average ${average} \
        --extremes ${extremes} \
        --scaling ${scaling}
    done
done


type=seesaw
top=False
best=True
average=False
extremes=True
scaling=False

for ((num_samples=1281167; num_samples<=num_samples_for_fitting_model; num_samples += 200000)); do
    for OUT_DATA in $(seq 1 1 21); do
        CUDA_VISIBLE_DEVICES=2,3 python test_ours_ensemble_nas.py \
        --name NAS_DETECTION_${adjust_type}_test_${MODEL}_${METHOD}/${type}/${OUT_DATA}/num_samples${num_samples}/ \
        --logdir /home/radhika/mos1/checkpoints/test_log \
        --ood_scores_path /home/radhika/mos1/checkpoints/test_log/${name_base_models}/${OUT_DATA}/num_samples${num_samples} \
        --id_scores_path /home/radhika/mos1/checkpoints/test_log/${name_base_models}/1/num_samples${num_samples} \
        --top ${top} \
        --best ${best} \
        --average ${average} \
        --extremes ${extremes} \
        --scaling ${scaling}
    done
done


type=seesaw_scaling
top=False
best=True
average=False
extremes=True
scaling=True

for ((num_samples=1281167; num_samples<=num_samples_for_fitting_model; num_samples += 200000)); do
    for OUT_DATA in $(seq 1 1 21); do
        CUDA_VISIBLE_DEVICES=2,3 python test_ours_ensemble_nas.py \
        --name NAS_DETECTION_${adjust_type}_test_${MODEL}_${METHOD}/${type}/${OUT_DATA}/num_samples${num_samples}/ \
        --logdir /home/radhika/mos1/checkpoints/test_log \
        --ood_scores_path /home/radhika/mos1/checkpoints/test_log/${name_base_models}/${OUT_DATA}/num_samples${num_samples} \
        --id_scores_path /home/radhika/mos1/checkpoints/test_log/${name_base_models}/1/num_samples${num_samples} \
        --top ${top} \
        --best ${best} \
        --average ${average} \
        --extremes ${extremes} \
        --scaling ${scaling}
    done
done


type=top
top=True
best=False
average=False
extremes=True
scaling=False

for ((num_samples=1281167; num_samples<=num_samples_for_fitting_model; num_samples += 200000)); do
    for OUT_DATA in $(seq 1 1 21); do
        CUDA_VISIBLE_DEVICES=2,3 python test_ours_ensemble_nas.py \
        --name NAS_DETECTION_${adjust_type}_test_${MODEL}_${METHOD}/${type}/${OUT_DATA}/num_samples${num_samples}/ \
        --logdir /home/radhika/mos1/checkpoints/test_log \
        --ood_scores_path /home/radhika/mos1/checkpoints/test_log/${name_base_models}/${OUT_DATA}/num_samples${num_samples} \
        --id_scores_path /home/radhika/mos1/checkpoints/test_log/${name_base_models}/1/num_samples${num_samples} \
        --top ${top} \
        --best ${best} \
        --average ${average} \
        --extremes ${extremes} \
        --scaling ${scaling}
    done
done


type=top_scaling
top=True
best=False
average=False
extremes=True
scaling=True

for ((num_samples=1281167; num_samples<=num_samples_for_fitting_model; num_samples += 200000)); do
    for OUT_DATA in $(seq 1 1 21); do
        CUDA_VISIBLE_DEVICES=2,3 python test_ours_ensemble_nas.py \
        --name NAS_DETECTION_${adjust_type}_test_${MODEL}_${METHOD}/${type}/${OUT_DATA}/num_samples${num_samples}/ \
        --logdir /home/radhika/mos1/checkpoints/test_log \
        --ood_scores_path /home/radhika/mos1/checkpoints/test_log/${name_base_models}/${OUT_DATA}/num_samples${num_samples} \
        --id_scores_path /home/radhika/mos1/checkpoints/test_log/${name_base_models}/1/num_samples${num_samples} \
        --top ${top} \
        --best ${best} \
        --average ${average} \
        --extremes ${extremes} \
        --scaling ${scaling}
    done
done


type=bottom
top=False
best=False
average=False
extremes=True
scaling=False

for ((num_samples=1281167; num_samples<=num_samples_for_fitting_model; num_samples += 200000)); do
    for OUT_DATA in $(seq 1 1 21); do
        CUDA_VISIBLE_DEVICES=2,3 python test_ours_ensemble_nas.py \
        --name NAS_DETECTION_${adjust_type}_test_${MODEL}_${METHOD}/${type}/${OUT_DATA}/num_samples${num_samples}/ \
        --logdir /home/radhika/mos1/checkpoints/test_log \
        --ood_scores_path /home/radhika/mos1/checkpoints/test_log/${name_base_models}/${OUT_DATA}/num_samples${num_samples} \
        --id_scores_path /home/radhika/mos1/checkpoints/test_log/${name_base_models}/1/num_samples${num_samples} \
        --top ${top} \
        --best ${best} \
        --average ${average} \
        --extremes ${extremes} \
        --scaling ${scaling}
    done
done


type=bottom_scaling
top=False
best=False
average=False
extremes=True
scaling=True

for ((num_samples=1281167; num_samples<=num_samples_for_fitting_model; num_samples += 200000)); do
    for OUT_DATA in $(seq 1 1 21); do
        CUDA_VISIBLE_DEVICES=2,3 python test_ours_ensemble_nas.py \
        --name NAS_DETECTION_${adjust_type}_test_${MODEL}_${METHOD}/${type}/${OUT_DATA}/num_samples${num_samples}/ \
        --logdir /home/radhika/mos1/checkpoints/test_log \
        --ood_scores_path /home/radhika/mos1/checkpoints/test_log/${name_base_models}/${OUT_DATA}/num_samples${num_samples} \
        --id_scores_path /home/radhika/mos1/checkpoints/test_log/${name_base_models}/1/num_samples${num_samples} \
        --top ${top} \
        --best ${best} \
        --average ${average} \
        --extremes ${extremes} \
        --scaling ${scaling}
    done
done
