#!/usr/bin/env bash

METHOD=GMM_FULL_Mahala_Ensemble
MODEL=BiT-S-R101x1
num_of_clusters_=32
num_samples_for_fitting_model=1281167
adjust_type=bright

type=top
top=True
best=False
average=False
extremes=True
scaling=False

for ((num_samples=1281167; num_samples<=num_samples_for_fitting_model; num_samples += 200000)); do
    for OUT_DATA in iNaturalist dtd Places SUN; do
        name_base_models=test_${MODEL}_GMM_FULL_Mahala_${OUT_DATA}
        CUDA_VISIBLE_DEVICES=3 python test_ours_ensemble.py \
        --name test_${MODEL}_${METHOD}/${type}/${OUT_DATA}/num_samples${num_samples}/ \
        --logdir checkpoints/test_log \
        --ood_scores_path checkpoints/test_log/${name_base_models}/num_samples${num_samples} \
        --id_scores_path checkpoints/test_log/${name_base_models}/num_samples${num_samples} \
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
    for OUT_DATA in iNaturalist dtd Places SUN; do
        name_base_models=test_${MODEL}_GMM_FULL_Mahala_${OUT_DATA}
        CUDA_VISIBLE_DEVICES=3 python test_ours_ensemble.py \
        --name test_${MODEL}_${METHOD}/${type}/${OUT_DATA}/num_samples${num_samples}/ \
        --logdir checkpoints/test_log \
        --ood_scores_path checkpoints/test_log/${name_base_models}/num_samples${num_samples} \
        --id_scores_path checkpoints/test_log/${name_base_models}/num_samples${num_samples} \
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
    for OUT_DATA in iNaturalist dtd Places SUN; do
        name_base_models=test_${MODEL}_GMM_FULL_Mahala_${OUT_DATA}
        CUDA_VISIBLE_DEVICES=3 python test_ours_ensemble.py \
        --name test_${MODEL}_${METHOD}/${type}/${OUT_DATA}/num_samples${num_samples}/ \
        --logdir checkpoints/test_log \
        --ood_scores_path checkpoints/test_log/${name_base_models}/num_samples${num_samples} \
        --id_scores_path checkpoints/test_log/${name_base_models}/num_samples${num_samples} \
        --top ${top} \
        --best ${best} \
        --average ${average} \
        --extremes ${extremes} \
        --scaling ${scaling}
    done
done


type=average
top=False
best=False
average=True
extremes=False
scaling=False

for ((num_samples=1281167; num_samples<=num_samples_for_fitting_model; num_samples += 200000)); do
    for OUT_DATA in iNaturalist dtd Places SUN; do
        name_base_models=test_${MODEL}_GMM_FULL_Mahala_${OUT_DATA}
        CUDA_VISIBLE_DEVICES=3 python test_ours_ensemble.py \
        --name test_${MODEL}_${METHOD}/${type}/${OUT_DATA}/num_samples${num_samples}/ \
        --logdir checkpoints/test_log \
        --ood_scores_path checkpoints/test_log/${name_base_models}/num_samples${num_samples} \
        --id_scores_path checkpoints/test_log/${name_base_models}/num_samples${num_samples} \
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
    for OUT_DATA in iNaturalist dtd Places SUN; do
        name_base_models=test_${MODEL}_GMM_FULL_Mahala_${OUT_DATA}
        CUDA_VISIBLE_DEVICES=3 python test_ours_ensemble.py \
        --name test_${MODEL}_${METHOD}/${type}/${OUT_DATA}/num_samples${num_samples}/ \
        --logdir checkpoints/test_log \
        --ood_scores_path checkpoints/test_log/${name_base_models}/num_samples${num_samples} \
        --id_scores_path checkpoints/test_log/${name_base_models}/num_samples${num_samples} \
        --top ${top} \
        --best ${best} \
        --average ${average} \
        --extremes ${extremes} \
        --scaling ${scaling}
    done
done