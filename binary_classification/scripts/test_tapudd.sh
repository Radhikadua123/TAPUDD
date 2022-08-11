#!/usr/bin/env bash

device=3
METHOD=TAPUDD
adjust_type=bright
gmm_random_state=0

type=average
top=False
best=False
average=True
extremes=False
scaling=False

for input_seed in 0 1 2 3 4 663 1458 1708 1955 7130 ; do
    loader_index=1
    for adjust_scale in $(seq 0.0 0.1 1.9) $(seq 2.0 0.5 7.5); do
        CUDA_VISIBLE_DEVICES=${device} python test_tapudd.py \
        --name NAS_DETECTION_${adjust_type}_test_${METHOD}/${type}/seed_${input_seed}/ \
        --logdir results/logs_ood_scores \
        --gmm_random_state ${gmm_random_state} \
        --adjust_scale ${adjust_scale} \
        --adjust_type ${adjust_type} \
        --seed ${input_seed} \
        --top ${top} \
        --best ${best} \
        --average ${average} \
        --extremes ${extremes} \
        --scaling ${scaling}
        ((loader_index=loader_index+1))
    done
done


type=trimmed_average
top=False
best=False
average=False
extremes=False
scaling=False

for input_seed in 0 1 2 3 4 663 1458 1708 1955 7130 ; do
    loader_index=1
    for adjust_scale in $(seq 0.0 0.1 1.9) $(seq 2.0 0.5 7.5); do
        CUDA_VISIBLE_DEVICES=${device} python test_tapudd.py \
        --name NAS_DETECTION_${adjust_type}_test_${METHOD}/${type}/seed_${input_seed}/ \
        --logdir results/logs_ood_scores \
        --gmm_random_state ${gmm_random_state} \
        --adjust_scale ${adjust_scale} \
        --adjust_type ${adjust_type} \
        --seed ${input_seed} \
        --top ${top} \
        --best ${best} \
        --average ${average} \
        --extremes ${extremes} \
        --scaling ${scaling}
        ((loader_index=loader_index+1))
    done
done



type=seesaw
top=False
best=True
average=False
extremes=True
scaling=False

for input_seed in 0 1 2 3 4 663 1458 1708 1955 7130 ; do
    loader_index=1
    for adjust_scale in $(seq 0.0 0.1 1.9) $(seq 2.0 0.5 7.5); do
        CUDA_VISIBLE_DEVICES=${device} python test_tapudd.py \
        --name NAS_DETECTION_${adjust_type}_test_${METHOD}/${type}/seed_${input_seed}/ \
        --logdir results/logs_ood_scores \
        --gmm_random_state ${gmm_random_state} \
        --adjust_scale ${adjust_scale} \
        --adjust_type ${adjust_type} \
        --seed ${input_seed} \
        --top ${top} \
        --best ${best} \
        --average ${average} \
        --extremes ${extremes} \
        --scaling ${scaling}
        ((loader_index=loader_index+1))
    done
done




type=top
top=True
best=False
average=False
extremes=True
scaling=False

for input_seed in 0 1 2 3 4 663 1458 1708 1955 7130 ; do
    loader_index=1
    for adjust_scale in $(seq 0.0 0.1 1.9) $(seq 2.0 0.5 7.5); do
        CUDA_VISIBLE_DEVICES=${device} python test_tapudd.py \
        --name NAS_DETECTION_${adjust_type}_test_${METHOD}/${type}/seed_${input_seed}/ \
        --logdir results/logs_ood_scores \
        --gmm_random_state ${gmm_random_state} \
        --adjust_scale ${adjust_scale} \
        --adjust_type ${adjust_type} \
        --seed ${input_seed} \
        --top ${top} \
        --best ${best} \
        --average ${average} \
        --extremes ${extremes} \
        --scaling ${scaling}
        ((loader_index=loader_index+1))
    done
done




type=bottom
top=False
best=False
average=False
extremes=True
scaling=False

for input_seed in 0 1 2 3 4 663 1458 1708 1955 7130 ; do
    loader_index=1
    for adjust_scale in $(seq 0.0 0.1 1.9) $(seq 2.0 0.5 7.5); do
        CUDA_VISIBLE_DEVICES=${device} python test_tapudd.py \
        --name NAS_DETECTION_${adjust_type}_test_${METHOD}/${type}/seed_${input_seed}/ \
        --logdir results/logs_ood_scores \
        --gmm_random_state ${gmm_random_state} \
        --adjust_scale ${adjust_scale} \
        --adjust_type ${adjust_type} \
        --seed ${input_seed} \
        --top ${top} \
        --best ${best} \
        --average ${average} \
        --extremes ${extremes} \
        --scaling ${scaling}
        ((loader_index=loader_index+1))
    done
done
