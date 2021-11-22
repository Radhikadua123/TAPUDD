#!/bin/bash

trap "kill 0" EXIT  # IMPORANT

echo "start"
#CUDA_VISIBLE_DEVICES=2 python ./get_ood_score.py --loss 'ce' --data 'oval' --metric 'mahalanobis';
#CUDA_VISIBLE_DEVICES=2 python ./get_ood_score.py --loss 'ours' --data 'oval' --metric 'mahalanobis';

#CUDA_VISIBLE_DEVICES=2 python ./get_ood_score.py --loss 'ce' --data 'moons' --metric 'mahalanobis';
#CUDA_VISIBLE_DEVICES=2 python ./get_ood_score.py --loss 'ours' --data 'moons' --metric 'mahalanobis';
#CUDA_VISIBLE_DEVICES=2 python ./get_ood_score.py --loss 'ours_var' --data 'moons' --metric 'mahalanobis';

#CUDA_VISIBLE_DEVICES=2 python ./get_ood_score.py --loss 'ce' --data 'circles' --metric 'mahalanobis';
#CUDA_VISIBLE_DEVICES=2 python ./get_ood_score.py --loss 'ours' --data 'circles' --metric 'mahalanobis';
#CUDA_VISIBLE_DEVICES=2 python ./get_ood_score.py --loss 'ours_var' --data 'circles' --metric 'mahalanobis';
#
#for loss in 'ce' 'ours'
#do
#  for metric in 'baseline' 'mahalanobis' 'odin'
#  do
#    CUDA_VISIBLE_DEVICES=2 python ./get_ood_score.py --loss $loss --data 'moons' --metric $metric;
#    CUDA_VISIBLE_DEVICES=2 python ./get_ood_score.py --loss $loss --data 'oval' --metric $metric;
#  done
#done
# for seed in 1118
# do
# for data in 'flowers'
# do
# #   CUDA_VISIBLE_DEVICES=2 python ./get_ood_score.py --loss 'ce' --data $data --metric 'baseline' --seed $seed;
# #   CUDA_VISIBLE_DEVICES=2 python ./get_ood_score.py --loss 'ce' --data $data --metric 'odin' --seed $seed;
# #   CUDA_VISIBLE_DEVICES=2 python ./get_ood_score.py --loss 'ce' --data $data --metric 'mahalanobis' --seed $seed;
# #   CUDA_VISIBLE_DEVICES=2 python ./get_ood_score.py --loss 'ce' --data $data --metric 'baseline' --flag_indist --seed $seed;
# #   CUDA_VISIBLE_DEVICES=2 python ./get_ood_score.py --loss 'ce' --data $data --metric 'odin' --flag_indist --seed $seed;
# #   CUDA_VISIBLE_DEVICES=2 python ./get_ood_score.py --loss 'ce' --data $data --metric 'mahalanobis' --flag_indist --seed $seed;
# #   CUDA_VISIBLE_DEVICES=2 python ./get_ood_score.py --loss 'ce' --data $data --metric 'mahalanobis' --seed $seed;
#     CUDA_VISIBLE_DEVICES=2 python ./get_ood_score.py --loss 'ours' --data $data --metric 'mahalanobis' --flag_indist --seed $seed;
#   CUDA_VISIBLE_DEVICES=2 python ./get_ood_score.py --loss 'ours' --data $data --metric 'mahalanobis' --seed $seed;
# #   CUDA_VISIBLE_DEVICES=2 python ./get_ood_score.py --loss 'ours' --data $data --metric 'mahalanobis' --flag_indist --seed $seed;
# done
# done



# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ce' --data 'multioval_10dim' --metric 'baseline' --seed 1113 --input_dim 10 --flag_indist;
# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ce' --data 'multioval_10dim' --metric 'odin' --seed 1113 --input_dim 10 --flag_indist;
# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ce' --data 'multioval_10dim' --metric 'mahalanobis' --seed 1113 --input_dim 10 --flag_indist;

# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ce' --data 'multicircle_10dim' --metric 'baseline' --seed 1113 --input_dim 10 --flag_indist;
# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ce' --data 'multicircle_10dim' --metric 'odin' --seed 1113 --input_dim 10 --flag_indist;
# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ce' --data 'multicircle_10dim' --metric 'mahalanobis' --seed 1113 --input_dim 10 --flag_indist;


# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ce' --data 'multioval_10dim' --metric 'baseline' --seed 1113 --input_dim 10;
# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ce' --data 'multioval_10dim' --metric 'odin' --seed 1113 --input_dim 10;
# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ce' --data 'multioval_10dim' --metric 'mahalanobis' --seed 1113 --input_dim 10;

# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ce' --data 'multicircle_10dim' --metric 'baseline' --seed 1113 --input_dim 10;
# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ce' --data 'multicircle_10dim' --metric 'odin' --seed 1113 --input_dim 10;
# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ce' --data 'multicircle_10dim' --metric 'mahalanobis' --seed 1113 --input_dim 10;



# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ours' --data 'multioval_10dim' --metric 'baseline' --seed 1118 --input_dim 10;
# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ours' --data 'multioval_10dim' --metric 'odin' --seed 1118 --input_dim 10;
# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ours' --data 'multioval_10dim' --metric 'mahalanobis' --seed 1113 --input_dim 10;

# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ours' --data 'multicircle_10dim' --metric 'baseline' --seed 1118 --input_dim 10;
# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ours' --data 'multicircle_10dim' --metric 'odin' --seed 1118 --input_dim 10;
# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ours' --data 'multicircle_10dim' --metric 'mahalanobis' --seed 1113 --input_dim 10;


# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ours' --data 'multioval_10dim' --metric 'baseline' --seed 1118 --input_dim 10 --flag_indist;
# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ours' --data 'multioval_10dim' --metric 'odin' --seed 1118 --input_dim 10 --flag_indist;
# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ours' --data 'multioval_10dim' --metric 'mahalanobis' --seed 1113 --input_dim 10 --flag_indist;

# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ours' --data 'multicircle_10dim' --metric 'baseline' --seed 1118 --input_dim 10 --flag_indist;
# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ours' --data 'multicircle_10dim' --metric 'odin' --seed 1118 --input_dim 10 --flag_indist;
# CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ours' --data 'multicircle_10dim' --metric 'mahalanobis' --seed 1113 --input_dim 10 --flag_indist;

CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ce' --data 'ellipse_binary' --metric 'baseline' --seed 1113;
CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ce' --data 'ellipse_binary' --metric 'odin' --seed 1113;
CUDA_VISIBLE_DEVICES=5 python ./get_ood_score.py --loss 'ce' --data 'ellipse_binary' --metric 'mahalanobis' --seed 1113;