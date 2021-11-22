#!/bin/bash

trap "kill 0" EXIT  # IMPORANT

echo "start"

# CUDA_VISIBLE_DEVICES=2 python ./train.py --loss 'ce' --data 'flowers' --seed 1118;
# CUDA_VISIBLE_DEVICES=2 python ./train.py --loss 'ours' --data 'flowers' --flag_mu --flag_entropy --flag_var --seed 1118;
# CUDA_VISIBLE_DEVICES=2 python ./train.py --loss 'ce' --data 'circles' --seed 1118;
#CUDA_VISIBLE_DEVICES=2 python ./train.py --loss 'ce' --data 'moons';
# CUDA_VISIBLE_DEVICES=2 python ./train.py --loss 'ce' --data 'oval';

#CUDA_VISIBLE_DEVICES=2 python ./train.py --loss 'ours' --data 'circles' --flag_mu --flag_entropy;
#CUDA_VISIBLE_DEVICES=2 python ./train.py --loss 'ours' --data 'moons' --flag_mu --flag_entropy;
#CUDA_VISIBLE_DEVICES=2 python ./train.py --loss 'ours' --data 'oval' --flag_mu --flag_entropy;

#CUDA_VISIBLE_DEVICES=2 python ./train.py --loss 'ours_var' --data 'circles' --flag_mu --flag_var;
#CUDA_VISIBLE_DEVICES=2 python ./train.py --loss 'ours_var' --data 'moons' --flag_mu --flag_var;
#CUDA_VISIBLE_DEVICES=2 python ./train.py --loss 'ours_var' --data 'oval' --flag_mu --flag_var;

# CUDA_VISIBLE_DEVICES=5 python ./train.py --loss 'ce' --data 'multioval_10dim' --seed 1113 --input_dim 10;
# CUDA_VISIBLE_DEVICES=5 python ./train.py --loss 'ce' --data 'multicircle_10dim' --seed 1113 --input_dim 10;

# CUDA_VISIBLE_DEVICES=5 python ./train.py --loss 'ours' --data 'multioval_10dim' --seed 1113 --input_dim 10  --flag_mu --flag_entropy --flag_var;
# CUDA_VISIBLE_DEVICES=5 python ./train.py --loss 'ours' --data 'multicircle_10dim' --seed 1113 --input_dim 10  --flag_mu --flag_entropy --flag_var;
CUDA_VISIBLE_DEVICES=5 python ./train.py --loss 'ce' --data 'ellipse_binary' --seed 1113;
# CUDA_VISIBLE_DEVICES=5 python ./train.py --loss 'ce' --data 'multioval' --seed 1118;
# CUDA_VISIBLE_DEVICES=5 python ./train.py --loss 'ce' --data 'multicircle' --seed 1118;