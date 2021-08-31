#!/bin/bash

########################
### MAIN EXPERIMENTS ###
########################

#################################################################
## NON-IMAGE ONLY ##
# export CUDA_VISIBLE_DEVICES=0
nohup python train.py --seed 0 --model non-image-only --n_TTA 0 --use_class_weights --label_smoothing 0.1 \
    > non-image-only_ls0.1.out &

# export CUDA_VISIBLE_DEVICES=1
nohup python train.py --seed 1 --model non-image-only --n_TTA 0 --use_class_weights --label_smoothing 0.1 \
    > non-image-only_ls0.1_seed1.out &

# export CUDA_VISIBLE_DEVICES=2
nohup python train.py --seed 2 --model non-image-only --n_TTA 0 --use_class_weights --label_smoothing 0.1 \
    > non-image-only_ls0.1_seed2.out &

# export CUDA_VISIBLE_DEVICES=3
nohup python train.py --seed 3 --model non-image-only --n_TTA 0 --use_class_weights --label_smoothing 0.1 \
    > non-image-only_ls0.1_seed3.out &

# export CUDA_VISIBLE_DEVICES=2
nohup python train.py --seed 4 --model non-image-only --n_TTA 0 --use_class_weights --label_smoothing 0.1 \
    > non-image-only_ls0.1_seed4.out &
## END NON-IMAGE ONLY ##
#################################################################


#################################################################
## IMAGE ONLY ##
# export CUDA_VISIBLE_DEVICES=0
nohup python train.py --seed 0 --model image-only --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > image-only_aug_ls0.1.out &

# export CUDA_VISIBLE_DEVICES=1
nohup python train.py --seed 1 --model image-only --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > image-only_aug_ls0.1_seed1.out &

# export CUDA_VISIBLE_DEVICES=2
nohup python train.py --seed 2 --model image-only --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > image-only_aug_ls0.1_seed2.out &

# export CUDA_VISIBLE_DEVICES=3
nohup python train.py --seed 3 --model image-only --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > image-only_aug_ls0.1_seed3.out &

# export CUDA_VISIBLE_DEVICES=0
nohup python train.py --seed 4 --model image-only --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > image-only_aug_ls0.1_seed4.out &
## END IMAGE ONLY ##
#################################################################


#################################################################
## PROBABILITY FUSION ##
# export CUDA_VISIBLE_DEVICES=0
nohup python train.py --seed 0 --model probability-fusion --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > probability-fusion_aug_ls0.1.out &

# export CUDA_VISIBLE_DEVICES=1
nohup python train.py --seed 1 --model probability-fusion --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > probability-fusion_aug_ls0.1_seed1.out &

# export CUDA_VISIBLE_DEVICES=2
nohup python train.py --seed 2 --model probability-fusion --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > probability-fusion_aug_ls0.1_seed2.out &

# export CUDA_VISIBLE_DEVICES=3
nohup python train.py --seed 3 --model probability-fusion --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > probability-fusion_aug_ls0.1_seed3.out &

# export CUDA_VISIBLE_DEVICES=1
nohup python train.py --seed 4 --model probability-fusion --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > probability-fusion_aug_ls0.1_seed4.out &
## END PROBABILITY FUSION ##
#################################################################


#################################################################
## FEATURE FUSION ##
# export CUDA_VISIBLE_DEVICES=0
nohup python train.py --seed 0 --model feature-fusion --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > feature-fusion_aug_ls0.1.out &

# export CUDA_VISIBLE_DEVICES=1
nohup python train.py --seed 1 --model feature-fusion --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > feature-fusion_aug_ls0.1_seed1.out &

# export CUDA_VISIBLE_DEVICES=2
nohup python train.py --seed 2 --model feature-fusion --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > feature-fusion_aug_ls0.1_seed2.out &

# export CUDA_VISIBLE_DEVICES=0
nohup python train.py --seed 3 --model feature-fusion --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > feature-fusion_aug_ls0.1_seed3.out &

# export CUDA_VISIBLE_DEVICES=2
nohup python train.py --seed 4 --model feature-fusion --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > feature-fusion_aug_ls0.1_seed4.out &
## END FEATURE FUSION ##
#################################################################


#################################################################
## LEARNED FEATURE FUSION (CONCAT) ##
# export CUDA_VISIBLE_DEVICES=0
nohup python train.py --seed 0 --model learned-feature-fusion --fusion_mode concat --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion_aug_ls0.1.out &

# export CUDA_VISIBLE_DEVICES=1
nohup python train.py --seed 1 --model learned-feature-fusion --fusion_mode concat --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion_aug_ls0.1_seed1.out &

# export CUDA_VISIBLE_DEVICES=2
nohup python train.py --seed 2 --model learned-feature-fusion --fusion_mode concat --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion_aug_ls0.1_seed2.out &

# export CUDA_VISIBLE_DEVICES=1
nohup python train.py --seed 3 --model learned-feature-fusion --fusion_mode concat --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion_aug_ls0.1_seed3.out &

# export CUDA_VISIBLE_DEVICES=2
nohup python train.py --seed 4 --model learned-feature-fusion --fusion_mode concat --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion_aug_ls0.1_seed4.out &
## END LEARNED FEATURE FUSION (CONCAT) ##
#################################################################

############################
### END MAIN EXPERIMENTS ###
############################


#############################
### AUXILIARY EXPERIMENTS ###
#############################

#################################################################
## LEARNED FEATURE FUSION (MULTIPLY) ##
# export CUDA_VISIBLE_DEVICES=0
nohup python train.py --seed 0 --model learned-feature-fusion --fusion_mode multiply --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion-multiply_aug_ls0.1.out &

# export CUDA_VISIBLE_DEVICES=1
nohup python train.py --seed 1 --model learned-feature-fusion --fusion_mode multiply --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion-multiply_aug_ls0.1_seed1.out &

# export CUDA_VISIBLE_DEVICES=2
nohup python train.py --seed 2 --model learned-feature-fusion --fusion_mode multiply --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion-multipy_aug_ls0.1_seed2.out &

# export CUDA_VISIBLE_DEVICES=1
nohup python train.py --seed 3 --model learned-feature-fusion --fusion_mode multiply --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
   > learned-feature-fusion-multiply_aug_ls0.1_seed3.out &

# export CUDA_VISIBLE_DEVICES=2
nohup python train.py --seed 4 --model learned-feature-fusion --fusion_mode multiply --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
   > learned-feature-fusion-multiply_aug_ls0.1_seed4.out &
## END LEARNED FEATURE FUSION (MULTIPLY) ##
#################################################################


#################################################################
## LEARNED FEATURE FUSION (ADD) ##
# export CUDA_VISIBLE_DEVICES=0
nohup python train.py --seed 0 --model learned-feature-fusion --fusion_mode add --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion_aug_ls0.1.out &

# export CUDA_VISIBLE_DEVICES=1
nohup python train.py --seed 1 --model learned-feature-fusion --fusion_mode add --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion_aug_ls0.1_seed1.out &

# export CUDA_VISIBLE_DEVICES=2
nohup python train.py --seed 2 --model learned-feature-fusion --fusion_mode add --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion_aug_ls0.1_seed2.out &

# export CUDA_VISIBLE_DEVICES=1
nohup python train.py --seed 3 --model learned-feature-fusion --fusion_mode add --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion_aug_ls0.1_seed3.out &

# export CUDA_VISIBLE_DEVICES=2
nohup python train.py --seed 4 --model learned-feature-fusion --fusion_mode add --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion_aug_ls0.1_seed4.out &
## END LEARNED FEATURE FUSION (ADD) ##
#################################################################


#################################################################
## LEARNED FEATURE FUSION VARIANT (EACH SUBNETWORK OPTIMIZED INDEPENDENTLY) ##
# export CUDA_VISIBLE_DEVICES=0
nohup python train.py --seed 0 --model learned-feature-fusion --train_mode multiopt --fusion_mode concat --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion_aug_ls0.1_multiopt.out &

# export CUDA_VISIBLE_DEVICES=1
nohup python train.py --seed 1 --model learned-feature-fusion --train_mode multiopt --fusion_mode concat --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion_aug_ls0.1_multiopt_seed1.out &

# export CUDA_VISIBLE_DEVICES=0
nohup python train.py --seed 2 --model learned-feature-fusion --train_mode multiopt --fusion_mode concat --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion_aug_ls0.1_multiopt_seed2.out &

# export CUDA_VISIBLE_DEVICES=3
nohup python train.py --seed 3 --model learned-feature-fusion --train_mode multiopt --fusion_mode concat --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion_aug_ls0.1_multiopt_seed3.out &

# export CUDA_VISIBLE_DEVICES=1
nohup python train.py --seed 4 --model learned-feature-fusion --train_mode multiopt --fusion_mode concat --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion_aug_ls0.1_multiopt_seed4.out &
## END LEARNED FEATURE FUSION VARIANT (EACH SUBNETWORK OPTIMIZED INDEPENDENTLY) ##
#################################################################


#################################################################
## LEARNED FEATURE FUSION VARIANT (TRAINED ON SUM OF THREE LOSSES) ##
# export CUDA_VISIBLE_DEVICES=1
nohup python train.py --seed 0 --model learned-feature-fusion --train_mode multiloss --fusion_mode concat --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion_aug_ls0.1_multiloss.out &

# export CUDA_VISIBLE_DEVICES=1
nohup python train.py --seed 1 --model learned-feature-fusion --train_mode multiloss --fusion_mode concat --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion_aug_ls0.1_multiloss_seed1.out &

# export CUDA_VISIBLE_DEVICES=2
nohup python train.py --seed 2 --model learned-feature-fusion --train_mode multiloss --fusion_mode concat --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion_aug_ls0.1_multiloss_seed2.out &

# export CUDA_VISIBLE_DEVICES=2
nohup python train.py --seed 3 --model learned-feature-fusion --train_mode multiloss --fusion_mode concat --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion_aug_ls0.1_multiloss_seed3.out &

# export CUDA_VISIBLE_DEVICES=3
nohup python train.py --seed 4 --model learned-feature-fusion --train_mode multiloss --fusion_mode concat --n_TTA 5 --augment --use_class_weights --label_smoothing 0.1 \
    > learned-feature-fusion_aug_ls0.1_multiloss_seed4.out &
## LEARNED FEATURE FUSION (TRAINED ON SUM OF THREE LOSSES) ##
#################################################################

#################################
### END AUXILIARY EXPERIMENTS ###
#################################
