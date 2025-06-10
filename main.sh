#!/bin/bash
trial_run=2
echo "Runing${trial_run} Training Script, trial_run${trial_run}"
dataset="SZU_R1"  # SZU_R1, SZU_R2 or Houston
epoch=30
seed=13
T=1000
image_size=32
log_dir=""
lr1=0.001 # learning rate for noise predictor
lr2=0.001 # learning rate for classifier
lr3=0.00002 # learning rate for GAN
betas=0.5,0.999
bs=16
feature_channels=1
dr=0.5
log_dir="./logs/run${trial_run}/lr1_${lr1}/lr2_${lr2}/lr3_${lr3}_bs${bs}"
python main.py --feature_channels $feature_channels --trial_run $trial_run --dataset $dataset --epoch $epoch --seed $seed --T $T --image_size $image_size --lr1 $lr1 --lr2 $lr2 --lr3 $lr3 --bs $bs --dr $dr --log_dir $log_dir
for lr2 in 0.0001 0.001 0.01 0.1;do
  for lr3 in 0.00001 0.0001 0.001 0.01;do
    log_dir="./logs/run${trial_run}/lr1_${lr1}/lr2_${lr2}/lr3_${lr3}_bs${bs}"
    python main.py --feature_channels $feature_channels --trial_run $trial_run --dataset $dataset --epoch $epoch --seed $seed --T $T --image_size $image_size --lr1 $lr1 --lr2 $lr2 --lr3 $lr3 --bs $bs --dr $dr --log_dir $log_dir
  done
done
# 遍历不同学习率和批量大小
#for lr1 in 0.001 0.01 0.1; do
#  for lr2 in 0.001 0.01 0.1; do
#    for lr3 in 0.001 0.01 0.1; do
#      for bs in 32 64 128; do
#        log_dir="./logs/run${trial_run}/lr1_${lr1}/lr2_${lr2}/lr3_${lr3}_bs${bs}"
#        CUDA_VISIBLE_DEVICES=0 python main.py --trial_run $trial_run --dataset $dataset --epoch $epoch --seed $seed --T $T --image_size $image_size --if_small_dataset $if_small_dataset --lr1 $lr1 --lr2 $lr2 --lr3 $lr3 --bs $bs --log_dir $log_dir
#      done
#    done
#  done
#done