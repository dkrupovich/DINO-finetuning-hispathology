#!/bin/bash

#SBATCH -J train_dino

#SBATCH --output=slurm-%x.%j.out

#SBATCH -N 1

#SBATCH --ntasks-per-node=4

#SBATCH -t 6-12:00:00

#SBATCH --mem=40G

#SBATCH --mail-type=ALL
#SBATCH --mail-user=dkrupovich99@gmail.com

#SBATCH --partition=gpu

#SBATCH --gres=gpu:tesla:4
#SBATCH --exclude=falcon3

module load python/3.8.6

source mt_proj/bin/activate


python -m torch.distributed.launch --nproc_per_node=4 MT_project/dino/main_dino.py --arch vit_small \
                                                                                --patch_size 16 \
                                                                                --out_dim 65536 \
                                                                                --norm_last_layer True \
                                                                                --warmup_teacher_temp 0.04 \
                                                                                --teacher_temp 0.07 \
                                                                                --warmup_teacher_temp_epochs 5 \
                                                                                --use_fp16 True \
                                                                                --weight_decay 0.2 \
                                                                                --weight_decay_end 0.4 \
                                                                                --clip_grad 0 \
                                                                                --batch_size_per_gpu 64 \
                                                                                --epochs 30 \
                                                                                --freeze_last_layer 5 \
                                                                                --lr 5e-6 \
                                                                                --min_lr 1e-7 \
                                                                                --global_crops_scale 0.25 1.0 \
                                                                                --local_crops_number 10 \
                                                                                --local_crops_scale 0.05 0.25 \
                                                                                --seed 0 \
                                                                                --num_workers 4 \
                                                                                --optimizer adamw \
                                                                                --momentum_teacher 0.996 \
                                                                                --use_bn_in_head False \
                                                                                --drop_path_rate 0.1 \
                                                                                --saveckp_freq 5 \
                                                                                --data_path testis/train/ \
                                                                                --output_dir MT_project/train_dino