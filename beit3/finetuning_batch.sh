#!/bin/bash

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
 
# conda 환경 활성화.
source  ~/.bashrc
conda   activate   svqa2
 
# cuda 11.0 환경 구성.
ml purge
ml load cuda/11.0
 
# 활성화된 환경에서 코드 실행.

python -m torch.distributed.launch --nproc_per_node=4 run_beit3_finetuning.py \
        --model beit3_large_indomain_patch16_224 \
        --input_size 224 \
        --task vqacustom \
        --batch_size 128 \
        --num_workers 32 \
        --layer_decay 1.0 \
        --lr 2e-5 \
        --update_freq 1 \
        --randaug \
        --epochs 10 \
        --warmup_epochs 1 \
        --drop_path 0.15 \
        --sentencepiece_model /home/seanlee/class/SpeechVQAPipeline/models/smp.model \
        --finetune /home/seanlee/class/SpeechVQAPipeline/models/beit3_large_indomain_patch16_224.zip \
        --data_path /scratch/seanlee/ \
        --output_dir /home/seanlee/class/SpeechVQAPipeline/beit3/finetune_checkpoint_large_indomain \
        --log_dir /home/seanlee/class/SpeechVQAPipeline/beit3/log \
        --weight_decay 0.01 \
        --seed 42 \
        --save_ckpt_freq 1 \
        --task_head_lr_weight 20 \
        --opt_betas 0.9 0.98 \
        --checkpoint_activations \
        --resume /home/seanlee/class/SpeechVQAPipeline/beit3/finetune_checkpoint_large_indomain/checkpoint-1.pth

echo "###"
echo "### END DATE=$(date)"
