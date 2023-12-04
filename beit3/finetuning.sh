export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 run_beit3_finetuning.py \
        --model beit3_base_patch16_384 \
        --input_size 384 \
        --task vqacustom \
        --batch_size 128 \
        --num_workers 4 \
        --layer_decay 1.0 \
        --lr 2e-5 \
        --update_freq 1 \
        --randaug \
        --epochs 10 \
        --warmup_epochs 1 \
        --drop_path 0.15 \
        --sentencepiece_model /home/seanlee/class/SpeechVQAPipeline/models/smp.model \
        --finetune /home/seanlee/class/SpeechVQAPipeline/models/beit3_base_patch16_224.zip \
        --data_path /home/seanlee/class/SpeechVQAPipeline/ \
        --output_dir /home/seanlee/class/SpeechVQAPipeline/beit3/finetune_checkpoint \
        --log_dir /home/seanlee/class/SpeechVQAPipeline/beit3/log \
        --weight_decay 0.01 \
        --seed 42 \
        --save_ckpt_freq 1 \
        --task_head_lr_weight 20 \
        --opt_betas 0.9 0.98 \
        --checkpoint_activations \