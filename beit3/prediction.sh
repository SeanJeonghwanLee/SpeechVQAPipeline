export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
python -m torch.distributed.launch --nproc_per_node=6 run_beit3_finetuning.py \
        --model beit3_base_patch16_384 \
        --input_size 384 \
        --task vqacustom \
        --batch_size 72 \
        --sentencepiece_model /home/seanlee/class/SpeechVQAPipeline/models/smp.model \
        --finetune /home/seanlee/class/SpeechVQAPipeline/beit3/finetune_checkpoint/checkpoint-best.pth \
        --data_path /home/seanlee/class/SpeechVQAPipeline/ \
        --output_dir /home/seanlee/class/SpeechVQAPipeline/beit3/results \
        --dist_eval