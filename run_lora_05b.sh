#!/bin/bash

# Export environment variables
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OMP_NUM_THREADS="8"
export NCCL_IB_DISABLE="0"
export NCCL_IB_GID_INDEX="3"
export NCCL_SOCKET_IFNAME="eth0"
export NCCL_DEBUG="INFO"
export ACCELERATE_CPU_AFFINITY="1"
# export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libffi.so.7"
export WANDB_API_KEY="65aeda82a75f1eed29c8e9250b175fcc73dca0d7"

# Run the command using torchrun
torchrun --nproc_per_node=4 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr=127.0.0.1 \
         --master_port=29500 \
         llava/train/train_mem.py \
         --deepspeed scripts/zero3.json \
         --model_name_or_path lmms-lab/llava-onevision-qwen2-0.5b-ov \
         --version qwen_1_5 \
         --data_path scripts/train/onevision.yaml \
         --image_folder /media/data/haozhe/VFM/onevision/llava_data/geo3k/ \
         --video_folder /media/data/haozhe/VFM/onevision/llava_video \
         --mm_tunable_parts mm_vision_tower,mm_mlp_adapter,mm_language_model \
         --mm_vision_tower_lr 2e-6 \
         --vision_tower google/siglip-so400m-patch14-384 \
         --mm_projector_type mlp2x_gelu \
         --mm_vision_select_layer -2 \
         --mm_use_im_start_end False \
         --mm_use_im_patch_token False \
         --group_by_modality_length True \
         --image_aspect_ratio anyres_max_9 \
         --image_grid_pinpoints "(1x1),...,(6x6)" \
         --mm_patch_merge_type spatial_unpad \
         --bf16 True \
         --run_name EK100_lora_05b_new \
         --output_dir experiments/EK100_lora_05b_new \
         --num_train_epochs 1 \
         --per_device_train_batch_size 1 \
         --per_device_eval_batch_size 4 \
         --gradient_accumulation_steps 2 \
         --evaluation_strategy no \
         --save_strategy steps \
         --save_steps 1000 \
         --save_total_limit 1 \
         --learning_rate 1e-4 \
         --weight_decay 0. \
         --warmup_ratio 0.03 \
         --lr_scheduler_type cosine \
         --logging_steps 1 \
         --tf32 True \
         --model_max_length 32768 \
         --gradient_checkpointing True \
         --dataloader_num_workers 4 \
         --lazy_preprocess True \
         --report_to wandb \
         --torch_compile True \
         --torch_compile_backend inductor \
         --dataloader_drop_last True \
         --frames_upbound 32 \
         --lora_enable True \
         --lora_r 128 \
         --lora_alpha 256 \
         --mm_projector_lr 2e-5 > train_kitchen_lora_0.5b_new.out 2>&1