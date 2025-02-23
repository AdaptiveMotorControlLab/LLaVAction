# Export environment variables
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export OMP_NUM_THREADS="8"
export NCCL_IB_DISABLE="0"
export NCCL_IB_GID_INDEX="3"
export NCCL_SOCKET_IFNAME="eth0"
export NCCL_DEBUG="INFO"
export ACCELERATE_CPU_AFFINITY="1"
export WANDB_API_KEY="4474ec79de023b0c3ffb43588ab6163264f875db"
export HF_HOME=/media/data/haozhe/VFM/huggingface
export PYTHONPATH=/media/data/haozhe/VFM/LLaVA-NeXT:$PYTHONPATH

# pip install moviepy spacy==3.7.5 numpy==1.26.1
# python -m spacy download en_core_web_sm

# torchrun --nproc_per_node=8 \
#          --nnodes=1 \
#          llava/train/train_mem.py \
#          --deepspeed scripts/zero3.json \
#          --model_name_or_path  lmms-lab/llava-onevision-qwen2-0.5b-ov\
#          --version qwen_1_5 \
#          --data_path scripts/train/tim.yaml \
#          --video_folder /media/data/haozhe/VFM/onevision/llava_video \
#          --mm_tunable_parts mm_vision_tower,mm_mlp_adapter,mm_language_model \
#          --mm_vision_tower_lr 2e-6 \
#          --vision_tower google/siglip-so400m-patch14-384 \
#          --mm_projector_type mlp2x_gelu \
#          --mm_vision_select_layer -2 \
#          --mm_use_im_start_end False \
#          --mm_use_im_patch_token False \
#          --group_by_modality_length True \
#          --image_aspect_ratio anyres_max_9 \
#          --image_grid_pinpoints "(1x1),...,(6x6)" \
#          --mm_patch_merge_type spatial_unpad \
#          --bf16 True \
#          --run_name dev_0.5b_4f_tim_base_haozhe \
#          --output_dir experiments/dev_0.5b_4f_tim_base_haozhe\
#          --num_train_epochs 1 \
#          --per_device_train_batch_size 8 \
#          --per_device_eval_batch_size 4 \
#          --gradient_accumulation_steps 2 \
#          --evaluation_strategy epoch \
#          --eval_steps 1\
#          --save_strategy steps \
#          --save_steps 2000 \
#          --learning_rate 1e-5 \
#          --weight_decay 0. \
#          --warmup_ratio 0.03 \
#          --lr_scheduler_type cosine \
#          --logging_steps 1 \
#          --tf32 True \
#          --model_max_length 32768 \
#          --gradient_checkpointing True \
#          --dataloader_num_workers 4 \
#          --lazy_preprocess True \
#          --report_to wandb \
#          --torch_compile True \
#          --torch_compile_backend inductor \
#          --dataloader_drop_last True \
#          --frames_upbound 4 \
#          --root /media/data/haozhe/VFM/onevision/llava_video/EK100 \
#          --action_predictions /media/data/haozhe/VFM/EK100/EK100_in_LLAVA/TIM/tim_pred_ids_val.json \
#          --val_metadata  /media/data/haozhe/VFM/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv \
#          --add_time_instruction False \
#          --llava_num_frames 4 \
#          --clip_length 4 \
#          --action_representation official_key \
#          --topk_predictions 5 > dev_0.5b_4f_tim_base_haozhe.out 2>&1

# torchrun --nproc_per_node=4 \
#          --nnodes=1 \
#          llava/train/train_mem.py \
#          --deepspeed scripts/zero3.json \
#          --model_name_or_path  lmms-lab/llava-onevision-qwen2-0.5b-ov\
#          --version qwen_1_5 \
#          --data_path scripts/train/tim_cross.yaml \
#          --video_folder /media/data/haozhe/VFM/onevision/llava_video \
#          --mm_tunable_parts mm_vision_tower,mm_mlp_adapter,mm_language_model \
#          --mm_vision_tower_lr 2e-6 \
#          --vision_tower google/siglip-so400m-patch14-384 \
#          --mm_projector_type mlp2x_gelu \
#          --mm_vision_select_layer -2 \
#          --mm_use_im_start_end False \
#          --mm_use_im_patch_token False \
#          --group_by_modality_length True \
#          --image_aspect_ratio anyres_max_9 \
#          --image_grid_pinpoints "(1x1),...,(6x6)" \
#          --mm_patch_merge_type spatial_unpad \
#          --bf16 True \
#          --run_name dev_0.5b_4f_tim_cross_haozhe \
#          --output_dir experiments/dev_0.5b_4f_tim_cross_haozhe\
#          --num_train_epochs 1 \
#          --per_device_train_batch_size 8 \
#          --per_device_eval_batch_size 4 \
#          --gradient_accumulation_steps 2 \
#          --evaluation_strategy epoch \
#          --eval_steps 1\
#          --save_strategy steps \
#          --save_steps 2000 \
#          --learning_rate 1e-5 \
#          --weight_decay 0. \
#          --warmup_ratio 0.03 \
#          --lr_scheduler_type cosine \
#          --logging_steps 1 \
#          --tf32 True \
#          --model_max_length 32768 \
#          --gradient_checkpointing True \
#          --dataloader_num_workers 4 \
#          --lazy_preprocess True \
#          --report_to wandb \
#          --torch_compile True \
#          --torch_compile_backend inductor \
#          --dataloader_drop_last True \
#          --frames_upbound 4 \
#          --root /media/data/haozhe/VFM/onevision/llava_video/EK100 \
#          --action_predictions /media/data/haozhe/VFM/EK100/EK100_in_LLAVA/TIM/tim_pred_ids_val.json \
#          --val_metadata  /media/data/haozhe/VFM/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv \
#          --add_time_instruction False \
#          --llava_num_frames 4 \
#          --clip_length 4 \
#          --action_representation official_key \
#          --topk_predictions 5 > dev_0.5b_4f_tim_cross_haozhe.out 2>&1


# torchrun --nproc_per_node=4 \
#          --nnodes=1 \
#          llava/train/train_mem.py \
#          --deepspeed scripts/zero3.json \
#          --model_name_or_path  lmms-lab/llava-onevision-qwen2-0.5b-ov\
#          --version qwen_1_5 \
#          --data_path scripts/train/tim_cross.yaml \
#          --video_folder /media/data/haozhe/VFM/onevision/llava_video \
#          --mm_tunable_parts mm_vision_tower,mm_mlp_adapter,mm_language_model \
#          --mm_vision_tower_lr 2e-6 \
#          --vision_tower google/siglip-so400m-patch14-384 \
#          --mm_projector_type mlp2x_gelu \
#          --mm_vision_select_layer -2 \
#          --mm_use_im_start_end False \
#          --mm_use_im_patch_token False \
#          --group_by_modality_length True \
#          --image_aspect_ratio anyres_max_9 \
#          --image_grid_pinpoints "(1x1),...,(6x6)" \
#          --mm_patch_merge_type spatial_unpad \
#          --bf16 True \
#          --run_name dev_0.5b_4f_tim_cross_3_haozhe \
#          --output_dir experiments/dev_0.5b_4f_tim_cross_3_haozhe\
#          --num_train_epochs 1 \
#          --per_device_train_batch_size 8 \
#          --per_device_eval_batch_size 4 \
#          --gradient_accumulation_steps 2 \
#          --evaluation_strategy epoch \
#          --eval_steps 1 \
#          --save_strategy steps \
#          --save_steps 2000 \
#          --learning_rate 1e-5 \
#          --weight_decay 0. \
#          --warmup_ratio 0.03 \
#          --lr_scheduler_type cosine \
#          --logging_steps 1 \
#          --tf32 True \
#          --model_max_length 32768 \
#          --gradient_checkpointing True \
#          --dataloader_num_workers 4 \
#          --lazy_preprocess True \
#          --report_to wandb \
#          --torch_compile True \
#          --torch_compile_backend inductor \
#          --dataloader_drop_last True \
#          --frames_upbound 4 \
#          --root /media/data/haozhe/VFM/onevision/llava_video/EK100 \
#          --action_predictions /media/data/haozhe/VFM/EK100/EK100_in_LLAVA/TIM/tim_pred_ids_val.json \
#          --val_metadata  /media/data/haozhe/VFM/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv \
#          --add_time_instruction False \
#          --llava_num_frames 4 \
#          --clip_length 4 \
#          --action_representation official_key \
#          --topk_predictions 5 > dev_0.5b_4f_tim_cross_3_haozhe.out 2>&1


torchrun --nproc_per_node=8 \
         --nnodes=1 \
         llava/train/train_mem.py \
         --deepspeed scripts/zero3.json \
         --model_name_or_path  lmms-lab/LLaVA-Video-7B-Qwen2\
         --version qwen_1_5 \
         --data_path scripts/train/llava_video.yaml \
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
         --run_name dev_7b_4f_llavavideo_test_haozhe \
         --output_dir experiments/dev_7b_4f_llavavideo_test_haozhe\
         --num_train_epochs 1 \
         --per_device_train_batch_size 1 \
         --per_device_eval_batch_size 1 \
         --gradient_accumulation_steps 2 \
         --evaluation_strategy no \
         --eval_steps 100 \
         --save_strategy steps \
         --save_steps 2000 \
         --learning_rate 1e-5 \
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
         --frames_upbound 4 \
         --root /media/data/haozhe/VFM/onevision/llava_video/EK100 \
         --action_predictions /media/data/haozhe/VFM/EK100/EK100_in_LLAVA/TIM/tim_pred_ids_val.json \
         --val_metadata  /media/data/haozhe/VFM/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv \
         --add_time_instruction False \
         --llava_num_frames 4 \
         --clip_length 4 \
         --action_representation official_key \
         --topk_predictions 5 > dev_7b_4f_llavavideo_test_haozhe.out 2>&1