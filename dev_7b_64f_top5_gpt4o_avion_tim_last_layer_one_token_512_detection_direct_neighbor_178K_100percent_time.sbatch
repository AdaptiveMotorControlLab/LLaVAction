#!/bin/bash
#SBATCH --job-name multinode
#SBATCH -A a-a03
#SBATCH --hint nomultithread    
#SBATCH --cpus-per-task 288
#SBATCH --no-requeue
#SBATCH --nodes 32                   # number of Nodes
#SBATCH --ntasks-per-node 1         # number of MP tasks. IMPORTANT: torchrun represents just 1 Slurm task
#SBATCH --gres gpu:4                # Number of GPUs
#SBATCH --time 20:00:00             # maximum execution time (DD-HH:MM:SS). Mandatory field in MN5
#SBATCH --output temp/R-%x.%j_dev_7b_64f_top5_gpt4o_avion_tim_last_layer_one_token_detection_direct_neighbor_178K_100percent_time.out
#SBATCH --error temp/R-%x.%j_dev_7b_64f_top5_gpt4o_avion_tim_last_layer_one_token_detection_direct_neighbor_178K_100percent_time.err

mkdir -p logs

echo "START TIME: $(date)"

# auto-fail on any errors in this script
# set -eo pipefail

# logging script's variables/commands for future debug needs
set -x

######################
### Set enviroment ###
######################
# module purge
# module load singularity

GPUS_PER_NODE=4
echo "NODES: $SLURM_NNODES"
######################

######################
#### Set network #####
######################
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
######################

# note that we don't want to interpolate `\$SLURM_PROCID` till `srun` since otherwise all nodes will get
# 0 and the launcher will hang
#
# same goes for `\$(hostname -s|tr -dc '0-9')` - we want it to interpolate at `srun` time
LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank \$SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

PYTHON_FILE=llava/train/train_mem.py
PYTHON_ARGS=" \
    --deepspeed scripts/zero3.json \
    --model_name_or_path lmms-lab/LLaVA-Video-7B-Qwen2 \
    --version qwen_1_5 \
    --data_path scripts/train/avion_tim_top5_gpt4o_detection_direct_178K_100percent.yaml \
    --video_folder /iopsstor/scratch/anonymous_server/anonymous/VFM/onevision/llava_video/ \
    --mm_tunable_parts mm_vision_tower,mm_mlp_adapter,mm_language_model \
    --mm_vision_tower_lr 2e-6 \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints \"(1x1),...,(6x6)\" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name dev_7b_64f_top5_gpt4o_avion_tim_last_layer_one_token_detection_direct_neighbor_178K_100percent_time \
    --output_dir temp/dev_7b_64f_top5_gpt4o_avion_tim_last_layer_one_token_detection_direct_neighbor_178K_100percent_time \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy epoch \
    --eval_steps 1\
    --save_strategy steps \
    --save_steps 1000 \
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
    --frames_upbound 64 \
    --root /iopsstor/scratch/anonymous_server/anonymous/VFM/onevision/llava_video/EK100/ \
    --action_predictions /iopsstor/scratch/anonymous_server/anonymous/VFM/llava_data/TIM_PREDS/tim_pred_ids_val.json \
    --val_metadata /iopsstor/scratch/anonymous_server/anonymous/VFM/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv \
    --llava_num_frames 64 \
    --add_time_instruction True \
    --clip_length 64 \
    --topk_predictions 5 \
    --action_representation GT_random_narration \
    --vision_supervision one_token \
    --vision_token_training last_layer \
    --action_types 97,300,3806 \
    --learn_neighbor_actions prior \
    "

export CMD="$LAUNCHER $PYTHON_FILE $PYTHON_ARGS"
export HF_HOME=$SCRATCH/huggingface
export OMP_NUM_THREADS="8"
export ACCELERATE_CPU_AFFINITY="1"
export WANDB_API_KEY=""

echo $CMD

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
SRUN_ARGS=" \
    -ul \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --jobid $SLURM_JOB_ID \
    --wait 60 \
    --environment=llava-env \
    --container-workdir=$PWD \
    "
# SINGULARITY_CONTAINER=/path/to/singularity/.sif/file
# SINGULARITY_ARGS=" \
#     --bind /path/to/bind/folder \
#     $SINGULARITY_CONTAINER \
#     "  

# bash -c is needed for the delayed interpolation of env vars to work
srun $SRUN_ARGS numactl --membind=0-3 bash -c "
    source /iopsstor/scratch/anonymous_server/anonymous/VFM/llava_dependency/llava-venv/bin/activate
    $CMD"

echo "END TIME: $(date)"
