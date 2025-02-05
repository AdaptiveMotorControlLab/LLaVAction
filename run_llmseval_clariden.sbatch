#!/bin/bash
#SBATCH --job-name multinode
#SBATCH -A a-a03
#SBATCH --hint nomultithread    
#SBATCH --cpus-per-task 288
#SBATCH --no-requeue
#SBATCH --nodes 1                   # number of Nodes
#SBATCH --ntasks-per-node 1         # number of MP tasks. IMPORTANT: torchrun represents just 1 Slurm task
#SBATCH --gres gpu:4                # Number of GPUs
#SBATCH --time 23:00:00             # maximum execution time (DD-HH:MM:SS). Mandatory field in MN5
#SBATCH --output logs/R-%x.%j-lmmseval-dev_7b_4f_llavavideo_test_haozhe.out
#SBATCH --error logs/R-%x.%j-lmmseval-dev_7b_4f_llavavideo_test_haozhe.err

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
LAUNCHER="accelerate launch \
    --num_processes=$GPUS_PER_NODE \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

PYTHON_FILE="-m lmms_eval"


PYTHON_ARGS=" \
    --model llava_vid \
    --model_args pretrained=experiments/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_vid \
    --output_path ./logs/
    --verbosity=DEBUG \
    "

export CMD="$LAUNCHER $PYTHON_FILE $PYTHON_ARGS"
export HF_HOME=$SCRATCH/huggingface
export OMP_NUM_THREADS="8"
export ACCELERATE_CPU_AFFINITY="1"
export WANDB_API_KEY="65aeda82a75f1eed29c8e9250b175fcc73dca0d7"
export OPENAI_API_KEY=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

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
    source /iopsstor/scratch/cscs/hqi/VFM/llava_dependency/llava-venv/bin/activate
    $CMD"

echo "END TIME: $(date)"