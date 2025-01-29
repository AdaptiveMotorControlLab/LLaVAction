export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OMP_NUM_THREADS="8"
export NCCL_IB_DISABLE="0"
export NCCL_IB_GID_INDEX="3"
export NCCL_SOCKET_IFNAME="eth0"
export NCCL_DEBUG="INFO"
export ACCELERATE_CPU_AFFINITY="1"
export WANDB_API_KEY="4474ec79de023b0c3ffb43588ab6163264f875db"
# export HF_HOME=/media/data/haozhe/VFM/huggingface
export HF_HOME=/mnt/SV_storage/VFM/huggingface
# export PYTHONPATH=/media/data/haozhe/VFM/LLaVA-NeXT:$PYTHONPATH
export PYTHONPATH=/mnt/SV_storage/VFM/LLaVA-NeXT:$PYTHONPATH
export OPENAI_API_KEY=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

accelerate launch --num_processes=4 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-ov,conv_template=qwen_1_5,model_name=llava_qwen \
--tasks activitynetqa \
--batch_size 1 \
--log_samples \
--log_samples_suffix llava_onevision \
--output_path ./logs/ \
--verbosity=DEBUG > ./logs/llava_onevision_activitynetqa_1.log 2>&1