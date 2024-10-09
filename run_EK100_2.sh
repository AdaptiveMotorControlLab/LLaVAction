python3 action/dataset.py \
    --root /media/data/haozhe/VFM/onevision/llava_video/EK100 \
    --train-metadata /media/data/haozhe/VFM/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv \
    --val-metadata /media/data/haozhe/VFM/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv \
    --llm_size 0.5b \
    --llava_num_frames 32 --clip-length 32 \
    --llava_checkpoint experiments/EK100_lora_05b_new \
    --action_predictions /media/data/haozhe/VFM/EK100/predictions.json \
    --topk_predictions 10 > EK100_lora_05b_new.out 2>&1

