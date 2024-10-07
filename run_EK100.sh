python3 action/dataset.py \
    --root /media/data/haozhe/VFM/onevision/llava_video/EK100 \
    --train-metadata /media/data/haozhe/VFM/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv \
    --val-metadata /media/data/haozhe/VFM/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv \
    --llm_size 0.5b \
    --llava_num_frames 16 \
    --llava_checkpoint experiments/EK100_test/checkpoint-8402 \
    --action_predictions action/avaion_predictions.json \
    --topk_predictions 10 > kitchen_test.out 2>&1
