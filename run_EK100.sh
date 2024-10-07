python3 action/dataset.py \
    --root /media/data/haozhe/VFM/EK100/EK100_320p_15sec_30fps_libx264 \
    --train-metadata /media/data/haozhe/VFM/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv \
    --val-metadata /media/data/haozhe/VFM/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv \
    --llm_size 7b \
    --llava_num_frames 16 > kitchen_test.out 2>&1 \
    # --llava_checkpoint /data/epic_kitchen/EK100_test/checkpoint-8402						    
    # --action_predictions action/avaion_predictions.json \
    # --topk_predictions 10
