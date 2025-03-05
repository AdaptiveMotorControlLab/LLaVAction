pip install moviepy spacy==3.7.5 numpy==1.26.1 && python -m spacy download en_core_web_sm &&


# Create AVION-based MQA data. Note you can adjust n_options to 10, 15, 20 etc.
# You can optionally adjust action_representation to official_key, GT_random_narration

# python3 llava/action/generate_description.py \
#     --train_metadata /data/anonymous/epic-kitchens-100-annotations/EPIC_100_train.csv \
#     --out_folder /data/anonymous/EK100_inst_train/ \
#     --train_predictions /data/anonymous/AVION_PREDS/avion_pred_ids_train.json \
#     --gen_type avion_mc \
#     --action_representation GT_random_narration \
#     --n_options 5

# python3 llava/action/generate_description.py \
#     --train_metadata /data/anonymous/epic-kitchens-100-annotations/EPIC_100_train.csv \
#     --out_folder /data/anonymous/EK100_inst_train/ \
#     --train_predictions /data/anonymous/AVION_PREDS/avion_pred_ids_train.json \
#     --gen_type avion_mc \
#     --action_representation official_key \
#     --n_options 5    


# Create TIM-based MQA data. Note you can adjust n_options to 10, 15, 20 etc.
# You can optionally adjust action_representation to official_key, GT_random_narration

# python3 llava/action/generate_description.py \
#     --train_metadata /data/anonymous/epic-kitchens-100-annotations/EPIC_100_train.csv \
#     --out_folder /data/anonymous/EK100_inst_train/ \
#     --train_predictions /data/anonymous/TIM_PREDS/tim_pred_ids_train.json \
#     --gen_type tim_mc \
#     --action_representation GT_random_narration \
#     --n_options 5 


# python3 llava/action/generate_description.py \
#     --train_metadata /data/anonymous/epic-kitchens-100-annotations/EPIC_100_train.csv \
#     --out_folder /data/anonymous/EK100_inst_train/cross_validation \
#     --train_predictions /data/anonymous/TIM_PREDS/tim_pred_ids_train_cross.json \
#     --gen_type tim_mc \
#     --action_representation official_key \
#     --n_options 5 
   
# Create Random-based MQA data. Note you can adjust n_options to 10, 15, 20 etc.
# You can optionally adjust action_representation to official_key, GT_random_narration

# python3 llava/action/generate_description.py \
#     --train_metadata /data/anonymous/epic-kitchens-100-annotations/EPIC_100_train.csv \
#     --out_folder /data/anonymous/EK100_inst_train/ \
#     --gen_type random_mc \
#     --action_representation official_key \
#     --n_options 5

# python3 llava/action/generate_description.py \
#     --train_metadata /data/anonymous/epic-kitchens-100-annotations/EPIC_100_train.csv \
#     --out_folder /data/anonymous/EK100_inst_train/\
#     --gen_type random_mc \
#     --action_representation GT_random_narration \
#     --n_options 5    

# Create Direct prediction data. 
# You can optionally adjust action_representation to official_key, GT_random_narration

# python3 llava/action/generate_description.py \
#     --train_metadata /data/anonymous/epic-kitchens-100-annotations/EPIC_100_train.csv \
#     --out_folder /data/anonymous/EK100_inst_train/\
#     --gen_type direct_narration \
#     --action_representation GT_random_narration \

# python3 llava/action/generate_description.py \
#     --train_metadata /data/anonymous/epic-kitchens-100-annotations/EPIC_100_train.csv \
#     --out_folder /data/anonymous/EK100_inst_train/\
#     --gen_type direct_narration \
#     --action_representation official_key \

