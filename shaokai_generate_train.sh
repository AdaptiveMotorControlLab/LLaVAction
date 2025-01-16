pip install moviepy spacy==3.7.5 numpy==1.26.1 && python -m spacy download en_core_web_sm &&
export PYTHONPATH=/data/shaokai/LLaVA-NeXT:$PYTHONPATH
export PYTHONPATH=/usr/local/lib/python3.10/site-packages/decord-0.6.0-py3.10-linux-x86_64.egg/:$PYTHONPATH



# python3 llava/action/generate_description.py \
#     --train_metadata /data/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv \
#     --out_folder /data/shaokai/EK100_inst_train/ \
#     --train_predictions /data/shaokai/AVION_PREDS/avion_pred_ids_train.json \
#     --gen_type avion_mc \
#     --action_representation GT_random_narration \
#     --n_options 5 \
#     --with_neighbors


# python3 llava/action/generate_description.py \
#     --train_metadata /data/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv \
#     --out_folder /data/shaokai/EK100_inst_train/ \
#     --train_predictions /data/shaokai/AVION_PREDS/avion_pred_ids_train.json \
#     --gen_type avion_mc \
#     --action_representation GT_random_narration \
#     --n_options 5

# python3 llava/action/generate_description.py \
#     --train_metadata /data/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv \
#     --out_folder /data/shaokai/EK100_inst_train/ \
#     --train_predictions /data/shaokai/AVION_PREDS/avion_pred_ids_train.json \
#     --gen_type avion_mc \
#     --action_representation official_key \
#     --n_options 5    


# python3 llava/action/generate_description.py \
#     --train_metadata /data/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv \
#     --out_folder /data/shaokai/EK100_inst_train/ \
#     --train_predictions /data/shaokai/AVION_PREDS/avion_pred_ids_train.json \
#     --gen_type avion_mc \
#     --action_representation GT_random_narration \
#     --n_options 10


# python3 llava/action/generate_description.py \
#     --train_metadata /data/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv \
#     --out_folder /data/shaokai/EK100_inst_train/ \
#     --train_predictions /data/shaokai/AVION_PREDS/avion_pred_ids_train.json \
#     --gen_type avion_mc \
#     --action_representation GT_random_narration \
#     --n_options 20

# python3 llava/action/generate_description.py \
#     --train_metadata /data/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv \
#     --out_folder /data/shaokai/EK100_inst_train/ \
#     --train_predictions /data/shaokai/TIM_PREDS/tim_pred_ids_train.json \
#     --gen_type tim_mc \
#     --action_representation GT_random_narration \
#     --n_options 5 

# python3 llava/action/generate_description.py \
#     --train_metadata /data/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv \
#     --out_folder /data/shaokai/EK100_inst_train/ \
#     --train_predictions /data/shaokai/TIM_PREDS/tim_pred_ids_train.json \
#     --gen_type tim_mc \
#     --action_representation topk_narration_cut_key \
#     --n_narrations 5 \
#     --n_options 5 

python3 llava/action/generate_description.py \
    --train_metadata /data/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv \
    --out_folder /data/shaokai/EK100_inst_train/cross_validation \
    --train_predictions /data/shaokai/TIM_PREDS/tim_pred_ids_train_cross.json \
    --gen_type tim_mc \
    --action_representation GT_random_narration \
    --n_options 5 


python3 llava/action/generate_description.py \
    --train_metadata /data/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv \
    --out_folder /data/shaokai/EK100_inst_train/cross_validation \
    --train_predictions /data/shaokai/TIM_PREDS/tim_pred_ids_train_cross.json \
    --gen_type tim_mc \
    --action_representation official_key \
    --n_options 5 

python3 llava/action/generate_description.py \
    --train_metadata /data/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv \
    --out_folder /data/shaokai/EK100_inst_train/cross_validation \
    --train_predictions /data/shaokai/TIM_PREDS/tim_pred_ids_train_cross.json \
    --gen_type tim_mc \
    --action_representation GT_random_narration \
    --n_options 10   

python3 llava/action/generate_description.py \
    --train_metadata /data/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv \
    --out_folder /data/shaokai/EK100_inst_train/cross_validation \
    --train_predictions /data/shaokai/TIM_PREDS/tim_pred_ids_train_cross.json \
    --gen_type tim_mc \
    --action_representation official_key \
    --n_options 10 


