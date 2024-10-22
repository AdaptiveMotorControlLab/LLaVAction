python3 llava/action/generate_description.py  \
    --train_metadata /data/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv \
    --out_folder /data/shaokai/EK100_inst_train/ \
    --avion_train_predictions /data/shaokai/avion_predictions_train.json \
    --gen_type avion_mc \
    --n_options 10 

# python3 action/generate_description.py \
#    --train_metadata /capstor/scratch/cscs/hqi/llava/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv \
#    --out_folder /capstor/scratch/cscs/hqi/llava/EK100/EK100_inst_train \
#    --avion_train_predictions /capstor/scratch/cscs/hqi/llava/EK100/avion_predictions_train.json \
#    --gen_type avion_mc \
#    --n_options 10

