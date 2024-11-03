pip install moviepy spacy==3.7.5 numpy==1.26.1 && python -m spacy download en_core_web_sm &&
export PYTHONPATH=/data/shaokai/LLaVA-NeXT:$PYTHONPATH
export PYTHONPATH=/usr/local/lib/python3.10/site-packages/decord-0.6.0-py3.10-linux-x86_64.egg/:$PYTHONPATH


python3 llava/action/generate_description.py \
    --train_metadata /data/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv \
    --out_folder /data/shaokai/EK100_inst_train/ \
    --avion_train_predictions /data/shaokai/AVION_PREDS/avion_pred_ids_train.json \
    --gen_type avion_mc \
    --action_representation top1_narration \
    --n_options 5


# python3 action/generate_description.py \
#    --train_metadata /capstor/scratch/cscs/hqi/llava/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv \
#    --out_folder /capstor/scratch/cscs/hqi/llava/EK100/EK100_inst_train \
#    --avion_train_predictions /capstor/scratch/cscs/hqi/llava/EK100/avion_predictions_train.json \
#    --gen_type avion_mc \
#    --n_options 10

