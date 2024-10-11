# python3 action/generate_description.py  \
#     --train_metadata /data/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv \
#     --out_folder /data/shaokai/EK100_avion_mc/ \
#  > train_gen.out 2>&1

python3 action/generate_description.py \
   --train_metadata /storage-rcp-pure/upmwmathis_scratch/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv \
   --out_folder /storage-rcp-pure/upmwmathis_scratch/shaokai/EK100_inst_train \
   --avion_train_predictions /storage-rcp-pure/upmwmathis_scratch/shaokai/avion_predictions_train.json \
   --gen_type avion_mc \
   --n_options 10

