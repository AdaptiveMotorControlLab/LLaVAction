import os
import json
import pandas as pd

def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)

annotation_path = "/mediaPFM/data/haozhe/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv"

original_jsonl_path = "/mediaPFM/data/haozhe/EK100/EK100_in_LLAVA/avion_mc_top5_GT_random_narration_shaokai/train_convs_narration.jsonl"
modifiled_jsonl_name = "train_convs_narration_frames.jsonl"
modifiled_jsonl_path = os.path.join(os.path.dirname(original_jsonl_path), modifiled_jsonl_name)

anno_df = pd.read_csv(annotation_path)

cur_data_dict = []
with open(original_jsonl_path, "r") as json_file:
    for line in json_file:
        cur_data_dict.append(json.loads(line.strip()))

new_data_dict = []
for i, data in enumerate(cur_data_dict):
    anno_data = anno_df.iloc[i]
    anno_start_time = datetime2sec(anno_data["start_timestamp"])
    anno_end_time = datetime2sec(anno_data["stop_timestamp"])
    assert anno_start_time == data["start_timestamp"] and anno_end_time == data["end_timestamp"]

    new_data = data.copy()
    # modify dataset_name to EKframes
    new_data["dataset_name"] = "EKframes"
    # add strat_frame and stop_frame in the new_data
    new_data["start_frame"] = int(anno_data["start_frame"])
    new_data["end_frame"] = int(anno_data["stop_frame"])

    new_data_dict.append(new_data)

with open(modifiled_jsonl_path, "w") as json_file:
    for data in new_data_dict:
        json_file.write(json.dumps(data) + "\n")

