import json
import os

json_root = '/mediaPFM/data/haozhe/onevision/llava_instruct_old'
save_root = '/mediaPFM/data/haozhe/onevision/llava_instruct'

json_list = os.listdir(json_root)
for json_name in json_list:
    json_path = os.path.join(json_root, json_name)
    if json_path.endswith(".jsonl"):
        cur_data_dict = []
        with open(json_path, "r") as json_file:
            for line in json_file:
                cur_data_dict.append(json.loads(line.strip()))
    elif json_path.endswith(".json"):
        with open(json_path, "r") as json_file:
            cur_data_dict = json.load(json_file)
    else:
        raise ValueError(f"Unsupported file type: {json_path}")
    
    dataset_name = json_path.split('/')[-1].split('.')[0]
    for data in cur_data_dict:
        data['dataset_name'] = dataset_name

    # save back
    save_path = os.path.join(save_root, json_name)
    with open(save_path, "w") as json_file:
        if json_path.endswith(".jsonl"):
            for data in cur_data_dict:
                json_file.write(json.dumps(data) + "\n")
        elif json_path.endswith(".json"):
            json.dump(cur_data_dict, json_file, indent=4)
    aa = 1