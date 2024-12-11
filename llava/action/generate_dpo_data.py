"""
From avion or tim predicions, we choose the gt as the
chosen sample and randomly sample one wrong answer as the rejected sample
"""

import json
import os

def remove_option_letter(answer):
    if '. ' in answer:
        return answer.split('. ')[1]
    else:
        return answer

path = '/mnt/upmwmathis/scratch/shaokai/EK100_inst_train/avion_mc_top5_GT_random_narration/train_convs_narration.jsonl'

def create_sample(path_to_jsonl):
    # load the jsonl
    with open(path_to_jsonl, 'r') as f:
        data = f.readlines()
    # create a dictionary
    dpo_pairs = []
    for line in data:
        temp = {}
        line = json.loads(line)
        candidates = line['conversations'][0]['value']
        gt = remove_option_letter(line['conversations'][1]['value'])
        candidates = eval(candidates)
        candidates = [remove_option_letter(c) for c in candidates if remove_option_letter(c) != gt]
        if len(candidates) < 3:
            continue
        for i in range(3):
            temp = {'id': line['id'], 
                    'prompt': '',
                    'answer': gt,
                    'chosen': gt,
                    'rejected': candidates[i],
                    'video': line['video'],
                    'split': line['split'],
                    'dataset_name': line['dataset_name'],
                    'start_timestamp': line['start_timestamp'],
                    'num_samples': line['num_samples'],
                    'question_type': 'dpo',                
                    'task_instruction': line['task_instruction'],
                    'end_timestamp': line['end_timestamp']}

            dpo_pairs.append(temp)
    with open('/mnt/upmwmathis/scratch/shaokai/EK100_inst_train/avion_mc_top5_GT_random_narration/dpo_pairs.jsonl', 'w') as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair)+'\n')
    

create_sample(path)