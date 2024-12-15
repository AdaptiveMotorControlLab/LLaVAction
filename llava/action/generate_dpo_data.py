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

def create_simple_sample(path_to_jsonl):
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


def create_cot_pairs(path_to_simple_jsonl, path_to_cot_jsonl):
    # load the jsonl
    with open(path_to_simple_jsonl, 'r') as f:
        simple_data = f.readlines()
    with open(path_to_cot_jsonl, 'r') as f:
        cot_data = f.readlines()
    # create a dictionary
    dpo_pairs = []
    for simple_line, cot_line in zip(simple_data, cot_data):
        temp = {}
        simple_line = json.loads(simple_line)
        cot_line = json.loads(cot_line)

        simple_candidates = simple_line['conversations'][0]['value']
        simple_gt = remove_option_letter(simple_line['conversations'][1]['value'])        
        candidates = eval(simple_candidates)
        candidates = [remove_option_letter(c) for c in candidates if remove_option_letter(c) != simple_gt]
        cot_answer = cot_line['conversations'][1]['value']

        if len(candidates) < 3:
            continue
        
        temp = {'id': simple_line['id'], 
                'prompt': '',
                'answer': simple_gt,
                'chosen': cot_answer,
                'rejected': simple_gt,
                'video': simple_line['video'],
                'split': simple_line['split'],
                'dataset_name': simple_line['dataset_name'],
                'start_timestamp': simple_line['start_timestamp'],
                'num_samples': simple_line['num_samples'],
                'question_type': 'dpo',                
                'task_instruction': simple_line['task_instruction'],
                'end_timestamp': simple_line['end_timestamp']}
        dpo_pairs.append(temp)
        
    with open('/mnt/upmwmathis/scratch/shaokai/EK100_inst_train/avion_mc_top5_GT_random_narration/cot_dpo_pairs.jsonl', 'w') as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair)+'\n')

#create_simple_sample(path)

simple_path = '/mnt/upmwmathis/scratch/shaokai/EK100_inst_train/avion_mc_top5_GT_random_narration/train_convs_narration.jsonl'
cot_path = '/mnt/upmwmathis/scratch/shaokai/first_person_annos/train_anno_gpt-gt-reason_4_first_person_all_action_idx.jsonl'
create_cot_pairs(simple_path, cot_path)