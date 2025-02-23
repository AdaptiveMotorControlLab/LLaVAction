"""
From avion or tim predicions, we choose the gt as the
chosen sample and randomly sample one wrong answer as the rejected sample
"""

import json
import os
import csv
from collections import defaultdict
def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)

def get_annotated_intervals(file_path):
    csv_reader = csv.reader(open(file_path))
    _ = next(csv_reader)
    vid_to_intervals = defaultdict(list)
    vid_to_gt_narration = defaultdict(list)
    uid_to_gt_narration = {}
    for row in csv_reader:
        pid, vid = row[1:3]
        narration = row[8]
        start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])
        if end_timestamp <= start_timestamp:
            raise ValueError("End timestamp is less than or equal to start timestamp")
        if end_timestamp - start_timestamp > 50:
            pass
            #print(f"{vid} has a long duration of action {narration} {end_timestamp - start_timestamp:.2f}")

        vid_to_intervals[vid].append((start_timestamp, end_timestamp))
        vid_to_gt_narration[vid].append(narration)
        uid = vid+'_'+str(round(start_timestamp,2))+'_'+str(round(end_timestamp,2))
        uid_to_gt_narration[uid] = narration
    return vid_to_intervals, vid_to_gt_narration, uid_to_gt_narration



def sample_uid_overlaps(anno_file):
    vid_to_intervals, vid_to_gt_narration, _ = get_annotated_intervals(anno_file)
    time_gaps = []
    count = 0
    overlap_count = 0
    overlap_but_different = 0
    ret = []
    for vid, intervals in vid_to_intervals.items():
        end_times = [end for _, end in intervals]
        start_times = [start for start, _ in intervals]

        # Step 2: Calculate the time gaps
        # Time gap between end of previous event and start of next event
        for i in range(1, len(start_times)):
            
            id = vid.split('_')[0] + '-' + vid
            
            time_diff = start_times[i] - end_times[i - 1]
            gt_narartion = vid_to_gt_narration[vid][i]
            prev_gt_narartion = vid_to_gt_narration[vid][i-1]
            
            cur_uid = id+'_'+str(round(start_times[i],2))+'_'+str(round(end_times[i],2))
            prev_uid = id+'_'+str(round(start_times[i-1],2))+'_'+str(round(end_times[i-1],2))
            if time_diff <= 0:               
                #print ('t', gt_narartion, 't-1', prev_gt_narartion)
                if gt_narartion != prev_gt_narartion:
                    overlap_but_different +=1
                    template = {'id': id,
                        'prompt': '',
                        'answer': gt_narartion,
                        'chosen': cur_uid,
                        'rejected': prev_uid,
                        'video': id,
                        'split': 'train',
                        'dataset_name': 'EK100',
                        'start_timestamp': start_times[i],
                        'end_timestamp': end_times[i],
                        'num_samples': 1,
                        'question_type': 'dpo',                
                        'task_instruction': ''
                        }
                    ret.append(template)                    
                overlap_count +=1

            else:
                time_gaps.append(time_diff)

            count+=1
    print ('overlap_but_different', overlap_but_different)
    print ('overlap_count, total_count, overlap_count/total_count', overlap_count, count, round(overlap_count/count,2))
    return ret    

def create_uid_to_cot_dict(path_to_jsonl):
    with open(path_to_jsonl, 'r') as f:
        data = f.readlines()
        
    uid_to_cot = {}
    for line in data:
        line = json.loads(line)
        start_timestamp = line['start_timestamp']
        end_timestamp = line['end_timestamp']
        vid = line['video']

        uid = vid+'_'+str(round(start_timestamp,2))+'_'+str(round(end_timestamp,2))
        cot = line['conversations'][1]['value']
        # print ('debug')
        # print (line['conversations'][1]['value'])
        uid_to_cot[uid] = cot
    return uid_to_cot
    


def sample_overlaps(anno_file):
    vid_to_intervals, vid_to_gt_narration, _ = get_annotated_intervals(anno_file)
    time_gaps = []
    count = 0
    overlap_count = 0
    overlap_but_different = 0
    ret = []
    for vid, intervals in vid_to_intervals.items():
        end_times = [end for _, end in intervals]
        start_times = [start for start, _ in intervals]

        # Step 2: Calculate the time gaps
        # Time gap between end of previous event and start of next event
        for i in range(1, len(start_times)):
            id = vid.split('_')[0] + '-' + vid
            
            time_diff = start_times[i] - end_times[i - 1]
            gt_narartion = vid_to_gt_narration[vid][i]
            prev_gt_narartion = vid_to_gt_narration[vid][i-1]
            if time_diff <= 0:               
                #print ('t', gt_narartion, 't-1', prev_gt_narartion)
                if gt_narartion != prev_gt_narartion:
                    overlap_but_different +=1
                    template = {'id': id,
                        'prompt': '',
                        'answer': gt_narartion,
                        'chosen': gt_narartion,
                        'rejected': prev_gt_narartion,
                        'video': id,
                        'split': 'train',
                        'dataset_name': 'EK100',
                        'start_timestamp': str(start_times[i]),
                        'end_timestamp': str(end_times[i]),
                        'num_samples': 1,
                        'question_type': 'dpo',                
                        'task_instruction': ''
                        }
                    ret.append(template)                    
                overlap_count +=1

            else:
                time_gaps.append(time_diff)

            count+=1
    print ('overlap_but_different', overlap_but_different)
    print ('overlap_count, total_count, overlap_count/total_count', overlap_count, count, round(overlap_count/count,2))
    return ret

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

def create_temporal_pairs(train_ann_file):
    """
    We focus on those pairs that are close in time.
    They might contain important 
    """    
    res = sample_overlaps(train_ann_file)
    
    with open('/mnt/upmwmathis/scratch/shaokai/EK100_inst_train/temporal_dpo_pairs.jsonl', 'w') as f:
        for pair in res:
            f.write(json.dumps(pair)+'\n')

def create_temporal_cot_pairs(train_ann_file, cot_jsonl_file):
    uid_to_cot_dict = create_uid_to_cot_dict(cot_jsonl_file)
    res = sample_uid_overlaps(train_ann_file)
    print (list(uid_to_cot_dict.keys())[:10])
   
    for e in res:
        #uid = e['video']+'_'+e['start_timestamp']+'_'+e['end_timestamp']
        chosen_uid = e['chosen']
        rejected_uid = e['rejected']        
        if chosen_uid not in uid_to_cot_dict or rejected_uid not in uid_to_cot_dict:
            continue
        
        e['chosen'] = uid_to_cot_dict[chosen_uid]
        e['rejected'] = uid_to_cot_dict[rejected_uid]

    with open('/mnt/upmwmathis/scratch/shaokai/EK100_inst_train/temporal_dpo_cot_pairs.jsonl', 'w') as f:
        for pair in res:
            f.write(json.dumps(pair)+'\n')
    

#create_temporal_pairs('/mnt/upmwmathis/scratch/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv')


#create_simple_sample(path)

simple_path = '/mnt/upmwmathis/scratch/shaokai/EK100_inst_train/avion_mc_top5_GT_random_narration/train_convs_narration.jsonl'
cot_path = '/mnt/upmwmathis/scratch/shaokai/first_person_annos/train_anno_gpt-gt-reason_4_first_person_all_action_idx.jsonl'
#create_cot_pairs(simple_path, cot_path)

create_temporal_cot_pairs('/mnt/upmwmathis/scratch/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv', cot_path)