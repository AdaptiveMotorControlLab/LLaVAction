"""
The goal is to merge overlapped intervals
"""

import pandas as pd
import json
import os
import csv
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from llava.action.utils import generate_label_map

def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def sort_correspondance(vid_to_intervals, vid_to_gt_narration):
    sorted_vid_to_gt_narration = {}
    
    for vid, intervals in vid_to_intervals.items():
        # Use the same sorting key as in the original sorting of intervals
        sorted_indices = sorted(range(len(intervals)), key=lambda i: intervals[i][1])
        
        # Apply the same sorting to the narrations
        sorted_vid_to_gt_narration[vid] = [vid_to_gt_narration[vid][i] for i in sorted_indices]
    
    return sorted_vid_to_gt_narration


def get_annotated_intervals(file_path):
    csv_reader = csv.reader(open(file_path))
    _ = next(csv_reader)
    vid_to_intervals = defaultdict(list)
    vid_to_gt_narration = defaultdict(list)
    vid_to_action_ids = defaultdict(list)
    
    labels, mapping_vn2narration, mapping_vn2act, verb_maps, noun_maps = generate_label_map(Path(file_path).parent, 'GT_random_narration')
    for row in csv_reader:       
        pid, vid = row[1:3]
        narration = row[8]
        verb_id = int(row[10])
        noun_id = int(row[12])
        vn_str = f'{row[10]}:{row[12]}'
        action_id = mapping_vn2act[vn_str]
        start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])
        if end_timestamp <= start_timestamp:
            raise ValueError("End timestamp is less than or equal to start timestamp")
        if end_timestamp - start_timestamp > 50:
            pass
            #print(f"{vid} has a long duration of action {narration} {end_timestamp - start_timestamp:.2f}")

        vid_to_intervals[vid].append((start_timestamp, end_timestamp))
        vid_to_gt_narration[vid].append(narration)


        vid_to_action_ids[vid].append((verb_id, noun_id, action_id))
    
    return vid_to_intervals, vid_to_gt_narration, vid_to_action_ids


def build_uid_pad_dict(ann_file,
                       delta = 3):
    """
    every uid corresponds to two neighboring actions    
    """
    
    uid_to_neighbors = {}
    
    vid_to_intervals, vid_to_gt_narration, vid_to_action_ids = get_annotated_intervals(ann_file)
    ret = []    
    
    
    for vid, intervals in vid_to_intervals.items():
        # Sort intervals by end time
        sorted_intervals = sorted(intervals, key=lambda x: x[1])
        
        end_times = [end for _, end in sorted_intervals]
        start_times = [start for start, _ in sorted_intervals]
        
        # Look for consecutive triples
        for i in range(len(sorted_intervals)-2):  # -2 because we need 3 consecutive intervals
            id = vid.split('_')[0] + '-' + vid
            
            # Get time differences between consecutive intervals
            time_diff1 = start_times[i+1] - end_times[i]
            time_diff2 = start_times[i+2] - end_times[i+1]
            
            # Check if both time differences are less than 3 seconds
            if time_diff1 <= delta and time_diff2 <= delta: 
                
                narration_prev_2 = vid_to_gt_narration[vid][i]
                narration_prev_1 = vid_to_gt_narration[vid][i+1]
                uid = f"{id}_{round(start_times[i+2],2)}_{round(end_times[i+2],2)}"
                uid_to_neighbors[uid] = {
                    'narration_prev_2': narration_prev_2,
                    'narration_prev_1': narration_prev_1,
                    'padded_start_time': start_times[i],
                    'padded_end_time': end_times[i+2]
                }
    return uid_to_neighbors
                
    
def get_pseudo_dict(pseudo_folder,  delta = 3):
    import glob
    

    files = glob.glob(os.path.join(pseudo_folder, 'prediction*.json'))
    
    pseudo_data = {}
    ret = {}
    for file in files:
        with open(file, 'r') as f:
            pseudo_data.update(json.load(f))
    for k,v in pseudo_data.items():
        start_timestamp = round(float(v['start_second']),2)
        end_timestamp = round(float(v['end_second']), 2)
        vid = v['vid_path'].replace('/', '-')
        uid = f"{vid}_{start_timestamp}_{end_timestamp}"
        ret[uid] = v['llava_pred']
            
    assert len(ret) == len(pseudo_data)
    return ret

def get_lookup_dict(ann_file, test_type = 'base', delta = 3, pseudo_folder = None):
    
    vid_to_intervals, vid_to_gt_narration, _ = get_annotated_intervals(ann_file)
    table = {}
    
    pseudo_dict = None
    if test_type == 'temporal_cot':
        pseudo_dict = get_pseudo_dict(pseudo_folder)
    
    for vid, intervals in vid_to_intervals.items():
                
        sorted_indices = sorted(range(len(intervals)), key=lambda i: intervals[i][1])
        
        sorted_intervals = [intervals[i] for i in sorted_indices]
        sorted_narrations = [vid_to_gt_narration[vid][i] for i in sorted_indices]
        
        end_times = [end for _, end in sorted_intervals]
        start_times = [start for start, _ in sorted_intervals]
        
        # Look for consecutive triples
        for i in range(len(sorted_intervals)-2):  # -2 because we need 3 consecutive intervals
            id = vid.split('_')[0] + '-' + vid
            
            # Get time differences between consecutive intervals
            time_diff1 = start_times[i+1] - end_times[i]
            time_diff2 = start_times[i+2] - end_times[i+1]
            
            # Check if both time differences are less than 3 seconds
            if time_diff1 <= delta and time_diff2 <= delta:
                # Create UIDs for each interval in the triple
                uid1 = f"{id}_{round(start_times[i],2)}_{round(end_times[i],2)}"
                uid2 = f"{id}_{round(start_times[i+1],2)}_{round(end_times[i+1],2)}"
                uid3 = f"{id}_{round(start_times[i+2],2)}_{round(end_times[i+2],2)}"
                             
                if test_type == 'base':
                    narration1 = sorted_narrations[i]
                    narration2 = sorted_narrations[i+1]
                    narration3 = sorted_narrations[i+2]
                elif test_type == 'temporal_cot':
                    narration1 = pseudo_dict[uid1]
                    narration2 = pseudo_dict[uid2]
                    narration3 = sorted_narrations[i+2]
                
                table[uid3] = {'prev2_narration': narration1,
                               'prev2_offset': round(start_times[i+2] - start_times[i],2),
                                'prev1_narration': narration2,
                                'prev1_offset': round(start_times[i+2] - start_times[i+1],2),
                                'cur_narration': narration3}
    return table
                                

def sample_uid_triples(anno_file, 
                       delta = 3, 
                       question_type="triple_direct_answer"):
    vid_to_intervals, vid_to_gt_narration, vid_to_action_ids = get_annotated_intervals(anno_file)
    ret = []
    
    for vid, intervals in vid_to_intervals.items():
        # Sort intervals by end time
        sorted_intervals = sorted(intervals, key=lambda x: x[1])
        
        end_times = [end for _, end in sorted_intervals]
        start_times = [start for start, _ in sorted_intervals]
        
        # Look for consecutive triples
        for i in range(len(sorted_intervals)-2):  # -2 because we need 3 consecutive intervals
            id = vid.split('_')[0] + '-' + vid
            
            # Get time differences between consecutive intervals
            time_diff1 = start_times[i+1] - end_times[i]
            time_diff2 = start_times[i+2] - end_times[i+1]
            
            # Check if both time differences are less than 3 seconds
            if time_diff1 <= delta and time_diff2 <= delta:
                # Create UIDs for each interval in the triple
                uid1 = f"{id}_{round(start_times[i],2)}_{round(end_times[i],2)}"
                uid2 = f"{id}_{round(start_times[i+1],2)}_{round(end_times[i+1],2)}"
                uid3 = f"{id}_{round(start_times[i+2],2)}_{round(end_times[i+2],2)}"
                
                # Get corresponding narrations
                verb_id1, noun_id1, action_id1 = vid_to_action_ids[vid][i]
                verb_id2, noun_id2, action_id2 = vid_to_action_ids[vid][i+1]
                verb_id3, noun_id3, action_id3 = vid_to_action_ids[vid][i+2]

                
                narration1 = vid_to_gt_narration[vid][i]
                narration2 = vid_to_gt_narration[vid][i+1]
                narration3 = vid_to_gt_narration[vid][i+2]
                                
                if question_type == "triple_multiple_choice":
                    pass
                elif question_type == "triple_direct_answer":
                    target = narration1 + ', ' + narration2 + ', ' + narration3                
                
                triple = {
                    'id': id,
                    'video': id,
                    'start_timestamp': start_times[i],
                    'end_timestamp': end_times[i+2],
                    'gt_narration_triple': [narration1, narration2, narration3],  
                    'conversations': [{"from": "human", "value":""},
                                    {"from": "gpt", "value": target}
                                        ], 
                    "question_type": question_type,              
                    'split': 'train',
                    'dataset_name': 'EK100',
                    'triple_meta':
                        [
                            {   'uid': uid1,
                                'narration': narration1,
                                'start_timestep': start_times[i],
                                'end_timestep': end_times[i],
                                'duration': round(end_times[i] - start_times[i],2),
                                'verb_id': verb_id1,
                                'noun_id': noun_id1,
                                'action_id': action_id1
                                
                            },                        
                            {   'uid': uid2,
                                'narration': narration2,
                                'start_timestep': start_times[i+1],
                                'end_timestep': end_times[i+1],
                                'duration': round(end_times[i+1] - start_times[i+1],2),
                                'verb_id': verb_id2,
                                'noun_id': noun_id2,
                                'action_id': action_id2
                            },                        
                            {   'uid': uid3,
                                'narration': narration3,
                                'start_timestep': start_times[i+2],
                                'end_timestep': end_times[i+2],
                                'duration': round(end_times[i+2] - start_times[i+2],2),
                                'verb_id': verb_id3,
                                'noun_id': noun_id3,            
                                'action_id': action_id3
                            }
                        ]
                }
                ret.append(triple)
    
    print(f'Found {len(ret)} triples with gaps <= {delta} seconds')
    return ret



def create_merged_intervals(train_ann_file):
    """
    interval of 2, 3, 4? We also do some stats to figure it out
    """
    pass


def create_merged_captions(triple_file, caption_file):
    # both files are jsonl
    with open(caption_file, 'r') as f:
        caption_lines = f.readlines()
    # get uid from each caption dict
    


if __name__ == '__main__':

    # res = sample_uid_triples('/mnt/upmwmathis/scratch/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv')

    # # save to jsonl
    # with open('ek100_triples.jsonl', 'w') as f:
    #     for item in res:
    #         f.write(json.dumps(item) + '\n')
    triple_file_path = 'ek100_triples.jsonl'
    caption_file_path = '/data/shaokai/first_person_annos/train_anno_gpt-gt-reason_4_first_person_all_action_idx.jsonl'
    create_merged_captions(triple_file_path, caption_file_path)
    
