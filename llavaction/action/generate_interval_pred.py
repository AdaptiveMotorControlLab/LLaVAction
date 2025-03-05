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
from llavaction.action.utils import generate_label_map

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


def get_annotated_intervals(file_path, action_representation):
    csv_reader = csv.reader(open(file_path))
    _ = next(csv_reader)
    vid_to_intervals = defaultdict(list)
    vid_to_action_representation = defaultdict(list)
    vid_to_action_ids = defaultdict(list)
    
    labels, mapping_vn2narration, mapping_vn2act, verb_maps, noun_maps = generate_label_map(Path(file_path).parent, action_representation)
    print (verb_maps)
    print (noun_maps)
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
        if action_representation == 'GT_random_narration':
            vid_to_action_representation[vid].append(narration)
        elif action_representation == 'official_key':
            vid_to_action_representation[vid].append(f'{verb_maps[str(verb_id)]} {noun_maps[str(noun_id)]}')

        vid_to_action_ids[vid].append((verb_id, noun_id, action_id))
    
    return vid_to_intervals, vid_to_action_representation, vid_to_action_ids


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
                
    
def get_pseudo_dict(pseudo_folder):
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

def get_lookup_dict(ann_file, action_representation, test_type = 'base', delta = 3, pseudo_folder = None):
    
    vid_to_intervals, vid_to_action_representation, _ = get_annotated_intervals(ann_file, action_representation)
    table = {}
    
    pseudo_dict = None
    if test_type == 'temporal_cot_pseudo':
        assert os.path.exists(pseudo_folder), f"Folder {pseudo_folder} does not exist"
        pseudo_dict = get_pseudo_dict(pseudo_folder)
    
    for vid, intervals in vid_to_intervals.items():
                
        sorted_indices = sorted(range(len(intervals)), key=lambda i: intervals[i][1])
        
        sorted_intervals = [intervals[i] for i in sorted_indices]
        sorted_narrations = [vid_to_action_representation[vid][i] for i in sorted_indices]
        
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
                             
                if test_type == 'base' or test_type.startswith('temporal_cot') and test_type != 'temporal_cot_pseudo':
                    narration1 = sorted_narrations[i]
                    narration2 = sorted_narrations[i+1]
                    narration3 = sorted_narrations[i+2]
                elif test_type == 'temporal_cot_pseudo':
                    narration1 = pseudo_dict[uid1]
                    narration2 = pseudo_dict[uid2]
                    narration3 = sorted_narrations[i+2]
                
                table[uid3] = {'prev2_narration': narration1,
                               'prev2_offset': round(start_times[i+2] - start_times[i],2),
                                'prev1_narration': narration2,
                                'prev1_offset': round(start_times[i+2] - start_times[i+1],2),
                                'cur_narration': narration3,
                                'prev2_start': start_times[i],
                                'prev2_end': end_times[i],
                                'prev1_start': start_times[i+1],
                                'prev1_end': end_times[i+1],
                                'cur_start': start_times[i+2],
                                'cur_end': end_times[i+2],
                                'prev2_uid': uid1,
                                'prev1_uid': uid2,
                                'cur_uid': uid3}
    return table
                                


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

    ann_file = '/data/anonymous/epic-kitchens-100-annotations/EPIC_100_train.csv'
    actoin_representation = 'GT_random_narration'
    ret = get_lookup_dict(ann_file, actoin_representation)
    
    print(list(ret.items())[:10])