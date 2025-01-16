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


def generate_label_map(anno_root, action_representation):
    print("Preprocess ek100 action label space")
    vn_list = []
    mapping_vn2narration = {}
    
    # Load CSVs
    noun_classes_pd = pd.read_csv(os.path.join(anno_root, 'EPIC_100_noun_classes_v2.csv'))
    verb_classes_pd = pd.read_csv(os.path.join(anno_root, 'EPIC_100_verb_classes.csv'))
    
    # Initialize maps
    verb_maps = {} if 'key' in action_representation or action_representation == 'first_sample' else None
    noun_maps = {} if 'key' in action_representation or action_representation == 'first_sample' else None
    
    # Process verb and noun maps
    if 'key' in action_representation:
        for _, row in verb_classes_pd.iterrows():
            verb_maps[str(row['id'])] = row['key']
        for _, row in noun_classes_pd.iterrows():
            elements = row['key'].split(':')
            noun_maps[str(row['id'])] = ' '.join(elements[1:] + [elements[0]]) if len(elements) > 1 else row['key']

    # Batch processing setup
    if 'cut' in action_representation:
        import spacy
        nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
        
        def process_batch_of_rows(rows_batch):
            # Prepare data for batch processing
            narrations = []
            verbs = []
            nouns = []
            vns = []
            
            for row in rows_batch:
                narrations.append(row[8])
                verbs.append(row[9])
                nouns.append(row[13])
                vn = '{}:{}'.format(int(row[10]), int(row[12]))
                vns.append(vn)
            
            # Process all narrations in batch
            processed_narrations = []
            for doc, verb, noun in zip(nlp.pipe(narrations, batch_size=1000), verbs, nouns):
                processed_narration = remove_sub_nouns_with_doc(doc, verb, noun)
                processed_narrations.append(processed_narration)
            
            return zip(vns, processed_narrations)

    # Process files
    batch_size = 1000
    current_batch = []
    
    for f in [
        os.path.join(anno_root, 'EPIC_100_train.csv'),
        os.path.join(anno_root, 'EPIC_100_validation.csv'),
    ]:
        csv_reader = csv.reader(open(f))
        next(csv_reader)  # skip header
        
        for row in tqdm(csv_reader):
            vn = '{}:{}'.format(int(row[10]), int(row[12]))
            if vn not in vn_list:
                vn_list.append(vn)
                
            if action_representation == 'first_sample':
                if row[10] not in verb_maps:
                    verb_maps[row[10]] = row[9]
                if row[12] not in noun_maps:
                    noun_maps[row[12]] = row[11]
            
            if 'cut' in action_representation:
                current_batch.append(row)
                
                if len(current_batch) >= batch_size:
                    # Process batch
                    for batch_vn, processed_narration in process_batch_of_rows(current_batch):
                        if batch_vn not in mapping_vn2narration:
                            mapping_vn2narration[batch_vn] = [processed_narration]
                        else:
                            mapping_vn2narration[batch_vn].append(processed_narration)
                    current_batch = []
            else:
                narration = row[8]
                if vn not in mapping_vn2narration:
                    mapping_vn2narration[vn] = [narration]
                else:
                    mapping_vn2narration[vn].append(narration)
        
        # Process remaining batch
        if current_batch and 'cut' in action_representation:
            for batch_vn, processed_narration in process_batch_of_rows(current_batch):
                if batch_vn not in mapping_vn2narration:
                    mapping_vn2narration[batch_vn] = [processed_narration]
                else:
                    mapping_vn2narration[batch_vn].append(processed_narration)
    
    # Finalize results
    vn_list = sorted(vn_list)
    print('# of action= {}'.format(len(vn_list)))
    mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
    
    # Create labels with frequency sorting
    labels = {}
    for vn, narrations in mapping_vn2narration.items():
        frequency_count = Counter(narrations)
        sorted_unique_list = [item for item, count in frequency_count.most_common()]
        labels[vn] = sorted_unique_list
    
    return labels, mapping_vn2narration, mapping_vn2act, verb_maps, noun_maps



def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)

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
                
                narration_prev = vid_to_gt_narration[vid][i]
                narration_after = vid_to_gt_narration[vid][i+2]
                uid = f"{id}_{round(start_times[i+1],2)}_{round(end_times[i+1],2)}"
                uid_to_neighbors[uid] = {
                    'narration_prev': narration_prev,
                    'narration_after': narration_after,
                    'padded_start_time': start_times[i],
                    'padded_end_time': end_times[i+2]
                }
    return uid_to_neighbors
                
    

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


if __name__ == '__main__':

    res = sample_uid_triples('/mnt/upmwmathis/scratch/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv')

    # save to jsonl
    with open('ek100_triples.jsonl', 'w') as f:
        for item in res:
            f.write(json.dumps(item) + '\n')