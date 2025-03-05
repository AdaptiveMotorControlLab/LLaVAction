
import csv
from llavaction.action.dataset import datetime2sec
import random
from llavaction.action.utils import generate_label_map
from pathlib import Path
import json

def get_temporal_detection(train_ann, delta = 5):
    
    labels, mapping_vn2narration, mapping_vn2act, verb_maps, noun_maps = generate_label_map(Path(train_ann).parent, 'GT_random_narration')

    csv_reader = csv.reader(open(train_ann))
    
    _ = next(csv_reader)
    
    ret = []
    
    for idx, row in enumerate(csv_reader):
        
        start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])
        pid, vid = row[1:3]
        vn_str = f'{row[10]}:{row[12]}'
        vid_path = '{}-{}'.format(pid, vid)
        process = lambda x: str(round(x, 2))
        
        action_duration = process(end_timestamp - start_timestamp)
        
        action_start_timestamp = process(start_timestamp)
        action_end_timestamp = process(end_timestamp)        
        action_gt_narration = row[8]        
        start_padding = random.uniform(0, delta)        
        end_padding = delta - start_padding        
        start_timestamp = process(max(0, start_timestamp - start_padding))
        end_timestamp = process(end_timestamp + end_padding)
            
        relative_start_time = process(float(action_start_timestamp) - float(start_timestamp))
        relative_end_time = process(float(action_end_timestamp) - float(start_timestamp))
        # print ('action_star_timestamp', action_start_timestamp)
        # print ('video start_timestamp', start_timestamp)
        # print ('relative_start_time', relative_start_time)
        
        # print ('action_end_timestamp', action_end_timestamp)
        # print ('video end_timestamp', end_timestamp)
        # print ('relative_end_time', relative_end_time)
        
        
        
        conversation = [
            {"from": "human", "value": f"The provided video contains an action '{action_gt_narration}' that lasts {action_duration} seconds. What is the relative start and end time of the action in seconds? Format it as 'start_timestamp: end_timestamp' and round to 2 decimal places."},
            {"from": "gpt", "value": f"{relative_start_time}, {relative_end_time}"}
        ]
                

        data = {'video': vid_path,
            'conversations': conversation,
            'id': vid_path,
            'split': 'train',
            'task_instruction': '',
            'num_samples': 1,
            'question_type': f'temporal_detection',
            'dataset_name': 'EK100',
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp,
            'verb_id': int(row[10]),
            'noun_id': int(row[12]),
            'action_id': mapping_vn2act[vn_str]}
        
        ret.append(data)
        
    return ret
        
        
res = get_temporal_detection('/data/anonymous/epic-kitchens-100-annotations/EPIC_100_train.csv')

# write to jsonl

with open('/data/anonymous/EK100_inst_train/temporal_detection.jsonl', 'w') as f:
    for item in res:
        f.write(json.dumps(item) + '\n')

