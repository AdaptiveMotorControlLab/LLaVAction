from __future__ import annotations
import json
import csv
import os
import argparse
import sys
import numpy as np
sys.path[0] = os.path.dirname(os.path.dirname(sys.path[0]))
from llavaction.action.utils import generate_label_map, RandomMultiChoiceGenerator, AvionMultiChoiceGenerator, format_llava_prompt, remove_sub_nouns
from llavaction.action.dataset import datetime2sec
from pathlib import Path
from llavaction.action.utils import hand_obj_ann_loader
from llavaction.action.generate_interval_pred import build_uid_pad_dict
import ast

def generate_train_ann(ann_file, labels, mapping_vn2narration, mapping_vn2act, verb_maps, noun_maps, gen_type = 'naive', prediction_path = '', n_options = 5,
                       action_representation = 'official_key', with_neighbors = False, n_narrations=-1):
    # epic kitchen uses csv
    csv_reader = csv.reader(open(ann_file))
    _ = next(csv_reader)
    ret = []
    ann_root = Path(ann_file).parent
    if gen_type == "random_mc":
        # DEPRECATED
        mc_generator = RandomMultiChoiceGenerator(ann_root)
    elif gen_type == 'avion_mc' or gen_type == 'tim_mc':
        mc_generator = AvionMultiChoiceGenerator(ann_root)
        with open(prediction_path, 'r') as f:
            train_predictions = json.load(f)
    import spacy
    nlp = spacy.load('en_core_web_sm')

    if with_neighbors:
        uid_pad_dict = build_uid_pad_dict(ann_file)

    for idx, row in enumerate(csv_reader):
        start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])
        
        pid, vid = row[1:3]
        vid_path = '{}-{}'.format(pid, vid)

        if gen_type == 'naive':
            # here we directly ask the model to output the action representation
            verb_noun = f'{verb_maps[row[10]]} {noun_maps[row[12]]}'
            conversation = generate_naive_conversation(verb_noun)
        elif gen_type == 'direct_narration':
            vn_str = f'{row[10]}:{row[12]}'
            # here we directly use the model to predict gt narration
            narration = row[8]
            conversation = generate_direct_conversation(narration)
        
        elif gen_type == "temporal_detection":
            """
            The action is X and it lasts for XX seconds, what is the start and end time
            """
            narration = row[8]
            
        
        elif gen_type == "random_mc":
            # DEPRECATED
            vn_str = f'{row[10]}:{row[12]}'
            narration = row[8]
            if 'cut' in action_representation:
                narration = remove_sub_nouns(nlp, narration, row[9], row[13])            
            mc_data = mc_generator.generate_multi_choice(vn_str,
                                    narration, 
                                    n_options, 
                                    action_representation, 
                                    n_narrations, 
                                    labels, 
                                    mapping_vn2narration, 
                                    verb_maps, 
                                    noun_maps,
                                    is_train = True)
            options = mc_data['options'][0]
            gt_answer_letter = mc_data['gt_answer_letter'][0]
            gt_answer_name = mc_data['gt_answer_name'][0]
            conversation = generate_random_mc_conversation(options, gt_answer_letter, gt_answer_name )
            
        elif gen_type == "avion_mc" or gen_type == "tim_mc":
            vn_str = f'{row[10]}:{row[12]}'
            action_preds = train_predictions[str(idx)]['predictions']
            gt_from_model = train_predictions[str(idx)]['target']
            assert gt_from_model == vn_str
            narration = row[8]
            if 'cut' in action_representation:
                narration = remove_sub_nouns(nlp, narration, row[9], row[13])
            mc_data = mc_generator.generate_multi_choice(vn_str, 
                                                         action_preds, 
                                                         narration, 
                                                         n_options, 
                                                         action_representation, 
                                                         n_narrations, 
                                                         labels, 
                                                         mapping_vn2narration, 
                                                         verb_maps, 
                                                         noun_maps,
                                                         is_train = True)
            options = mc_data['options'][0]
            gt_answer_letter = mc_data['gt_answer_letter'][0]
            gt_answer_name = mc_data['gt_answer_name'][0]
            conversation = generate_random_mc_conversation(options, gt_answer_letter, gt_answer_name )

        pad = {}
        if with_neighbors:
            _id = vid.split('_')[0]
            uid = f'{_id}-{vid}_{round(start_timestamp,2)}_{round(end_timestamp,2)}'
            # print ('uid', uid)
            # print ('key', list(uid_pad_dict.keys())[0])        
            pad = uid_pad_dict.get(uid, {})
            if len(pad) != 0:               
                pad['start_timestamp'] = pad['padded_start_time']
                pad['end_timestamp'] = pad['padded_end_time']

        if 'direct_narration' not in gen_type:
            question_type = f'mc_{action_representation}'
        else:
            question_type = 'direct_narration'
        
        data = {'video': vid_path,
                'conversations': conversation,
                'id': vid_path,
                'split': 'train',
                'task_instruction': '',
                'num_samples': 1,
                'question_type': question_type,
                'dataset_name': 'EK100',
                'start_timestamp': start_timestamp,
                'end_timestamp': end_timestamp,
                'verb_id': int(row[10]),
                'noun_id': int(row[12]),
                'action_id': mapping_vn2act[vn_str]}
        if with_neighbors:
            data.update(pad)
            
        ret.append(data)
    return ret

def append_action_idx_to_existing_ann(instruct_ann_file, ek100_ann_file,  mapping_vn2act):
                       
    csv_reader = csv.reader(open(ek100_ann_file))
    _ = next(csv_reader)
    
    map_video_2_action = {}

    for idx, row in enumerate(csv_reader):
        start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])
        
        pid, vid = row[1:3]
        vid_path = '{}-{}'.format(pid, vid)  
        vn_str = f'{row[10]}:{row[12]}'

        verb_id = int(row[10])
        noun_id = int(row[12])
        action_id = mapping_vn2act[vn_str]

        _key= f'{vid_path}-{start_timestamp}-{end_timestamp}'
        map_video_2_action[_key] = (verb_id, noun_id, action_id)

    ret = []
    instruct_ann_root = Path(instruct_ann_file).parent
    # instruction file is jsonl
    with open(instruct_ann_file, 'r') as f:
        instructions = f.readlines()

        for instruct in instructions:
            instruct_dict = json.loads(instruct)
            vid_path = instruct_dict['video']
            start_timestamp = instruct_dict['start_timestamp']
            end_timestamp = instruct_dict['end_timestamp']
            _key = f'{vid_path}-{start_timestamp}-{end_timestamp}'
            verb_id, noun_id, action_id = map_video_2_action[_key]
            instruct_dict['verb_id'] = verb_id
            instruct_dict['noun_id'] = noun_id
            instruct_dict['action_id'] = action_id
            ret.append(instruct_dict)
    
    # write a new instruct ann file in the same folder but with a fixed suffix
    # use the same filename of the original instruct ann
    out_path = os.path.join(instruct_ann_root, f'{Path(instruct_ann_file).stem}_action_idx.jsonl')
    with open(out_path, 'w') as f:
        for instruct in ret:
            f.write(json.dumps(instruct) + '\n')
    


def generate_naive_conversation(vn_str:str):
    # DEPRECATED. As this is hard-coding the prompt into the data
    # in this version, we do not care about diversifying the questions
    return [
        {"from": "human", "value": "The video is taken from egocentric view. What action is the person performing? Hint: provide your answer in verb-noun pair. "},
        {"from": "gpt", "value": f"{vn_str}"}    
    ]

def generate_direct_conversation(narration:str):
    return [
        {"from": "human", "value": ""},
        {"from": "gpt", "value": f"{narration}"}    
    ]

def generate_random_mc_conversation(options:list[str], gt_answer_letter, gt_answer_name):
    return [
        {"from": "human", "value": f"{options}"},
        {"from": "gpt", "value": f"{gt_answer_letter}. {gt_answer_name}"} 
    ]


def combine_reason_and_mc(reason_path, mc_path, out_folder):
    """
    Looks like that it's hard to balance mc and reason if we train it separately. So we just cmoine them together
    """
    
    # reason_path and mc_path are jsonl
    os.makedirs(out_folder, exist_ok=True)
    with open(reason_path, 'r') as f:
        reasons = f.readlines()
    with open(mc_path, 'r') as f:
        mcs = f.readlines()
    
    assert len(reasons) == len(mcs)

    ret = []
    for reason_conv, mc_conv in zip(reasons, mcs):
        reason_traj = json.loads(reason_conv)['conversations'][1]['value']
        mc_dict = json.loads(mc_conv)
        mc_answer = mc_dict['conversations'][1]['value']
        combined_traj = reason_traj + ' The answer is ' + mc_answer
        mc_dict['conversations'][1]['value'] = combined_traj
        mc_dict['question_type'] = 'cot_mc'
        ret.append(mc_dict)
    
    
    out_path = os.path.join(out_folder, 'train_convs_narration.jsonl')
    with open(out_path, 'w') as f:
        for conv in ret:
            f.write(json.dumps(conv) + '\n')




def generate_hand_object_instruction_tuning_data(root, ann_file, hand_obj_root, image_out_folder):
    """
    iterate through the training dataset.
    take a few frames from each action and use opencv to save them into a folder
    load the corresponding hand-object annotations and use chatGPT to annotate it
    finally save it to a jsonl file
    """

    csv_reader = csv.reader(open(ann_file))
    _ = next(csv_reader)
    ret = []
    ann_root = Path(ann_file).parent

    for idx, row in enumerate(csv_reader):
        start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])

        pid, vid = row[1:3]
        vid_path = '{}/{}'.format(pid, vid)

        frames, hand_dets_list, obj_dets_list = hand_obj_ann_loader(root,
                                                                    hand_obj_root,
                                                                    vid_path,
                                                                    'MP4',
                                                                    start_timestamp,
                                                                    end_timestamp,
                                                                    chunk_len = 15,                       
                                                                    clip_length = 16)

        def contains_nan(lst):
            # Check each element in the list individually for NaN
            return any(isinstance(x, float) and np.isnan(x) for x in lst)
        if contains_nan(hand_dets_list) or contains_nan(obj_dets_list):
            continue
        print (hand_dets_list)
        print (obj_dets_list)        
    

def get_args():
    parser = argparse.ArgumentParser(description="For generating VQA for EPIC-KITCHEN")
    parser.add_argument('--train_metadata', default='/data/anonymous/epic-kitchens-100-annotations/EPIC_100_train.csv', type=str)
    parser.add_argument('--out_folder', default = '/data/anonymous/EK100_in_LLAVA/', type = str)
    parser.add_argument('--train_predictions', default = '/data/anonymous/avion_predictions_train.json', type = str)
    parser.add_argument('--gen_type', default = 'avion_mc', type = str, choices = ['naive', 'direct_narration', 'random_mc', 'avion_mc', 'tim_mc'])
    parser.add_argument('--n_options', default = 5, type = int)
    parser.add_argument('--action_representation', default = 'GT_random_narration_cut', type = str, 
                                            choices = ['first_sample', 'official_key', 
                                                       'random_narration_cut', 'top1_narration', 'top1_narration_cut', 'topk_narration_cut_key',
                                                       'GT_key', 'GT_random_narration', 'GT_random_narration_cut'])
    parser.add_argument('--n_narrations', default = -1, type = int)
    parser.add_argument('--with_neighbors', action = 'store_true', default = False)
    return parser.parse_args()

def main(): 
    args = get_args()    
    ann_file = args.train_metadata
    
    if 'direct_narration' in args.gen_type:
        inst_train_folder = os.path.join(args.out_folder, f'{args.gen_type}_{args.action_representation}')
    else:
        inst_train_folder = os.path.join(args.out_folder, f'{args.gen_type}_top{args.n_options}_{args.action_representation}')

    print ('train_metadata', args.train_metadata)
    print ('out_folder', args.out_folder)
    print ('loading predictions from ', args.train_predictions)
    print ('gen_type is ', args.gen_type)
    print ('n_options', args.n_options)
    print ('with_neighbors', args.with_neighbors)

    os.makedirs(inst_train_folder, exist_ok=True)    

    anno_path = Path(ann_file).parent
    labels, mapping_vn2narration, mapping_vn2act, verb_maps, noun_maps = generate_label_map(anno_path, args.action_representation)
    conv_lst = generate_train_ann(ann_file,
                                  labels,
                                  mapping_vn2narration,
                                  mapping_vn2act,
                                  verb_maps, 
                                  noun_maps, 
                                  gen_type = args.gen_type, 
                                  prediction_path = args.train_predictions,
                                  n_options = args.n_options,
                                  with_neighbors = args.with_neighbors,
                                  action_representation = args.action_representation,
                                  n_narrations = args.n_narrations)
        
    # save it to a jsonl
    
    if args.with_neighbors:
        filename = os.path.join(inst_train_folder,'train_convs_narration_actionids_padded.jsonl')
    else:
        filename = os.path.join(inst_train_folder,'train_convs_narration_actionids.jsonl')

    
    with open(filename, 'w') as f:
        for conv in conv_lst:
            f.write(json.dumps(conv) + '\n')

def fix_annotations():

    instruct_files = [
        '/data/anonymous/EK100_inst_train/avion_mc_top5_GT_random_narration/train_convs_narration.jsonl',
        '/data/anonymous/EK100_inst_train/avion_mc_top5_official_key/train_convs_narration.jsonl',
        '/data/anonymous/EK100_inst_train/tim_mc_top5_GT_random_narration/train_convs_narration.jsonl',
        '/data/anonymous/EK100_inst_train/tim_mc_top5_official_key/train_convs_narration.jsonl',
        '/data/anonymous/first_person_annos/train_anno_gpt-gt-reason_4_first_person_all.jsonl',
        '/data/anonymous/first_person_annos/train_anno_gpt-gt-instruct-reason_4_first_person_all.jsonl'
    ]
    for instruct_ann_file in instruct_files:
        _, _, mapping_vn2act, _, _ = generate_label_map('/data/anonymous/epic-kitchens-100-annotations/', 'official_key')
        append_action_idx_to_existing_ann(instruct_ann_file, '/data/anonymous/epic-kitchens-100-annotations/EPIC_100_train.csv', mapping_vn2act)
   

if __name__ == "__main__":

    main()