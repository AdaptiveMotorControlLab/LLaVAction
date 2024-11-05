from __future__ import annotations
import json
import csv
import os
import argparse
import sys
import numpy as np
sys.path[0] = os.path.dirname(os.path.dirname(sys.path[0]))
import llava
from llava.action.utils import generate_label_map, MultiChoiceGenerator, AvionMultiChoiceGenerator, format_llava_prompt, remove_sub_nouns
from llava.action.dataset import datetime2sec
from pathlib import Path
from llava.action.utils import hand_obj_ann_loader
import ast

def generate_train_ann(ann_file, labels, mapping_vn2narration, verb_maps, noun_maps, gen_type = 'naive', avion_prediction_path = '', n_options = 5,
                       action_representation = 'official_key', n_narrations=-1):
    # epic kitchen uses csv
    csv_reader = csv.reader(open(ann_file))
    _ = next(csv_reader)
    ret = []
    ann_root = Path(ann_file).parent
    if gen_type == "random_mc":
        # DEPRECATED
        mc_generator = MultiChoiceGenerator(ann_root)
    elif gen_type == 'avion_mc':
        mc_generator = AvionMultiChoiceGenerator(ann_root)
        with open(avion_prediction_path, 'r') as f:
            avion_train_predictions = json.load(f)

    nlp = spacy.load('en_core_web_sm')

    for idx, row in enumerate(csv_reader):
        start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])
        
        pid, vid = row[1:3]
        vid_path = '{}-{}'.format(pid, vid)

        if gen_type == 'naive':
            # here we directly ask the model to output the action representation
            verb_noun = f'{verb_maps[row[10]]} {noun_maps[row[12]]}'
            conversation = generate_naive_conversation(verb_noun)
        elif gen_type == 'direct_narration':
            # here we directly use the model to predict gt narration
            narration = row[8]
            conversation = generate_direct_conversation(narration)
        elif gen_type == "random_mc":
            # DEPRECATED
            vn_str = f'{row[10]}:{row[12]}'
            mc_data = mc_generator.generate_multi_choice(vn_str, n_options, verb_maps, noun_maps)
            options = mc_data['options'][0]
            gt_answer_letter = mc_data['gt_answer_letter'][0]
            gt_answer_name = mc_data['gt_answer_name'][0]
            conversation = generate_random_mc_conversation(options, gt_answer_letter, gt_answer_name )
        elif gen_type == "avion_mc":
            vn_str = f'{row[10]}:{row[12]}'
            avion_preds = avion_train_predictions[str(idx)]['predictions']
            gt_from_avion = avion_train_predictions[str(idx)]['target']
            assert gt_from_avion == vn_str
            narration = row[8]
            if 'cut' in action_representation:
                narration = remove_sub_nouns(nlp, narration, row[9], row[13])
            mc_data = mc_generator.generate_multi_choice(vn_str, 
                                                         avion_preds, 
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

        data = {'video': vid_path,
                'conversations': conversation,
                'id': vid_path,
                'split': 'train',
                'task_instruction': '',
                'num_samples': 1,
                'question_type': f'mc_{action_representation}',
                'dataset_name': 'EK100',
                'start_timestamp': start_timestamp,
                'end_timestamp': end_timestamp}
        ret.append(data)
    return ret

def generate_naive_conversation(vn_str:str):
    # DEPRECATED. As this is hard-coding the prompt into the data
    # in this version, we do not care about diversifying the questions
    return [
        {"from": "human", "value": "<image>\n the video is taken from egocentric view. What action is the person performing? Hint: provide your answer in verb-noun pair. "},
        {"from": "gpt", "value": f"{vn_str}"}    
    ]

def generate_direct_conversation(vn_str:str):
    # DEPRECATED. As this is hard-coding the prompt into the data
    # in this version, we do not care about diversifying the questions
    return [
        {"from": "human", "value": "<image>\n the video is taken from egocentric view. What action is the person performing? "},
        {"from": "gpt", "value": f"{vn_str}"}    
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
    parser.add_argument('--train_metadata', default='/data/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv', type=str)
    parser.add_argument('--out_folder', default = '/data/shaokai/EK100_in_LLAVA/', type = str)
    parser.add_argument('--avion_train_predictions', default = '/data/shaokai/avion_predictions_train.json', type = str)
    parser.add_argument('--gen_type', default = 'avion_mc', type = str, choices = ['naive', 'direct_narration', 'random_mc', 'avion_mc'])
    parser.add_argument('--n_options', default = 5, type = int)
    parser.add_argument('--action_representation', default = 'GT_random_narration_cut', type = str, 
                                            choices = ['first_sample', 'official_key', 
                                                       'random_narration_cut', 'top1_narration', 'top1_narration_cut', 'topk_narration_cut_key',
                                                       'GT_key', 'GT_random_narration', 'GT_random_narration_cut'])
    parser.add_argument('--n_narrations', default = -1, type = int)
    return parser.parse_args()

def main(): 
    args = get_args()    
    ann_file = args.train_metadata
    inst_train_folder = os.path.join(args.out_folder, f'{args.gen_type}_top{args.n_options}_{args.action_representation}')

    print ('train_metadata', args.train_metadata)
    print ('out_folder', args.out_folder)
    print ('loading predictions from ', args.avion_train_predictions)
    print ('gen_type is ', args.gen_type)
    print ('n_options', args.n_options)

    os.makedirs(inst_train_folder, exist_ok=True)    

    anno_path = Path(ann_file).parent
    labels, mapping_vn2narration, mapping_vn2act, verb_maps, noun_maps = generate_label_map(anno_path, args.action_representation)
    conv_lst = generate_train_ann(ann_file,
                                  labels,
                                  mapping_vn2narration,
                                  verb_maps, 
                                  noun_maps, 
                                  gen_type = args.gen_type, 
                                  avion_prediction_path = args.avion_train_predictions,
                                  n_options = args.n_options,
                                  action_representation = args.action_representation,
                                  n_narrations = args.n_narrations)
        
    # save it to a jsonl
    with open(os.path.join(inst_train_folder,'train_convs_narration.jsonl'), 'w') as f:
        for conv in conv_lst:
            f.write(json.dumps(conv) + '\n')

   
if __name__ == "__main__":
    #main()
    
    # reason_path = "/storage-rcp-pure/upmwmathis_scratch/shaokai/train_anno_gpt-gt-reason_4_all.jsonl"
    # mc_path = "/storage-rcp-pure/upmwmathis_scratch/shaokai/EK100_inst_train/avion_mc_top5_GT_random_narration/train_convs_narration.jsonl"
    # out_folder = "/storage-rcp-pure/upmwmathis_scratch/shaokai/EK100_inst_train/avion_mc_top5_GT_random_narration_cot/"
    # combine_reason_and_mc(reason_path, mc_path, out_folder)

    data_root = '/data/EK100/EK100_320p_15sec_30fps_libx264'
    ann_file_path = '/data/epic_kitchen/epic-kitchens-100-annotations/EPIC_100_train.csv'
    hand_obj_root = '/data/epic_kitchen/hand_obj_anns/'
    out_image_folder = ''

    generate_hand_object_instruction_tuning_data(data_root, ann_file_path, hand_obj_root, out_image_folder)