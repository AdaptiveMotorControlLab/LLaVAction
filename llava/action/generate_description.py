from __future__ import annotations
import json
import csv
import os
import argparse
import sys
from llava.action.utils import generate_label_map, MultiChoiceGenerator, AvionMultiChoiceGenerator, format_task_related_prompt
from pathlib import Path


GEN_TYPES = ['naive', 'random_mc', 'avion_mc']

def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)

def generate_train_ann(ann_file, verb_ids, noun_ids, gen_type = 'naive', avion_prediction_path = '', n_options = 5):
    assert gen_type in GEN_TYPES
    # epic kitchen uses csv
    csv_reader = csv.reader(open(ann_file))
    _ = next(csv_reader)
    ret = []
    ann_root = Path(ann_file).parent
    if gen_type == "random_mc":
        mc_generator = MultiChoiceGenerator(ann_root)
    elif gen_type == 'avion_mc':
        mc_generator = AvionMultiChoiceGenerator(ann_root)
        with open(avion_prediction_path, 'r') as f:
            avion_train_predictions = json.load(f)

    for idx, row in enumerate(csv_reader):
        start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])
        
        pid, vid = row[1:3]
        vid_path = '{}-{}'.format(pid, vid)

        if gen_type == 'naive':
            # here we directly use the names
            verb_noun = f'{verb_ids[row[10]]} {noun_ids[row[12]]}'
            conversation = generate_naive_conversation(verb_noun)
        elif gen_type == "random_mc":
            # here we use the index
            vn_str = f'{row[10]}:{row[12]}'
            mc_data = mc_generator.generate_multi_choice(vn_str, n_options)
            options = mc_data['options'][0]
            gt_answer_letter = mc_data['gt_answer_letter'][0]
            gt_answer_name = mc_data['gt_answer_name'][0]
            conversation = generate_random_mc_conversation(options, gt_answer_letter, gt_answer_name )
        elif gen_type == "avion_mc":
            vn_str = f'{row[10]}:{row[12]}'
            avion_preds = avion_train_predictions[str(idx)]['predictions']
            gt_from_avion = avion_train_predictions[str(idx)]['target']
            mc_data = mc_generator.generate_multi_choice(vn_str, avion_preds, n_options)
            options = mc_data['options'][0]
            gt_answer_letter = mc_data['gt_answer_letter'][0]
            gt_answer_name = mc_data['gt_answer_name'][0]
            assert gt_answer_name.replace(' ', ':') == gt_from_avion
            conversation = generate_random_mc_conversation(options, gt_answer_letter, gt_answer_name )

        data = {'video': vid_path,
                'conversations': conversation,
                'id': vid_path,
                'split': 'train',
                'task_instruction': '',
                'num_samples': 1,
                'question_type': 'open-ended',
                'dataset_name': 'EK100',
                'start_timestamp': start_timestamp,
                'end_timestamp': end_timestamp}
        ret.append(data)
    return ret

def generate_naive_conversation(vn_str:str):
    # in this version, we do not care about diversifying the questions
    return [
        {"from": "human", "value": "<image>\n the video is taken from egocentric view. What action is the person performing? Hint: provide your answer in verb-noun pair. "},
        {"from": "gpt", "value": f"{vn_str}"}    
    ]

def generate_random_mc_conversation(options:list[str], gt_answer_letter, gt_answer_name):
    return [
        {"from": "human", "value": f"{options}"},
        {"from": "gpt", "value": f"{gt_answer_letter}. {gt_answer_name}"} 
    ]


def get_args():
    parser = argparse.ArgumentParser(description="For generating VQA for EPIC-KITCHEN")
    parser.add_argument('--train_metadata', default='/data/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv', type=str)
    parser.add_argument('--out_folder', default = '/data/shaokai/EK100_in_LLAVA/', type = str)
    parser.add_argument('--avion_train_predictions', default = '/data/shaokai/avion_predictions_train.json', type = str)
    parser.add_argument('--gen_type', default = 'avion_mc', type = str, choices = GEN_TYPES)
    parser.add_argument('--n_options', default = 5, type = int)
    return parser.parse_args()

def main(): 
    args = get_args()    
    ann_file = args.train_metadata
    inst_train_folder = os.path.join(args.out_folder, f'{args.gen_type}_top{args.n_options}')

    print ('train_metadata', args.train_metadata)
    print ('out_folder', args.out_folder)
    print ('loading predictions from ', args.avion_train_predictions)
    print ('gen_type is ', args.gen_type)
    print ('n_options', args.n_options)

    os.makedirs(inst_train_folder, exist_ok=True)    

    anno_path = Path(ann_file).parent
    _, _, verb_ids, noun_ids = generate_label_map(anno_path)
    conv_lst = generate_train_ann(ann_file, 
                                  verb_ids, 
                                  noun_ids, 
                                  gen_type = args.gen_type, 
                                  avion_prediction_path = args.avion_train_predictions,
                                  n_options = args.n_options)
        
    # save it to a jsonl
    with open(os.path.join(inst_train_folder,'train_convs_narration.jsonl'), 'w') as f:
        for conv in conv_lst:
            f.write(json.dumps(conv) + '\n')

   
if __name__ == "__main__":
    main()