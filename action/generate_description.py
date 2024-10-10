import json
import csv
import os
import argparse
from action.utils import generate_label_map, MultiChoiceGenerator
from pathlib import Path


GEN_TYPES = ['naive', 'random_mc', 'avion_mc']

def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)

def generate_train_ann(ann_file, verb_ids, noun_ids, gen_type = 'naive'):
    assert gen_type in GEN_TYPES
    # epic kitchen uses csv
    csv_reader = csv.reader(open(ann_file))
    _ = next(csv_reader)
    ret = []
    ann_root = Path(ann_file).parent
    if gen_type == "random_mc":
        mc_generator = MultiChoiceGenerator(ann_root)

    for row in csv_reader:
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
            mc_data = mc_generator.generate_multi_choice(vn_str, 5)
            options = mc_data['option'][0]
            gt_answer_letter = mc_data['gt_answer_letter'][0]
            gt_answer_name = mc_data['gt_answer_name'][0]
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
        {"from": "human", "value": f"<image>\n the video is taken from egocentric view. What action is the person performing? Please select the letter for the right answer {options}"},
        {"from": "gpt", "value": f"{gt_answer_letter}. {gt_answer_name}"} 
    ]


def get_args():
    parser = argparse.ArgumentParser(description="For generating VQA for EPIC-KITCHEN")
    parser.add_argument('--train_metadata', default='/data/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv', type=str)
    parser.add_argument('--out_folder', default = '/data/shaokai/EK100_in_LLAVA/', type = str)
    return parser.parse_args()

def main():    
    args = get_args()    
    ann_file = args.train_metadata
    inst_train_folder = args.out_folder
    print (ann_file)
    anno_path = Path(ann_file).parent
    labels, mapping_vn2act, verb_ids, noun_ids = generate_label_map(anno_path)
    conv_lst = generate_train_ann(ann_file, verb_ids, noun_ids, gen_type = 'random_mc')
    
    os.makedirs(inst_train_folder, exist_ok=True)

    # save it to a jsonl
    with open(os.path.join(inst_train_folder,'train_convs_narration.jsonl'), 'w') as f:
        for conv in conv_lst:
            f.write(json.dumps(conv) + '\n')

   
if __name__ == "__main__":
    main()