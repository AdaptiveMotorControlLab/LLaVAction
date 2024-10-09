import json
import csv
import os

def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)

def generate_label_map(dataset):
    if dataset == 'ek100_cls':
        print("Preprocess ek100 action label space")
        vn_list = []
        mapping_vn2narration = {}
        verb_ids = {}
        noun_ids = {}
        for f in [
            '/data/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv',
            '/data/shaokai/epic-kitchens-100-annotations/EPIC_100_validation.csv',
        ]:
            csv_reader = csv.reader(open(f))
            _ = next(csv_reader)  # skip the header
            for row in csv_reader:
                vn = '{}:{}'.format(int(row[10]), int(row[12]))
                narration = row[8]
                if row[10] not in verb_ids.keys():
                    verb_ids[row[10]] = row[9]
                if row[12] not in noun_ids.keys():
                    noun_ids[row[12]] = row[11]
                if vn not in vn_list:
                    vn_list.append(vn)
                if vn not in mapping_vn2narration:
                    mapping_vn2narration[vn] = [narration]
                else:
                    mapping_vn2narration[vn].append(narration)
                # mapping_vn2narration[vn] = [narration]
        vn_list = sorted(vn_list)
        print('# of action= {}'.format(len(vn_list)))
        mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
        labels = [list(set(mapping_vn2narration[vn_list[i]])) for i in range(len(mapping_vn2act))]
        print(labels[:5])
    elif dataset == 'charades_ego':
        print("=> preprocessing charades_ego action label space")
        vn_list = []
        labels = []
        with open('datasets/CharadesEgo/CharadesEgo/Charades_v1_classes.txt') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                vn = row[0][:4]
                vn_list.append(vn)
                narration = row[0][5:]
                labels.append(narration)
        mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
        print(labels[:5])
    elif dataset == 'egtea':
        print("=> preprocessing egtea action label space")
        labels = []
        with open('datasets/EGTEA/action_idx.txt') as f:
            for row in f:
                row = row.strip()
                narration = ' '.join(row.split(' ')[:-1])
                labels.append(narration.replace('_', ' ').lower())
                # labels.append(narration)
        mapping_vn2act = {label: i for i, label in enumerate(labels)}
        print(len(labels), labels[:5])
    else:
        raise NotImplementedError
    return labels, mapping_vn2act, verb_ids, noun_ids


def parse_train_ann(ann_file, verb_ids, noun_ids):
    # epic kitchen uses csv
    csv_reader = csv.reader(open(ann_file))
    _ = next(csv_reader)
    ret = []
    for row in csv_reader:
        # start_frame, end_frame = row[6], row[7]
        start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])
        narration = f'{verb_ids[row[10]]} {noun_ids[row[12]]}'
        pid, vid = row[1:3]
        vid_path = '{}-{}'.format(pid, vid)        
        conversation = generate_naive_conversation(narration)
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

def generate_naive_conversation(narration):
    # in this version, we do not care about diversifying the questions
    return [
        {"from": "human", "value": "<image>\n the video is taken from egocentric view. What action is the person performing? Hint: provide your answer in verb-noun pair. "},
        {"from": "gpt", "value": f"{narration}"}    
    ]

def main():
    
    ann_file = "/data/shaokai/epic-kitchens-100-annotations/EPIC_100_train.csv"
    labels, mapping_vn2act, verb_ids, noun_ids = generate_label_map('ek100_cls')
    conv_lst = parse_train_ann(ann_file, verb_ids, noun_ids)
    inst_train_folder = '/data/shaokai/EK100_in_LLAVA/'
    os.makedirs(inst_train_folder, exist_ok=True)

    # save it to a jsonl
    with open(os.path.join(inst_train_folder,'train_convs_narration.jsonl'), 'w') as f:
        for conv in conv_lst:
            f.write(json.dumps(conv) + '\n')

   

if __name__ == "__main__":
    main()