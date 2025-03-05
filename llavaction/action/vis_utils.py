import csv
from llavaction.action.dataset import datetime2sec
from llavaction.action.utils import generate_label_map  
from pathlib import Path
import glob
import os
import json


class Prediction:
    def __init__(self, folder = None, file = None, type = 'gpt', task = 'mqa'):
        """
        uid:{"pred": "",
        "options": ["", "", "", "", ""],
        "gt": ""
        }                        
        """
        if type == 'gpt':
            self.data = self.load_from_gpt_file(file)
        if type == 'llava':
            self.data = self.load_from_llava_folder(folder)
    def load_from_gpt_file(self, file):
        """
        "1212": {"uid": xx,
        "gt_name": "",
        "options": ["", "", "", "", ""],
        "chatgpt_answer": ""
        }
        """
        with open(file, 'r') as f:
            data = json.load(f)
        ret = {}
        for k,v in data.items():
            uid = v['uid']
            pred = v['chatgpt_answer']
            options = v['options']
            ret[uid] = {'pred': pred, 'options': options,
                        "gt": v['gt_name']}
        print ('len', len(ret))
        return ret 
                
    def __getitem__(self, key):

        return self.data[key]
       
    def load_from_llava_folder(self, folder):
        """
        "9667": {"llava_pred", "",
        "gt_name": "",
        "avion_preds": [],
        "start_second": 0.0,
        "end_second": 0.0,
        "vid_path": ""
        """
        files = glob.glob(os.path.join(folder, '*.json'))
        ret = {}
        data = {}
        for file in files:
            with open(file, 'r') as f:
                data.update(json.load(f))
        sorted_keys = sorted(data.keys(), key = lambda k: int(k))
        for key in sorted_keys:
            value = data[key]
            start_second = value['start_second']
            end_second = value['end_second']
            vid_path = value['vid_path']
            gt_name = value['gt_name']
            left = vid_path.split('/')[0]
            right = vid_path.split('/')[1]
            uid = f'{left}-{right}_{start_second}_{end_second}'
           
            options = value["avion_preds"]

            ret[uid] = {'pred': value['llava_pred'], 'options': options,
                        'gt': gt_name}
        print ('len', len(ret))
        return ret

def get_uid_options_map_from_prediction_folder(uid, prediction_folder):
    """
    look for where llava makes mistakes
    """
    files = glob.glob(os.path.join(prediction_folder, '*.json'))
    ret = {}
    data = {}
    for file in files:
        with open(file, 'r') as f:
            data.update(json.load(f))
    sorted_keys = sorted(data.keys(), key = lambda k: int(k))
    for key in sorted_keys:
        value = data[key]
        start_second = value['start_second']
        end_second = value['end_second']
        vid_path = value['vid_path']
        gt_name = value['gt_name']
        left = vid_path.split('/')[0]
        right = vid_path.split('/')[1]
        uid = f'{left}-{right}_{start_second}_{end_second}'
        options = value["avion_preds"]
        ret[uid] = options   
    return ret
def get_uid_official_map(ann_file):
    csv_reader = csv.reader(open(ann_file, 'r'))
    _ = next(csv_reader)
    anno_root = Path(ann_file).parent
    labels, mapping_vn2narration, mapping_vn2act, verb_maps, noun_maps = generate_label_map(anno_root,
                                                                                            'official_key')    
    ret = {}
    for idx, row in enumerate(csv_reader):
        pid, vid = row[1:3]
        
        start_second, end_second = datetime2sec(row[4]), datetime2sec(row[5])
        start_second = round(float(start_second),2)
        end_second = round(float(end_second),2)
        vid_path = '{}/{}'.format(pid, vid)
        left = vid_path.split('/')[0]
        right = vid_path.split('/')[1]
        uid = f'{left}-{right}_{start_second}_{end_second}'
                  
        verb, noun = int(row[10]), int(row[12])
        gt_vn = '{}:{}'.format(verb, noun) 
        narration = row[8]
        official_key = verb_maps[str(verb)] + ' ' + noun_maps[str(noun)]
        ret[uid] = official_key
    return ret

def get_uid_narration_map(ann_file):
    csv_reader = csv.reader(open(ann_file, 'r'))
    _ = next(csv_reader)
    anno_root = Path(ann_file).parent
    labels, mapping_vn2narration, mapping_vn2act, verb_maps, noun_maps = generate_label_map(anno_root,
                                                                                            'official_key')    
    ret = {}
    for idx, row in enumerate(csv_reader):
        pid, vid = row[1:3]
        
        start_second, end_second = datetime2sec(row[4]), datetime2sec(row[5])
        start_second = round(float(start_second),2)
        end_second = round(float(end_second),2)
        vid_path = '{}/{}'.format(pid, vid)
        left = vid_path.split('/')[0]
        right = vid_path.split('/')[1]
        uid = f'{left}-{right}_{start_second}_{end_second}'
                  
        verb, noun = int(row[10]), int(row[12])
        gt_vn = '{}:{}'.format(verb, noun) 
        narration = row[8]
        official_key = verb_maps[str(verb)] + ' ' + noun_maps[str(noun)]
        ret[uid] = narration
    return ret



def get_narration_by_uid(uid, ann_file):
    csv_reader = csv.reader(open(ann_file, 'r'))
    _ = next(csv_reader)
    anno_root = Path(ann_file).parent
    labels, mapping_vn2narration, mapping_vn2act, verb_maps, noun_maps = generate_label_map(anno_root,
                                                                                            'official_key')    
    ret = {}
    for idx, row in enumerate(csv_reader):
        pid, vid = row[1:3]
        
        start_second, end_second = datetime2sec(row[4]), datetime2sec(row[5])
        start_second = round(float(start_second),2)
        end_second = round(float(end_second),2)
        vid_path = '{}/{}'.format(pid, vid)
        left = vid_path.split('/')[0]
        right = vid_path.split('/')[1]
        uid = f'{left}-{right}_{start_second}_{end_second}'
                  
        verb, noun = int(row[10]), int(row[12])
        gt_vn = '{}:{}'.format(verb, noun) 
        narration = row[8]
        official_key = verb_maps[str(verb)] + ' ' + noun_maps[str(noun)]
        ret[uid] = official_key
    return ret    



    
def get_uid_official_map(uid, ann_file):
    csv_reader = csv.reader(open(ann_file, 'r'))
    _ = next(csv_reader)
    anno_root = Path(ann_file).parent
    labels, mapping_vn2narration, mapping_vn2act, verb_maps, noun_maps = generate_label_map(anno_root,
                                                                                            'GT_random_narration')    
    ret = {}
    for idx, row in enumerate(csv_reader):
        pid, vid = row[1:3]
        
        start_second, end_second = datetime2sec(row[4]), datetime2sec(row[5])
        start_second = round(float(start_second),2)
        end_second = round(float(end_second),2)
        vid_path = '{}/{}'.format(pid, vid)
        left = vid_path.split('/')[0]
        right = vid_path.split('/')[1]
        uid = f'{left}-{right}_{start_second}_{end_second}'
                  
        verb, noun = int(row[10]), int(row[12])
        gt_vn = '{}:{}'.format(verb, noun) 
        narration = row[8]
        official_key = verb_maps[str(verb)] + ' ' + noun_maps[str(noun)]
        ret[uid] = official_key
    return ret   



 
def compare_caption_generation(chatgpt_file, llava_file):
    "Do we have llava file for this yet?"
    pass 
 
def compare_open_ended_question_answering(chatgpt_file, llava_file):
    "Do we have llava file for this yet?"
    pass

def search_llavaction_win(tim_chatgpt_file, 
                    random_chatgpt_file, 
                    llava_zeroshot_folder, 
                    llavaction_folder):
    # it has to be: chatgpt wrong -> need chatgpt prediction folder
    # and llava zeroshot wrong -> need access to zeroshot 
    # and llavaction gets right: need access to ours
    # note that they have to have same number of options
    tim_chatgpt_pred = Prediction(file = tim_chatgpt_file, type = 'gpt')
    random_chatgpt_pred = Prediction(file = random_chatgpt_file, type = 'gpt')
    llava_pred = Prediction(folder = llava_zeroshot_folder, type = 'llava')
    llavaction_pred = Prediction(folder = llavaction_folder, type = 'llava')
    results = {}
    for uid in tim_chatgpt_pred.data.keys():
        tim_chatgpt_options = tim_chatgpt_pred[uid]['options']
        random_chatgpt_options = random_chatgpt_pred[uid]['options']
        if uid not in llava_pred.data:
            continue
        llava_options = llava_pred[uid]['options']
        llavaction_options = llavaction_pred[uid]['options']
        if llavaction_pred[uid]['pred'] == llavaction_pred[uid]['gt'] and \
            tim_chatgpt_pred[uid]['pred'] != tim_chatgpt_pred[uid]['gt'] and \
            llava_pred[uid]['pred'] != llava_pred[uid]['gt'] and \
                random_chatgpt_pred[uid]['pred'] == random_chatgpt_pred[uid]['gt']:        
        
            results[uid] = {'gt': tim_chatgpt_pred[uid]['gt'],
                            'tim_chatgpt_pred': tim_chatgpt_pred[uid]['pred'],
                            'random_chatgpt_pred': random_chatgpt_pred[uid]['pred'],
                            'llava_pred': llava_pred[uid]['pred'],
                            'llavaction_pred': llavaction_pred[uid]['pred'],
                            'tim_chatgpt_options': tim_chatgpt_options,
                            'llava_options': llava_options,
                            'llavaction_options': llavaction_options,
                            'random_chatgpt_options': random_chatgpt_options}
    # write results to a file
    with open('llavaction_win.json', 'w') as f:
        json.dump(results, f, indent = 4)

def get_wrong_prediction_uids(prediction_folder, ann_file):
    """
    look for where llava makes mistakes
    """
    files = glob.glob(os.path.join(prediction_folder, '*.json'))
    uid_narration_map = get_uid_narration_map(ann_file)
    
    data = {}
    for file in files:
        with open(file, 'r') as f:
            data.update(json.load(f))
    sorted_keys = sorted(data.keys(), key = lambda k: int(k))
    results = {}
    for key in sorted_keys:
        value = data[key]
        start_second = value['start_second']
        end_second = value['end_second']
        vid_path = value['vid_path']
        gt_name = value['gt_name']
        official_key = value['gt_name']
        left = vid_path.split('/')[0]
        right = vid_path.split('/')[1]
        uid = f'{left}-{right}_{start_second}_{end_second}'
        narration = uid_narration_map[uid]
       
        if value['gt_name'] not in value['avion_preds']:
            continue
        
        if value['llava_pred'] != value['gt_name']:
            print ('uid', uid)
            print ('llava_pred', value['llava_pred'])
            print ('official key gt', gt_name)
            print ('gt narration', narration)
            print ('options', value['avion_preds'])
            # put everything i printed in a dictionary
            results[uid] = {'llava_pred': value['llava_pred'],
                            'gt': gt_name,
                            'narration': narration,
                            'official_key': official_key,
                            'options': value['avion_preds']}
       
        # write results to a file
    with open('llava_gets_confused_by_key.json', 'w') as f:
        json.dump(results, f, indent = 4)

def walk_through(ann_file):
    csv_reader = csv.reader(open(ann_file, 'r'))
    _ = next(csv_reader)
    anno_root = Path(ann_file).parent
    labels, mapping_vn2narration, mapping_vn2act, verb_maps, noun_maps = generate_label_map(anno_root,
                                                                                            'official_key')    
    count = 0
    for idx, row in enumerate(csv_reader):
        pid, vid = row[1:3]
        
        start_second, end_second = datetime2sec(row[4]), datetime2sec(row[5])
        start_second = round(float(start_second),2)
        end_second = round(float(end_second),2)
        vid_path = '{}/{}'.format(pid, vid)
        left = vid_path.split('/')[0]
        right = vid_path.split('/')[1]
        uid = f'{left}-{right}_{start_second}_{end_second}'
                  
        verb, noun = int(row[10]), int(row[12])
        gt_vn = '{}:{}'.format(verb, noun) 
        narration = row[8]
        print ('----')
        print ('uid', uid)
        noun = noun_maps[str(noun)]
        verb = verb_maps[str(verb)]
        print ('official key', f'{verb}:{noun}')
        print ('narration', narration)
        print ('----')
        count+=1

        
if __name__ == '__main__':
    ann_file = '/data/anonymous/epic-kitchens-100-annotations/EPIC_100_validation.csv'
    prediction_folder = '/data/anonymous/predictions_for_vis/dev_7b_16f_top5_full_includes_tim/'
    #walk_through(ann_file)
    #get_wrong_prediction_uids(prediction_folder, ann_file)
    root = '/data/anonymous/predictions_for_vis/'
    chatgpt_tim_file = os.path.join(root, 'gpt-4o-2024-08-06_tim_GT_random_narration_top5_8f_9668samples.json')
    chatgpt_random_file = os.path.join(root, 'gpt-4o-2024-08-06_random_GT_random_narration_top5_8f_9668samples.json')
    llava_zeroshot_folder = os.path.join(root, 'LLaVA_Video_7B')
    llavaction_folder = os.path.join(root, 'LLaVAction_7B')
    search_llavaction_win(chatgpt_tim_file, 
                    chatgpt_random_file, 
                    llava_zeroshot_folder, 
                    llavaction_folder)
                                