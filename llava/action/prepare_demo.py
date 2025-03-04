import json
from llava.action.make_visualizations import visualize_with_uid
from llava.action.selective_inference import SelectiveInferencer
import random
import os
# 1) iterate through llava_win json, retrieve a list of uids
# 2) save the corresponding video clips
# 3) add caption and free-end question answering 
# after 1) and 3), there should be one single json file that uses uid as the key
# and it contains: caption (chatgpt, llavaction), mqa (chatgpt, llavaction, gt)


def load_llava_wins(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_video_clips_with_uids(data_root, llava_win_path, vis_folder, checkpoint_folder):
    llava_wins = load_llava_wins(llava_win_path)
    uids = list(llava_wins.keys())
    random.shuffle(uids)
    sample_uids = uids[:20]  
    ret = {}  
    inferencer = SelectiveInferencer(data_root, 
                                     checkpoint_folder,
                                     include_time_instruction = False,
                                     n_frames = 32)
    count = 0
    for uid in sample_uids:
        if count > 10:
            break        
        data = llava_wins[uid]
        if data['tim_chatgpt_pred'] not in data['llavaction_options']:
            continue
        data.pop('llava_pred')
        data.pop('llava_options')
        #data.pop('tim_chatgpt_pred')
        data.pop('random_chatgpt_pred')
        data.pop('tim_chatgpt_options')
        data.pop('random_chatgpt_options')
        visualize_with_uid(data_root, uid, vis_folder)
        open_ended = get_open_ended_question(inferencer, uid, checkpoint_folder)
        caption = get_caption(inferencer, uid, checkpoint_folder)
        data['open_ended'] = open_ended
        data['caption'] = caption
        ret[uid] = data
        count+=1
    with open('demo_videos/demo.json', 'w') as f:
        json.dump(ret, f, indent=4)

def get_open_ended_question(inferencer,
                            uid, 
                            checkpoint_folder):
    mqa =  inferencer.inference('what objects are visible in the video?', 
                                 uid, 
                                 'open-ended')
    return mqa
        
def get_caption(inferencer,
                uid, 
                checkpoint_folder):
    caption =  inferencer.inference('', 
                                     uid, 
                                     'caption')
    return caption    


if __name__ == '__main__':
    llava_win_path = 'llavaction_win.json'
    vis_folder = 'demo_videos'
    os.makedirs(vis_folder, exist_ok = True)
    checkpoint_folder = 'experiments/dev_7b_16f_top5_strong_first_layer_three_tokens_detection_and_direct_llava_video_10percent/checkpoint-15000/'
    data_root = '/data/shaokai/EK100_512/EK100'
    save_video_clips_with_uids(data_root, 
                               llava_win_path, 
                               vis_folder, 
                               checkpoint_folder)