"""
Instead of running the whole validation set, 
"""
from llava.action.ek_eval import prepare_llava
from llava.action.generate_interval_pred import  get_lookup_dict
from llava.action.inference import llava_inference
 
val_metadata = '/data/shaokai/epic-kitchens-100-annotations/EPIC_100_validation.csv'  
root = '/data/shaokai/EK100_512/EK100'
n_frames = 32
action_representation = 'GT_random_narration'

def get_frames_by_uid(uid, root):
    from llava.action.utils import avion_video_loader
    vid_path = '_'.join(uid.split('_')[:2]).replace('-', '/')
    start_timestamp, end_timestamp = uid.split('_')[2:]
    start_timestamp = float(start_timestamp)
    end_timestamp = float(end_timestamp)
    print (vid_path, start_timestamp, end_timestamp)
    # split uid to video path and start, end second
    frames, time_meta = avion_video_loader(root,
                                           vid_path,
                                           'MP4',
                                            start_timestamp,
                                            end_timestamp,
                                            chunk_len = 15,
                                            clip_length = n_frames,
                                            threads = 1,
                                            fast_rrc=False,
                                            fast_rcc = False,
                                            jitter = False) 
    return frames   

def inference_task_by_uid(checkpoint_folder, uid, task):
    
    tokenizer, model, image_processor, max_length = prepare_llava(checkpoint_folder)
    
    frames = get_frames_by_uid(uid, root)

    if 'temporal_cot' in task:
        get_lookup_dict(val_metadata, 
                        action_representation,
                        test_type = task, 
                        pseudo_folder = '')
        pred = llava_inference(
                            frames, 
                            tokenizer, 
                            model, 
                            image_processor,  
                            mc_data,  
                            test_type = test_type,
                            clip_length = clip_length, 
                            num_frames=num_frames, 
                            temperature = temperature,
                            time_meta = time_meta,
                            learn_neighbor_actions = learn_neighbor_actions,
                            meta_data = meta_data,
                            perspective = perspective,
                            include_time_instruction = include_time_instruction
                            )