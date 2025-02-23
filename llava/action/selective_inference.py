"""
Instead of running the whole validation set, 
"""
from llava.action.ek_eval import prepare_llava
from llava.action.generate_interval_pred import  get_lookup_dict
from llava.action.llava_inference import llava_inference

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
# val_metadata = '/data/shaokai/epic-kitchens-100-annotations/EPIC_100_validation.csv'  
# root = '/data/shaokai/EK100_512/EK100'
val_metadata = '/iopsstor/scratch/cscs/hqi/VFM/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv'
root = '/iopsstor/scratch/cscs/hqi/VFM/onevision/EK100_512/EK100'

n_frames = 32
action_representation = 'GT_random_narration'
perspective = 'first_person'
include_time_instruction = False
image_token = DEFAULT_IMAGE_TOKEN



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
    return frames, time_meta   
#

 
                        
                        
                        

# for prior actions
def get_meta_data():
    pass


def inference_task_by_uid(question, checkpoint_folder, uid, task):
    
    tokenizer, model, image_processor, max_length = prepare_llava(checkpoint_folder)
    
    frames, time_meta = get_frames_by_uid(uid, root)
    
    meta_data = None
    learn_neighbor_actions = ""
    if 'temporal_cot' in task:
        lookup_table = get_lookup_dict(val_metadata, 
                        action_representation,
                        test_type = task, 
                        pseudo_folder = '')
        meta_data = lookup_table.get(uid, None)
        learn_neighbor_actions = "prior"
    
    video_duration = time_meta['duration']
            
        
    pred = llava_inference(
                        [frames], 
                        tokenizer, 
                        model, 
                        image_processor,  
                        question,  
                        test_type = task,
                        clip_length = n_frames, 
                        num_frames= n_frames, 
                        temperature = 0,
                        time_meta = time_meta,
                        learn_neighbor_actions = learn_neighbor_actions,
                        meta_data = meta_data,
                        perspective = perspective,
                        include_time_instruction = include_time_instruction
                        )
    print (pred)
    
if __name__ == '__main__':
    pretrained_model_folder = 'experiments/dev_LLaVA-Video-7B-Qwen2_64f_top5_gpt4o_avion_tim_last_layer_one_token_detection_direct_neighbor_178K_100percent_time'
    uid = 'P28-P28_15_50.66_51.69'
    task = 'open-ended'
    question = "What is the object that is to the left of the knife?"
    
    inference_task_by_uid(question,
                          pretrained_model_folder,
                          uid,
                          task)